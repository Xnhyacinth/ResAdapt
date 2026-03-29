import argparse
import os
import re
import numpy as np
import torch
import torchvision
from PIL import Image
import datasets
from datasets import Dataset, DatasetDict
from qwen_vl_utils import fetch_video

# Note: 'verl.utils.hdfs_io' is a custom internal library.
try:
    from verl.utils.hdfs_io import copy, makedirs
except ImportError:
    def copy(src, dst): pass
    def makedirs(path): pass

def extract_answer_content(text):
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.S)
    if match:
        return match.group(1).strip()  
    else:
        return ""

# [NEW] Helper function to convert video tensor to list of PIL Images
def tensor_to_pil_list(video_tensor):
    """
    Args:
        video_tensor: (T, C, H, W) or (T, H, W, C) tensor, range [0, 255] or [0, 1]
    Returns:
        List[PIL.Image]
    """
    images = []
    # torchvision read_video returns (T, H, W, C) in [0, 255]
    if video_tensor.ndim == 4:
        # Check if channel is last (T, H, W, C)
        if video_tensor.shape[-1] == 3:
            # Permute to (T, C, H, W) for torchvision transform if needed, 
            # OR just use numpy/PIL directly which expects (H, W, C)
            pass
        elif video_tensor.shape[1] == 3:
            # (T, C, H, W) -> (T, H, W, C)
            video_tensor = video_tensor.permute(0, 2, 3, 1)
            
    for frame in video_tensor:
        # frame is (H, W, C)
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
                
        img = Image.fromarray(frame)
        images.append(img)
    return images

# [NEW] Function to load video
def load_video_frames(video_path, min_frames=1, max_frames=64):
    try:
        # fetch_video usually returns a local path if it's a file
        videos = fetch_video({"video": video_path, "min_frames": 1, "max_frames": 32, "nframes": 4})
        
        # Use torchvision to read video. 
        # returns: vframes (T, H, W, C), aframes, info
        # pts_unit='sec' is more robust
        # vframes, _, info = torchvision.io.read_video(local_path, pts_unit='sec', output_format="TCHW")
        
        # Sampling logic (simple uniform sampling)
        # total_frames = vframes.shape[0]
        # if total_frames > max_frames:
        #     indices = torch.linspace(0, total_frames - 1, max_frames).long()
        #     vframes = vframes[indices]
        
        return videos # (T, C, H, W)
        
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default="/mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/TSPO-10K/TSPO_10k.jsonl", help="The local path to the JSONL dataset file.")
    parser.add_argument("--local_save_dir", default="/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-4frames", help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None, help="Optional: HDFS directory to copy the final dataset to.")
    parser.add_argument("--num_proc", type=int, default=16, help="Number of processes for the map function.")
    parser.add_argument("--val_size", type=int, default=500, help="Number of samples to use for validation.")
    parser.add_argument("--save_images_to_disk", action="store_true", help="Save frames as image files and store paths instead of PIL images.")
    parser.add_argument("--image_dir", default=None, help="Directory to save frames when --save_images_to_disk is enabled.")

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    print(f"Loading dataset from JSONL: {local_dataset_path}")
    dataset = datasets.load_dataset("json", data_files=local_dataset_path)

    if isinstance(dataset, DatasetDict):
        full_dataset = dataset["train"]
    elif isinstance(dataset, Dataset):
        full_dataset = dataset
    else:
        raise TypeError(f"Loaded object is of an unexpected type: {type(dataset)}")

    print(f"Total samples loaded: {len(full_dataset)}")

    print(f"Splitting dataset: reserving {args.val_size} samples for validation...")
    split_dataset = full_dataset.train_test_split(test_size=args.val_size, seed=42)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags and the answer MUST BE enclosed within <answer> </answer> tags. "
        r"The final answer MUST BE put in \boxed{} and the \boxed{} expression MUST BE contained entirely within the <answer> </answer> tags. "
        r"Do not include any reasoning or explanations outside these tags."
    )

    def make_map_fn(split):
        def process_fn(example, idx):
            video_path = example.pop("video", None) 
            if video_path is None:
                video_path = example.pop("video_path", None)
            
            pil_images = []
            image_paths = []
            
            # [MODIFIED] Load video and convert to images immediately
            if video_path:
                full_path = os.path.join("/mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/LLaVA-Video-178K", video_path)
                
                # Check exist
                if os.path.exists(full_path):
                    # Load frames
                    video_tensor = load_video_frames(full_path, max_frames=32) # You can adjust max_frames
                    if video_tensor is not None:
                        pil_images = tensor_to_pil_list(video_tensor)
                        print(len(pil_images))
                else:
                    # print(f"Warning: Video not found {full_path}")
                    pass
            
            # Construct Videos list (Empty now, as we use images)
            videos = [] 

            problem = example.pop("problem", None)
            if problem is None:
                problem = example.pop("original_question", "") 
            
            # [MODIFIED] Replace <image> or <video> with multiple <image> tags based on frame count
            if problem:
                # Normalize placeholders to <image>
                # problem = problem.replace("<video>", "<image>")
                
                if len(pil_images) > 0:
                    # Construct string like "<image><image><image>..."
                    image_tokens = "<image>" * len(pil_images)
                    # Replace the single existing <image> tag with multiple tags
                    # Use count=1 to replace only the first occurrence if multiple exist (unlikely for video tasks)
                    problem = problem.replace("<image>", image_tokens, 1)
                else:
                    # If no images loaded (e.g. text-only or missing video), remove the tag
                    problem = problem.replace("<image>", "")

            answer = example.pop("solution", None)
            if answer is None:
                answer = example.pop("original_answer", "")
            
            if answer:
                answer = extract_answer_content(answer)

            doc_id = example.pop("id", None)
            if doc_id in (None, ""):
                doc_id = str(idx)

            prompt = (problem or "") + " " + instruction_following

            if args.save_images_to_disk and pil_images:
                if args.image_dir is None:
                    raise ValueError("--image_dir must be set when --save_images_to_disk is enabled.")
                split_dir = os.path.join(args.image_dir, split)
                os.makedirs(split_dir, exist_ok=True)
                for frame_idx, img in enumerate(pil_images):
                    filename = f"{doc_id}_{frame_idx:04d}.png"
                    out_path = os.path.join(split_dir, filename)
                    img.save(out_path, format="PNG", quality=95)
                    image_paths.append(out_path)
                pil_images = []
            if args.save_images_to_disk:
                images_field = image_paths
            else:
                images_field = pil_images

            data = {
                "data_source": "TSPO-10K",
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],

                "images": images_field,   # PIL images or list of paths
                # "videos": videos,       # Now empty
                
                "ability": "video_reasoning", 
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                    "doc_id": doc_id,
                    "video_path": video_path if video_path else "",
                    **example,
                },
            }
            return data

        return process_fn

    # --- PROCESS TRAIN ---
    print("Starting dataset transformation for TRAIN...")
    # [CRITICAL] Set num_proc=1 if using simple threading within map, or ensure functions are picklable.
    # Reading video is heavy, multiple processes are recommended, 
    # but be careful with memory usage as PIL images are raw pixels in memory before parquet save.
    processed_train = train_dataset.map(
        function=make_map_fn("train"), 
        with_indices=True, 
        num_proc=args.num_proc, 
        writer_batch_size=100, # Reduce batch size to control memory
    )
    
    # --- PROCESS VAL ---
    print("Starting dataset transformation for VAL...")
    processed_val = val_dataset.map(
        function=make_map_fn("test"), 
        with_indices=True, 
        num_proc=args.num_proc,
        writer_batch_size=100,
    )

    print("Transformation complete.")

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    
    # Save the processed datasets to parquet files
    train_output_path = os.path.join(local_save_dir, "train.parquet")
    val_output_path = os.path.join(local_save_dir, "val.parquet")

    print(f"Saving processed TRAIN data to: {train_output_path}")
    processed_train.to_parquet(train_output_path)

    print(f"Saving processed VAL data to: {val_output_path}")
    processed_val.to_parquet(val_output_path)
    
    print("Save complete.")

    if args.hdfs_dir is not None:
        print(f"Copying data to HDFS at: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
        print("HDFS copy complete.")
