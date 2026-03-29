# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocesses a local Video JSONL dataset (TSPO-10K) into a standardized parquet format.
Splits 500 samples for validation.
"""

import argparse
import os
import re

import datasets
from datasets import Dataset, DatasetDict

# Note: 'verl.utils.hdfs_io' is a custom internal library.
# If you don't have this, you can comment out the HDFS related lines.
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path to your JSONL file
    parser.add_argument("--local_dataset_path", default="/mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/Video-R1-data/Video-R1-260k.json", help="The local path to the JSONL dataset file.")
    parser.add_argument("--local_save_dir", default="~/data/processed_video_data", help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None, help="Optional: HDFS directory to copy the final dataset to.")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes for the map function.")
    parser.add_argument("--val_size", type=int, default=500, help="Number of samples to use for validation.")

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    # Load the dataset from a local JSONL file
    print(f"Loading dataset from JSONL: {local_dataset_path}")
    dataset = datasets.load_dataset("json", data_files=local_dataset_path)

    # Standardize to a single Dataset object
    if isinstance(dataset, DatasetDict):
        full_dataset = dataset["train"]
    elif isinstance(dataset, Dataset):
        full_dataset = dataset
    else:
        raise TypeError(f"Loaded object is of an unexpected type: {type(dataset)}")

    print(f"Total samples loaded: {len(full_dataset)}")

    # --- SPLIT DATASET ---
    # Randomly shuffle and split the dataset
    print(f"Splitting dataset: reserving {args.val_size} samples for validation...")
    # seed=42 ensures reproducibility
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

    # This function now maps your specific fields to the target format
    def make_map_fn(split):
        def process_fn(example, idx):
            video_path = example.pop("video", None) 
            if video_path is None:
                video_path = example.pop("video_path", None)
            
            # Safe handling of video path join
            if video_path:
                video_path = os.path.join("/mnt/bn/jiangzhongtao/users/liaohuanxuan/vlm_datasets/LLaVA-Video-178K", video_path)
                # videos = [video_path]
                videos = [
                    {
                        "type": "video",
                        "video": video_path,
                        "min_frames": 1,
                        "max_frames": 32
                    }
                ]
            else:
                # Handle cases where video_path might be missing
                videos = []

            problem = example.pop("problem", None)
            if problem is None:
                problem = example.pop("original_question", "") # Fallback
            if problem:
                problem = problem.replace("<image>", "<video>")

            answer = example.pop("solution", None)
            if answer is None:
                answer = example.pop("original_answer", "") # Fallback
            
            if answer:
                answer = extract_answer_content(answer)

            doc_id = example.pop("id", str(idx)) 

            prompt = (problem or "") + " " + instruction_following

            data = {
                "data_source": "TSPO-10K",
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],

                "images": [], # Empty if no static images
                "videos": videos, # Path to video files
                
                "ability": "video_reasoning", 
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                    "doc_id": doc_id,
                    "video_path": video_path if video_path else "",
                    # Add any other remaining fields from the original example
                    **example,
                },
            }
            return data

        return process_fn

    # --- PROCESS TRAIN ---
    # print("Starting dataset transformation for TRAIN...")
    # processed_train = train_dataset.map(
    #     function=make_map_fn("train"), 
    #     with_indices=True, 
    #     num_proc=args.num_proc,
    # )
    
    # # --- PROCESS VAL ---
    # print("Starting dataset transformation for VAL...")
    # processed_val = val_dataset.map(
    #     function=make_map_fn("test"), 
    #     with_indices=True, 
    #     num_proc=args.num_proc,
    # )

    # print("Transformation complete.")

    # local_save_dir = os.path.expanduser(args.local_save_dir)
    # os.makedirs(local_save_dir, exist_ok=True)
    
    # # Save the processed datasets to parquet files
    # train_output_path = os.path.join(local_save_dir, "train.parquet")
    # val_output_path = os.path.join(local_save_dir, "val.parquet")

    # print(f"Saving processed TRAIN data to: {train_output_path}")
    # processed_train.to_parquet(train_output_path)

    # print(f"Saving processed VAL data to: {val_output_path}")
    # processed_val.to_parquet(val_output_path)
    
    # print("Save complete.")

    # # Optional: copy to HDFS
    # if args.hdfs_dir is not None:
    #     print(f"Copying data to HDFS at: {args.hdfs_dir}")
    #     makedirs(args.hdfs_dir)
    #     copy(src=local_save_dir, dst=args.hdfs_dir)
    #     print("HDFS copy complete.")

    dataframe = datasets.load_dataset("parquet", data_files="/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/videoautor1_mixed/train.parquet")["train"]
    # dataframe = datasets.load_dataset("parquet", data_files="/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/videoautor1_mixed/train.parquet")["train"]
    # dataframe = datasets.load_dataset("parquet", data_files="/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/VisionThink-Mixedpath/train.parquet")["train"]
    # dataframe = datasets.load_dataset("parquet", data_files="/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-10K/train.parquet")["train"]
    # dataframe = datasets.load_dataset("parquet", data_files="/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/TSPO-4frames-path/train.parquet")["train"]
    # dataframe = datasets.load_dataset("parquet", data_files="/mnt/bn/jiangzhongtao/users/liaohuanxuan/visionthink/data/VisionThink-General-Train/val.parquet")["train"]
    breakpoint()
    # videos={"type": "video", "video": dataframe[0]['videos']}
    # videos = process_video(dataframe[0]["videos"])