"""
Preprocesses a local Video dataset into a standardized parquet format.
"""

import argparse
import json
import os
import re

try:
    import datasets
    from datasets import Dataset, DatasetDict
except ImportError:
    datasets = None
    Dataset = None
    DatasetDict = None


def extract_answer_content(text):
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.S)
    if match:
        return match.group(1).strip()
    return ""


def iter_records_from_json_file(file_path: str):
    lower = file_path.lower()
    if lower.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    if not lower.endswith(".json"):
        return

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        for item in obj:
            yield item
        return

    if isinstance(obj, dict):
        if isinstance(obj.get("data"), list):
            for item in obj["data"]:
                yield item
            return
        if isinstance(obj.get("annotations"), list):
            for item in obj["annotations"]:
                yield item
            return
        yield obj
        return

    yield obj


def collect_shard_files(dataset_root: str, folder_regex: str) -> list[str]:
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"dataset_root not found or not a directory: {dataset_root}")

    pattern = re.compile(folder_regex)
    shard_files: list[str] = []

    for name in sorted(os.listdir(dataset_root)):
        subdir = os.path.join(dataset_root, name)
        if not os.path.isdir(subdir):
            continue
        if pattern.match(name) is None:
            continue

        for root, _, files in os.walk(subdir):
            for filename in sorted(files):
                lower = filename.lower()
                if not (lower.endswith(".json") or lower.endswith(".jsonl")):
                    continue
                shard_files.append(os.path.join(root, filename))

    return shard_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dataset_root",
        default="/path/to/LLaVA-Video-178K",
        help="Root directory that contains numeric-named subfolders with JSON/JSONL shards and video files.",
    )
    parser.add_argument(
        "--folder_regex",
        default=r"^\d+",
        help="Regex for shard subfolder names to include (matched against folder basename).",
    )
    parser.add_argument("--local_save_dir", default="~/data/processed_video_data", help="The save directory for the preprocessed dataset.")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes for the map function.")
    parser.add_argument("--val_size", type=int, default=500, help="Number of samples to use for validation.")
    parser.add_argument("--data_source", default="LLaVA-Video-178K", help="data_source field in the output samples.")

    args = parser.parse_args()
    if datasets is None:
        raise ModuleNotFoundError("Missing dependency: datasets. Please install `datasets` (with parquet support) in the runtime environment.")

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    shard_files = collect_shard_files(args.local_dataset_root, args.folder_regex)
    if not shard_files:
        raise FileNotFoundError(f"No .json/.jsonl found under numeric folders in: {args.local_dataset_root}")

    print(f"Found {len(shard_files)} shard files.")
    all_records: list[dict] = []
    for file_path in shard_files:
        for record in iter_records_from_json_file(file_path):
            all_records.append(record)

    full_dataset = Dataset.from_list(all_records)

    print(f"Total samples loaded: {len(full_dataset)}")

    print(f"Splitting dataset: reserving {args.val_size} samples for validation...")
    split_dataset = full_dataset.train_test_split(test_size=args.val_size, seed=42)

    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

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
            example = dict(example)

            video_path = example.pop("video", None)
            if video_path is None:
                video_path = example.pop("video_path", None)

            if video_path and not os.path.isabs(video_path):
                video_path = os.path.join(args.local_dataset_root, video_path)

            if video_path:
                videos = [
                    {
                        "type": "video",
                        "video": video_path,
                        "min_frames": 1,
                        "max_frames": 32,
                    }
                ]
            else:
                videos = []

            problem = example.pop("problem", None)
            if problem is None:
                problem = example.pop("original_question", "")
            if problem:
                problem = problem.replace("<image>", "<video>")

            answer = example.pop("solution", None)
            if answer is None:
                answer = example.pop("original_answer", "")
            if answer:
                answer = extract_answer_content(answer)

            doc_id = example.pop("id", str(idx))
            prompt = (problem or "") + " " + instruction_following

            data = {
                "data_source": example.pop("data_source", args.data_source),
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "images": [],
                "videos": videos,
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

    print("Starting dataset transformation for TRAIN...")
    processed_train = train_dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        num_proc=args.num_proc,
    )

    print("Starting dataset transformation for VAL...")
    processed_val = val_dataset.map(
        function=make_map_fn("test"),
        with_indices=True,
        num_proc=args.num_proc,
    )

    print("Transformation complete.")

    train_output_path = os.path.join(local_save_dir, "train.parquet")
    val_output_path = os.path.join(local_save_dir, "val.parquet")

    print(f"Saving processed TRAIN data to: {train_output_path}")
    processed_train.to_parquet(train_output_path)

    print(f"Saving processed VAL data to: {val_output_path}")
    processed_val.to_parquet(val_output_path)

    print("Save complete.")