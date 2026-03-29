import argparse
import os
import re

import datasets
from datasets import Dataset, DatasetDict


def extract_answer_content(text: str) -> str:
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.S)
    if match:
        return match.group(1).strip()
    return ""


instruction_following = (
    r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    r"The reasoning process MUST BE enclosed within <think> </think> tags and the answer MUST BE enclosed within <answer> </answer> tags. "
    r"The final answer MUST BE put in \boxed{} and the \boxed{} expression MUST BE contained entirely within the <answer> </answer> tags. "
    r"Do not include any reasoning or explanations outside these tags."
)

QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
)

TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) in \\boxed{} within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) in \\boxed{} within the <answer> </answer> tags.",
    "ocr": " Please transcribe text from the image/video clearly and provide your text answer in \\boxed{} within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer in \\boxed{} within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) in \\boxed{} within the <answer> </answer> tags.",
}

COT_SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> </think> tags, and then give your answer between the <answer> </answer> tags. "
    "The final answer MUST BE wrapped in \\boxed{} and the \\boxed{} expression MUST BE contained entirely within the <answer> </answer> tags."
)


def join_media_path(root: str, path: str | None) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    if path.startswith("./"):
        path = path[2:]
    return os.path.join(root, path)


def build_question(example: dict) -> str:
    problem = example.get("problem") or example.get("original_question") or ""
    if (example.get("problem_type") or "").lower() == "multiple choice":
        options = example.get("options") or []
        if options:
            problem = problem + "\nOptions:\n" + "\n".join(options)
    return problem


def build_prompt_text(question: str, problem_type: str) -> str:
    type_key = (problem_type or "").strip().lower()
    suffix = TYPE_TEMPLATE.get(type_key, "")
    # return QUESTION_TEMPLATE.format(Question=question) + suffix
    return question + '\n' + suffix


def make_map_fn(split: str, local_dataset_root: str):
    def process_fn(example: dict, idx: int):
        example = dict(example)

        raw_data_source = example.get("data_source", "")
        data_type = example.get("data_type") or ""
        raw_path = example.get("path") or ""

        image_path = ""
        video_path = ""

        if isinstance(example.get("image_path"), str):
            image_path = join_media_path(local_dataset_root, example.get("image_path"))
        if isinstance(example.get("video_path"), str):
            video_path = join_media_path(local_dataset_root, example.get("video_path"))

        if not image_path and not video_path and isinstance(raw_path, str) and raw_path:
            p = join_media_path(local_dataset_root, raw_path)
            if data_type == "image":
                image_path = p
            elif data_type == "video":
                video_path = p
            else:
                lower = p.lower()
                if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    image_path = p
                elif lower.endswith((".mp4", ".mkv", ".webm", ".mov", ".avi")):
                    video_path = p

        question = build_question(example)

        tags = ""
        if image_path:
            tags += "<image>"
        if video_path:
            tags += "<video>"
        question_with_tags = tags + question

        answer_raw = example.get("solution") or example.get("original_answer") or ""
        answer = extract_answer_content(answer_raw) if isinstance(answer_raw, str) else ""

        doc_id = (
            str(example.get("doc_id"))
            if example.get("doc_id") is not None
            else str(example.get("problem_id")) if example.get("problem_id") is not None else str(idx)
        )

        images = [{"image": image_path}] if image_path else []
        videos = [{"video": video_path}] if video_path else []

        prompt = build_prompt_text(question_with_tags, example.get("problem_type") or "")

        messages = [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        data = {
            "data_source": raw_data_source,
            "prompt": messages,
            "images": images,
            "videos": videos,
            "ability": example.get("problem_type") or "",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "problem_type": example.get("problem_type") or "",
                "index": idx,
                "answer": answer,
                "question": question_with_tags,
                "doc_id": doc_id,
                "data_type": data_type,
            },
        }
        return data

    return process_fn

def load_json_or_jsonl(path: str) -> Dataset:
    ds = datasets.load_dataset("json", data_files=path)
    if isinstance(ds, DatasetDict):
        return ds["train"]
    if isinstance(ds, Dataset):
        return ds
    raise TypeError(f"Loaded object is of an unexpected type: {type(ds)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", required=True)
    parser.add_argument("--local_dataset_root", default=".")
    parser.add_argument("--local_save_dir", required=True)
    parser.add_argument("--num_proc", type=int, default=8)
    args = parser.parse_args()

    full_dataset = load_json_or_jsonl(args.local_dataset_path)
    processed_train = full_dataset.map(
        function=make_map_fn("train", args.local_dataset_root),
        with_indices=True,
        num_proc=args.num_proc,
    )

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    processed_train.to_parquet(os.path.join(local_save_dir, "train.parquet"))
