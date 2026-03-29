import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from vllm import LLM, SamplingParams


def _uniform_indices(total: int, k: int) -> List[int]:
    if total <= 0:
        return []
    k = min(k, total)
    return np.linspace(0, total - 1, k, dtype=int).tolist()


def build_prompt(prompt_text: str, n_images: int) -> str:
    vision_tags = "<image>" * n_images
    return f"USER: {vision_tags}\n{prompt_text}\nASSISTANT:"


def build_video_prompt(prompt_text: str) -> str:
    return f"USER: <video>\n{prompt_text}\nASSISTANT:"


def build_scale_prompt(question: str, data_type: str, num_frames: int) -> str:
    return (
        "You are a cost-aware visual perception policy for long videos/images.\n"
        "Given a **video (sequence of frames)** or a **single image**, and a **question**, your job is to choose an appropriate **scale factor** for **each frame** (or the single image) to control the resolution used by the vision encoder.\n\n"
        "### Goal\n\n"
        "Minimize visual compute and token cost by using **small scales whenever possible**, while ensuring that **question-relevant, small, or critical details remain readable**.\n"
        "Use **larger scales only when necessary** to preserve important information.\n\n"
        "### Output\n\n"
        "Return a list of scale factors `s_t` in **[0.2, 2.0]** for frames `t = 1..T` (or a single value for one image).\n"
        "Scales may be discrete or continuous, but must stay within the range.\n\n"
        "### What you must consider (multi-factor decision)\n\n"
        "1. **Question relevance**\n\n"
        "* If the frame likely contains evidence needed to answer the question, increase scale.\n"
        "* If irrelevant, reduce scale.\n\n"
        "2. **Information density / visual complexity**\n\n"
        "* Frames with dense text, small objects, fine-grained actions, tiny UI elements, or multiple entities may require higher scale.\n"
        "* Simple scenes with large objects can use lower scale.\n\n"
        "3. **Redundancy and similarity across time**\n\n"
        "* If a frame is highly similar to nearby frames (little new information), prefer a smaller scale.\n"
        "* Allocate higher scale only to representative/key frames within repetitive segments.\n\n"
        "4. **Uncertainty / ambiguity**\n\n"
        "* If the frame contains ambiguous or hard-to-read content that might matter (e.g., blurred text, small signs, distant objects), increase scale.\n"
        "* If confidence is high that details are unneeded, reduce scale.\n\n"
        "5. **Budget awareness (cost-first)**\n\n"
        "* Prefer minimal scale by default.\n"
        "* Only “spend” higher scale when it is likely to change the answer.\n\n"
        "### Decision Rules (practical heuristics)\n\n"
        "* Start with a low default (e.g., `0.4–0.7`) for most frames.\n"
        "* Increase scale (`>1.0`) when:\n\n"
        "  * The question requires **reading text/numbers**.\n"
        "  * The evidence is **small / far away / fine-grained**.\n"
        "  * The frame is a **key transition** (new scene, new object, new action).\n"
        "* Decrease scale (`<0.5`) when:\n\n"
        "  * The frame is redundant with neighbors.\n"
        "  * The frame is clearly irrelevant to the question.\n"
        "  * The scene is low-detail and large-structure dominated.\n\n"
        "### Required format\n\n"
        "Output **ONLY** valid JSON, no extra text.\n\n"
        "For a video:\n\n"
        "{\n"
        '  "scales": [0.6, 0.4, 0.4, 1.2, 0.8, 0.4, 0.4, 0.6]\n'
        "}\n\n"
        "For a single image:\n\n"
        "{\n"
        '  "scale": 0.8\n'
        "}\n\n"
        f"Question: {question}\n"
        f"Modality: {data_type}\n"
        f"Frames: {num_frames}\n"
    )


def parse_scales(text: str) -> Optional[List[float]]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            if isinstance(parsed.get("scales"), list):
                return [float(x) for x in parsed["scales"]]
            if isinstance(parsed.get("scale"), (int, float)):
                return [float(parsed["scale"])]
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except Exception:
        numbers = []
        for token in text.replace(",", " ").replace("[", " ").replace("]", " ").split():
            try:
                numbers.append(float(token))
            except Exception:
                continue
        return numbers if numbers else None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise TypeError("Input JSON must be a list of examples.")

    llm = LLM(model=args.model, tensor_parallel_size=1, gpu_memory_utilization=args.gpu_memory_utilization)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for ex in raw:
            if not isinstance(ex, dict):
                continue
            rel = ex.get("path") or ex.get("video") or ex.get("image") or ""
            if not rel:
                continue
            media_path = os.path.normpath(os.path.join(args.data_root, rel))
            if not os.path.exists(media_path):
                continue
            is_video = media_path.lower().endswith((".mp4", ".mkv", ".webm", ".mov", ".avi"))
            question = ex.get("problem") or ""
            if is_video:
                prompt_text = build_scale_prompt(question, "video", args.num_frames)
                prompt = build_video_prompt(prompt_text)
                request = {
                    "prompt": prompt,
                    "multi_modal_data": {"video": media_path},
                }
            else:
                image = Image.open(media_path).convert("RGB")
                prompt_text = build_scale_prompt(question, "image", 1)
                prompt = build_prompt(prompt_text, 1)
                request = {"prompt": prompt, "multi_modal_data": {"image": image}}

            outputs = llm.generate(request, sampling_params)
            text = outputs[0].outputs[0].text
            scales = parse_scales(text)

            record = {
                "data_source": "Video-R1-COT-165k",
                "messages": [{"role": "user", "content": [{"type": "text", "text": question}]}],
                "response": ex.get("solution") or "",
                "problem_type": ex.get("problem_type") or "",
                "scales": scales,
                "model_text": text,
                "video_path": media_path if is_video else "",
                "image_path": media_path if not is_video else "",
                "raw_example": ex,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
