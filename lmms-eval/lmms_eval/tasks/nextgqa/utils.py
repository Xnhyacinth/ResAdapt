import os
import re
import sys
from pathlib import Path

import yaml
from lmms_eval.tasks._task_utils.eval_utils import extract_final_boxed_content


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# cache_dir = os.path.join(hf_home, cache_dir)
# base_cache_dir = config["dataset_kwargs"]["cache_dir"]
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "nextgqa_boxed.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


# Pass in video path here
# Can only work correctly with video llm
def gqa_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    video_path = doc["video"]
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = os.path.join(cache_dir, "nextgqa_videos", video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif "s3://" not in video_path:
        sys.exit(f"video path:{video_path} does not exist, please check")

    return [video_path]


def gqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc["question"]
    options = "\n".join([f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(doc["candidates"])])

    return f"{pre_prompt}Question: {question}\nOptions:\n{options}\n{post_prompt}"


def gqa_doc_to_answer(doc):
    return {
        "answer": chr(ord("A") + doc["candidates"].index(doc["answer"])),
        "segment": doc["solution"],
    }


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return ""

    matches = re.search(r"[ABCDE]", s)
    if matches is None:
        return ""
    return matches[0]


def parse_timestamps_from_string(pred_text):
    """
    Parse a single interval from free text.

    Supports the following interval forms (numbers only; no time like 1:23):
      - [t1, t2], [t1 - t2], [t1 — t2], [t1 to t2], [t1 and t2]
      - (t1, t2) and similar with the same separators
      - t1, t2 (without brackets), with the same separators

    Returns:
      - [start, end] as floats if a match is found (start <= end; swapped if needed)
      - None if no valid interval is found
    """
    text = (pred_text or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()
    text = text.strip()
    if text:
        boxed_matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
        if boxed_matches:
            joined = " ".join(boxed_matches)
            nums = re.findall(r"-?\d+(?:\.\d+)?", joined)
            if len(nums) >= 2:
                start = float(nums[0])
                end = float(nums[1])
                if start > end:
                    start, end = end, start
                return [start, end]
            text = boxed_matches[-1].strip()
    lowered = text.lower()
    if not re.search(r"\d", text) and any(k in lowered for k in ("none", "null", "not found", "no answer", "no timestamps")):
        return None
    if text:
        try:
            if text[0] in "[{":
                data = json.loads(text)
                if isinstance(data, dict):
                    data = [data]
                if isinstance(data, list) and data:
                    item = data[0]
                    if isinstance(item, dict):
                        for start_key, end_key in (
                            ("start_time", "end_time"),
                            ("start", "end"),
                            ("begin", "finish"),
                        ):
                            if start_key in item and end_key in item:
                                start = float(item[start_key])
                                end = float(item[end_key])
                                if start > end:
                                    start, end = end, start
                                return [start, end]
        except Exception:
            pass
        try:
            start_match = re.search(r'"start_time"\s*:\s*"?(-?\d+(?:\.\d+)?)"?', text)
            end_match = re.search(r'"end_time"\s*:\s*"?(-?\d+(?:\.\d+)?)"?', text)
            if start_match and end_match:
                start = float(start_match.group(1))
                end = float(end_match.group(1))
                if start > end:
                    start, end = end, start
                return [start, end]
        except Exception:
            pass
    m = re.search(r"start(?:_time)?\s*[:=]?\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    n = re.search(r"end(?:_time)?\s*[:=]?\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m and n:
        start = float(m.group(1))
        end = float(n.group(1))
        if start > end:
            start, end = end, start
        return [start, end]
    m = re.search(r"starts?\s+at\s+(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    n = re.search(r"ends?\s+at\s+(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m and n:
        start = float(m.group(1))
        end = float(n.group(1))
        if start > end:
            start, end = end, start
        return [start, end]

    NUM = r"(?:\d+(?:\.\d+)?)"
    SEP = r"(?:,|-|–|—|\bto\b|\band\b)"
    INTERVAL_RE = re.compile(
        rf"""
        (?<![\d.])
        (?:
            \[\s*(?P<sb>{NUM})\s*{SEP}\s*(?P<eb>{NUM})\s*\]
          | \(\s*(?P<sp>{NUM})\s*{SEP}\s*(?P<ep>{NUM})\s*\)
          | (?P<s>{NUM})\s*{SEP}\s*(?P<e>{NUM})
        )
        (?![\d.])
        """,
        re.IGNORECASE | re.VERBOSE,
    )
    m = INTERVAL_RE.search(text)
    if not m:
        return None
    start_str = m.group("sb") or m.group("sp") or m.group("s")
    end_str = m.group("eb") or m.group("ep") or m.group("e")
    start = float(start_str)
    end = float(end_str)
    if start > end:
        start, end = end, start
    return [start, end]


def compute_iou(gt_timestamps, pred_timestamps, eps: float = 1e-9):
    # Compute IoU for single intervals
    gs, ge = gt_timestamps[0], gt_timestamps[1]
    ps, pe = pred_timestamps[0], pred_timestamps[1]

    if ge < gs:
        gs, ge = ge, gs

    if pe < ps:
        ps, pe = pe, ps

    # If predicted interval is degenerate (point), return 0
    len_p = max(0.0, pe - ps)
    if len_p <= eps:
        return 0.0

    len_g = max(0.0, ge - gs)
    inter = max(0.0, min(ge, pe) - max(gs, ps))
    union = len_g + len_p - inter
    return 0.0 if union <= eps else inter / union


def gqa_process_results_generation(doc, result):
    pred = result[0]
    pred_answer = pred.split("<>")[0].strip()
    pred_segment = pred.split("<>")[-1].strip()
    pred_timestamps = parse_timestamps_from_string(pred_segment)
    if pred_timestamps is None:
        pred_timestamps = [0.0, 0.0]

    pred_answer = extract_characters_regex(pred_answer)
    gt_answer = chr(ord("A") + doc["candidates"].index(doc["answer"]))

    gt_timestamps = doc["solution"]
    gt_timestamps = [[float(gt[0]), float(gt[1])] for gt in gt_timestamps]

    result = {
        "query": f'{doc["video"]}>>>{doc["candidates"]}',
        "gt_answer": gt_answer,
        "gt_segment": gt_timestamps,
        "pred": pred,
        "score": float(gt_answer == pred_answer),
        "iou": max([compute_iou(gt, pred_timestamps) for gt in gt_timestamps]),
    }
    return {
        "accuracy": result,
        "iou_0.3": result,
        "iou_0.5": result,
        "iou_0.7": result,
        "m_iou": result,
    }


def gqa_process_boxed_results_generation(doc, result):
    pred = extract_final_boxed_content(result[0])
    pred_answer = pred.split("<>")[0].strip()
    pred_segment = pred.split("<>")[-1].strip()
    pred_timestamps = parse_timestamps_from_string(pred_segment)
    if pred_timestamps is None:
        pred_timestamps = [0.0, 0.0]

    pred_answer = extract_characters_regex(pred_answer)
    gt_answer = chr(ord("A") + doc["candidates"].index(doc["answer"]))

    gt_timestamps = doc["solution"]
    gt_timestamps = [[float(gt[0]), float(gt[1])] for gt in gt_timestamps]

    result = {
        "query": f'{doc["video"]}>>>{doc["candidates"]}',
        "gt_answer": gt_answer,
        "gt_segment": gt_timestamps,
        "pred": pred,
        "score": float(gt_answer == pred_answer),
        "iou": max([compute_iou(gt, pred_timestamps) for gt in gt_timestamps]),
    }

    return {
        "accuracy": result,
        "iou_0.3": result,
        "iou_0.5": result,
        "iou_0.7": result,
        "m_iou": result,
    }


def temporal_grounding_aggregate_nextgqa_iou_threshold(results, args, threshold):
    ious = []
    for result in results:
        ious.append(result["iou"])

    success_cnt = 0
    for cur_iou in ious:
        if cur_iou >= threshold:
            success_cnt += 1

    return float(success_cnt * 100 / len(ious))


def temporal_grounding_aggregate_nextgqa_iou_03(results, args):
    return temporal_grounding_aggregate_nextgqa_iou_threshold(results, args, 0.3)


def temporal_grounding_aggregate_nextgqa_iou_05(results, args):
    return temporal_grounding_aggregate_nextgqa_iou_threshold(results, args, 0.5)


def temporal_grounding_aggregate_nextgqa_iou_07(results, args):
    return temporal_grounding_aggregate_nextgqa_iou_threshold(results, args, 0.7)


def temporal_grounding_aggregate_nextgqa_m_iou(results, args):
    ious = []
    for result in results:
        ious.append(result["iou"])

    return float(sum(ious) * 100 / len(ious))


def multi_choice_aggregate_nextgqa_accuracy(results, args):
    accuracy = []
    for result in results:
        accuracy.append(result["score"])

    return float(sum(accuracy) * 100 / len(accuracy))
