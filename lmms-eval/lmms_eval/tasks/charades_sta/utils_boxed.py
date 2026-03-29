import datetime
import json
import os
import re
import sys
from pathlib import Path

import yaml
import lmms_eval.tasks._task_utils.file_utils as file_utils
from lmms_eval.tasks._task_utils.eval_utils import extract_final_boxed_content
from loguru import logger as eval_logger

# with open(Path(__file__).parent / "_default_template.yaml", "r") as f:
#     raw_data = f.readlines()
#     safe_data = []
#     for i, line in enumerate(raw_data):
#         # remove function definition since yaml load cannot handle it
#         if "!function" not in line:
#             safe_data.append(line)

#     config = yaml.safe_load("".join(safe_data))


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# cache_dir = os.path.join(hf_home, cache_dir)
# base_cache_dir = config["dataset_kwargs"]["cache_dir"]
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "charades.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


# DATA_LIST = {
#     "charades": 'your_data_dir/Charades/',
# }
# Pass in video path here
# Can only work correctly with video llm
def temporal_grounding_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    video_path = doc["video"]
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = os.path.join(cache_dir, "Charades_v1_480", video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif "s3://" not in video_path:
        sys.exit(f"video path:{video_path} does not exist, please check")

    return [video_path]


# This is the place where you format your question
def temporal_grounding_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc["caption"]

    return f"{pre_prompt}{question}{post_prompt}"


def temporal_grounding_doc_to_answer(doc):
    return doc["timestamp"]


def iou(A, B):
    max0 = max((A[0]), (B[0]))
    min0 = min((A[0]), (B[0]))
    max1 = max((A[1]), (B[1]))
    min1 = min((A[1]), (B[1]))
    # Ensure result is a regular Python float, not float16
    return float(max(min1 - max0, 0) / (max1 - min0))

    # # hacked!
    # return float(max(min1 - max0, 0) / (B[1] - B[0]))


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


def temporal_grounding_process_results_generation(doc, result):
    """
    Parse the predicted text to get (start_timestamp, end_timestamp) in seconds.
    Supports:
      - [mm:ss(.ms), mm:ss(.ms)]
      - Natural language with HH:MM(:SS)(.ms) or explicit seconds

    Notes:
      - Expects a global logger named `eval_logger` available in the caller's scope.
      - Returns (0.0, 0.0) if parsing fails; emits warnings accordingly.
    """
    pred = result[0]
    try:
        start_timestamp, end_timestamp = parse_timestamps_from_string(pred)
    except Exception as e:
        eval_logger.warning(f"Failed to extract timestamps from pred: {pred}, doc: {doc}, error: {e}")
        start_timestamp, end_timestamp = 0, 0

    if start_timestamp == 0 and end_timestamp == 0:
        eval_logger.warning(f"Failed to extract timestamps from pred: {pred}, doc: {doc}")

    # Convert float16 to regular Python floats to make them JSON serializable
    gt_timestamp = doc["timestamp"]
    if hasattr(gt_timestamp, "tolist"):
        # Handle numpy arrays or tensors
        gt_timestamp = gt_timestamp.tolist()
    elif isinstance(gt_timestamp, (list, tuple)):
        # Handle lists/tuples that might contain float16 values
        gt_timestamp = [float(x) for x in gt_timestamp]
    else:
        # Handle single values
        gt_timestamp = float(gt_timestamp)

    result = {
        "query": f'{doc["video"]}>>>{doc["caption"]}>>>{gt_timestamp}',
        "gt": gt_timestamp,
        "pred": [start_timestamp, end_timestamp],
        "iou": iou(gt_timestamp, [start_timestamp, end_timestamp]),
    }

    return {
        "iou_0.3": result,
        "iou_0.5": result,
        "iou_0.7": result,
        "m_iou": result,
    }


def temporal_grounding_process_boxed_results_generation(doc, result):
    """
    Parse the predicted text to get (start_timestamp, end_timestamp) in seconds.
    Supports:
      - \boxed{[mm:ss(.ms), mm:ss(.ms)]}
      - [mm:ss(.ms), mm:ss(.ms)]
      - Natural language with HH:MM(:SS)(.ms) or explicit seconds

    Notes:
      - Expects a global logger named `eval_logger` available in the caller's scope.
      - Returns (0.0, 0.0) if parsing fails; emits warnings accordingly.
    """
    pred = extract_final_boxed_content(result[0])

    try:
        start_timestamp, end_timestamp = parse_timestamps_from_string(pred)
    except Exception as e:
        eval_logger.warning(f"Failed to extract timestamps from pred: {pred}, doc: {doc}, error: {e}")
        start_timestamp, end_timestamp = 0, 0

    if start_timestamp == 0 and end_timestamp == 0:
        eval_logger.warning(f"Failed to extract timestamps from pred: {pred}, doc: {doc}")

    # Convert float16 to regular Python floats to make them JSON serializable
    gt_timestamp = doc["timestamp"]
    if hasattr(gt_timestamp, "tolist"):
        # Handle numpy arrays or tensors
        gt_timestamp = gt_timestamp.tolist()
    elif isinstance(gt_timestamp, (list, tuple)):
        # Handle lists/tuples that might contain float16 values
        gt_timestamp = [float(x) for x in gt_timestamp]
    else:
        # Handle single values
        gt_timestamp = float(gt_timestamp)

    result = {
        "query": f'{doc["video"]}>>>{doc["caption"]}>>>{gt_timestamp}',
        "gt": gt_timestamp,
        "pred": [start_timestamp, end_timestamp],
        "iou": iou(gt_timestamp, [start_timestamp, end_timestamp]),
    }

    return {
        "iou_0.3": result,
        "iou_0.5": result,
        "iou_0.7": result,
        "m_iou": result,
    }


def temporal_grounding_aggregate_charades_iou_threshold(results, args, threshold):
    ious = []
    for result in results:
        ious.append(result["iou"])

    success_cnt = 0
    for cur_iou in ious:
        if cur_iou >= threshold:
            success_cnt += 1

    return float(success_cnt * 100 / len(ious))


def temporal_grounding_aggregate_charades_iou_03(results, args):
    return temporal_grounding_aggregate_charades_iou_threshold(results, args, 0.3)


def temporal_grounding_aggregate_charades_iou_05(results, args):
    return temporal_grounding_aggregate_charades_iou_threshold(results, args, 0.5)


def temporal_grounding_aggregate_charades_iou_07(results, args):
    return temporal_grounding_aggregate_charades_iou_threshold(results, args, 0.7)


def temporal_grounding_aggregate_charades_m_iou(results, args):
    ious = []
    for result in results:
        ious.append(result["iou"])

    return float(sum(ious) * 100 / len(ious))


def temporal_grounding_aggregate_submissions(results, args, task):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"inference_results_temporal_grounding_{task}_{now_date_time}.json"
    path = file_utils.generate_submission_file(submission_file_name, args)

    # results is a list of 5031 dict,
    # need to convert results into a single dict with 5031 key-value pairs
    combined_submission = {}

    for submission_dict in results:
        combined_submission.update(submission_dict)

    with open(path, "w") as f:
        json.dump(combined_submission, f, indent=4)

    eval_logger.info(f"Submission file saved to {path}")
