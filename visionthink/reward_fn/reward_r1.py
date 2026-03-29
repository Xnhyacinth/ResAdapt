import re
from typing import List, Optional

from visionthink.reward_fn import reward as _reward
from visionthink.reward_fn.mc_grader import equal_answer
from visionthink.reward_fn.tg_grader import compute_iou_reward


try:
    from rouge_score import rouge_scorer as _rouge_scorer
except Exception:
    _rouge_scorer = None

_ROUGE_SCORER = None
_RE_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_RE_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_RE_FORMAT = re.compile(r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*", re.DOTALL | re.MULTILINE)
_RE_FLOAT = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")

def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _last_boxed_only_string(s: str) -> Optional[str]:
    idx = s.rfind("\\boxed")
    if idx < 0:
        idx = s.rfind("\\fbox")
        if idx < 0:
            return None
    i, open_braces = idx, 0
    while i < len(s):
        if s[i] == "{":
            open_braces += 1
        elif s[i] == "}":
            open_braces -= 1
            if open_braces == 0:
                return s[idx:i + 1]
        i += 1
    return None


def _remove_boxed(s: str) -> str:
    s = s.strip()
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{"):-1]
    if s.startswith("\\boxed "):
        return s[len("\\boxed "):]
    return s


def extract_answer(text: str) -> str:
    if not text:
        return ""

    matches = _RE_BOXED.findall(text)
    if matches:
        return matches[-1].strip()

    match = _RE_ANSWER_TAG.search(text)
    if match:
        return match.group(1).strip()

    boxed = _last_boxed_only_string(text)
    if boxed:
        return _remove_boxed(boxed).strip()

    patterns = [
        r"(?:final\s+answer|answer\s+is|答案|结论|结果)\s*[:：]?\s*(.+)",
        r"(?:the\s+answer)\s+is\s*(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).split("\n")[0].strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _strip_code_fence(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            return parts[1].strip()
    return s


def _strip_unit(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"[°℃]", "", s)
    s = re.sub(
        r"(?<=\d)\s*"
        r"(?:(?:mm/s|m/s|km/h|m/s²|m/s\^2|m/s2)|"
        r"(?:mm|cm|km|m)|"
        r"(?:mg|kg|g|t|lb|lbs)|"
        r"(?:hour|sec|min|day|hr|h|s)|"
        r"(?:ghz|mhz|khz|hz)|"
        r"(?:kw|mw|w|j|v|a|pa)|"
        r"(?:gb|mb|tb|byte|b))\b",
        "",
        s,
        flags=re.IGNORECASE,
    )
    return s.strip()


def _parse_number(s: str) -> Optional[float]:
    if not s:
        return None

    s = _strip_unit(s.strip())
    s = s.replace(" ", "")

    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return None

    frac_match = re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", s)
    if frac_match:
        try:
            num, den = s.split("/", 1)
            den_f = float(den)
            if den_f == 0:
                return None
            return float(num) / den_f
        except Exception:
            return None

    sci_match = re.search(r"(-?\d+\.?\d*)\s*[\*x]\s*10\^?\{?(-?\d+)\}?", s)
    if sci_match:
        try:
            coef = float(sci_match.group(1))
            exp = int(sci_match.group(2))
            return coef * (10.0 ** exp)
        except Exception:
            return None

    if "," in s and "." not in s:
        if re.fullmatch(r"\d{1,3}(,\d{3})+", s):
            s = s.replace(",", "")
        elif re.fullmatch(r"\d+,\d+", s):
            s = s.replace(",", ".")

    try:
        return float(s)
    except Exception:
        return None


def normalize_number(num_str: str) -> Optional[float]:
    if not num_str:
        return None
    return _parse_number(num_str)


def _normalize_answer(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def wer(reference: str, hypothesis: str) -> float:
    ref = _normalize_text(reference).split()
    hyp = _normalize_text(hypothesis).split()
    m, n = len(ref), len(hyp)
    if m == 0:
        return 1.0 if n > 0 else 0.0
    prev = list(range(n + 1))
    cur = [0] * (n + 1)
    for i in range(1, m + 1):
        cur[0] = i
        ref_i = ref[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ref_i == hyp[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, cur = cur, prev
    return prev[n] / max(1, m)


def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    cur = [0] * (n + 1)
    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = prev[j] if prev[j] >= cur[j - 1] else cur[j - 1]
        prev, cur = cur, prev
    return prev[n]


def compute_rouge_score(reference: str, hypothesis: str) -> float:
    global _ROUGE_SCORER

    if _rouge_scorer is not None:
        if _ROUGE_SCORER is None:
            _ROUGE_SCORER = _rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = _ROUGE_SCORER.score(reference or "", hypothesis or "")
        return float((scores["rouge1"].fmeasure + scores["rouge2"].fmeasure + scores["rougeL"].fmeasure) / 3.0)

    ref_tokens = _normalize_text(reference).split()
    hyp_tokens = _normalize_text(hypothesis).split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = _lcs_len(ref_tokens, hyp_tokens)
    precision = lcs / max(1, len(hyp_tokens))
    recall = lcs / max(1, len(ref_tokens))
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _extract_choice_set(ans: str) -> List[str]:
    if not ans:
        return []
    s = ans.strip().upper()
    letters = re.findall(r"\b([A-Z])\b", s)
    if letters:
        return sorted(set(letters))
    letters = re.findall(r"(?:option|choice)\s*[:：]?\s*([A-Z])", s)
    if letters:
        return sorted(set(letters))
    return []


def format_reward_fn(predict_str: str) -> float:
    return 1.0 if _RE_FORMAT.fullmatch(predict_str) else 0.0


def _numeric_close(pred: Optional[float], gt: Optional[float], abs_tol: float = 1e-2, rel_tol: float = 1e-2) -> bool:
    if pred is None or gt is None:
        return False
    return abs(pred - gt) <= max(abs_tol, rel_tol * abs(gt))

def _parse_intervals(s: str):
    if not s:
        return None
    nums = _RE_FLOAT.findall(s)
    if len(nums) < 2:
        return None
    try:
        vals = [float(x) for x in nums]
    except Exception:
        return None
    if len(vals) < 2:
        return None
    if len(vals) >= 4 and len(vals) % 2 == 0:
        intervals: list[list[float]] = []
        for i in range(0, len(vals), 2):
            start, end = vals[i], vals[i + 1]
            if start > end:
                start, end = end, start
            intervals.append([start, end])
        return intervals

    start, end = vals[0], vals[1]
    if start > end:
        start, end = end, start
    return [start, end]


def _compute_acc_reward(output_ans: str, gt_ans: str, gt, question_type: str) -> float:
    question_type = _normalize_text(str(question_type).replace("-", " ").replace("_", " "))
    if not question_type:
        try:
            return float(_reward.compute_score_general(output_ans, gt_ans))
        except Exception:
            return 1.0 if _normalize_answer(output_ans) == _normalize_answer(gt_ans) else 0.0
        
    if question_type == "multiple choice":
        gt_choices = _extract_choice_set(gt_ans)
        pred_choices = _extract_choice_set(output_ans)
        if gt_choices:
            if not pred_choices:
                return 0.0
            gt_set = set(gt_choices)
            pred_set = set(pred_choices)
            inter = len(gt_set & pred_set)
            if inter == 0:
                return 0.0
            return (2.0 * inter) / (len(gt_set) + len(pred_set))
        return 1.0 if _normalize_answer(output_ans) == _normalize_answer(gt_ans) else 0.0
    elif question_type == "numerical":
        gt_number = normalize_number(gt_ans)
        out_number = normalize_number(output_ans)
        return 1.0 if _numeric_close(out_number, gt_number) else float(_reward.compute_score_general(output_ans, gt_ans))
    elif question_type == "ocr":
        error_rate = wer(gt_ans, output_ans)
        reward = 1.0 - error_rate
        return max(0.0, min(1.0, reward))
    elif question_type in ("free-form", "open-end", "openend", "open end", "open ended"):
        score = compute_rouge_score(gt_ans, output_ans)
        return max(0.0, min(1.0, score))
    elif question_type == "regression":
        gt_number = normalize_number(gt_ans)
        out_number = normalize_number(output_ans)
        if gt_number is None or out_number is None:
            return 0.0
        rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
        rel_diff = min(1.0, max(0.0, rel_diff))
        return 1.0 - rel_diff
    elif question_type == "exact_match":
        return 1.0 if equal_answer(gt_ans, output_ans) else float(_reward.compute_score_general(output_ans, gt_ans))
    elif question_type == "math":
        gt_number = normalize_number(gt_ans)
        out_number = normalize_number(output_ans)
        if _numeric_close(out_number, gt_number):
            return 1.0
        if gt_number is None or out_number is None:
            return 0.0
        return float(_reward.compute_score_general(output_ans, gt_ans))
    elif question_type == "iou":
        if isinstance(gt, (list, tuple)):
            return compute_iou_reward(gt, output_ans)
        if isinstance(gt, dict):
            for key in ("segment", "timestamps", "timestamp"):
                if key in gt:
                    return compute_iou_reward(gt[key], output_ans)
        if isinstance(gt, str):
            gt_intervals = _parse_intervals(gt)
            if gt_intervals is None:
                return 0.0
            return compute_iou_reward(gt_intervals, output_ans)
        return 0.0
    elif question_type == "gqa":
        if not isinstance(gt, dict):
            return float(_reward.compute_score_general(output_ans, gt_ans))
        gt_answer = str(gt.get("answer", "")).strip()
        gt_segment = gt.get("segment", None)
        if not gt_answer or gt_segment is None:
            return float(_reward.compute_score_general(output_ans, gt_ans))
        if isinstance(gt_segment, str):
            gt_segment = _parse_intervals(gt_segment)
            if gt_segment is None:
                return float(_reward.compute_score_general(output_ans, gt_ans))
        if not isinstance(gt_segment, (list, tuple)):
            return float(_reward.compute_score_general(output_ans, gt_ans))
        parts = output_ans.split("<>")
        output_answer = parts[0].strip() if parts else ""
        pred_segment = parts[-1].strip() if len(parts) > 1 else ""
        reward_answer = 1.0 if equal_answer(gt_answer, output_answer) else 0.0
        reward_segment = compute_iou_reward(gt_segment, pred_segment)
        return reward_answer + reward_segment
    
    return float(_reward.compute_score_general(output_ans, gt_ans))


def compute_score(*args, **kwargs) -> dict:
    kwargs.pop("data_source", None)
    solution_str = kwargs.pop("solution_str", None)
    ground_truth = kwargs.pop("ground_truth", None)
    extra_info = kwargs.pop("extra_info", None)

    if solution_str is None and ground_truth is None and args:
        if len(args) >= 4:
            _, solution_str, ground_truth, extra_info = args[:4]
        elif len(args) == 3:
            solution_str, ground_truth, extra_info = args
        elif len(args) == 2:
            solution_str, ground_truth = args
        elif len(args) == 1:
            solution_str = args[0]

    solution_str = "" if solution_str is None else str(solution_str)
    extra_info = extra_info if isinstance(extra_info, dict) else {}

    problem_type = ""
    if isinstance(extra_info, dict):
        problem_type = extra_info.get("problem_type") or extra_info.get("question_type") or extra_info.get("type") or ""
    if not problem_type:
        problem_type = kwargs.get("problem_type") or kwargs.get("question_type") or ""
    if isinstance(problem_type, (list, tuple)):
        problem_type = problem_type[0] if problem_type else ""
    question_type = _normalize_text(str(problem_type).replace("-", " ").replace("_", " "))

    is_format_error = (
        solution_str.count("<think>") != solution_str.count("</think>")
        or solution_str.count("<answer>") != solution_str.count("</answer>")
    )

    predict_no_think = solution_str.split("</think>")[-1].strip() if "</think>" in solution_str else solution_str.strip()
    answer_match = _RE_ANSWER_TAG.search(predict_no_think)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    else:
        answer_text = predict_no_think

    if question_type in ("iou", "gqa"):
        output_ans = _strip_code_fence(answer_text or "")
        boxed_matches = _RE_BOXED.findall(output_ans)
        if boxed_matches:
            output_ans = boxed_matches[-1].strip()
        else:
            boxed = _last_boxed_only_string(output_ans)
            if boxed:
                output_ans = _remove_boxed(boxed).strip()
        output_ans = output_ans.strip()
    else:
        output_ans = extract_answer(answer_text)
    if isinstance(ground_truth, dict):
        gt_text = str(ground_truth.get("answer", ""))
    else:
        gt_text = "" if ground_truth is None else str(ground_truth)
    gt_ans = extract_answer(gt_text)
    acc_reward = _compute_acc_reward(output_ans, gt_ans, ground_truth, question_type)
    format_reward = -1.0 if is_format_error else format_reward_fn(solution_str)

    acc_reward_weight = float(kwargs.get("acc_reward_weight", 1.0))
    format_reward_weight = float(kwargs.get("format_reward_weight", 0.2))
    final_score = acc_reward_weight * float(acc_reward) + format_reward_weight * float(format_reward)
    return {
        "score": final_score,
        "acc_reward": acc_reward,
        "format_reward": format_reward,
    }
