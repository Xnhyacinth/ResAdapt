import re
import string
import math
from typing import Optional, Union, List

try:
    import sympy as sp
    SYMPY_OK = True
except Exception:
    SYMPY_OK = False

import logging
logger = logging.getLogger(__name__)


# ------------------------
# Core utility functions
# ------------------------

def format_reward_fn(predict_str: str) -> float:
    """Check if output strictly follows <think>...<answer>...</answer> format."""
    pattern = re.compile(r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*", re.DOTALL | re.MULTILINE)
    return 1.0 if re.fullmatch(pattern, predict_str) else 0.0


def last_boxed_only_string(s: str) -> Optional[str]:
    """Find last \boxed{...} expression in LaTeX string."""
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


def remove_boxed(s: str) -> str:
    """Remove \boxed{} wrapper."""
    s = s.strip()
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{"):-1]
    if s.startswith("\\boxed "):
        return s[len("\\boxed "):]
    return s


def extract_final_answer(text: str) -> Optional[str]:
    """Extract final numeric/symbolic answer from text."""
    # Try boxed answer
    boxed = last_boxed_only_string(text)
    if boxed:
        return remove_boxed(boxed)

    # Try keyword cues
    patterns = [
        r"(?:The\s+final\s+answer\s+is)\s*[:：]?\s*(.+)",
        r"(?:final\s+answer|answer\s+is|答案|结论|结果)\s*[:：]?\s*(.+)",
        r"(?:最终答案|最后答案)\s*[:：]?\s*(.+)",
        r"(?:the\s+answer)\s+is\s*(.+)"
    ]
    for pat in patterns:
        match = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).split("\n")[0].strip()

    # Fallback: last line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def normalize_answer(s: str) -> str:
    """Normalize string by removing punctuation, case, and articles."""
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def normalize_math_expr(expr: str) -> str:
    """Normalize LaTeX-like math expressions to sympy-compatible format."""
    if not expr:
        return ""
    expr = expr.strip().replace("^", "**").replace("\\times", "*")
    expr = re.sub(r"\\frac\s*{([^{}]+)}\s*{([^{}]+)}", r"(\1)/(\2)", expr)
    expr = re.sub(r"\\sqrt\s*{([^{}]+)}", r"sqrt(\1)", expr)
    expr = expr.replace("\\", "")
    return expr


# ------------------------
# Numeric & Symbolic checking
# ------------------------

def is_numeric_like(s: str) -> bool:
    """Detect if string looks like a number (integer, float, sqrt, fraction, etc.)."""
    if not s:
        return False
    s = s.strip()
    numeric_unit = r'(?:(?:\d+(?:\.\d*)?|\.\d+)|(?:\d*√\d+))'
    if re.fullmatch(rf'^[+-]?{numeric_unit}(?:[eE][+-]?\d+)?$', s):
        return True
    if re.fullmatch(rf'^[+-]?{numeric_unit}/{numeric_unit}$', s):
        return True
    return False


def sympy_equiv(a: str, b: str) -> Optional[bool]:
    """Check symbolic or numeric equivalence using sympy, including sqrt/sin/log cases."""
    if not SYMPY_OK:
        return None
    try:
        a, b = normalize_math_expr(a), normalize_math_expr(b)

        # Define allowed symbols for safety
        syms = sorted(set(re.findall(r"[A-Za-z]\w*", a + b)))
        locals_dict = {x: sp.symbols(x) for x in syms}

        # Try symbolic simplification
        expr_a, expr_b = sp.sympify(a, locals=locals_dict), sp.sympify(b, locals=locals_dict)
        diff = sp.simplify(expr_a - expr_b)
        if diff == 0:
            return True

        # Try numeric evaluation (e.g., sqrt(16) vs 4)
        try:
            val_a = expr_a.evalf()
            val_b = expr_b.evalf()
            if abs(val_a - val_b) < 1e-6:
                return True
        except Exception:
            pass

    except Exception:
        # fallback: try evaluate both as numeric if possible
        try:
            val_a = sp.sympify(a).evalf()
            val_b = sp.sympify(b).evalf()
            if abs(val_a - val_b) < 1e-6:
                return True
        except Exception:
            return None

    return False


_number_re = re.compile(
    r"[+-]?(?:\d{1,3}(?:,\d{3})*(?:\.\d*)?|\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)
def strip_unit(s: str) -> str:
    """Remove common units or symbols (°, %, kg, m/s, GB, etc.) from numeric-like strings."""
    if not s:
        return ""
    s = s.strip()

    # m = _number_re.search(s)
    # if m:
    #     num_str = m.group(0)
    #     num_str = num_str.replace(",", "")
    #     return num_str

    s = re.sub(r"[°%℃]", "", s)
    s = re.sub(
        r"(?<=\d)\s*"
        r"(?:(?:mm/s|m/s|km/h|m/s²|m/s\^2|m/s2)|"
        r"(?:mm|cm|km|m)|"
        r"(?:mg|kg|g|t|lb|lbs)|"
        r"(?:hour|sec|min|day|hr|h|s)|"
        r"(?:ghz|mhz|khz|hz)|"
        r"(?:kw|mw|w|j|v|a|pa)|"
        r"(?:gb|mb|tb|byte|b))\b",
        "", s, flags=re.IGNORECASE
    )
    # print(s)
    s = s.strip()
    return s

def parse_number(s: str) -> Optional[float]:
    """
    Attempt to parse a string into a float.
    Handles:
    - Integers/Floats: 123, 12.34
    - Scientific: 1.2e-5, 1.2*10^5
    - Thousands separators: 1,000 (careful with lists)
    - Fractions: 1/2, 3/4
    - Percentages: 50% -> 0.5
    """
    s = normalize_answer(s)
    
    # 1. Handle Percentage
    if s.endswith('%'):
        try:
            return float(s[:-1]) / 100
        except:
            pass
            
    # 2. Handle Fraction (e.g., "1/4")
    if re.match(r'^-?\d+/\d+$', s):
        try:
            num, den = s.split('/')
            return float(num) / float(den)
        except:
            pass

    # 3. Clean string for standard float conversion
    # Remove ',' if it looks like a thousand separator (e.g. 1,000)
    # Heuristic: If comma is followed by 3 digits and not at end.
    if re.match(r'.*\d{1,3},\d{3}', s):
        s_clean = s.replace(',', '')
    else:
        s_clean = s

    try:
        return float(s_clean)
    except ValueError:
        pass
        
    # 4. Handle Scientific Notation like "1.2 * 10^5" or "1.2 x 10^5"
    sci_match = re.search(r'(-?\d+\.?\d*)\s*[\*x]\s*10\^?\{?(-?\d+)\}?', s)
    if sci_match:
        try:
            coef = float(sci_match.group(1))
            exp = float(sci_match.group(2))
            return coef * (10 ** exp)
        except:
            pass
            
    return None

def extract_numeric_core(s: str) -> Optional[float]:
    """
    Aggressively extract the main number from a string containing text/units.
    Examples: 
    - "$500" -> 500.0
    - "approx 5.2 meters" -> 5.2
    - "x = 5" -> 5.0
    """
    s = normalize_answer(s)
    # Regex for finding float/int patterns, including negatives
    # This pattern looks for numbers that might be standalone
    # Non-capturing groups for look-behinds/aheads to avoid grabbing version numbers etc.
    patterns = [
        r'(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?)',  # 1,234.56
        r'(-?\d+\.\d+)',                      # 123.456
        r'(-?\d+)'                            # 123
    ]
    
    for pat in patterns:
        matches = re.findall(pat, s)
        if matches:
            # Usually the last number is the answer (e.g. "x = 5"), but sometimes not.
            # Heuristic: Pick the one that parses successfully. 
            # If multiple, prefer the last one as it's often the result.
            try:
                # Need to handle the comma removal inside parse_number
                val = parse_number(matches[-1]) 
                if val is not None:
                    return val
            except:
                continue
    return None

def normalize_set_list(s: str) -> Optional[List[str]]:
    """Convert '1, 2, 3' or '{1, 2}' into a sorted list of strings."""
    s = normalize_answer(s)
    # Remove wrapping brackets if present
    if (s.startswith('{') and s.endswith('}')) or \
       (s.startswith('[') and s.endswith(']')) or \
       (s.startswith('(') and s.endswith(')')):
        s = s[1:-1]
    
    # Split by comma or semicolon
    parts = re.split(r'[,;]\s*', s)
    if len(parts) > 1:
        return sorted([p.strip() for p in parts if p.strip()])
    return None

def is_option_match(pred: str, gt: str) -> bool:
    def clean_opt(x):
        return re.sub(r'[\(\)]', '', x).strip().lower()
    
    def extract_opt_char(x):
        match = re.search(r'\b(option|choice)\s+([a-z])\b', x.lower())
        if match: return match.group(2)
        return clean_opt(x)

    return extract_opt_char(pred) == extract_opt_char(gt)

# ------------------------
# Scoring logic
# ------------------------

def compute_score_general(pred: str, gt: str) -> float:
    """Main scoring logic: numeric / symbolic / textual comparison."""
    try:
        pred_raw = extract_final_answer(pred) or pred
        gt_raw = extract_final_answer(gt) or gt
    except NameError:
        pred_raw, gt_raw = pred, gt

    pred_norm = normalize_answer(pred_raw)
    gt_norm = normalize_answer(gt_raw)

    # ✅ 1. Exact text match
    if pred_norm == gt_norm:
        return 1.0

    # ✅ 2. Numeric / Unit Match (The most common failure case)
    pred_val = parse_number(pred_norm)
    gt_val = parse_number(gt_norm)

    # If direct parse failed, try extracting core number from text (e.g. "5 kg")
    if pred_val is None: 
        pred_val = extract_numeric_core(pred_norm)
    if gt_val is None:
        gt_val = extract_numeric_core(gt_norm)

    # Compare floats if both exist
    if pred_val is not None and gt_val is not None:
        try:
            # Relative tolerance for large numbers, absolute for small
            if math.isclose(pred_val, gt_val, rel_tol=1e-3, abs_tol=1e-4):
                return 1.0
        except:
            pass

    try:
        pred_num = float(strip_unit(pred_raw))
        gt_num = float(strip_unit(gt_raw))
        if abs(pred_num - gt_num) < 1e-4:
            return 1.0
    except Exception:
        pass

    if (pred_norm in gt_norm or gt_norm in pred_norm):
        if any(c.isdigit() for c in pred_norm) and any(c.isdigit() for c in gt_norm):
             if pred_val is not None and gt_val is not None and math.isclose(pred_val, gt_val, rel_tol=1e-3):
                 return 1.0

    # ✅ 3. Set / List Match
    # Handles "x=1, y=2" vs "y=2, x=1" or "{1,2}" vs "1, 2"
    pred_list = normalize_set_list(pred_norm)
    gt_list = normalize_set_list(gt_norm)
    if pred_list and gt_list:
        if pred_list == gt_list:
            return 1.0
        # Check if numeric lists match (e.g. "1.0, 2.0" vs "1, 2")
        try:
            pred_nums = sorted([float(x) for x in pred_list])
            gt_nums = sorted([float(x) for x in gt_list])
            if len(pred_nums) == len(gt_nums):
                if all(math.isclose(p, g, rel_tol=1e-3) for p, g in zip(pred_nums, gt_nums)):
                    return 1.0
        except:
            pass

    # ✅ 4. Symbolic match (via sympy)
    # Good for latex expressions: \frac{1}{2} vs 0.5 (handled by sympy often)
    # or sqrt(2) vs 1.414 (handled if evaluated)
    if sympy_equiv(pred_raw, gt_raw):
        return 1.0

    number_part = r'[+-]?(?:\d+(?:[.,]\d*)?|\.\d+)'
    unit_patterns = [
        rf'^({number_part})\s*[°%]$',                                 # 65°, 100%
        rf'^({number_part})\s*[a-zA-Z]+/[a-zA-Z]+$',                 # 3 mm/s
        rf'^({number_part})\\s*[a-zA-Z]+$',                           # 100kg
        rf'^[€$£]\s*({number_part})$',                                # $3300
    ]

    for pat in unit_patterns:
        match = re.fullmatch(pat, gt_raw)
        if match:
            # print(match)
            if match.group(1) == pred_raw or match.group(1) == pred_norm or match.group(1) == strip_unit(pred_norm):
                return 1.0
            match_ans = re.fullmatch(pat, pred_raw)
            if match_ans and match.group(1) == match_ans.group(1):
                return 1.0

    if is_option_match(pred_raw, gt_raw):
        return 1.0

    yes_synonyms = {'yes', 'true', 'correct', 'yeah', 'yep'}
    no_synonyms = {'no', 'false', 'incorrect', 'wrong', 'nope'}
    if pred_norm in yes_synonyms and gt_norm in yes_synonyms:
        return 1.0
    if pred_norm in no_synonyms and gt_norm in no_synonyms:
        return 1.0

    # ✅ 6. Numeric-like fallback check
    if is_numeric_like(pred_norm) and is_numeric_like(gt_norm):
        if normalize_math_expr(pred_norm) == normalize_math_expr(gt_norm):
            return 1.0

    return 0.0


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> dict:
    """Unified scoring interface combining accuracy and format reward."""
    # Detect formatting issues
    is_format_error = (
        solution_str.count("<think>") != solution_str.count("</think>")
        or solution_str.count("<answer>") != solution_str.count("</answer>")
    )

    # Try to extract from <answer> tags
    answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    else:
        # Strategy 2: Use the entire solution_str as fallback
        answer_text = solution_str.strip()

    acc_reward = compute_score_general(answer_text, ground_truth)
    format_reward = -1.0 if is_format_error else format_reward_fn(solution_str)

    # Log debug information for problematic cases
    if is_format_error or not answer_text:
        logger.debug(
            f"Format issue detected:\n"
            f"Solution: {solution_str[:200]}...\n"
            f"Extracted answer: '{answer_text}'\n"
            f"Format error: {is_format_error}\n"
        )

    final_score = 1.0 * acc_reward + 0.2 * format_reward
    return {
        "score": final_score,
        "acc_reward": acc_reward,
        "format_reward": format_reward,
    }


# ------------------------
# Quick test
# ------------------------
if __name__ == '__main__':
    test_cases = [
        ("<think>Reasoning...</think><answer>64.27</answer>", "64.27"),
        ("<think>...</think><answer>a^2*sin(B)*sin(C)/(2*sin(A))</answer>", "a^2*sin(B)*sin(C)/(2*sin(A))"),
        ("The final answer is 13th Century.", "13th century"),
        ("<think>...</think><answer>\\frac{1}{2}</answer>", "0.5"),
        ("<think>...</think><answer>sqrt(16)</answer>", "4"),
        ("<think>...</think><answer>3000</answer>", "$3,000"),
        ("<think>...</think><answer>4WD</answer>", "4WD"),
        ("<think>...</think><answer>√34</answer>", "√34"),
        ("<think>...</think><answer>65</answer>", "65°"),
        ("<think>...</think><answer>34</answer>", "34GB"),
        ("<think>...</think><answer>34</answer>", "34 KG"),
        ("<think>...</think><answer>34 mm/s</answer>", "34"),
        ('<think>The image shows a book cover with the title "Steve Jobs: American Genius" by Amanda Ziller. The title and the name "Steve Jobs" suggest that the book is likely a biography or a biography-style book focusing on the life and achievements of Steve Jobs, the co-founder of Apple Inc. The subtitle "American Genius" implies that it is a book about his contributions and impact on American culture and technology.</think><answer>The book is likely a biography or a biography-style book focusing on the life and achievements of Steve Jobs.</answer>\\boxed{Biography}', "33")
    ]
    for i, (pred, gt) in enumerate(test_cases):
        result = compute_score(pred, gt)
        print(f"Case {i+1}:\n  Pred: {pred}\n  GT: {gt}\n  Result: {result}\n")
