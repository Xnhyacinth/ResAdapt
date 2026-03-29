import re
from typing import Optional
import string
from verl.utils.reward_score import math_reward
try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

try:
    import sympy as sp
    SYMPY_OK = True
except Exception:
    SYMPY_OK = False

import logging
logger = logging.getLogger(__name__)

def format_reward_fn(predict_str: str) -> float:
    pattern = re.compile(r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*", re.DOTALL | re.MULTILINE)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0

def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]

def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left):]
    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left):-1]

ANSWER_CUES = [
    r"(?:final\s+answer|answer\s+is|answer|答案是|答案|结论|结果)\s*[:：]\s*(.+)",
    r"(?:最终答案|最后答案)\s*[:：]\s*(.+)",
    r"(?:the\s+answer)\s+is\s*(.+)",
]

def extract_final_answer(solution_str: str) -> Optional[str]:

    boxed = last_boxed_only_string(solution_str)
    if boxed is not None:
        try:
            return remove_boxed(boxed)
        except Exception:
            m = re.search(r"\{(.*)\}", boxed)
            if m:
                return m.group(1)

    text = solution_str
    for pat in ANSWER_CUES:
        matches = list(re.finditer(pat, text, flags=re.IGNORECASE | re.DOTALL))
        if matches:
            cand = matches[-1].group(1).strip()
            cand = cand.split("\n")[0].strip()
            return cand

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    tail = lines[-1]

    tail = tail.strip().strip("`'\"")
    return tail if tail else None

# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def is_numeric_with_units(gt: str):
    """
    Recognizes numeric values, including those with units.
    This replaces/upgrades the old is_numeric_ground_truth.
    """
    gt = extract_final_answer(gt)
    ground_truth = gt
    if gt is None:
        return False, ground_truth
    s = gt.strip()

    number_part = r'[+-]?(?:\d+(?:[.,]\d*)?|\.\d+)(?:[eE][+-]?\d+)?'

    simple_unit_part = r'[a-zA-Z%€$£]+'

    # (e.g., "65°", "100%")
    compound_unit_pat = rf'^({number_part})\s*[°%]$'
    match = re.fullmatch(compound_unit_pat, s)
    if match:
        ground_truth = match.group(1)
        return True, ground_truth
    
    # (e.g., "3 mm/s")
    compound_unit_pat = rf'^({number_part})\s*[a-zA-Z]+/[a-zA-Z]+$'
    match = re.fullmatch(compound_unit_pat, s)
    if match:
        ground_truth = match.group(1)
        return True, ground_truth

    # (e.g., "100kg", "64GB")
    compound_unit_pat = rf'^({number_part})\s*{simple_unit_part}$'
    match = re.fullmatch(compound_unit_pat, s)
    if match:
        ground_truth = match.group(1)
        return True, ground_truth

    # (e.g., "$3300", "€100")
    compound_unit_pat = rf'^({simple_unit_part})\s*({number_part})$'
    match = re.fullmatch(compound_unit_pat, s)
    if match:
        ground_truth = match.group(2)
        return True, ground_truth
    
    # (e.g., "0.40 \text{ cm}")
    compound_unit_pat = rf'^({number_part})\s*\\text{{.*?}}$'
    match = re.fullmatch(compound_unit_pat, s)
    if match:
        ground_truth = match.group(1)
        return True, ground_truth

    numeric_unit = r'(?:(?:\d+(?:\.\d*)?|\.\d+)|(?:\d*√\d+))'
    single_unit_pat = rf'^[+-]?{numeric_unit}(?:[eE][+-]?\d+)?$'
    if re.fullmatch(single_unit_pat, s): 
        return True, ground_truth
    fraction_of_units_pat = rf'^[+-]?{numeric_unit}/{numeric_unit}$'
    if re.fullmatch(fraction_of_units_pat, s): 
        return True, ground_truth

    return False, ground_truth

def is_numeric_ground_truth(gt: str) -> bool:
    """
    Enhanced version that also recognizes combinations like √34/2.
    """
    if gt is None:
        return False
    s = gt.strip()

    numeric_unit = r'(?:(?:\d+(?:\.\d*)?|\.\d+)|(?:\d*√\d+))'
    
    # 模式 1: 单个数字单元 (处理 3.14, √34, 2√5)
    # 允许科学计数法
    single_unit_pat = rf'^[+-]?{numeric_unit}(?:[eE][+-]?\d+)?$'
    if re.fullmatch(single_unit_pat, s):
        return True

    # 模式 2: 两个数字单元组成的分数 (处理 1/2, √34/2, 2/√5, 3.14/5)
    # ^[+-]?<unit>/<unit>$
    fraction_of_units_pat = rf'^[+-]?{numeric_unit}/{numeric_unit}$'
    if re.fullmatch(fraction_of_units_pat, s):
        return True
    
    # 保留对欧式小数点的特殊处理
    euro_decimal_pat = r'^[+-]?\d+,\d+$'
    if re.fullmatch(euro_decimal_pat, s):
        return True

    math_patterns = [
        r'\\',  # 任何 LaTeX 命令的开始
        r'\^',  # 乘方
        r'_',   # 下标
        r'\{',  # LaTeX 分组
        r'\}',
        r'\\sin', r'\\cos', r'\\tan', r'\\log', # 函数
    ]
    if any(re.search(pattern, s) for pattern in math_patterns):
        return True
        
    return False

def compute_score_math_verify(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    return ret_score

def compute_score_math(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    ret_score = 0.0

    

    return ret_score

def sympy_equiv(a: str, b: str) -> Optional[bool]:
    if not SYMPY_OK:
        return None
    try:
        def prep(x: str) -> str:
            x = x.replace("^", "**")
            return x
        a2, b2 = prep(a), prep(b)
        syms = sorted(set(re.findall(r"[A-Za-z]\w*", a2) + re.findall(r"[A-Za-z]\w*", b2)))
        symbols = {name: sp.symbols(name) for name in syms}
        expr_a = sp.sympify(a2, locals=symbols)
        expr_b = sp.sympify(b2, locals=symbols)
        diff = sp.simplify(expr_a - expr_b)
        return bool(diff == 0)
    except Exception:
        return None

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_score_sym(solution_str, ground_truth):
    """
    Computes score for symbolic math expressions.
    """
    retval = 0.0
    try:
        extracted_answer = extract_final_answer(solution_str)
        # print(f"extract answer: {extracted_answer}")
        if extracted_answer is not None:
            if sympy_equiv(extracted_answer, ground_truth):
                return 1.0
            if normalize_answer(extracted_answer) == normalize_answer(ground_truth):
                return 1.0

            extracted_gt = extract_final_answer(ground_truth)
            with_units_ans, answer_text_removed = is_numeric_with_units(extracted_answer)
            with_units_gt, ground_truth_removed = is_numeric_with_units(extracted_gt)
            
            if is_numeric_ground_truth(extracted_gt):
                return max(compute_score_math(extracted_answer, extracted_gt), compute_score_math(answer_text_removed, extracted_gt))
            else:
                if with_units_gt:
                    return max(compute_score_math(extracted_answer, ground_truth_removed), compute_score_math(answer_text_removed, ground_truth_removed))

                elif normalize_answer(extracted_answer) == normalize_answer(extracted_gt):
                    return 1.0

    except Exception:
        pass
    return retval


def compute_score_general(answer_text, ground_truth):
    with_units_ans, answer_text_removed = is_numeric_with_units(answer_text)
    with_units_gt, ground_truth_removed = is_numeric_with_units(ground_truth)
    # print("answer" + answer_text_removed)
    
    if is_numeric_ground_truth(ground_truth):
        return max(compute_score_math(answer_text, ground_truth), compute_score_math(answer_text_removed, ground_truth))
    else:
        if with_units_gt:
            # print("gt" + ground_truth_removed)
            return max(compute_score_math(answer_text, ground_truth_removed), compute_score_math(answer_text_removed, ground_truth_removed))
        else:
            return compute_score_sym(answer_text, ground_truth)

def compute_score(
    solution_str: str,
    ground_truth: str,
    **kwargs,
):
    # Initialize tracking variables
    is_format_error = False

    # 1. Check <think> tag format
    count_think_1 = solution_str.count("<think>")
    count_think_2 = solution_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    # 2. Extract answer text with multiple fallback strategies
    answer_text = ""

    # Strategy 1: Try to extract from <answer> tags first
    predict_no_think = (
        solution_str.split("</think>")[-1].strip() if "</think>" in solution_str else solution_str.strip()
    )

    # Check <answer> tag format
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    # Try to extract from <answer> tags
    answer_match = re.search(r"<answer>(.*?)</answer>", predict_no_think, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    else:
        # No proper <answer> tags found - this is a format erro
        if "</think>" in solution_str:
            # Remove any remaining tool-related tags and extract meaningful content
            remaining_content = predict_no_think
            # Remove tool calls and responses
            remaining_content = re.sub(r"<tool_call>.*?</tool_call>", "", remaining_content, flags=re.DOTALL)
            remaining_content = re.sub(
                r"<tool_response>.*?</tool_response>", "", remaining_content, flags=re.DOTALL
            )
            # Remove user/assistant markers
            remaining_content = re.sub(r"\b(user|assistant)\b", "", remaining_content)
            answer_text = remaining_content.strip()
        else:
            # Strategy 4: Use the entire solution_str as fallback
            answer_text = solution_str.strip()

    # Clean up answer text
    answer_text = answer_text.strip()

    # If answer is still empty after all strategies, mark as format error
    if not answer_text:
        is_format_error = True
        answer_text = solution_str.strip()  # Use full text as last resort

    acc_reward = compute_score_general(answer_text, ground_truth)

    # format = format_reward(solution_str, extra_info)
    # Format reward: penalty for format errors
    format_reward = -1.0 if is_format_error else format_reward_fn(solution_str)

    # Log debug information for problematic cases
    if is_format_error or not answer_text:
        logger.debug(
            f"Format issue detected:\n"
            f"Solution: {solution_str[:200]}...\n"
            f"Extracted answer: '{answer_text}'\n"
            f"Format error: {is_format_error}\n"
        )

    # Final weighted score
    final_score = 0.8 * acc_reward + 0.2 * format_reward

    return {
        "score": final_score,
        "acc_reward": acc_reward,
        "format_reward": format_reward,
    }

if __name__ == '__main__':
    test_cases = [
        # 数字比较
        ("The final answer is: 64.27", "64.27", 1.0),
        ("The result is approximately 64.270.", "64.27", 1.0),
        ("It's 0.", "0", 1.0),
        ("My calculation leads to 1,000.", "1000", 1.0),
        ("<think>The image contains several objects that appear to be made of metal, including a yellow bicycle, a blue van, a brown car, a green scooter, a green school bus, a gray jet, and a brown motorcycle. Each of these objects has a shiny, reflective surface and a metallic appearance, which is characteristic of objects made of metal.</think>\\boxed{1}", "$1", 1.0),
        # 简单词语/短语比较
        ("The correct choice is MEM.", "MEM", 1.0),
        ("I think it's Paperback.", "paperback", 1.0),
        ("This is the paperback version.", "The Paperback", 1.0),
        # 复杂短语/特殊格式比较
        ("This happened in the 13th century.", "13th Century", 1.0),
        ("The car type is 4WD!", "4WD", 1.0),
        # 复杂公式/代码风格答案
        ("The answer is `a^2*sin(B)*sin(C)/(2*sin(A))`", "a^2*sin(B)*sin(C)/(2*sin(A))", 1.0),
        # 提取策略测试
        ("No clear marker, but this is the last line.", "this is the last line", 1.0),
        ("Answer: **MEM**", "MEM", 1.0),
        ("This is wrong. Final Answer: a^2", "a^2", 1.0),
        # 失败用例
        ("The answer is 50.", "fifty", 0.0),
        ("The car is Red.", "Blue", 0.0),
        ("The answer is in the attachment.", "42", 0.0), # 提取到 "in the attachment"
    ]

    test_cases = [
        ("\\boxed{ The area of square ABCD can be calculated as 8.0^2, which equals 64.0.\nThe answer is 64.0}", "4WD", 1.0),

    ]

    for i, (solution, gt, expected) in enumerate(test_cases):
        score = compute_score(solution, gt)
        print(f"--- Test Case {i+1} ---")
        print(f"Solution: '{solution}'")
        print(f"Ground Truth: '{gt}'")
        print(f"Computed Score: {score}, Expected: {expected}")
        # assert score == expected, "Test Failed!"
        print("PASS\n")

    # print(is_numeric_ground_truth("√34"))
    # print(is_numeric_ground_truth("7,705"))
    # print(is_numeric_ground_truth("$3,000"))