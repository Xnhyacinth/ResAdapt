import re
# from mathruler.grader import extract_boxed_content, grade_answer
# from math_verify import LatexExtractionConfig, parse, verify, ExprExtractionConfig
import os
from datetime import datetime
import time
import asyncio
import aiohttp
import json


QWEN_2_5_JUDGE_SYSTEM_PROMPT = """You are an expert in verifying if two answers are the same.
Your input is a problem and two answers, Answer 1 (**Model Predicted Answer**) and Answer 2 (**Ground Truth Answer**).
Your need to evaluate the model's predicted answer against the ground truth answer.
Your task is to determine if two answers are equivalent, without attempting to solve the original problem.
Compare the answers to verify they represent identical values or meaning, even when written in different forms or notations.

Your output must follow the following format:
1) Provide an explanation for why the answers are equivalent or not.
2) Then provide your final answer in the form of: [[YES]] or [[NO]]
"""

QWEN3_JUDGE_SYSTEM_PROMPT = """You are an expert in verifying if two answers are the same.
Your input is a problem and two answers, Answer 1 (**Model Predicted Answer**) and Answer 2 (**Ground Truth Answer**).
Your need to evaluate the model's predicted answer against the ground truth answer.
Your task is to determine if two answers are equivalent, without attempting to solve the original problem.
Compare the answers to verify they represent identical values or meaning, even when written in different forms or notations.

Your output should be only: [[YES]] if the answers are equivalent, or [[NO]] if they are not.
"""

QUERY_PROMPT = """
Problem: {question}
Answer 1 (**Model Predicted Answer**): {prediction}
Answer 2 (**Ground Truth Answer**): {ground_truth}
"""

# ADDRESS = "22.1.90.78"
# ADDRESS = "22.1.141.78"
ADDRESS = "22.0.156.69"
MODEL_NAME = "qwen3"

def extract_solution(solution_str: str) -> str:
    answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    extract_final_answer = (answer_match.group(1) or "").strip() if answer_match else ""
    return extract_final_answer if extract_final_answer else "Invalid Answer. Please output [[NO]]."

def qwen2_5_extract_judge(cur_judge: str) -> str:
    return cur_judge

def qwen3_extract_judge(cur_judge: str) -> str:
    match = re.search(r"</think>\s*(.*)", cur_judge, re.DOTALL)
    result = match.group(1).strip() if match else ""
    return result


def format_reward(predict_str: str, extra_info: dict = None) -> float:
    pattern = re.compile(r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*", re.DOTALL | re.MULTILINE)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


async def chat_completions_aiohttp(address, port=80, max_retries=5, **chat_complete_request):
    for attempt in range(max_retries):
        try:
            request_url = f"http://{address}:{port}/v1/chat/completions"
            timeout = aiohttp.ClientTimeout(total=None)
            session = aiohttp.ClientSession(timeout=timeout)
            async with session.post(
                url=request_url,
                json=chat_complete_request,
            ) as resp:
                output = await resp.text()
                try:
                    output = json.loads(output)
                    return output["choices"][0]["message"]["content"]
                except Exception as e:
                    print(f"Error: {e}. Output: {output}")
                    return ""
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(2 ** attempt)
        finally:
            await session.close()

async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    **kwargs,
):
    question = extra_info["question"]
    SYSTEM_PROMPT = QWEN3_JUDGE_SYSTEM_PROMPT
    extract_judge = qwen3_extract_judge

    acc_reward_weight = kwargs.get('acc_reward_weight', 1.0) if kwargs else 1.0
    format_reward_weight = kwargs.get('format_reward_weight', 1.0) if kwargs else 0.5
    # print(format_reward_weight)
    # print("*"*50)

    prompt = QUERY_PROMPT.format(question=question.replace("<image>", ""), prediction=solution_str, ground_truth=ground_truth)
    messages = [
        {
            "role": "system",
            "content":[
                    {"type": "text", "text": SYSTEM_PROMPT},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        },
    ]
    # print(messages)
    response = await chat_completions_aiohttp(
        ADDRESS,
        port=8000,
        messages=messages,
        model=MODEL_NAME,
        max_tokens=16384,
    )
    # print(response)
    if response is not None and len(response) > 0:  
        verification = extract_judge(response.strip())
        if "YES" in verification:
            acc = 1.0
        else:
            acc = 0.0
            if "NO" not in verification:
                print(f"Fail to judge response: {verification} Set score to 0.0.")
    else:
        return {
        "score": -1,
        "acc": -1,
        "format": -1,
    }

    format = format_reward(solution_str, extra_info)

    acc_score = acc_reward_weight * acc
    format_score = format_reward_weight * format
    score = acc_score + format_score

    return {
        "score": score,
        "acc": acc_score,
        "format": format_score,
    }

