import re
# from mathruler.grader import extract_boxed_content, grade_answer
# from math_verify import LatexExtractionConfig, parse, verify, ExprExtractionConfig
import os
from datetime import datetime
import time
import openai
import random
import requests
import json

SYSTEM_PROMPT = "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs.\nYour task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:\n------\n##INSTRUCTIONS:\n- Focus on the meaningful match between the predicted answer and the correct answer.\n- Consider synonyms or paraphrases as valid matches.\n- Evaluate the correctness of the prediction compared to the answer."

QUERY_PROMPT = """I will give you a question related to an image and the following text as inputs:\n\n1. **Question Related to the Image**: {question}\n2. **Ground Truth Answer**: {ground_truth}\n3. **Model Predicted Answer**: {prediction}\n\nYour task is to evaluate the model's predicted answer against the ground truth answer, based on the context provided by the question related to the image. Consider the following criteria for evaluation:\n- **Relevance**: Does the predicted answer directly address the question posed, considering the information provided by the given question?\n- **Accuracy**: Compare the predicted answer to the ground truth answer. You need to evaluate from the following two perspectives:\n(1) If the ground truth answer is open-ended, consider whether the prediction accurately reflects the information given in the ground truth without introducing factual inaccuracies. If it does, the prediction should be considered correct.\n(2) If the ground truth answer is a definitive answer, strictly compare the model's prediction to the actual answer. Pay attention to unit conversions such as length and angle, etc. As long as the results are consistent, the model's prediction should be deemed correct.\n**Output Format**:\nYour response should include an integer score indicating the correctness of the prediction: 1 for correct and 0 for incorrect. Note that 1 means the model's prediction strictly aligns with the ground truth, while 0 means it does not.\nThe format should be \"Score: 0 or 1\""""

class GPT4VisionClient:
    """Client for interacting with GPT-4 Vision API"""

    def __init__(self, endpoint=None, api_key=None):
        self.api_key = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
        self.endpoint = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
        self.api_version = os.getenv("AZURE_API_VERSION", "2023-07-01-preview")
        self.client = openai.AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
            api_key=self.api_key,
        )

    def query(
        self, images, prompt: str, system_prompt: str = None, max_retries=5, initial_delay=8
    ) -> str:
        """Query GPT-4 Vision with an image and prompt"""
        if system_prompt is not None:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_prompt},
                    ],
                },
            ]
        else:
            messages = []
        messages.append(
            {
                "role": "user",
                "content": [
                    # {"type": "text", "text": prompt},
                    # {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        )

        messages[-1]["content"].append({"type": "text", "text": prompt})

        attempt = 0
        while attempt < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-2024-11-20",
                    messages=messages,
                    temperature=min(0.2*attempt, 1.0),
                    max_tokens=16384,
                    timeout=120,
                )

                if "1" not in response.choices[0].message.content and '0' not in response.choices[0].message.content:
                    raise ValueError("No '0' nor '1' in the response: {}".format(response.choices[0].message.content))
                return response.choices[0].message.content
            except openai.RateLimitError as e:
                print(str(e))
                time.sleep(3)
                continue
            except Exception as e:
                print("="*100)
                print(str(e))
                print("messages: ", messages)
                print("="*100)
                delay = 10
                time.sleep(delay)
            attempt += 1
        print(f"Warning: Failed after {max_retries} attempts")
        return ""

client = GPT4VisionClient()

# ADDRESS = "22.1.90.78"
# ADDRESS = "22.1.141.78"
ADDRESS = "22.0.156.69"
MODEL_NAME = "qwen3"

def chat_completions_requests(address, port=80, max_retries=5, **chat_complete_request):
    attempt = 0
    while attempt < max_retries:
        try:
            request_url = f"http://{address}:{port}/v1/chat/completions"
            # 同步会话管理，使用with语句自动关闭会话
            with requests.Session() as session:
                # 发送POST请求（同步），不设置超时（对应原代码的total=None）
                resp = session.post(
                    url=request_url,
                    json=chat_complete_request,
                    timeout=None  # 不限制超时时间，与原代码一致
                )
                output = resp.text
                try:
                    output = json.loads(output)
                    return output["choices"][0]["message"]["content"]
                except Exception as e:
                    print(f"Error: {e}. Output: {output}")
                    return ""
        except Exception as e:
            print(f"Request failed: {e}")
            delay = 10
            time.sleep(delay)
            attempt += 1
    print(f"Warning: Failed after {max_retries} attempts")
    return ""

def format_reward(predict_str: str, extra_info: dict = None) -> float:
    pattern = re.compile(r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*", re.DOTALL | re.MULTILINE)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0

def extract_last_score(response: str) -> float:
    pattern = r"score:\s*(1(?:\.0+)?|0(?:\.0+)?)\b"
    matches = re.findall(pattern, response, flags=re.IGNORECASE)
    return float(matches[-1]) if matches else 0.0


def inner_acc_reward(prompt:str, predict_str: str, original_answer: str, use_gpt=False, gpt_extract_answer=False):
    original_predict_str = predict_str
    
    if gpt_extract_answer:
        original_predict_str = original_predict_str.split("<answer>")[-1].split("</answer>")[0].strip()

    question = prompt

    prompt = QUERY_PROMPT.format(question=question, ground_truth=original_answer, prediction=original_predict_str)
    # response = client.query(images=[], prompt=prompt, system_prompt=SYSTEM_PROMPT)
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

    response = chat_completions_requests(
        ADDRESS,
        port=8000,
        messages=messages,
        model=MODEL_NAME,
        max_tokens=16384,
    )

    if len(response) == 0:
        reward = {
                "score": -1,
                "acc": -1,
                "format": -1,
            }
    else:
        reward = extract_last_score(response)
        # reward = 1.0 if '1' in response else 0.0
    return reward


def acc_reward(prompt: str, predict_str: str, solution: str, extra_info: dict = None) -> float:
    gpt_extract_answer = extra_info.get("gpt_extract_answer", False)
    reward = inner_acc_reward(prompt, predict_str, solution, use_gpt=True, gpt_extract_answer=gpt_extract_answer)
    return reward

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    **kwargs,
):  
    question = extra_info["question"]
    acc_reward_weight = extra_info.get('acc_reward_weight', 1.0) if extra_info else 1.0
    format_reward_weight = extra_info.get('format_reward_weight', 1.0) if extra_info else 1.0
    acc = acc_reward(question, solution_str, ground_truth, extra_info)
    
    if isinstance(acc, dict):
        return acc
    
    format = format_reward(solution_str, extra_info)

    acc_score = acc_reward_weight * acc
    format_score = format_reward_weight * format
    score = acc_score + format_score

    return score, acc_score, format_score


if __name__ == '__main__':
    question = "<image>\nHint: Please answer the question and provide the final answer at the end.\nQuestion: How many states are represented by the lightest color on the map?" #"<image>What is the output score when the first input is 4 and the second input is 5 according to the Hamlet Evaluation System shown in Figure 2?" #"<image>Who wrote this book?\nAnswer the question with a short phrase."
    predict_str = "<think> The image shows an agenda for a design showcasing with five teams listed: Team 1, Team 2, Team 3, Team 4, and Team 5. This indicates that there are five teams presenting. </think>\n<answer> China </answer>" #"<think> Since triangles ACD and ABC share the same height (the perpendicular distance from point C to line AB), the ratio of their areas is equal to the ratio of their bases. The base of triangle ACD is AB, and the base of triangle ABC is also AB. Therefore, the ratio of the area of triangle ACD to the area of triangle ABC is 1:1. </think>\n<answer> $ 2\\sqrt{26} $ </answer>" #"<think> The book is a 2016 calendar featuring sloths, as indicated by the title and the image of sloths on the cover. </think> <answer>Calendars</answer>" #"<think> Since triangles ACD and ABC share the same height (the perpendicular distance from point C to line AB), the ratio of their areas is equal to the ratio of their bases. The base of triangle ACD is AB, and the base of triangle ABC is also AB. Therefore, the ratio of the area of triangle ACD to the area of triangle ABC is 1:1. </think>\n<answer> 2 </answer>" #" <think>...xxx</think> <answer>\\boxed{1/2}</answer>\n" #"2\sqrt{26}" #"1/2" #" <think> abcd\n</think>\nSo the answer is \\boxed{abc} This" #"Given that AB is 9.0. Since ABCD is a rectangle, the opposite sides must be equal. Thus, DC is also 9.0. DC has a length of 9.0. As DCEF is a square, its perimeter is 9.0 times 4, giving 36.0.\nThe answer is 36.0" #"Twelfth Edition" #"<think>...</think>\n<answer>\\boxed{f'(3)}</answer>"
    ground_truth = "Martha White" #"china" #"$ 2 $" #"A" #"1:3" #"0.5 cm" #"0.5"
    extra_info = {
        "acc_reward_weight": 1.0,
        "format_reward_weight": 1.0,
        "extract_answer_tags": "split",
    }
    s1 = compute_score(question, predict_str, ground_truth, extra_info)
    print(s1)