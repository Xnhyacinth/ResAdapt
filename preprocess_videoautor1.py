import json
import logging
import os
import re
import string
from typing import Any, Dict, List, Literal, Optional, Tuple

from math_verify import parse
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DIRECT_SYSTEM_PROMPT = "You are a helpful assistant."

COT_SYSTEM_PROMPT0 = (
    "You are a helpful assistant. FIRST, think through the reasoning process as an internal monologue, and THEN provide the final answer. "
    "The reasoning process MUST be enclosed within <think> </think> tags, and the final answer MUST be wrapped in \\boxed{}."
)

COT_SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> </think> tags, and then give your answer between the <answer> </answer> tags. "
    "The final answer MUST BE wrapped in \\boxed{} and the \\boxed{} expression MUST BE contained entirely within the <answer> </answer> tags."
)

COT_SYSTEM_PROMPT_ANSWER_TWICE = (
    "You are a helpful assistant.\n"
    "FIRST: Output your initial answer inside the first \\boxed{...} without any analysis or explanations. "
    "If you cannot determine the answer without reasoning, output \\boxed{Let's analyze the problem step by step.} instead.\n"
    "THEN: Think through the reasoning as an internal monologue enclosed within <think>...</think>.\n"
    "AT LAST: Output the final answer again inside \\boxed{...}. If you believe the previous answer was correct, repeat it; otherwise, correct it.\n"
    "Output format: \\boxed{...}<think>...</think>\\boxed{...}"
)


def extract_boxed_content(text):
    # Find all occurrences of \boxed{...} with regex
    # This handles one level of nested braces by using a non-greedy match
    boxed_matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)

    # Return the last match if any matches were found
    if boxed_matches:
        return boxed_matches[-1]
    else:
        return text


def extract_answer(text):
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    logger.warning(f"<answer>...</answer> format not found: {text}, skip")
    return None


def _find_first_user_text(messages: List[Dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    return item["text"]
    return ""


def _find_first_system_text(messages: List[Dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "system" and isinstance(msg.get("content"), str):
            return msg["content"]
    return ""


def _extract_media_paths(messages: List[Dict[str, Any]]) -> Tuple[str, str]:
    image_path = ""
    video_path = ""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image" and isinstance(item.get("image"), str) and not image_path:
                image_path = item["image"]
            if item.get("type") == "video" and isinstance(item.get("video"), str) and not video_path:
                video_path = item["video"]
    return image_path, video_path


def autor1_to_preprocess_data_record(example: Dict[str, Any], idx: int, split: str) -> Dict[str, Any]:
    example = dict(example)
    messages = example.get("messages") or []
    if not isinstance(messages, list):
        messages = []

    system_text = _find_first_system_text(messages)
    question_text = _find_first_user_text(messages)
    image_path, video_path = _extract_media_paths(messages)

    tags = ""
    if image_path:
        tags += "<image>"
    if video_path:
        tags += "<video>"
    question_with_tags = tags + question_text

    # prompt = tags + (system_text + "\n" if system_text else "") + question_text
    prompt = tags + question_text

    response = example.get("response", "")
    if isinstance(response, (dict, list)):
        answer = json.dumps(response, ensure_ascii=False)
    elif isinstance(response, str):
        answer = response
    else:
        answer = str(response)

    doc_id = str(example.get("doc_id")) if example.get("doc_id") is not None else str(idx)

    raw_data_source = str(example.get("data_source") or "")
    images = [{"image": image_path}] if image_path else []
    videos = [{"video": video_path}] if video_path else []

    # raw_example = json.dumps(example, ensure_ascii=False, default=str)

    messages = [
        {"role": "system", "content": system_text},
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
        },
    }
    return data


class LazyGRPODataset(Dataset):
    """Dataset for GRPO fine-tuning."""

    def __init__(
        self,
        dataset_name: List[str],
        dataset_config: Dict[str, Dict[str, str]],
        image_min_pixels: Optional[int] = 4 * 28 * 28,
        image_max_pixels: Optional[int] = 16384 * 28 * 28,
        video_min_pixels: Optional[int] = 128 * 28 * 28,
        video_max_pixels: Optional[int] = 768 * 28 * 28,
        video_total_pixels: Optional[int] = 115200 * 28 * 28,
        max_frames: Optional[int] = 768,
        nframes: Optional[int] = None,
        fps: Optional[float] = 2.0,
        rl_mode: Literal["direct_rl", "cot_rl", "answer_twice_rl"] = "cot_rl",
    ):
        """
        Args:
            video_min_pixels (int, optional): The minimum pixels number for a frame of a video. Defaults to 128 * 28 * 28.
            video_max_pixels (int, optional): The maximum pixels number for a frame of a video. Defaults to 768 * 28 * 28.
            video_total_pixels (int, optional): The total pixels number for a video. Defaults to 115200 * 28 * 28.
            max_frames (int, optional): The maximum frames number allowed for a video. Defaults to 768.
            nframes (int, optional): The exact number of frames to extract for a video. Defaults to None, meaning the frame number is decided by fps and max_frames.
            fps (float, optional): The fps to extract frames for a video. Defaults to None.
        """

        super(LazyGRPODataset, self).__init__()

        # system prompt
        if rl_mode == "direct_rl":
            self.system_prompt = DIRECT_SYSTEM_PROMPT
        elif rl_mode == "cot_rl":
            self.system_prompt = COT_SYSTEM_PROMPT
        elif rl_mode == "answer_twice_rl":
            self.system_prompt = COT_SYSTEM_PROMPT_ANSWER_TWICE
        else:
            raise ValueError(f"rl_mode {rl_mode} is not supported.")
        logger.info(f"System Prompt:\n{self.system_prompt}")

        # data configuration for fetch_image
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        logger.info(
            f"image_min_pixels: {self.image_min_pixels}, "
            f"image_max_pixels: {self.image_max_pixels}"
        )

        # data configuration for fetch_video
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels
        self.video_total_pixels = video_total_pixels
        self.max_frames = max_frames
        self.nframes = nframes
        self.fps = fps
        logger.info(
            f"video_min_pixels: {self.video_min_pixels}, "
            f"video_max_pixels: {self.video_max_pixels}, "
            f"video_total_pixels: {self.video_total_pixels}, "
            f"max_frames: {self.max_frames}, "
            f"nframes: {self.nframes}, "
            f"fps: {self.fps}"
        )

        # load dataset
        list_data_dict = []
        logger.info(f"Dataset name: {dataset_name}")
        for name in dataset_name:
            logger.info(f"Loading dataset {name}")
            if name == "DAPO-Math":
                new_data_dict = self.load_dapo_math(dataset_config[name])
            elif name == "VIRL":
                new_data_dict = self.load_virl(dataset_config[name])
            elif name == "ThinkLite-VL-Hard":
                new_data_dict = self.load_thinklite(dataset_config[name])
            elif name == "VideoR1":
                new_data_dict = self.load_videor1(dataset_config[name])
            elif name == "LongVideoReason":
                new_data_dict = self.load_longvideoreason(dataset_config[name])
            elif name == "TVBench":
                new_data_dict = self.load_tvbench(dataset_config[name])
            elif name == "STI-Bench":
                new_data_dict = self.load_stibench(dataset_config[name])
            elif name == "MMR-VBench":
                new_data_dict = self.load_mmrv(dataset_config[name])
            elif name == "CG-Bench":
                new_data_dict = self.load_cgbench(dataset_config[name])
            elif name == "Charades-STA":
                new_data_dict = self.load_charades(dataset_config[name])
            elif name == "NeXT-GQA":
                new_data_dict = self.load_nextgqa(dataset_config[name])
            elif name == "ActivityNet-TVG":
                new_data_dict = self.load_anet(dataset_config[name])
            elif name == "TimeR1":
                new_data_dict = self.load_timer1(dataset_config[name])
            else:
                raise NotImplementedError(f"Dataset {name} is not supported.")
            logger.info(f"Dataset {name} loaded {len(new_data_dict)} examples")
            for ex in new_data_dict:
                if isinstance(ex, dict) and "data_source" not in ex:
                    ex["data_source"] = name
            list_data_dict.extend(new_data_dict)

        self.list_data_dict = list_data_dict
        logger.info(f"Total loaded {len(self.list_data_dict)} examples")

    def convert_to_message(
        self,
        question: str,
        answer: str,
        image: Optional[Any] = None,
        video: Optional[Any] = None,
        video_segment_info: Optional[Dict] = None,
        grounding: Optional[bool] = False,
    ) -> Tuple[List[Dict], str]:
        if (image is not None) and (video is not None):
            raise ValueError("`image` and `video` cannot be provided simultaneously.")

        user_content = []
        if image is not None:
            visual_dict = {
                "type": "image",
                "image": image,
                "min_pixels": self.image_min_pixels,
                "max_pixels": self.image_max_pixels,
            }
            user_content.append(visual_dict)

        if video is not None:
            visual_dict = {
                "type": "video",
                "video": video,
                "min_pixels": self.video_min_pixels,
                "max_pixels": self.video_max_pixels,
                "total_pixels": self.video_total_pixels,
                "max_frames": self.max_frames,
                "fps": self.fps,
            }

            # if nframes is provided, sample fixed number of frames
            if self.nframes is not None:
                visual_dict["nframes"] = self.nframes
                visual_dict.pop("fps")

            # if video_segment_info is provided, sample frames from the specified segment
            if video_segment_info is not None:
                visual_dict.update(video_segment_info)
            user_content.append(visual_dict)

        if not grounding:
            question = question + "\nPut your final answer in \\boxed{}."
        user_content.append({"type": "text", "text": question})

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        response = answer  # f"\\boxed{{{answer}}}"
        return messages, response

    def load_dapo_math(self, dataset_config):
        dataset = json.load(open(dataset_config["anno_path"], "r"))
        list_data_dict = []

        for ex in dataset:
            question = ex["prompt"]
            answer = ex["solution"]
            messages, response = self.convert_to_message(question, answer)
            list_data_dict.append(
                {
                    "messages": messages,
                    "response": response,
                    "problem_type": "math",
                }
            )
        return list_data_dict

    def load_virl(self, dataset_config):
        dataset = json.load(open(dataset_config["anno_path"], "r"))
        list_data_dict = []

        for ex in dataset:
            question = ex["question"].replace("<image>", "").strip()

            # skip if there are multiple images
            if len(ex["image"]) > 1:
                continue

            image_path = os.path.join(dataset_config["data_path"], ex["image"][0])

            if not os.path.exists(image_path):
                logger.warning(f"Missing file: {image_path}, skip it.")
                continue

            answer = extract_boxed_content(ex["answer"])
            messages, response = self.convert_to_message(
                question, answer, image=image_path
            )

            if len(parse(response)) == 0:  # eg: "Cartoonish", "A"
                problem_type = "exact_match"
            else:
                problem_type = "math"

            list_data_dict.append(
                {
                    "messages": messages,
                    "response": response,
                    "problem_type": problem_type,
                }
            )
        return list_data_dict

    def load_thinklite(self, dataset_config):
        dataset = json.load(open(dataset_config["anno_path"], "r"))

        list_data_dict = []
        for ex in dataset:
            question = ex["problem"].replace("<image>", "").strip()
            answer = ex["ground_truth"]
            image_path = os.path.join(dataset_config["data_path"], ex["image"])

            if not os.path.exists(image_path):
                logger.warning(f"Missing file: {image_path}, skip it.")
                continue

            messages, response = self.convert_to_message(
                question, answer, image=image_path
            )

            list_data_dict.append(
                {
                    "messages": messages,
                    "response": response,
                    "problem_type": ex["problem_type"],
                }
            )
        return list_data_dict

    def load_videor1(self, dataset_config):
        dataset = json.load(open(dataset_config["anno_path"], "r"))
        list_data_dict = []

        for ex in dataset:
            if ex["problem_type"] not in ["multiple choice", "numerical"]:
                continue

            if ex["problem_type"] == "multiple choice":
                question = ex["problem"] + "Options:\n"
                for op in ex["options"]:
                    question += op + "\n"
            else:
                question = ex["problem"]

            data_path = os.path.join(dataset_config["data_path"], ex["path"])
            data_path = os.path.normpath(data_path)

            if not os.path.exists(data_path):
                logger.warning(f"Missing file: {data_path}, skip it.")
                continue

            answer = extract_answer(ex["solution"].lstrip("\n"))
            if answer is None:
                continue

            visual_kwargs = {ex["data_type"]: data_path}
            messages, response = self.convert_to_message(
                question, answer, **visual_kwargs
            )

            # reward function
            if ex["problem_type"] in ["multiple choice"]:
                problem_type = "exact_match"
            elif ex["problem_type"] in ["numerical"]:
                problem_type = "math"
            else:
                raise NotImplementedError(
                    f"Problem type {ex['problem_type']} is not supported."
                )
            list_data_dict.append(
                {
                    "messages": messages,
                    "response": response,
                    "problem_type": problem_type,
                }
            )
        return list_data_dict

    def load_longvideoreason(self, dataset_config):
        dataset = []
        with open(dataset_config["anno_path"], "r") as f:
            for line in f:
                dataset.append(json.loads(line))
        list_data_dict = []

        for ex in dataset:
            question = ex["problem"]
            data_path = os.path.join(dataset_config["data_path"], ex["videos"])
            data_path = os.path.normpath(data_path)

            assert ex["data_type"] == "video"
            if not os.path.exists(data_path):
                logger.warning(f"Missing file: {data_path}, skip it.")
                continue

            answer = extract_answer(ex["answer"])
            messages, response = self.convert_to_message(
                question, answer, video=data_path
            )
            list_data_dict.append(
                {
                    "messages": messages,
                    "response": response,
                    "problem_type": "exact_match",
                }
            )
        return list_data_dict

    def load_tvbench(self, dataset_config):
        list_data_dict = []
        for file in os.listdir(dataset_config["anno_path"]):
            assert file.endswith(".json")
            dataset = json.load(
                open(os.path.join(dataset_config["anno_path"], file), "r")
            )

            for ex in dataset:
                option_prompt = ""
                option_letters = string.ascii_uppercase
                for char_index, option in enumerate(ex["candidates"]):
                    option_letter = option_letters[char_index]
                    option_prompt += f"{option_letter}. {option}\n"

                question = (
                    "Question: " + ex["question"] + "\nOptions:\n" + option_prompt
                )
                answer = option_letters[ex["candidates"].index(ex["answer"])]

                data_path = os.path.join(
                    dataset_config["data_path"], file[:-5], ex["video"]
                )
                data_path = os.path.normpath(data_path)

                if not os.path.exists(data_path):
                    logger.warning(f"Missing file: {data_path}, skip it.")
                    continue

                # add video_start, video_end
                if file in ["action_sequence.json", "action_localization.json"]:
                    video_segment_info = {
                        "video_start": ex["start"],
                        "video_end": ex["end"],
                    }
                else:
                    video_segment_info = None

                messages, response = self.convert_to_message(
                    question,
                    answer,
                    video=data_path,
                    video_segment_info=video_segment_info,
                )
                list_data_dict.append(
                    {
                        "messages": messages,
                        "response": response,
                        "problem_type": "exact_match",
                    }
                )
        return list_data_dict

    def load_stibench(self, dataset_config):
        dataset = json.load(open(dataset_config["anno_path"], "r"))
        list_data_dict = []

        for ex in dataset:
            cand_str = "\n".join([f"({k}) {v}" for k, v in ex["Candidates"].items()])
            ts, te = ex["time_start"], ex["time_end"]
            question = f"From {ts} s to {te} s. {ex['Question']}\n{cand_str}"
            answer = ex["Answer"]

            data_path = os.path.join(dataset_config["data_path"], ex["Video"])
            data_path = os.path.normpath(data_path)

            if not os.path.exists(data_path):
                logger.warning(f"Missing file: {data_path}, skip it.")
                continue

            messages, response = self.convert_to_message(
                question, answer, video=data_path
            )
            list_data_dict.append(
                {
                    "messages": messages,
                    "response": response,
                    "problem_type": "exact_match",
                }
            )
        return list_data_dict

    def load_mmrv(self, dataset_config):
        dataset = json.load(open(dataset_config["anno_path"], "r"))
        list_data_dict = []

        for ex in dataset:
            options = "\n".join(
                [op[1] + "." + op[3:] for op in ex["options"] if op != ""]
            )
            question = ex["question"] + "\nOptions:\n" + options
            answer = ex["correctAnswer"][1]

            data_path = os.path.join(dataset_config["data_path"], ex["video"])
            data_path = os.path.normpath(data_path)

            if not os.path.exists(data_path):
                logger.warning(f"Missing file: {data_path}, skip it.")
                continue

            messages, response = self.convert_to_message(
                question, answer, video=data_path
            )
            list_data_dict.append(
                {
                    "messages": messages,
                    "response": response,
                    "problem_type": "exact_match",
                }
            )
        return list_data_dict

    def load_cgbench(self, dataset_config):
        dataset = json.load(open(dataset_config["anno_path"], "r"))
        list_data_dict = []
        option_letters = string.ascii_uppercase

        for ex in dataset:
            options = [
                f"{option_letters[idx]}. {op}" for idx, op in enumerate(ex["choices"])
            ]
            question = ex["question"] + "\nOptions:\n" + "\n".join(options)
            answer = ex["right_answer"]

            data_path = os.path.join(
                dataset_config["data_path"], "filtered_videos", ex["video_uid"] + ".mp4"
            )
            data_path = os.path.normpath(data_path)

            if not os.path.exists(data_path):
                logger.warning(f"Missing file: {data_path}, skip it.")
                continue

            messages, response = self.convert_to_message(
                question, answer, video=data_path
            )
            list_data_dict.append(
                {
                    "messages": messages,
                    "response": response,
                    "problem_type": "exact_match",
                }
            )
        return list_data_dict

    def load_charades(self, dataset_config):
        dataset = json.load(open(dataset_config["anno_path"], "r"))
        list_data_dict = []

        pre_prompt = "Locate the start and end timestamps of the video segment corresponding to the description: "
        post_prompt = " Please provide the start and end timestamps (in seconds, precise to one decimal places) in the format \\boxed{[start, end]}."

        for ex in dataset:
            question = f"{pre_prompt}{ex['description']}{post_prompt}"
            answer = ex["timestamps"]

            data_path = os.path.join(dataset_config["data_path"], ex["video"])
            data_path = os.path.normpath(data_path)

            if not os.path.exists(data_path):
                logger.warning(f"Missing file: {data_path}, skip it.")
                continue

            messages, response = self.convert_to_message(
                question,
                answer,
                video=data_path,
                grounding=True,
            )
            list_data_dict.append(
                {
                    "messages": messages,
                    "response": response,
                    "problem_type": "iou",
                }
            )
        return list_data_dict

    def load_nextgqa(self, dataset_config):
        dataset = json.load(open(dataset_config["anno_path"], "r"))
        list_data_dict = []

        post_prompt = (
            "Please answer the question, and then provide the video segment that best supports your answer. "
            "The start and end timestamps must be in seconds with exactly one decimal place.\n"
            "Output format: \\boxed{answer <> [start, end]}. For example: \\boxed{A <> [20.3, 30.8]}."
        )
        for ex in dataset:
            options = "\n".join(ex["options"])
            question = f"Question: {ex['question']}\nOptions:\n{options}\n{post_prompt}"
            answer = {"answer": ex["answer"], "segment": ex["timestamps"]}

            data_path = os.path.join(dataset_config["data_path"], ex["video"])
            data_path = os.path.normpath(data_path)

            if not os.path.exists(data_path):
                logger.warning(f"Missing file: {data_path}, skip it.")
                continue

            messages, response = self.convert_to_message(
                question,
                answer,
                video=data_path,
                grounding=True,
            )
            list_data_dict.append(
                {
                    "messages": messages,
                    "response": response,
                    "problem_type": "gqa",
                }
            )
        return list_data_dict

    def load_anet(self, dataset_config):
        dataset = json.load(open(dataset_config["anno_path"], "r"))
        list_data_dict = []

        pre_prompt = "Locate the start and end timestamps of the video segment corresponding to the description: "
        post_prompt = " Please provide the start and end timestamps (in seconds, precise to one decimal places) in the format \\boxed{[start, end]}."

        for ex in dataset:
            question = f"{pre_prompt}{ex['description']}{post_prompt}"
            answer = ex["timestamps"]

            data_path = os.path.join(dataset_config["data_path"], ex["video"])
            data_path = os.path.normpath(data_path)

            if not os.path.exists(data_path):
                logger.warning(f"Missing file: {data_path}, skip it.")
                continue

            messages, response = self.convert_to_message(
                question,
                answer,
                video=data_path,
                grounding=True,
            )
            list_data_dict.append(
                {
                    "messages": messages,
                    "response": response,
                    "problem_type": "iou",
                }
            )
        return list_data_dict

    def load_timer1(self, dataset_config):
        dataset = json.load(open(dataset_config["anno_path"], "r"))
        list_data_dict = []

        pre_prompt = "Locate the start and end timestamps of the video segment corresponding to the description: "
        post_prompt = " Please provide the start and end timestamps (in seconds, precise to one decimal places) in the format \\boxed{[start, end]}."

        for ex in dataset:
            question = f"{pre_prompt}{ex['sentence']}{post_prompt}"
            answer = ex["timestamp"]

            data_path = os.path.join(dataset_config["data_path"], ex["video"])
            data_path = os.path.normpath(data_path)

            if not os.path.exists(data_path):
                logger.warning(f"Missing file: {data_path}, skip it.")
                continue

            messages, response = self.convert_to_message(
                question,
                answer,
                video=data_path,
                grounding=True,
            )
            list_data_dict.append(
                {
                    "messages": messages,
                    "response": response,
                    "problem_type": "iou",
                }
            )
        return list_data_dict

    def __getitem__(self, i: int):
        return self.list_data_dict[i]

    def __len__(self) -> int:
        return len(self.list_data_dict)


if __name__ == "__main__":
    import argparse

    try:
        from datasets import Dataset as HFDataset
    except Exception as e:
        raise ModuleNotFoundError("Missing dependency: datasets (pip install datasets[parquet])") from e

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_names", default=None, help="Comma-separated dataset names, e.g. VideoR1,TVBench")
    parser.add_argument("dataset_names_pos", nargs="*", help="Dataset names separated by spaces")
    parser.add_argument("--dataset_config_path", required=True, help="JSON file mapping dataset name -> config dict")
    parser.add_argument("--local_save_dir", required=True)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--split", default="train")
    parser.add_argument("--rl_mode", default="cot_rl")
    args = parser.parse_args()

    dataset_config = json.load(open(args.dataset_config_path, "r"))
    if args.dataset_names:
        dataset_names = [x.strip() for x in args.dataset_names.split(",") if x.strip()]
    else:
        dataset_names = []
        for item in args.dataset_names_pos:
            dataset_names.extend([x.strip() for x in item.split(",") if x.strip()])
    if not dataset_names:
        raise ValueError("No dataset names provided. Use --dataset_names or positional dataset names.")

    ds = LazyGRPODataset(dataset_name=dataset_names, dataset_config=dataset_config, rl_mode=args.rl_mode)
    records = [autor1_to_preprocess_data_record(ex, i, args.split) for i, ex in enumerate(ds.list_data_dict)]

    out_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.split}.parquet")
    HFDataset.from_list(records).to_parquet(out_path)
