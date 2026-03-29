import time
from typing import List

import torch
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.simple.qwen3_vl_custom import Qwen3_VL_Custom as Qwen3_VLSimpleCustom
from lmms_eval.protocol import ChatMessages

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen3_vl_custom_chat")
class Qwen3_VL_Custom(Qwen3_VLSimpleCustom):
    is_simple = False

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            return x[0], x[0]

        req_and_args = [(req, req.args) for req in requests]
        re_ords = utils.Collator(
            req_and_args,
            lambda x: _collate(x[1]),
            group_fn=lambda x: x[1][2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (
            len(requests) // self.batch_size
            if len(requests) % self.batch_size == 0
            else len(requests) // self.batch_size + 1
        )
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0
        total_tokens = 0
        ttfts = []
        tpops = []
        merged_videos = []
        visual_merged_ratios = []
        input_merged_ratios = []

        for chunk in chunks:
            reqs, args_list = zip(*chunk)
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*args_list)
            chat_messages = [
                doc_to_messages[idx](self.task_dict[task][split][ids])
                for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))
            ]
            chat_messages = [ChatMessages(**{"messages": message}) for message in chat_messages]
            gen_kwargs = all_gen_kwargs[0]

            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
            # if self.fps is not None:
            #     video_kwargs["fps"] = self.fps
            #     video_kwargs["max_frames"] = self.max_num_frames
            # else:
            #     video_kwargs["nframes"] = self.max_num_frames
            video_kwargs["max_frames"] = self.max_num_frames

            batched_messages = [
                chat_message.to_hf_messages(video_kwargs=video_kwargs)
                for chat_message in chat_messages
            ]
            texts = self.processor.apply_chat_template(
                batched_messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs, video_kwargs_qwen = process_vision_info(
                batched_messages,
                return_video_kwargs=True,
                image_patch_size=16,
                return_video_metadata=True,
            )
            video_kwargs = {**video_kwargs, **video_kwargs_qwen}

            video_metadatas = None
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs, video_metadatas = (
                    list(video_inputs),
                    list(video_metadatas),
                )

            padding_side = "left" if self.batch_size > 1 else "right"
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadatas,
                **video_kwargs,
                do_resize=False,
                padding=True,
                padding_side=padding_side,
                return_tensors="pt",
            )

            input_ids = inputs["input_ids"][0]
            video_pad_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
            target_idxs = torch.where(input_ids == video_pad_id)[0]
            if target_idxs.numel() > 0 and "video_grid_thw" in inputs:
                target_start_idx = target_idxs[0].item()
                target_end_idx = target_idxs[-1].item()
                grid = inputs["video_grid_thw"].squeeze().tolist()
                T, H, W = grid
                merge_size = getattr(self.processor.video_processor, "merge_size", 1)
                H = H // merge_size
                W = W // merge_size
            else:
                target_start_idx = 0
                target_end_idx = 0
                T, H, W = 0, 0, 0

            prompt_stat = {
                "sys": target_start_idx,
                "inst": len(input_ids) - (target_end_idx + 1),
                "frame": T,
                "video": len(target_idxs),
                "T": T,
                "H": H,
                "W": W,
            }

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs["top_k"] = None

            start_time = time.time()
            try:
                cont = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=current_gen_kwargs["do_sample"],
                    temperature=current_gen_kwargs["temperature"],
                    top_p=current_gen_kwargs["top_p"],
                    num_beams=current_gen_kwargs["num_beams"],
                    max_new_tokens=current_gen_kwargs["max_new_tokens"],
                    top_k=current_gen_kwargs.get("top_k", None),
                    use_cache=self.use_cache,
                    prompt_stat=prompt_stat,
                )
            except TypeError:
                cont = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=current_gen_kwargs["do_sample"],
                    temperature=current_gen_kwargs["temperature"],
                    top_p=current_gen_kwargs["top_p"],
                    num_beams=current_gen_kwargs["num_beams"],
                    max_new_tokens=current_gen_kwargs["max_new_tokens"],
                    top_k=current_gen_kwargs.get("top_k", None),
                    use_cache=self.use_cache,
                )
            end_time = time.time()

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            e2e_latency += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)
            num_last = int(prompt_stat.get("num_last_layer_token", 0))
            sys_tokens = int(prompt_stat.get("sys", 0))
            inst_tokens = int(prompt_stat.get("inst", 0))
            video_tokens = int(prompt_stat.get("video", 0))
            merged_video = max(num_last - sys_tokens - inst_tokens, 0)
            visual_merged_ratio = 100.0 * merged_video / max(video_tokens, 1)
            denom = max(video_tokens + sys_tokens + inst_tokens, 1)
            input_merged_ratio = 100.0 * (num_last / denom)
            prompt_stat["merged_video"] = merged_video
            prompt_stat["visual_merged_ratio"] = visual_merged_ratio
            prompt_stat["input_merged_ratio"] = input_merged_ratio
            ttfts.append(float(prompt_stat.get("ttft", 0.0)))
            tpops.append(float(prompt_stat.get("tpop", 0.0)))
            merged_videos.append(merged_video)
            visual_merged_ratios.append(visual_merged_ratio)
            input_merged_ratios.append(input_merged_ratio)

            for ans, context in zip(answers, texts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
            pbar.update(1)

        res = re_ords.get_original(res)

        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
                "ttft": (sum(ttfts) / len(ttfts)) if ttfts else 0.0,
                "tpop": (sum(tpops) / len(tpops)) if tpops else 0.0,
                "merged_video": sum(merged_videos) / len(merged_videos),
                "visual_merged_ratio": sum(visual_merged_ratios) / len(visual_merged_ratios),
                "input_merged_ratio": sum(input_merged_ratios) / len(input_merged_ratios),
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res
