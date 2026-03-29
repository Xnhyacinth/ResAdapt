import base64
import time
import re
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from llava.model.multimodal_encoder.sttm.patch import replace_qwen_by_sparse_attn
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import parse_reasoning_model_answer
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL as Qwen2_5_VL_Base
from lmms_eval.models.simple.qwen2_vl_custom import ModelArguments
from llava.model.multimodal_encoder.sttm.flashvid_patch import apply_flashvid

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen2_5_vl_custom")
class Qwen2_5_VL_Custom(Qwen2_5_VL_Base):
    """
    Qwen2.5_VL custom model with sparse-attention patch support.
    Reuses the validated generate/multi-round methods from qwen2_vl_custom.
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        enable_flashvid: bool = False,
        retention_ratio: float = 0.25,
        do_segment: bool = True,
        segment_threshold: float = 0.9,
        min_segment_num: int = 8,
        complementary_segment: bool = True,
        token_selection_method: str = "attn_div_v2",
        alpha: float = 0.7,
        temporal_threshold: float = 0.8,
        expansion: float = 1.25,
        pruning_layer: int = 20,
        llm_retention_ratio: float = 0.3,
        **kwargs,
    ) -> None:
        lmms.__init__(self)
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "torch_dtype": "bfloat16",
            "device_map": self.device_map,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, **model_kwargs)
        if enable_flashvid:
            self._model = apply_flashvid(
                model=self._model,
                retention_ratio=retention_ratio,
                do_segment=do_segment,
                segment_threshold=segment_threshold,
                min_segment_num=min_segment_num,
                complementary_segment=complementary_segment,
                token_selection_method=token_selection_method,
                alpha=alpha,
                temporal_threshold=temporal_threshold,
                expansion=expansion,
                pruning_layer=pruning_layer,
                llm_retention_ratio=llm_retention_ratio,
            )
        self._model.eval()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)

        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        max_length = kwargs.pop("max_length", 2048)
        self._max_length = max_length
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        self.sa_pattern = kwargs.pop("sa_pattern", None)
        model_args_keys = ModelArguments.__annotations__.keys()
        model_args_dict = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in model_args_keys}
        if kwargs:
            raise ValueError(f"Unexpected kwargs: {kwargs}")

        if self.sa_pattern is not None:
            print(f"using sa_pattern: {self.sa_pattern}")
            model_args = ModelArguments(**model_args_dict)
            sa_start_layer_idx = model_args.sa_start_layer_idx
            vit_start_layer_idx = model_args.vit_start_layer_idx
            sa_prune_ratio = model_args.sa_prune_ratio

            if vit_start_layer_idx is not None and vit_start_layer_idx < 0:
                vit_start_layer_idx = self._config.vision_config.depth + vit_start_layer_idx

            sa_kwargs = {
                "sa_pattern": self.sa_pattern,
                "sa_start_layer_idx": sa_start_layer_idx,
                "vit_start_layer_idx": vit_start_layer_idx,
                "sa_prune_ratio": sa_prune_ratio,
            }

            if "quadtree" in self.sa_pattern:
                sa_tree_root_level = model_args.sa_tree_root_level
                threshold = model_args.threshold
                sa_tree_temporal_thresh = model_args.sa_tree_temporal_thresh
                sa_tree_weighted_avg = model_args.sa_tree_weighted_avg
                sttm_slow_ver = model_args.sttm_slow_ver
                sim_per_head = model_args.sim_per_head
                sa_tree_dist_topk = model_args.sa_tree_dist_topk
                sa_tree_dist_time = model_args.sa_tree_dist_time
                sa_tree_trk_thresh = model_args.sa_tree_trk_thresh
                sa_tree_trk_layer_idx = model_args.sa_tree_trk_layer_idx
                pos_emb_ver = model_args.pos_emb_ver
                pos_emb_weighted_avg = model_args.pos_emb_weighted_avg
                sa_kwargs.update(
                    {
                        "threshold": threshold,
                        "sa_tree_root_level": sa_tree_root_level,
                        "sa_tree_temporal_thresh": sa_tree_temporal_thresh,
                        "sa_tree_weighted_avg": sa_tree_weighted_avg,
                        "sa_tree_dist_topk": sa_tree_dist_topk,
                        "sa_tree_dist_time": sa_tree_dist_time,
                        "sa_tree_trk_thresh": sa_tree_trk_thresh,
                        "sa_tree_trk_layer_idx": sa_tree_trk_layer_idx,
                        "sttm_slow_ver": sttm_slow_ver,
                        "sim_per_head": sim_per_head,
                        "pos_emb_ver": pos_emb_ver,
                        "pos_emb_weighted_avg": pos_emb_weighted_avg,
                    }
                )

                if "new" in self.sa_pattern:
                    sa_var_thresh = model_args.sa_var_thresh
                    sa_tem_diff_thresh = model_args.sa_tem_diff_thresh
                    sa_kwargs.update(
                        {
                            "sa_var_thresh": sa_var_thresh,
                            "sa_tem_diff_thresh": sa_tem_diff_thresh,
                        }
                    )

            elif "dycoke-stage1" in self.sa_pattern:
                threshold = model_args.threshold
                sa_kwargs.update({"threshold": threshold})

            elif "dpmm_infer" in self.sa_pattern:
                sa_alpha = model_args.sa_alpha
                sa_tau = model_args.sa_tau
                sa_local_sim_thresh = model_args.sa_local_sim_thresh
                max_global_k = model_args.max_global_k
                local_max_k = model_args.local_max_k
                ema_momentum = model_args.ema_momentum
                chunk_size = model_args.chunk_size
                max_size_per_cluster = model_args.max_size_per_cluster
                sa_kwargs.update(
                    {
                        "sa_alpha": sa_alpha,
                        "sa_tau": sa_tau,
                        "sa_local_sim_thresh": sa_local_sim_thresh,
                        "ema_momentum": ema_momentum,
                        "max_global_k": max_global_k,
                        "local_max_k": local_max_k,
                        "chunk_size": chunk_size,
                        "max_size_per_cluster": max_size_per_cluster,
                        "dtype": self._model.dtype,
                    }
                )

            elif "dpmm" in self.sa_pattern:
                sa_alpha = model_args.sa_alpha
                local_max_k = model_args.local_max_k
                chunk_size = model_args.chunk_size
                prior_var = model_args.prior_var
                likelihood_var = model_args.likelihood_var
                sa_kwargs.update(
                    {
                        "sa_alpha": sa_alpha,
                        "feature_dim": self._config.hidden_size,
                        "local_max_k": local_max_k,
                        "chunk_size": chunk_size,
                        "device": self._model.device,
                        "dtype": self._model.dtype,
                        "prior_var": prior_var,
                        "likelihood_var": likelihood_var,
                    }
                )

            elif "tome" in self.sa_pattern:
                sa_tome_ver = model_args.sa_tome_ver
                sa_kwargs.update({"sa_tome_ver": sa_tome_ver})

            elif "framefusion" in self.sa_pattern:
                sa_ratio = 1.0 - sa_prune_ratio
                sa_kwargs = {
                    "sa_framefusion_cost": sa_ratio,
                    "model": self._model,
                }

            elif "router" in self.sa_pattern:
                threshold = model_args.threshold
                sa_kwargs.update({"threshold": threshold})
                if "seq" not in self.sa_pattern:
                    sa_kwargs.update({"dim": self._config.hidden_size})

            elif "win" in self.sa_pattern:
                win_size = model_args.chunk_size
                threshold = model_args.threshold
                sa_tree_temporal_thresh = model_args.sa_tree_temporal_thresh
                sa_kwargs.update(
                    {
                        "win_size": win_size,
                        "spatial_sim_threshold": threshold,
                        "temporal_sim_threshold": sa_tree_temporal_thresh,
                    }
                )

            elif "hnsw" in self.sa_pattern:
                chunk_size = model_args.chunk_size
                hnsw_threshold = model_args.threshold
                greedy_threshold = model_args.sa_tree_temporal_thresh
                greedy_layers = model_args.greedy_layers
                opt_name = model_args.opt_name
                sa_kwargs.update(
                    {
                        "chunk_size": chunk_size,
                        "hnsw_threshold": hnsw_threshold,
                        "greedy_threshold": greedy_threshold,
                        "greedy_layers": greedy_layers,
                        "opt_name": opt_name,
                    }
                )

            elif "streaming" in self.sa_pattern or "random" in self.sa_pattern or "saliency" in self.sa_pattern or "visionzip" in self.sa_pattern:
                sa_ratio = 1.0 - sa_prune_ratio
                sa_kwargs.update({"sa_ratio": sa_ratio})

            elif "semantic_consistent" in self.sa_pattern:
                sa_consistency_weight = model_args.sa_consistency_weight
                sa_spatial_keep_ratio = model_args.sa_spatial_keep_ratio
                sa_debug_stats = model_args.sa_debug_stats
                sa_kwargs.update(
                    {
                        "sa_consistency_weight": sa_consistency_weight,
                        "sa_spatial_keep_ratio": sa_spatial_keep_ratio,
                        "sa_debug_stats": sa_debug_stats,
                    }
                )

            elif "ml_select" in self.sa_pattern:
                selector_name = model_args.sa_selector_name
                if ":" in self.sa_pattern:
                    selector_name = self.sa_pattern.split(":", 1)[1] or selector_name
                sa_debug_stats = model_args.sa_debug_stats
                sa_selector_kwargs = {
                    "seed": model_args.sa_selector_seed,
                    "max_samples": model_args.sa_selector_max_samples,
                    "iters": model_args.sa_kmeanspp_iters,
                    "rank": model_args.sa_leverage_rank,
                    "power_iters": model_args.sa_leverage_power_iters,
                    "motion_thresh": model_args.sa_scene_motion_thresh,
                    "entropy_thresh": model_args.sa_scene_entropy_thresh,
                    "mix_kcenter": model_args.sa_mix_kcenter,
                    "mix_facility": model_args.sa_mix_facility,
                    "mix_leverage": model_args.sa_mix_leverage,
                }
                sa_kwargs.update(
                    {
                        "sa_selector_name": selector_name,
                        "sa_selector_kwargs": sa_selector_kwargs,
                        "sa_debug_stats": sa_debug_stats,
                    }
                )
        else:
            sa_kwargs = {}

        replace_qwen_by_sparse_attn(self.sa_pattern, "qwen2_5_vl", **sa_kwargs)

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        req_and_args = [(req, req.args) for req in requests]
        re_ords = utils.Collator(req_and_args, lambda x: _collate(x[1]), grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        e2e_latency = 0
        total_tokens = 0
        ttfts = []
        tpops = []
        merged_videos = []
        visual_merged_ratios = []
        input_merged_ratios = []
        for chunk in chunks:
            reqs, args_list = zip(*chunk)
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*args_list)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list] but got {type(until)}")
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": self.system_prompt}]
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                if visual_list[i] is not None:
                    for visual in visual_list[i]:
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                            vr = decord.VideoReader(visual)
                            _ = vr[0].asnumpy()
                            processed_visuals.append(
                                {
                                    "type": "video",
                                    "video": visual,
                                    "max_pixels": self.max_pixels,
                                    "min_pixels": self.min_pixels,
                                }
                            )
                        elif isinstance(visual, Image.Image):
                            base64_image = visual.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            processed_visuals.append(
                                {
                                    "type": "image",
                                    "image": f"data:image/jpeg;base64,{base64_string}",
                                    "max_pixels": self.max_pixels,
                                    "min_pixels": self.min_pixels,
                                }
                            )

                if self.interleave_visuals is False:
                    message.append(
                        {
                            "role": "user",
                            "content": processed_visuals + [{"type": "text", "text": context}],
                        }
                    )
                else:
                    image_placeholders = re.findall(r"<image \d+>", context)
                    content_parts = []
                    text_parts = re.split(r"<image \d+>", context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})

                    for j, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and image_idx < len(processed_visuals):
                            content_parts.append(processed_visuals[image_idx])
                        if j + 1 < len(text_parts) and text_parts[j + 1]:
                            content_parts.append({"type": "text", "text": text_parts[j + 1]})

                    message.append(
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    )

                batched_messages.append(message)

            texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(batched_messages)
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                indices = np.unique(indices)
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                    indices = np.unique(indices)
                video_inputs[0] = video_inputs[0][indices]
            padding_side = "left" if self.batch_size > 1 else "right"
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
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
                if merge_size is None:
                    merge_size = 1
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
                "max_new_tokens": 32768,
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

            start_time = time.time()
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
                prompt_stat=prompt_stat,
            )
            end_time = time.time()
            e2e_latency += end_time - start_time
            per_req_gen_s = (end_time - start_time) / len(reqs) if len(reqs) > 0 else 0.0
            for req in reqs:
                req.generation_s = per_req_gen_s

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

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
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for ans, context in zip(answers, contexts):
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
                "ttft": sum(ttfts) / len(ttfts),
                "tpop": sum(tpops) / len(tpops),
                "merged_video": sum(merged_videos) / len(merged_videos),
                "visual_merged_ratio": sum(visual_merged_ratios) / len(visual_merged_ratios),
                "input_merged_ratio": sum(input_merged_ratios) / len(input_merged_ratios),
            },
        }
        log_metrics(**metric_dict)
        pbar.close()
        return res
