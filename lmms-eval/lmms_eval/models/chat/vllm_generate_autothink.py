import math
import json
import os
import re
import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoProcessor

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.chat.vllm import VLLM as VLLMChat
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages

LLM, _ = optional_import("vllm", "LLM")
SamplingParams, _ = optional_import("vllm", "SamplingParams")
fetch_video, _ = optional_import("qwen_vl_utils", "fetch_video")
process_vision_info, _ = optional_import("qwen_vl_utils", "process_vision_info")

WORKERS = int(os.getenv("WORKERS", "32"))
MICRO_BATCH = int(os.getenv("MICRO_BATCH", "16"))
max_inflight_per_gpu = int(os.getenv("max_inflight_per_gpu", "4"))

COT_SYSTEM_PROMPT_ANSWER_TWICE = (
    "You are a helpful assistant.\n"
    "FIRST: Output your initial answer inside the first \\boxed{...} without any analysis or explanations. "
    "If you cannot determine the answer without reasoning, output \\boxed{Let's analyze the problem step by step.} instead.\n"
    "THEN: Think through the reasoning as an internal monologue enclosed within <think>...</think>.\n"
    "AT LAST: Output the final answer again inside \\boxed{...}. If you believe the previous answer was correct, repeat it; otherwise, correct it.\n"
    "Output format: \\boxed{...}<think>...</think>\\boxed{...}\n"
)

COT_SYSTEM_PROMPT_FIRST_ONLY = (
    "You are a helpful assistant.\n"
    "Output your initial answer inside \\boxed{...} without any analysis or explanations. "
    "If you cannot determine the answer without reasoning, output \\boxed{Let's analyze the problem step by step.} instead.\n"
    "Output format: \\boxed{...}\n"
)


def env_true(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).lower() in ("1", "true", "yes", "y", "t")

def default_scale_entry() -> Dict[str, Any]:
    return {"scale_means": 1.0, "scale_mask": None, "scales": None}

def _ensure_2d_scale_tensors(
    scales: Any,
    scale_mask: Any,
) -> Tuple[Any, Any]:
    try:
        import torch
        if torch.is_tensor(scales) and scales.dim() == 1:
            scales = scales.unsqueeze(0)
        if torch.is_tensor(scale_mask) and scale_mask.dim() == 1:
            scale_mask = scale_mask.unsqueeze(0)
    except Exception:
        pass
    return scales, scale_mask

def _align_scale_mask(scales: Any, scale_mask: Any) -> Any:
    if scales is None or scale_mask is None:
        return scale_mask
    try:
        import torch
        if torch.is_tensor(scales) and torch.is_tensor(scale_mask):
            return scale_mask if scales.shape == scale_mask.shape else None
    except Exception:
        pass
    if isinstance(scales, (list, tuple)) and isinstance(scale_mask, (list, tuple)):
        if len(scales) != len(scale_mask):
            return None
        for a, b in zip(scales, scale_mask):
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                if len(a) != len(b):
                    return None
    return scale_mask

def _masked_scale_stats(scales: Any, scale_mask: Any) -> Optional[Tuple[float, float, float]]:
    if scales is None:
        return None
    try:
        import torch
        if torch.is_tensor(scales):
            s = scales.detach().float()
            if torch.is_tensor(scale_mask) and scale_mask.shape == s.shape:
                m = scale_mask.detach().bool()
                vals = s[m]
            else:
                vals = s.flatten()
            if vals.numel() == 0:
                return None
            return float(vals.mean().item()), float(vals.min().item()), float(vals.max().item())
    except Exception:
        pass
    values: List[float] = []
    if isinstance(scales, (list, tuple)):
        if isinstance(scale_mask, (list, tuple)) and len(scales) == len(scale_mask):
            for row, mrow in zip(scales, scale_mask):
                if isinstance(row, (list, tuple)) and isinstance(mrow, (list, tuple)):
                    for v, mv in zip(row, mrow):
                        if mv and isinstance(v, (int, float)):
                            values.append(float(v))
                elif isinstance(mrow, (int, float)) and mrow and isinstance(row, (int, float)):
                    values.append(float(row))
        if not values:
            for row in scales:
                if isinstance(row, (list, tuple)):
                    for v in row:
                        if isinstance(v, (int, float)):
                            values.append(float(v))
                elif isinstance(row, (int, float)):
                    values.append(float(row))
    if not values:
        return None
    return float(sum(values) / len(values)), float(min(values)), float(max(values))


def _build_vllm_sampling_params(params: Dict[str, Any]):
    if SamplingParams is None:
        raise RuntimeError("vllm.SamplingParams is unavailable")
    try:
        return SamplingParams(**params)
    except TypeError as e:
        try:
            allowed = set(inspect.signature(SamplingParams).parameters.keys())
            filtered = {k: v for k, v in params.items() if k in allowed}
            return SamplingParams(**filtered)
        except Exception:
            msg = str(e)
            filtered = dict(params)
            for _ in range(16):
                m = re.search(r"Unexpected keyword argument '([^']+)'", msg)
                if not m:
                    break
                filtered.pop(m.group(1), None)
                try:
                    return SamplingParams(**filtered)
                except TypeError as e2:
                    msg = str(e2)
            return SamplingParams(**filtered)

from resadapt.utils.utils import (
    expand_image_prompt,
    apply_adaptive_scaling,
    maybe_expand_video_prompt,
    video2list,
    video2images,
)

if env_true("RESADAPT_MROPE_PATCH") or env_true("VLLM_MROPE_PATCH"):
    print("🚀 [Patching] Applying custom VLLM Qwen2.5-VL MROPE logic...")
    import vllm
    from vllm.model_executor.models import qwen2_5_vl
    from resadapt.allocator.vllm_patch import get_mrope_input_positions
    # , iter_mm_grid_thw

    # vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.get_mrope_input_positions = (
    #     get_mrope_input_positions
    # )
    # vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLMultiModalProcessor._get_mm_fields_config = _get_mm_fields_config
    vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.get_mrope_input_positions = get_mrope_input_positions
    # vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.iter_mm_grid_thw = iter_mm_grid_thw


@register_model("vllm_generate_autothink")
class VLLMGenerateAutoThink(VLLMChat):
    is_simple = False

    def __init__(
        self,
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        tensor_parallel_size=1,
        data_parallel_size=1,
        gpu_memory_utilization=0.8,
        batch_size=1,
        max_frame_num=768,
        trust_remote_code=True,
        chat_template=None,
        max_pixels: int = 1605632,
        min_image_pixels=28,
        fps: Optional[int] = None,
        nframes: Optional[int] = 32,
        early_exit_thresh: float = 0.97,
        inference_mode: str = "auto",
        log_visual_len: Optional[bool] = None,
        allocator_max_scale: Optional[float] = None,
        allocator_min_scale: Optional[float] = None,
        **kwargs,
    ):
        self.system_prompt = COT_SYSTEM_PROMPT_ANSWER_TWICE
        self.inference_mode = inference_mode
        if self.inference_mode not in ["first", "second", "auto"]:
            raise ValueError(f"Invalid inference_mode: {self.inference_mode}")
        self.early_exit_thresh = early_exit_thresh
        self.add_sys = env_true("ADD_SYS", "1")
        if log_visual_len is None:
            self.log_visual_len = env_true("LOG_VISUAL_LEN")
        elif isinstance(log_visual_len, str):
            self.log_visual_len = str(log_visual_len).lower() in ("1", "true", "yes", "y", "t")
        else:
            self.log_visual_len = bool(log_visual_len)
        self.allocator_path = os.getenv("ALLOCATOR_PATH", None)
        self.enable_baseline_scale = env_true("ENABLE_BASELINE_SCALE")
        self.scale_preprocess_retries = int(os.getenv("SCALE_PREPROCESS_RETRIES", "2"))
        self.scale_preprocess_timeout_s = int(os.getenv("SCALE_PREPROCESS_TIMEOUT_S", "21600"))
        self.scale_preprocess_chunk_size = int(os.getenv("SCALE_PREPROCESS_CHUNK_SIZE", "30000"))
        self.workers = int(os.getenv("WORKERS", str(WORKERS)))
        self.micro_batch = int(os.getenv("MICRO_BATCH", str(MICRO_BATCH)))
        self.max_inflight_per_gpu = int(os.getenv("max_inflight_per_gpu", str(max_inflight_per_gpu)))
        self.max_queue_per_gpu = int(os.getenv("MAX_QUEUE_PER_GPU", str(os.getenv("MAX_QUEUE_PER_GPU", "8"))))
        
        # Determine if this is the main rank (only rank 0 runs allocator pool)
        self._is_main_rank = int(os.getenv("RANK", "0")) == 0
        self._world_size_env = int(os.getenv("WORLD_SIZE", "1"))
        
        # Pool kwargs for allocator pool initialization
        self._pool_kwargs = dict(
            model_path=self.allocator_path,
            num_gpus=int(os.getenv("ALLOCATOR_NUM_GPUS", "8")),
            enable_batch=True,
            microbatch_ms=int(os.getenv("MICRO_BATCH_MS", "10")),
            microbatch_max=self.micro_batch,
            max_total_inflight=self.max_inflight_per_gpu * int(os.getenv("ALLOCATOR_NUM_GPUS", "8")),
            max_inflight_per_gpu=self.max_inflight_per_gpu,
            max_queue_per_gpu=self.max_queue_per_gpu,
            submit_threads=self.workers if self.allocator_path is not None and "smol" not in self.allocator_path else self.workers * 3,
            max_frames=max_frame_num // 2 if self.allocator_path is not None and "smol" not in self.allocator_path else max_frame_num,
            schedule_policy=os.getenv("ALLOCATOR_SCHED_POLICY", "least_inflight"),
        )
        if allocator_max_scale is None:
            try:
                allocator_max_scale = float(os.getenv("ALLOCATOR_MAX_SCALE", ""))
            except Exception:
                allocator_max_scale = None
        if allocator_min_scale is None:
            try:
                allocator_min_scale = float(os.getenv("ALLOCATOR_MIN_SCALE", ""))
            except Exception:
                allocator_min_scale = None
        if isinstance(allocator_max_scale, (int, float)):
            self._pool_kwargs["max_scale"] = float(allocator_max_scale)
        if isinstance(allocator_min_scale, (int, float)):
            self._pool_kwargs["min_scale"] = float(allocator_min_scale)
        self.pool = None
        self._scales_missing_max_warn = 5
        self.scale_preprocess_failures: List[Tuple[str, str]] = []
        self._allocator_scales_cache_base: Optional[str] = None
        self._allocator_scales_cache_loaded: Dict[str, bool] = {}
        self._allocator_scales_cache_mem: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._allocator_scales_cache_frames = int(self._pool_kwargs.get("max_frames", max_frame_num))
        if self.allocator_path is not None:
            try:
                expanded = os.path.expanduser(self.allocator_path)
                if os.path.isdir(expanded):
                    self._allocator_scales_cache_base = os.path.join(expanded, "benchmarks")
            except Exception:
                self._allocator_scales_cache_base = None

        super().__init__(
            model,
            tensor_parallel_size,
            data_parallel_size,
            gpu_memory_utilization,
            batch_size,
            max_frame_num,
            trust_remote_code,
            chat_template,
            max_pixels,
            min_image_pixels,
            fps,
            nframes,
            **kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(model)
        if self.chat_template is not None:
            with open(self.chat_template, "r") as f:
                chat_template = f.read()
                self.processor.chat_template = chat_template
        self.patch_size = self.processor.image_processor.patch_size
        self.image_factor = self.processor.video_processor.merge_size * self.patch_size

        # Only rank 0 starts allocator pool to avoid GPU competition
        if self._is_main_rank and (self.allocator_path is not None or self.enable_baseline_scale):
            print(f"[VLLMGenerateAutoThink] [Rank 0] Found allocator at {self.allocator_path}")
            if self._check_gpu_available():
                try:
                    from resadapt.eval.multi_model_limit_async import MultiGPUInferPool
                    self.pool = MultiGPUInferPool(**self._pool_kwargs)
                    self.pool.start()
                    print("[VLLMGenerateAutoThink] [Rank 0] Allocator pool started successfully")
                except Exception as e:
                    print(f"[VLLMGenerateAutoThink] [Rank 0] Warning: Failed to start allocator pool: {e}")
                    import traceback
                    traceback.print_exc()
                    self.pool = None
            else:
                print("[VLLMGenerateAutoThink] [Rank 0] Warning: GPU not available, skipping allocator pool")
        elif not self._is_main_rank and self.allocator_path is not None:
            print(f"[VLLMGenerateAutoThink] [Rank {os.getenv('RANK', '?')}] Waiting for rank 0 to run allocator")

    def _sanitize_benchmark_name(self, name: str) -> str:
        name = str(name)
        name = name.replace(os.sep, "_")
        name = name.replace("..", "_")
        return name

    def _normalize_benchmark_for_cache(self, benchmark: str) -> str:
        b = str(benchmark)
        return b[:-6] if b.endswith("_boxed") else b

    def _normalize_compound_key_for_cache(self, compound_key: Any) -> Any:
        if not isinstance(compound_key, str):
            return compound_key
        if "::" not in compound_key:
            return compound_key
        task, rest = compound_key.split("::", 1)
        task = self._normalize_benchmark_for_cache(task)
        return f"{task}::{rest}"

    def _format_scale_for_cache(self, v: Any) -> Optional[str]:
        if v is None:
            return None
        try:
            fv = float(v)
        except Exception:
            return None
        if not (fv > 0):
            return None
        s = f"{fv:.6f}".rstrip("0").rstrip(".")
        return s if s else None

    def _allocator_scales_cache_file_tag(self) -> str:
        min_s = self._format_scale_for_cache(self._pool_kwargs.get("min_scale", None))
        max_s = self._format_scale_for_cache(self._pool_kwargs.get("max_scale", None))
        tag = ""
        if min_s is not None:
            tag += f"_min{min_s}"
        if max_s is not None:
            tag += f"_max{max_s}"
        return tag

    def _allocator_scales_cache_key(self, benchmark: str) -> str:
        bench = self._sanitize_benchmark_name(benchmark)
        return f"{bench}::frames{self._allocator_scales_cache_frames}{self._allocator_scales_cache_file_tag()}"

    def _scale_entry_jsonable(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        def _to_jsonable(obj: Any) -> Any:
            if obj is None:
                return None
            if hasattr(obj, "tolist"):
                try:
                    return obj.tolist()
                except Exception:
                    pass
            try:
                import torch
                if torch.is_tensor(obj):
                    return obj.detach().cpu().tolist()
            except Exception:
                pass
            if isinstance(obj, dict):
                return {k: _to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_jsonable(v) for v in obj]
            if isinstance(obj, (int, float, str, bool)):
                return obj
            return str(obj)

        return {
            "scale_mask": _to_jsonable(entry.get("scale_mask", None)),
            "scales": _to_jsonable(entry.get("scales", None)),
        }

    def _compute_scale_means_from_scales(self, scales: Any) -> float:
        vals: List[float] = []
        if hasattr(scales, "tolist"):
            try:
                scales = scales.tolist()
            except Exception:
                pass
        if isinstance(scales, (int, float)):
            vals.append(float(scales))
        elif isinstance(scales, (list, tuple)):
            def _flatten(x: Any) -> None:
                if isinstance(x, (int, float)):
                    vals.append(float(x))
                    return
                if isinstance(x, (list, tuple)):
                    for y in x:
                        _flatten(y)
            _flatten(scales)
        return float(sum(vals) / len(vals)) if vals else 1.0

    def _cache_entry_to_runtime_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        scales = entry.get("scales", None)
        scale_mask = entry.get("scale_mask", None)
        return {
            "scale_means": self._compute_scale_means_from_scales(scales) if scales is not None else 1.0,
            "scale_mask": scale_mask,
            "scales": scales,
        }

    def _load_allocator_scales_cache(self, benchmark: str) -> Dict[str, Dict[str, Any]]:
        if self._allocator_scales_cache_base is None:
            return {}
        cache_key = self._allocator_scales_cache_key(benchmark)
        if self._allocator_scales_cache_loaded.get(cache_key, False):
            return self._allocator_scales_cache_mem.get(cache_key, {})
        cache: Dict[str, Dict[str, Any]] = {}
        try:
            tag = self._allocator_scales_cache_file_tag()
            candidates = [str(benchmark)]
            if not str(benchmark).endswith("_boxed"):
                candidates.append(str(benchmark) + "_boxed")

            found_any = False
            for cand in candidates:
                bench = self._sanitize_benchmark_name(cand)
                bench_dir = os.path.join(self._allocator_scales_cache_base, bench)
                cache_file = os.path.join(bench_dir, f"scales_{self._allocator_scales_cache_frames}{tag}.jsonl")
                if not os.path.exists(cache_file):
                    continue
                found_any = True
                with open(cache_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except Exception:
                            continue
                        key = payload.get("key", None)
                        entry = payload.get("entry", None)
                        if not isinstance(key, str) or not isinstance(entry, dict):
                            continue
                        if entry.get("scales", None) is None:
                            continue
                        cache[key] = self._cache_entry_to_runtime_entry(entry)

            if not found_any:
                bench = self._sanitize_benchmark_name(benchmark)
                bench_dir = os.path.join(self._allocator_scales_cache_base, bench)
                cache_file = os.path.join(bench_dir, f"scales_{self._allocator_scales_cache_frames}{tag}.jsonl")
                print(f"[VLLMGenerateAutoThink] No allocator scales cache for task={benchmark} frames={self._allocator_scales_cache_frames}: {cache_file}")
        except Exception:
            cache = {}
        self._allocator_scales_cache_mem[cache_key] = cache
        self._allocator_scales_cache_loaded[cache_key] = True
        return cache

    def _append_allocator_scales_cache(self, benchmark: str, entries: Dict[str, Dict[str, Any]]) -> None:
        if self._allocator_scales_cache_base is None:
            return
        if not entries:
            return
        bench = self._sanitize_benchmark_name(benchmark)
        bench_dir = os.path.join(self._allocator_scales_cache_base, bench)
        os.makedirs(bench_dir, exist_ok=True)
        cache_file = os.path.join(
            bench_dir,
            f"scales_{self._allocator_scales_cache_frames}{self._allocator_scales_cache_file_tag()}.jsonl",
        )
        cache_mem = self._load_allocator_scales_cache(bench)
        try:
            with open(cache_file, "a", encoding="utf-8") as f:
                for key, entry in entries.items():
                    if not isinstance(key, str) or not isinstance(entry, dict):
                        continue
                    if key in cache_mem:
                        continue
                    if entry.get("scales", None) is None:
                        continue
                    jsonable = self._scale_entry_jsonable(entry)
                    f.write(json.dumps({"key": key, "entry": jsonable}, ensure_ascii=False) + "\n")
                    cache_mem[key] = self._cache_entry_to_runtime_entry(jsonable)
        except Exception:
            return

    def _get_scales_with_disk_cache(self, payloads: List[dict], keys: List[str]) -> Dict[str, Dict[str, Any]]:
        if self.allocator_path is None or self._allocator_scales_cache_base is None or not keys:
            return self._submit_payloads_with_retries(payloads, keys)
        cached: Dict[str, Dict[str, Any]] = {}
        missing_payloads: List[dict] = []
        missing_keys: List[str] = []
        for key, payload in zip(keys, payloads):
            task = None
            if isinstance(payload, dict):
                task = payload.get("_task", None)
            if not isinstance(task, str) and isinstance(key, str) and "::" in key:
                task = key.split("::", 1)[0]
            if not isinstance(task, str) or not task:
                missing_payloads.append(payload)
                missing_keys.append(key)
                continue
            cache_task = self._normalize_benchmark_for_cache(task)
            cache_key = self._normalize_compound_key_for_cache(key)
            cache = self._load_allocator_scales_cache(cache_task)
            hit = cache.get(cache_key, None)
            if isinstance(hit, dict) and hit.get("scales", None) is not None:
                cached[key] = hit
            else:
                missing_payloads.append(payload)
                missing_keys.append(key)
        predicted: Dict[str, Dict[str, Any]] = {}
        if missing_keys:
            predicted = self._submit_payloads_with_retries(missing_payloads, missing_keys)
            by_task: Dict[str, Dict[str, Dict[str, Any]]] = {}
            for key, entry in predicted.items():
                if not isinstance(entry, dict) or entry.get("scales", None) is None:
                    continue
                task = key.split("::", 1)[0] if isinstance(key, str) and "::" in key else None
                if not isinstance(task, str) or not task:
                    continue
                cache_task = self._normalize_benchmark_for_cache(task)
                cache_key = self._normalize_compound_key_for_cache(key)
                by_task.setdefault(cache_task, {})[cache_key] = entry
            for task, task_entries in by_task.items():
                self._append_allocator_scales_cache(task, task_entries)
        out = dict(cached)
        out.update(predicted)
        return out

    def _compute_visual_len(self, text, images, videos, video_metadatas):
        if images is None and videos is None:
            return 0
        proc_kwargs: Dict[str, Any] = {}
        if images is not None:
            proc_kwargs["images"] = images
        if videos is not None:
            proc_kwargs["videos"] = videos
            if video_metadatas is not None:
                if isinstance(video_metadatas, (list, tuple)):
                    cleaned_metadatas = []
                    for meta in video_metadatas:
                        if isinstance(meta, dict) and "video_timestamps" in meta:
                            cleaned_metadatas.append({k: v for k, v in meta.items() if k != "video_timestamps"})
                        else:
                            cleaned_metadatas.append(meta)
                    proc_kwargs["videos_kwargs"] = {"video_metadata": cleaned_metadatas, "do_sample_frames": False}
                else:
                    proc_kwargs["videos_kwargs"] = {"video_metadata": video_metadatas, "do_sample_frames": False}
        try:
            outputs = self.processor(text=[text], **proc_kwargs)
        except Exception:
            return None
        merge_size = getattr(self.processor.image_processor, "merge_size", 1)
        merge_len = int(merge_size) ** 2 if merge_size else 1

        def _grid_token_count(grid):
            try:
                import torch
                if torch.is_tensor(grid):
                    if grid.numel() == 0:
                        return 0
                    prod = grid.long().prod(dim=-1)
                    return int((prod // merge_len).sum().item())
            except Exception:
                pass
            if isinstance(grid, (list, tuple)):
                total = 0
                for item in grid:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        try:
                            t, h, w = item[:3]
                            total += (int(t) * int(h) * int(w)) // merge_len
                        except Exception:
                            continue
                return total
            return 0

        image_tokens = _grid_token_count(outputs.get("image_grid_thw", None))
        video_tokens = _grid_token_count(outputs.get("video_grid_thw", None))
        total_tokens = image_tokens + video_tokens
        if total_tokens > 0:
            return total_tokens
        try:
            text_only = self.processor(text=[text])["input_ids"][0]
            full_ids = outputs.get("input_ids", None)
            if full_ids is not None:
                image_pad_count = text.count("<|image_pad|>")
                video_pad_count = text.count("<|video_pad|>")
                pad_count = image_pad_count + video_pad_count
                return max(int(len(full_ids[0]) - len(text_only) + pad_count), 0)
        except Exception:
            return None
        return None

    def _unload_vllm(self):
        if hasattr(self, "client") and self.client is not None:
            if hasattr(self.client, "llm_engine"):
                del self.client.llm_engine
            del self.client
            self.client = None
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()

    def _unload_allocator(self):
        if self.pool is not None:
            try:
                self.pool.close()
            except Exception:
                pass
            self.pool = None
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()

    def _check_gpu_available(self, min_free_gb: float = 1.0) -> bool:
        """Check if GPUs are available with sufficient free memory."""
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                return False
            min_free_bytes = min_free_gb * 1024**3
            for i in range(num_gpus):
                try:
                    mem_free, _ = torch.cuda.mem_get_info(i)
                    if mem_free < min_free_bytes:
                        print(f"[VLLMGenerateAutoThink] GPU {i}: only {mem_free / 1024**3:.2f}GB free")
                        return False
                except Exception as e:
                    print(f"[VLLMGenerateAutoThink] Cannot query GPU {i}: {e}")
                    return False
            return True
        except Exception as e:
            print(f"[VLLMGenerateAutoThink] GPU check failed: {e}")
            return False

    def _scales_dict_to_cpu(self, scales_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Convert tensor values in scales dict to CPU lists for distributed broadcast."""
        try:
            import torch
        except ImportError:
            return scales_dict
        result = {}
        for key, entry in scales_dict.items():
            new_entry = {}
            for k, v in entry.items():
                if torch.is_tensor(v):
                    new_entry[k] = v.detach().cpu().tolist()
                else:
                    new_entry[k] = v
            result[key] = new_entry
        return result

    def _ensure_vllm_client(self):
        if hasattr(self, "client") and self.client is not None:
            return
        if not hasattr(self, "_vllm_client_init"):
            init_kwargs = {
                "model": getattr(self, "model_name_or_path", "Qwen/Qwen2.5-VL-3B-Instruct"),
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "trust_remote_code": self.trust_remote_code,
                "max_model_len": 32768,
            }
        else:
            init_kwargs = self._vllm_client_init
        self.client = LLM(**init_kwargs)

    def _ensure_allocator_pool(self):
        if self.pool is not None:
            return
        if self.allocator_path is None and not self.enable_baseline_scale:
            return
        try:
            from resadapt.eval.multi_model_limit_async import MultiGPUInferPool
            self.pool = MultiGPUInferPool(**self._pool_kwargs)
            self.pool.start()
        except Exception:
            self.pool = None

    def make_compound_key_from_req(self, req: Instance) -> str:
        _, _, _, doc_id, task, _ = req.arguments
        return f"{task}::{doc_id}"

    def build_scales_dict_for_batch(
        self,
        batch_requests: List[Instance],
        all_scales_dict: Dict[str, Dict[str, Any]],
        *,
        warn_on_missing: bool = True,
        warn_only_if_dict_nonempty: bool = True,
        preview_n: int = 3,
    ) -> Tuple[Dict[Any, Dict[str, Any]], List[Any]]:
        scales_dict: Dict[Any, Dict[str, Any]] = {}
        missing: List[Any] = []
        for req in batch_requests:
            key = self.make_compound_key_from_req(req)
            entry = all_scales_dict.get(key, None)
            if entry is None:
                missing.append(key)
                scales_dict[key] = default_scale_entry()
            else:
                scales_dict[key] = dict(entry)
        if warn_on_missing and missing:
            if (not warn_only_if_dict_nonempty) or (len(all_scales_dict) > 0):
                print(
                    f"[VLLMGenerateAutoThink] WARNING: {len(missing)} doc_ids not found in all_scales_dict "
                    f"(showing first {min(preview_n, len(missing))}): {missing[:preview_n]}..."
                )
        return scales_dict, missing

    def _scale_entry_stats(self, scale_entry: Dict[str, Any]) -> Dict[str, Any]:
        scales = scale_entry.get("scales", None)
        scale_mask = scale_entry.get("scale_mask", None)
        scale_means = float(scale_entry.get("scale_means", 1.0))
        stats: Dict[str, Any] = {
            "scale_means": scale_means,
            "scales_len": None,
            "scales_min": None,
            "scales_max": None,
        }
        try:
            import torch
            if torch.is_tensor(scales):
                s = scales.detach().float().flatten()
                stats["scales_len"] = int(s.numel())
                if s.numel() > 0:
                    stats["scales_min"] = float(s.min().item())
                    stats["scales_max"] = float(s.max().item())
            if torch.is_tensor(scale_mask):
                m = scale_mask.detach().bool().flatten()
                stats["scale_mask_ratio"] = float(m.float().mean().item()) if m.numel() > 0 else None
            else:
                stats["scale_mask_ratio"] = None
        except Exception:
            if isinstance(scales, (list, tuple)):
                stats["scales_len"] = len(scales)
                if len(scales) > 0:
                    try:
                        stats["scales_min"] = float(min(scales))
                        stats["scales_max"] = float(max(scales))
                    except Exception:
                        pass
            stats["scale_mask_ratio"] = None
        masked = _masked_scale_stats(scales, scale_mask)
        if masked is not None:
            masked_mean, masked_min, masked_max = masked
            stats["scale_means_masked"] = masked_mean
            stats["scales_min_masked"] = masked_min
            stats["scales_max_masked"] = masked_max
        else:
            stats["scale_means_masked"] = None
            stats["scales_min_masked"] = None
            stats["scales_max_masked"] = None
        return stats

    def _normalize_scales(self, scales: Any) -> Optional[List[float]]:
        if scales is None:
            return None
        try:
            import torch
            if torch.is_tensor(scales):
                values = scales.detach().float().flatten().tolist()
                return values if values else None
        except Exception:
            pass
        if hasattr(scales, "tolist"):
            try:
                scales = scales.tolist()
            except Exception:
                pass
        if isinstance(scales, (int, float)):
            return [float(scales)]
        if isinstance(scales, (list, tuple)):
            flat: List[float] = []
            def _flatten(val):
                if isinstance(val, (list, tuple)):
                    for item in val:
                        _flatten(item)
                elif isinstance(val, (int, float)):
                    flat.append(float(val))
            _flatten(scales)
            return flat if flat else None
        return None

    def _build_video_kwargs(self) -> Dict[str, Any]:
        return {
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_image_pixels,
            "max_frames": self.max_frame_num,
        }

    # ... (other methods)

    # In generate_until (I will strictly target the lines to modify)



    def _prepare_scale_payload(self, request: Instance) -> Tuple[str, dict]:
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.arguments
        raw_messages = doc_to_messages(self.task_dict[task][split][doc_id])
        chat_messages = ChatMessages(messages=raw_messages)
        video_kwargs = self._build_video_kwargs()
        messages = chat_messages.to_hf_messages(video_kwargs=video_kwargs)
        if self.add_sys:
            messages.insert(0, {"role": "system", "content": COT_SYSTEM_PROMPT_ANSWER_TWICE})
        compound_key = f"{task}::{doc_id}"
        payload = {
            "messages": [messages],
            "eval_mode": True,
            "return_mm_data": False,
            "_task": task,
            "_doc_id": doc_id,
            "_compound_id": compound_key,
        }
        return compound_key, payload
    
    def _submit_payloads_with_retries(self, payloads: List[dict], keys: List[str]) -> Dict[str, Dict[str, Any]]:
        """Submit payloads to pool with retries, returning results dict."""
        self._ensure_allocator_pool()
        if self.pool is None:
            return {k: default_scale_entry() for k in keys}

        results: Dict[str, Dict[str, Any]] = {}
        pending_payloads = payloads
        pending_keys = keys
        attempt = 0
        mismatch_count = 0
        while pending_payloads and attempt <= self.scale_preprocess_retries:
            timeout_s = int(self.scale_preprocess_timeout_s) * max(1, attempt + 1)
            chunk_size = max(1, int(self.scale_preprocess_chunk_size))
            next_payloads = []
            next_keys = []
            for start in range(0, len(pending_payloads), chunk_size):
                chunk_payloads = pending_payloads[start : start + chunk_size]
                chunk_keys = pending_keys[start : start + chunk_size]
                try:
                    outs = self.pool.submit_many_sync(chunk_payloads, timeout=timeout_s)
                except Exception as e:
                    print(f"[VLLMGenerateAutoThink] Scale preprocess batch exception (attempt={attempt}) size={len(chunk_keys)}: {e}")
                    for ck, cp in zip(chunk_keys, chunk_payloads):
                        next_payloads.append(cp)
                        next_keys.append(ck)
                        self.scale_preprocess_failures.append((ck, str(e)))
                    continue
                for i, out in enumerate(outs):
                    compound = chunk_keys[i]
                    if isinstance(out, Exception):
                        next_payloads.append(chunk_payloads[i])
                        next_keys.append(compound)
                        self.scale_preprocess_failures.append((compound, str(out)))
                        continue
                    if isinstance(out, dict) and not out.get("ok", True):
                        next_payloads.append(chunk_payloads[i])
                        next_keys.append(compound)
                        continue
                    ret_compound = out.get("_compound_id", compound) if isinstance(out, dict) else compound
                    if ret_compound != compound:
                        mismatch_count += 1
                        if mismatch_count <= 5:
                            print(f"[VLLMGenerateAutoThink] WARNING: compound mismatch expected={compound} got={ret_compound}")
                    results[compound] = {
                        "scale_means": out.get("scale_means", 1.0),
                        "scale_mask": out.get("scale_mask", None),
                        "scales": out.get("scales", None),
                    }
            pending_payloads = next_payloads
            pending_keys = next_keys
            attempt += 1
        
        if mismatch_count > 5:
            print(f"[VLLMGenerateAutoThink] WARNING: Total {mismatch_count} compound mismatches detected (showing first 5)")

        if pending_keys:
            for compound in pending_keys:
                results[compound] = default_scale_entry()

        # Summary
        success_count = sum(1 for r in results.values() if r.get("scales", None) is not None)
        fail_count = len(results) - success_count
        if fail_count > 0:
            print("\n[VLLMGenerateAutoThink] Scale preprocessing summary:")
            print(f"  Total batch items: {len(keys)}")
            print(f"  Successful: {success_count}")
            print(f"  Failed (fallback to 1.0): {fail_count}")
            if len(keys) > 0:
                print(f"  Success rate: {(success_count / len(keys)) * 100:.1f}%")
        
        return results

    def _batch_scale_preprocess(self, requests: List[Instance]) -> Dict[str, Dict[str, Any]]:
        if self.allocator_path is None and not self.enable_baseline_scale:
            return {self.make_compound_key_from_req(req): default_scale_entry() for req in requests}
        
        payloads: List[dict] = []
        keys: List[str] = []
        for req in requests:
            compound_key, payload = self._prepare_scale_payload(req)
            payloads.append(payload)
            keys.append(compound_key)
        return self._get_scales_with_disk_cache(payloads, keys)

    def _make_one_request_with_scale(
        self,
        request: Instance,
        scale_entry: Dict[str, Any],
    ) -> Tuple[dict, dict, float, Optional[int]]:
        import torch
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.arguments
        raw_messages = doc_to_messages(self.task_dict[task][split][doc_id])
        if self.add_sys:
            raw_messages = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}] + raw_messages
        chat_messages = ChatMessages(messages=raw_messages)
        _gen = dict(gen_kwargs or {})
        _gen.setdefault("max_new_tokens", 4096)
        _gen.setdefault("temperature", 0)
        _gen.setdefault("top_p", 0.95)
        until = _gen.get("until", None)
        if isinstance(until, str):
            until = [until]
        if isinstance(until, list):
            until = [s for s in until if s != "\n\n"]
        params = {
            "temperature": _gen["temperature"],
            "max_tokens": _gen["max_new_tokens"],
            "top_p": _gen["top_p"],
        }
        if until:
            params["stop"] = until
        params.setdefault("logprobs", True)
        if isinstance(params.get("logprobs", None), bool):
            params["logprobs"] = 1
        video_kwargs = self._build_video_kwargs()
        messages = chat_messages.to_hf_messages(video_kwargs=video_kwargs)
        images, videos, audios = chat_messages.extract_media()
        video_inputs: Optional[List[Any]] = []
        video_metadatas: Optional[List[dict]] = []
        mm_kwargs: Dict[str, Any] = {}
        for video in videos:
            video_dict = {"type": "video", "video": video, **video_kwargs}
            final_video, fps = fetch_video(
                video_dict,
                return_video_metadata=True,
                return_video_sample_fps=True,
            )
            frames, video_metadata = final_video
            video_inputs.append(frames)
            video_metadatas.append(video_metadata)
            mm_kwargs["fps"] = fps
            mm_kwargs["do_sample_frames"] = False
        if len(videos) == 0:
            video_inputs = None
            video_metadatas = None
        if len(images) == 0:
            images = None
        if len(audios) == 0:
            audios = None
        packed_videos = None
        if video_inputs is not None:
            packed_videos = [(v, m) for v, m in zip(video_inputs, video_metadatas)]
            if env_true("VIDEO2IMAGE"):
                messages, images = video2images(messages, packed_videos, images)
                packed_videos = None
                video_inputs = None
                video_metadatas = None
            elif env_true("VIDEO2LIST"):
                messages, packed_videos = video2list(
                    messages,
                    packed_videos,
                    env_true("RESADAPT_MROPE_PATCH") or env_true("VLLM_MROPE_PATCH"),
                    self.processor.video_processor.temporal_patch_size,
                )
        scales = scale_entry.get("scales", None)
        scale_mask = scale_entry.get("scale_mask", None)
        scale_means = float(scale_entry.get("scale_means", 1.0))
        scales, scale_mask = _ensure_2d_scale_tensors(scales, scale_mask)
        scale_mask = _align_scale_mask(scales, scale_mask)
        if scales is not None:
            try:
                if hasattr(scales, "float") and hasattr(scales, "mean"):
                    scale_means = float(scales.float().mean().item())
                elif isinstance(scales, (list, tuple)):
                    flat_vals = []
                    for row in scales:
                        if isinstance(row, (list, tuple)):
                            flat_vals.extend([v for v in row if isinstance(v, (int, float))])
                        elif isinstance(row, (int, float)):
                            flat_vals.append(row)
                    if flat_vals:
                        scale_means = float(sum(flat_vals) / len(flat_vals))
            except Exception:
                pass
            masked = _masked_scale_stats(scales, scale_mask)
            if masked is not None:
                scale_means = masked[0]
        if scales is not None and len(scales) > 0:
            mm_data = {"images": images, "videos": packed_videos}
            mm_data = apply_adaptive_scaling(
                multi_modal_data=[mm_data],
                scales=scales,
                new_scale_mask=scale_mask,
                processor=self.processor,
                patch_size=self.patch_size,
                image_factor=self.image_factor,
                temporal_patch_size=self.processor.video_processor.temporal_patch_size,
            )[0]
            images = mm_data.get("images", None)
            videos_scaled_flat = mm_data.get("videos", None)
            if videos_scaled_flat is not None:
                video_inputs = [v[0] for v in videos_scaled_flat]
                video_metadatas = [v[1] for v in videos_scaled_flat]
            else:
                video_inputs = None
                video_metadatas = None
        elif self.allocator_path is not None:
            # Allocator enabled but scales missing: keep original media
            compound_key = f"{task}::{doc_id}"
            print(f"[VLLMGenerateAutoThink] WARNING: scales missing for request {compound_key}, keeping original media")
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if video_metadatas is not None and len(video_metadatas) > 0 and "video_timestamps" in video_metadatas[0]:
            text = maybe_expand_video_prompt(
                raw_prompt=text,
                videos=video_inputs,
                video_metadatas=video_metadatas,
                temporal_patch_size=self.processor.video_processor.temporal_patch_size,
            )
        if video_inputs and env_true("VIDEO2IMAGE") and env_true("REMOVEPAD"):
            text = expand_image_prompt(text, video_inputs)
        vllm_inputs: Dict[str, Any] = {"prompt": text, "multi_modal_data": {}}
        if images is not None:
            vllm_inputs["multi_modal_data"]["image"] = images
        if video_inputs is not None:
            vllm_inputs["multi_modal_data"]["video"] = []
            for video_input, video_metadata in zip(video_inputs, video_metadatas):
                if "Qwen3VL" in type(self.processor).__name__:
                    if isinstance(video_metadata, dict) and "video_timestamps" in video_metadata:
                        video_metadata = {k: v for k, v in video_metadata.items() if k != "video_timestamps"}
                    vllm_inputs["multi_modal_data"]["video"].append((video_input, video_metadata))
                else:
                    vllm_inputs["multi_modal_data"]["video"].append(video_input)
            vllm_inputs["mm_processor_kwargs"] = {**mm_kwargs}
        visual_len = self._compute_visual_len(text, images, video_inputs, video_metadatas) if self.log_visual_len else None
        if self.log_visual_len and (images is not None or video_inputs is not None):
            if visual_len is None or (isinstance(visual_len, int) and visual_len <= 0):
                visual_len = 1
        return vllm_inputs, params, scale_means, visual_len

    def _extract_boxed_segments(self, text: str) -> List[str]:
        segments: List[str] = []
        idx = 0
        while True:
            start = text.find("\\boxed{", idx)
            if start == -1:
                break
            i = start + len("\\boxed{")
            depth = 1
            j = i
            while j < len(text) and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            if depth == 0:
                segments.append(text[start:j])
                idx = j
            else:
                break
        return segments

    def _select_autothink_answer(self, ans: str, force_mode: Optional[str] = None) -> str:
        first_chunk = ans.split("<think>")[0]
        second_chunk = ans.split("</think>")[-1] if "</think>" in ans else ""
        segments = self._extract_boxed_segments(ans)
        first_box = segments[0] if len(segments) > 0 else ""
        second_box = segments[1] if len(segments) > 1 else ""
        first_answer = first_chunk.strip() if first_chunk.strip() else first_box
        second_answer = second_chunk.strip() if second_chunk.strip() else second_box
        mode = force_mode or self.inference_mode
        if mode == "first":
            return first_answer if first_answer else ans
        if mode == "second":
            return second_answer if second_answer else first_answer if first_answer else ans
        if first_box and second_box:
            first_box_probs = 1.0 if first_box.strip() == second_box.strip() else 0.0
        elif first_box:
            first_box_probs = 1.0
        else:
            first_box_probs = 0.0
        if self.early_exit_thresh > 0:
            if first_box_probs >= self.early_exit_thresh:
                return first_answer if first_answer else ans
            return second_answer if second_answer else first_answer if first_answer else ans
        return (second_answer if second_answer else first_answer if first_answer else ans) + f"\n\n\nprobs:{first_box_probs}"

    def _get_token_logprob(self, logprob_entry, token_id: int) -> Optional[float]:
        if logprob_entry is None:
            return None
        if isinstance(logprob_entry, dict):
            if token_id in logprob_entry:
                val = logprob_entry[token_id]
                if isinstance(val, (int, float)):
                    return float(val)
                if hasattr(val, "logprob"):
                    return float(val.logprob)
                if isinstance(val, dict) and "logprob" in val:
                    return float(val["logprob"])
            return None
        if hasattr(logprob_entry, "logprob"):
            return float(logprob_entry.logprob)
        if isinstance(logprob_entry, (int, float)):
            return float(logprob_entry)
        return None

    def _find_first_box_span(self, text: str) -> Optional[Tuple[int, int]]:
        start = text.find("\\boxed{")
        if start == -1:
            return None
        i = start + len("\\boxed{")
        depth = 1
        j = i
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth != 0:
            return None
        return start, j

    def _compute_first_boxed_answer_probs(self, output, text: str) -> float:
        span = self._find_first_box_span(text)
        if span is None:
            return 0.0
        start_char, end_char = span
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return 0.0
        prefix = text[:start_char]
        boxed = text[start_char:end_char]
        try:
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            boxed_ids = tokenizer.encode(boxed, add_special_tokens=False)
        except Exception:
            return 0.0
        start_idx = len(prefix_ids)
        end_idx = start_idx + len(boxed_ids)
        token_ids = getattr(output, "token_ids", None)
        logprobs = getattr(output, "logprobs", None)
        if token_ids is None or logprobs is None:
            return 0.0
        end_idx = min(end_idx, len(token_ids), len(logprobs))
        if start_idx >= end_idx:
            return 0.0
        probs = []
        for i in range(start_idx, end_idx):
            lp = self._get_token_logprob(logprobs[i], token_ids[i])
            if lp is None:
                continue
            if lp <= 0:
                prob = math.exp(lp)
            else:
                prob = lp
            if 0 <= prob <= 1:
                probs.append(prob)
        if not probs:
            return 0.0
        return float(sum(probs) / len(probs))

    def _needs_second_pass(self, ans: str) -> bool:
        if not ans or not ans.strip():
            return True
        if "\\boxed{" not in ans:
            return True
        if "Let's analyze the problem step by step." in ans:
            return True
        return False

    def make_one_request(self, request: Instance, system_prompt: Optional[str] = None) -> Tuple[dict, dict, Optional[int]]:
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.arguments
        raw_messages = doc_to_messages(self.task_dict[task][split][doc_id])
        if self.add_sys:
            sys_prompt = system_prompt or self.system_prompt
            if raw_messages and raw_messages[0].get("role") == "system":
                raw_messages[0]["content"] = [{"type": "text", "text": sys_prompt}]
            else:
                raw_messages = [{"role": "system", "content": [{"type": "text", "text": sys_prompt}]}] + raw_messages
        chat_messages = ChatMessages(messages=raw_messages)
        _gen = dict(gen_kwargs or {})
        _gen.setdefault("max_new_tokens", 4096)
        _gen.setdefault("temperature", 0)
        _gen.setdefault("top_p", 0.95)

        # map simple qwen2_5_vl_auto 'until' into vLLM stop list
        until = _gen.get("until", None)
        if isinstance(until, str):
            until = [until]
        if isinstance(until, list):
            # avoid using "\n\n" as a stopper (align with simple version)
            until = [s for s in until if s != "\n\n"]
        params = {
            "temperature": _gen["temperature"],
            "max_tokens": _gen["max_new_tokens"],
            "top_p": _gen["top_p"],
        }
        if until:
            params["stop"] = until
        # enable token logprobs for potential confidence-based selection
        params.setdefault("logprobs", True)
        if isinstance(params.get("logprobs", None), bool):
            params["logprobs"] = 1

        video_kwargs = {
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_image_pixels,
            "max_frames": self.max_frame_num,
        }
        if self.fps is not None:
            video_kwargs["fps"] = self.fps
        else:
            video_kwargs["nframes"] = self.nframes
        messages = chat_messages.to_hf_messages(video_kwargs=video_kwargs)
        images, videos, audios = chat_messages.extract_media()
        video_inputs = []
        video_metadatas = []
        kwargs = {}
        for video in videos:
            video_dict = {
                "type": "video",
                "video": video,
                **video_kwargs,
            }
            final_video, fps = fetch_video(video_dict, return_video_metadata=True, return_video_sample_fps=True)
            frames, video_metadata = final_video
            video_inputs.append(frames)
            video_metadatas.append(video_metadata)
            kwargs["fps"] = fps
            kwargs["do_sample_frames"] = False
        if len(videos) == 0:
            video_inputs = None
            video_metadatas = None
        if len(images) == 0:
            images = None
        if len(audios) == 0:
            audios = None

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        vllm_inputs = {"prompt": text, "multi_modal_data": {}}
        if images is not None:
            vllm_inputs["multi_modal_data"]["image"] = images
        if video_inputs is not None:
            vllm_inputs["multi_modal_data"]["video"] = []
            for video_input, video_metadata in zip(video_inputs, video_metadatas):
                if "Qwen3VL" in type(self.processor).__name__:
                    if isinstance(video_metadata, dict) and "video_timestamps" in video_metadata:
                        video_metadata = {k: v for k, v in video_metadata.items() if k != "video_timestamps"}
                    video_input = (video_input, video_metadata)
                vllm_inputs["multi_modal_data"]["video"].append(video_input)
            vllm_inputs["mm_processor_kwargs"] = {**kwargs}

        visual_len = self._compute_visual_len(text, images, video_inputs, video_metadatas) if self.log_visual_len else None
        if self.log_visual_len and (images is not None or video_inputs is not None):
            if visual_len is None or (isinstance(visual_len, int) and visual_len <= 0):
                visual_len = 1
        return vllm_inputs, params, visual_len

    def generate_until(self, requests) -> List[str]:
        res = []
        self.load_cache()
        res, requests = self.get_response_from_cache(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        use_scale = self.allocator_path is not None or self.enable_baseline_scale
        scale_preprocess_s = 0.0
        all_scales_dict: Dict[str, Dict[str, Any]] = {}
        
        # Check if distributed is available and initialized
        try:
            import torch
            dist_available = torch.distributed.is_available() and torch.distributed.is_initialized()
        except Exception:
            dist_available = False
        
        # Only process scales if allocator is enabled
        if len(requests) > 0 and use_scale:
            if self._is_main_rank:
                # Rank 0: Gather requests from all ranks, run allocator pool, and broadcast scales
                self._unload_vllm()
                
                # 1. Gather all requests (payloads only, to strictly avoid pickling issues)
                all_payloads: List[dict] = []
                all_keys: List[str] = []
                
                # Prepare local payloads first
                local_payloads: List[dict] = []
                local_keys: List[str] = []
                for req in requests:
                    try:
                        k, p = self._prepare_scale_payload(req)
                        local_payloads.append(p)
                        local_keys.append(k)
                    except Exception:
                        pass # Ignore failed prep (will use default scale later)
                
                if dist_available and self._world_size_env > 1:
                    gather_payloads = [None for _ in range(self._world_size_env)]
                    gather_keys = [None for _ in range(self._world_size_env)]
                    try:
                        torch.distributed.gather_object(local_payloads, gather_payloads, dst=0)
                        torch.distributed.gather_object(local_keys, gather_keys, dst=0)
                        
                        # Flatten list of lists
                        for p_list in gather_payloads:
                            if p_list: all_payloads.extend(p_list)
                        for k_list in gather_keys:
                            if k_list: all_keys.extend(k_list)
                    except Exception as e:
                        print(f"[VLLMGenerateAutoThink] [Rank 0] Warning: gather_object failed: {e}")
                        all_payloads = local_payloads
                        all_keys = local_keys
                else:
                    all_payloads = local_payloads
                    all_keys = local_keys

                # 2. Run allocator pool on all gathered payloads
                try:
                    t0 = time.time()
                    all_scales_dict = self._get_scales_with_disk_cache(all_payloads, all_keys)
                    scale_preprocess_s = time.time() - t0
                    print(f"[VLLMGenerateAutoThink] [Rank 0] Scale preprocessing took {scale_preprocess_s:.2f}s for {len(all_keys)} gathered items")
                except Exception as e:
                    print(f"[VLLMGenerateAutoThink] [Rank 0] Warning: scale preprocessing failed: {e}")
                    import traceback
                    traceback.print_exc()
                    all_scales_dict = {k: default_scale_entry() for k in all_keys}
                
                # Unload allocator pool to free memory before vLLM generation
                self._unload_allocator()
                
                # Convert tensors to CPU-serializable format for broadcast
                all_scales_dict = self._scales_dict_to_cpu(all_scales_dict)
            
            else:
                 # Non-Rank 0: Send local payloads to Rank 0
                local_payloads: List[dict] = []
                local_keys: List[str] = []
                for req in requests:
                    try:
                        k, p = self._prepare_scale_payload(req)
                        local_payloads.append(p)
                        local_keys.append(k)
                    except Exception:
                        pass

                if dist_available and self._world_size_env > 1:
                    try:
                        torch.distributed.gather_object(local_payloads, None, dst=0)
                        torch.distributed.gather_object(local_keys, None, dst=0)
                    except Exception as e:
                        print(f"[VLLMGenerateAutoThink] [Rank {os.getenv('RANK', '?')}] Warning: gather_object failed: {e}")

            # 3. Broadcast scales from rank 0 to all ranks
            if dist_available and self._world_size_env > 1:
                try:
                    broadcast_list = [all_scales_dict] if self._is_main_rank else [None]
                    torch.distributed.broadcast_object_list(broadcast_list, src=0)
                    all_scales_dict = broadcast_list[0]
                    if not self._is_main_rank:
                        # Optional: filter to only keep relevant keys to save memory? 
                        # But checking missing keys is useful.
                        pass
                except Exception as e:
                    print(f"[VLLMGenerateAutoThink] Warning: broadcast failed: {e}")
                    if not self._is_main_rank:
                        # Fallback: create default scales
                        all_scales_dict = {self.make_compound_key_from_req(req): default_scale_entry() for req in requests}

        # Sync all processes before vLLM initialization
        if dist_available:
            try:
                torch.distributed.barrier()
            except Exception:
                pass

        self._ensure_vllm_client()

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        e2e_latency = 0.0
        for batch_requests in batched_requests:
            batched_vllm_inputs: List[dict] = []
            visual_lens: List[Optional[int]] = []
            scales_means: List[float] = []
            scales_dict: Dict[Any, Dict[str, Any]] = {}
            if use_scale:
                scales_dict, _missing = self.build_scales_dict_for_batch(
                    batch_requests=batch_requests,
                    all_scales_dict=all_scales_dict,
                )
                with ThreadPoolExecutor(max_workers=min(self.workers, len(batch_requests))) as executor:
                    futures = [
                        executor.submit(
                            self._make_one_request_with_scale,
                            r,
                            scales_dict.get(self.make_compound_key_from_req(r), default_scale_entry()),
                        )
                        for r in batch_requests
                    ]
                    for future in futures:
                        vllm_inputs, sampling_params, scale_mean, visual_len = future.result()
                        batched_vllm_inputs.append(vllm_inputs)
                        visual_lens.append(visual_len)
                        scales_means.append(scale_mean)
            else:
                with ThreadPoolExecutor(max_workers=min(self.workers, len(batch_requests))) as executor:
                    futures = [
                        executor.submit(
                            self.make_one_request,
                            request,
                            None,
                        )
                        for request in batch_requests
                    ]
                    for future in futures:
                        vllm_inputs, sampling_params, visual_len = future.result()
                        batched_vllm_inputs.append(vllm_inputs)
                        visual_lens.append(visual_len)

            sampling_params = _build_vllm_sampling_params(sampling_params)
            start_time = time.time()
            response = self.client.generate(batched_vllm_inputs, sampling_params)
            end_time = time.time()
            e2e_latency += end_time - start_time

            response_text = [o.outputs[0].text for o in response]
            first_box_probs_list = [
                self._compute_first_boxed_answer_probs(o.outputs[0], text)
                for o, text in zip(response, response_text)
            ]

            per_req_gen_s = (end_time - start_time) / max(1, len(batch_requests))
            for req in batch_requests:
                req.generation_s = per_req_gen_s

            # Strict alignment: single-pass generation; selection handled by _select_autothink_answer
            if self.inference_mode == "auto":
                selected = []
                for text, first_box_probs in zip(response_text, first_box_probs_list):
                    if self.early_exit_thresh > 0:
                        if first_box_probs >= self.early_exit_thresh:
                            selected.append(self._select_autothink_answer(text, force_mode="first"))
                        else:
                            selected.append(self._select_autothink_answer(text, force_mode="second"))
                    else:
                        selected.append(
                            self._select_autothink_answer(text, force_mode="second") + f"\n\n\nprobs:{first_box_probs}"
                        )
                response_text = selected
            else:
                response_text = [self._select_autothink_answer(text) for text in response_text]

            if use_scale:
                per_req_scale_preprocess_s = scale_preprocess_s / max(1, len(requests))
                for req in requests:
                    req.scale_preprocess_s = per_req_scale_preprocess_s
            else:
                for req in requests:
                    req.scale_preprocess_s = 0.0

            for idx, (req, text, vlen) in enumerate(zip(batch_requests, response_text, visual_lens)):
                self.add_request_response_to_cache(req, text)
                req.visual_len = vlen
                if use_scale:
                    smean = scales_means[idx] if idx < len(scales_means) else 1.0
                    req.scale = smean
                    key = self.make_compound_key_from_req(req)
                    scale_entry = scales_dict.get(key, default_scale_entry())
                    req.scale_stats = self._scale_entry_stats(scale_entry)
                    req.scales = self._normalize_scales(scale_entry.get("scales", None))
                    req.has_allocator_scale = scale_entry.get("scales", None) is not None

            assert len(response_text) == len(batch_requests)
            res.extend(response_text)
            pbar.update(len(batch_requests))

        if not self.disable_log_stats:
            metrics = self.get_format_metrics()
            total_tokens = metrics["generation_tokens"]
            avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
            metric_dict = {
                "total_tokens": total_tokens,
                "e2e_latency": e2e_latency,
                "avg_speed": avg_speed,
                "additional_metrics": {
                    "ttft": metrics["ttft"],
                    "tpot": metrics["tpot"],
                    "rank": self.rank,
                    "scale_preprocess_s": scale_preprocess_s,
                },
            }
            log_metrics(**metric_dict)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "GPT4V not support"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")

    def get_format_metrics(self):
        metrics = self.client.get_metrics()
        ttft = 0
        tpot = 0
        generation_tokens = 0
        for metric in metrics:
            name = metric.name
            if "time_to_first_token" in name:
                ttft = metric.sum / metric.count
            if "time_per_output_token_seconds" in name:
                tpot = metric.sum / metric.count
            if name == "vllm:generation_tokens":
                generation_tokens = metric.value

        metrics = {
            "ttft": ttft,
            "tpot": tpot,
            "generation_tokens": generation_tokens,
        }

        return metrics
