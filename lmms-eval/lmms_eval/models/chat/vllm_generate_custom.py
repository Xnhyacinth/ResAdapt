import os
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

def env_true(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).lower() in ("1", "true", "yes", "y", "t")

def default_scale_entry() -> Dict[str, Any]:
    """Return a fresh default entry each time to avoid shared-mutation bugs."""
    return {"scale_means": 1.0, "scale_mask": None, "scales": None}

COT_SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> </think> tags, and then give your answer between the <answer> </answer> tags. "
    "The final answer MUST BE wrapped in \\boxed{} and the \\boxed{} expression MUST BE contained entirely within the <answer> </answer> tags."
)


def _ensure_2d_scale_tensors(
    scales: Any,
    scale_mask: Any,
) -> Tuple[Any, Any]:
    """
    Ensure scales and scale_mask are shaped like (1, T) if they are torch tensors of shape (T,).
    Keeps non-tensor values unchanged.
    """
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


def _count_video_frames(video_frames: Any) -> int:
    if hasattr(video_frames, "shape"):
        try:
            return int(video_frames.shape[0])
        except Exception:
            pass
    try:
        return len(video_frames)
    except Exception:
        return 0


def _build_baseline_scales(
    images: Optional[List[Any]],
    packed_videos: Optional[List[Tuple[Any, dict]]],
    temporal_patch_size: int,
    scale_factor: float,
) -> List[float]:
    scales: List[float] = []
    if images is not None:
        scales.extend([scale_factor] * len(images))
    if packed_videos is not None:
        for frames, _meta in packed_videos:
            num_frames = _count_video_frames(frames)
            if num_frames <= 0:
                continue
            num_chunks = int((num_frames + temporal_patch_size - 1) / temporal_patch_size)
            scales.extend([scale_factor] * max(num_chunks, 1))
    return scales

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


WORKERS = int(os.getenv("WORKERS", "32"))
MICRO_BATCH = int(os.getenv("MICRO_BATCH", "16"))
max_inflight_per_gpu = int(os.getenv("max_inflight_per_gpu", "4"))


from visionthink.adaptive.utils import (
    expand_image_prompt,
    apply_adaptive_scaling,
    maybe_expand_video_prompt,
    video2list,
    video2images,
    expand_video_prompt_blocks,
)

if env_true("VLLM_MROPE_PATCH"):
    print("🚀 [Patching] Applying custom VLLM Qwen2.5-VL MROPE logic...")
    import vllm
    from vllm.model_executor.models import qwen2_5_vl
    from visionthink.predictor.vllm_patch import get_mrope_input_positions
    # , iter_mm_grid_thw

    # vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.get_mrope_input_positions = (
    #     get_mrope_input_positions
    # )
    # vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLMultiModalProcessor._get_mm_fields_config = _get_mm_fields_config
    vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.get_mrope_input_positions = get_mrope_input_positions
    # vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.iter_mm_grid_thw = iter_mm_grid_thw


@register_model("vllm_generate_custom")
class VLLMGenerateCustom(VLLMChat):
    """
    Different from .chat, use generate method instead of chat method.
    """

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
        **kwargs,
    ):
        self.predictor_path = os.getenv("PREDICTOR_PATH", None)
        self.enable_baseline_scale = env_true("ENABLE_BASELINE_SCALE")
        try:
            self.baseline_scale_factor = float(os.getenv("BASELINE_SCALE_FACTOR", "1.0"))
        except Exception:
            self.baseline_scale_factor = 1.0
        self.add_sys = env_true("ADD_SYS")
        self.log_visual_len = env_true("LOG_VISUAL_LEN")
        self.scale_preprocess_retries = int(os.getenv("SCALE_PREPROCESS_RETRIES", "2"))
        self.workers = int(os.getenv("WORKERS", str(WORKERS)))
        self.micro_batch = int(os.getenv("MICRO_BATCH", str(MICRO_BATCH)))
        self.max_inflight_per_gpu = int(os.getenv("max_inflight_per_gpu", str(max_inflight_per_gpu)))
        self.max_queue_per_gpu = int(os.getenv("MAX_QUEUE_PER_GPU", str(os.getenv("MAX_QUEUE_PER_GPU", "8"))))

        # Determine if this is the main rank (only rank 0 runs predictor pool)
        self._is_main_rank = int(os.getenv("RANK", "0")) == 0
        self._world_size_env = int(os.getenv("WORLD_SIZE", "1"))

        # Pool kwargs for predictor pool initialization
        self._pool_kwargs = dict(
            model_path=self.predictor_path,
            num_gpus=int(os.getenv("PREDICTOR_NUM_GPUS", "8")),
            enable_batch=True,
            microbatch_ms=int(os.getenv("MICRO_BATCH_MS", "10")),
            microbatch_max=self.micro_batch,
            max_total_inflight=self.max_inflight_per_gpu * int(os.getenv("PREDICTOR_NUM_GPUS", "8")),
            max_inflight_per_gpu=self.max_inflight_per_gpu,
            max_queue_per_gpu=self.max_queue_per_gpu,
            submit_threads=self.workers if self.predictor_path is not None and "smol" not in self.predictor_path else self.workers * 3,
            max_frames=max_frame_num // 2 if self.predictor_path is not None and "smol" not in self.predictor_path else max_frame_num,
            schedule_policy=os.getenv("PREDICTOR_SCHED_POLICY", "least_inflight"),
        )
        self.pool = None
        self.scale_preprocess_failures: List[Tuple[str, str]] = []

        # Only rank 0 starts predictor pool to avoid GPU competition
        if self._is_main_rank and self.predictor_path is not None:
            print(f"[VLLMGenerateCustom] [Rank 0] Found predictor at {self.predictor_path}")
            
            # Check GPU availability before starting pool
            if self._check_gpu_available():
                try:
                    from visionthink.predictor.multi_model_limit_async import MultiGPUInferPool
                    print(
                        "[VLLMGenerateCustom] [Rank 0] Predictor pool config: "
                        f"num_gpus={self._pool_kwargs.get('num_gpus')} "
                        f"microbatch_max={self._pool_kwargs.get('microbatch_max')} "
                        f"max_inflight_per_gpu={self._pool_kwargs.get('max_inflight_per_gpu')} "
                        f"max_queue_per_gpu={self._pool_kwargs.get('max_queue_per_gpu')} "
                        f"max_total_inflight={self._pool_kwargs.get('max_total_inflight')} "
                        f"submit_threads={self._pool_kwargs.get('submit_threads')}"
                    )
                    self.pool = MultiGPUInferPool(**self._pool_kwargs)
                    self.pool.start()
                    print("[VLLMGenerateCustom] [Rank 0] Predictor pool started successfully")
                except Exception as e:
                    print(f"[VLLMGenerateCustom] [Rank 0] Warning: Failed to start predictor pool: {e}")
                    import traceback
                    traceback.print_exc()
                    self.pool = None
            else:
                print("[VLLMGenerateCustom] [Rank 0] Warning: GPU not available, skipping predictor pool")
        elif not self._is_main_rank and self.predictor_path is not None:
            print(f"[VLLMGenerateCustom] [Rank {os.getenv('RANK', '?')}] Waiting for rank 0 to run predictor")

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
                self.processor.chat_template = f.read()

        self.patch_size = self.processor.image_processor.patch_size
        self.image_factor = self.processor.video_processor.merge_size * self.patch_size

    def _check_gpu_available(self, min_free_gb: float = 1.0) -> bool:
        """Check if GPUs are available with sufficient free memory.
        
        Args:
            min_free_gb: Minimum free memory in GB required per GPU.
            
        Returns:
            True if all GPUs have sufficient free memory.
        """
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
                    mem_free, mem_total = torch.cuda.mem_get_info(i)
                    if mem_free < min_free_bytes:
                        print(f"[VLLMGenerateCustom] GPU {i}: only {mem_free / 1024**3:.2f}GB free")
                        return False
                except Exception as e:
                    print(f"[VLLMGenerateCustom] Cannot query GPU {i}: {e}")
                    return False
            return True
        except Exception as e:
            print(f"[VLLMGenerateCustom] GPU check failed: {e}")
            return False


    def _unload_vllm(self):
        """Forcefully unload VLLM to free GPU memory."""
        if hasattr(self, "client") and self.client is not None:
            print("[VLLMGenerateCustom] 🧹 Unloading VLLM Client...")
            if hasattr(self.client, "llm_engine"):
                del self.client.llm_engine
            del self.client
            self.client = None

            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()

    def _unload_predictor(self):
        """Forcefully stop predictor pool to free GPU memory."""
        if self.pool is not None:
            print("[VLLMGenerateCustom] 🧹 Stopping Predictor Pool...")
            try:
                self.pool.close()
            except Exception as e:
                print(f"[VLLMGenerateCustom] Warning during pool close: {e}")
            self.pool = None

            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()

    def _scales_dict_to_cpu(self, scales_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Convert tensor values in scales dict to CPU lists for distributed broadcast.
        
        torch.distributed.broadcast_object_list requires CPU-serializable objects.
        This converts any GPU tensors to CPU lists.
        """
        try:
            import torch
        except ImportError:
            return scales_dict
        
        result = {}
        for key, entry in scales_dict.items():
            new_entry = {}
            for k, v in entry.items():
                if torch.is_tensor(v):
                    # Move to CPU and convert to list
                    new_entry[k] = v.detach().cpu().tolist()
                else:
                    new_entry[k] = v
            result[key] = new_entry
        return result

    def _ensure_vllm_client(self):
        """Recreate vLLM client if it doesn't exist."""
        if hasattr(self, "client") and self.client is not None:
            return

        print("[VLLMGenerateCustom] 🚀 Initializing VLLM Client...")
        try:
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
            print("[VLLMGenerateCustom] ✅ VLLM Client ready.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize VLLM: {e}")

    def _ensure_predictor_pool(self):
        """Initialize predictor pool if needed."""
        if self.pool is not None:
            return
        if self.predictor_path is None:
            return

        print(f"[VLLMGenerateCustom] 🚀 Initializing Predictor Pool (Model: {self.predictor_path})...")
        try:
            from visionthink.predictor.multi_model_limit_async import MultiGPUInferPool
            self.pool = MultiGPUInferPool(**self._pool_kwargs)
            self.pool.start()
            print("[VLLMGenerateCustom] ✅ Predictor Pool ready.")
        except Exception as e:
            print(f"[VLLMGenerateCustom] ❌ Failed to start predictor pool: {e}")
            self.pool = None


    def make_compound_key_from_req(self, req: Instance) -> str:
        """
        req.arguments:
          (ctx, doc_to_messages, gen_kwargs, doc_id, task, split)
        """
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
        """
        Build scales_dict keyed by req compound key and collect missing keys.

        Returns:
          scales_dict: {compound_key: entry}
          missing: [compound_key, ...]
        """
        scales_dict: Dict[Any, Dict[str, Any]] = {}
        missing: List[Any] = []

        for req in batch_requests:
            key = self.make_compound_key_from_req(req)
            entry = all_scales_dict.get(key, None)

            if entry is None:
                missing.append(key)
                scales_dict[key] = default_scale_entry()
            else:
                # shallow copy to avoid accidental in-place mutation
                scales_dict[key] = dict(entry)

        if warn_on_missing and missing:
            if (not warn_only_if_dict_nonempty) or (len(all_scales_dict) > 0):
                print(
                    f"[VLLMGenerateCustom] WARNING: {len(missing)} doc_ids not found in all_scales_dict "
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

    def _normalize_scales(self, scales):
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

    def _to_cpu_mm_data(self, mm_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Recursively convert GPU tensors in rescaled_mm_data to CPU numpy arrays.
        This prevents OOM during distributed gather and makes data picklable.
        """
        if mm_data is None:
            return None
        try:
            import torch
            import numpy as np
        except ImportError:
            return mm_data

        def to_cpu(obj):
            if obj is None:
                return None
            if torch.is_tensor(obj):
                return obj.detach().cpu().numpy()
            if isinstance(obj, np.ndarray):
                return obj
            if isinstance(obj, (list, tuple)):
                converted = [to_cpu(item) for item in obj]
                return type(obj)(converted) if isinstance(obj, tuple) else converted
            if isinstance(obj, dict):
                return {k: to_cpu(v) for k, v in obj.items()}
            return obj

        return to_cpu(mm_data)

    def _compute_visual_len(self, text, images, videos, video_metadatas):
        if images is None and videos is None:
            return 0
        proc_kwargs: Dict[str, Any] = {}
        if images is not None:
            proc_kwargs["images"] = images
        if videos is not None:
            proc_kwargs["videos"] = videos
            if video_metadatas is not None:
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

    def _build_video_kwargs(self) -> Dict[str, Any]:
        return {
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_image_pixels,
            "max_frames": self.max_frame_num,
        }

    def _prepare_scale_payload(self, request: Instance) -> Tuple[str, dict]:
        """
        Prepare payload for predictor without calling it.
        Returns (compound_key, payload)
        """
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.arguments
        raw_messages = doc_to_messages(self.task_dict[task][split][doc_id])
        chat_messages = ChatMessages(messages=raw_messages)
        video_kwargs = self._build_video_kwargs()

        messages = chat_messages.to_hf_messages(video_kwargs=video_kwargs)
        if self.add_sys:
            messages.insert(0, {"role": "system", "content": COT_SYSTEM_PROMPT})

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
        self._ensure_predictor_pool()
        if self.pool is None:
            return {k: default_scale_entry() for k in keys}

        results: Dict[str, Dict[str, Any]] = {}
        pending_payloads = payloads
        pending_keys = keys
        attempt = 0
        mismatch_count = 0
        while pending_payloads and attempt <= self.scale_preprocess_retries:
            outs = self.pool.submit_many_sync(pending_payloads, timeout=9600)
            next_payloads: List[dict] = []
            next_keys: List[str] = []
            for i, out in enumerate(outs):
                compound = pending_keys[i]
                if isinstance(out, Exception):
                    next_payloads.append(pending_payloads[i])
                    next_keys.append(compound)
                    self.scale_preprocess_failures.append((compound, str(out)))
                    print(f"[VLLMGenerateCustom] Scale preprocess error (attempt={attempt}) {compound}: {out}")
                    continue
                if isinstance(out, dict) and not out.get("ok", True):
                    err = out.get("err", "unknown")
                    retry = out.get("retry", True)
                    self.scale_preprocess_failures.append((compound, str(err)))
                    if retry:
                        next_payloads.append(pending_payloads[i])
                        next_keys.append(compound)
                        print(f"[VLLMGenerateCustom] Scale preprocess retry (attempt={attempt}) {compound}: {err}")
                    else:
                        print(f"[VLLMGenerateCustom] Scale preprocess failed (no-retry) {compound}: {err}")
                    continue
                ret_compound = out.get("_compound_id", compound) if isinstance(out, dict) else compound
                if ret_compound != compound:
                    mismatch_count += 1
                    if mismatch_count <= 5:
                        print(f"[VLLMGenerateCustom] WARNING: compound mismatch expected={compound} got={ret_compound}")
                    # Note: Use original compound key, not ret_compound, to ensure proper matching
                # Always use original compound key for results storage
                results[compound] = {
                    "scale_means": out.get("scale_means", 1.0),
                    "scale_mask": out.get("scale_mask", None),
                    "scales": out.get("scales", None),
                }
            pending_payloads = next_payloads
            pending_keys = next_keys
            attempt += 1
        
        if mismatch_count > 5:
            print(f"[VLLMGenerateCustom] WARNING: Total {mismatch_count} compound mismatches detected (showing first 5)")

        if pending_keys:
            for compound in pending_keys:
                results[compound] = default_scale_entry()

        # Summary handled by caller or here? 
        # Let's handle summary here for gathered requests
        success_count = sum(1 for r in results.values() if r.get("scales", None) is not None)
        fail_count = len(results) - success_count
        if fail_count > 0:
            print("\n[VLLMGenerateCustom] Scale preprocessing summary:")
            print(f"  Total batch items: {len(keys)}")
            print(f"  Successful: {success_count}")
            print(f"  Failed (fallback to 1.0): {fail_count}")
            if len(keys) > 0:
                print(f"  Success rate: {(success_count / len(keys)) * 100:.1f}%")
        
        return results

    def _batch_scale_preprocess(self, requests: List[Instance]) -> Dict[str, Dict[str, Any]]:
        """
        Run predictor over all requests (one big batch), returning:
          all_scales_dict[compound_key] = {"scale_means","scale_mask","scales"}
        """
        # No predictor: return defaults with correct compound keys
        if self.predictor_path is None and not self.enable_baseline_scale:
            return {self.make_compound_key_from_req(req): default_scale_entry() for req in requests}

        payloads: List[dict] = []
        keys: List[str] = []
        for req in requests:
            compound_key, payload = self._prepare_scale_payload(req)
            payloads.append(payload)
            keys.append(compound_key)
            
        return self._submit_payloads_with_retries(payloads, keys)


    def _make_one_request_with_scale(
        self,
        request: Instance,
        scale_entry: Dict[str, Any],
    ) -> Tuple[dict, dict, float, Optional[int], Optional[Dict[str, Any]]]:
        """
        Build vLLM input using precomputed scale entry for this request.
        Returns: (vllm_inputs, sampling_params_dict, scale_means, visual_len, rescaled_mm_data)
        """
        import torch  # safe here (already in main proc)

        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.arguments
        raw_messages = doc_to_messages(self.task_dict[task][split][doc_id])
        chat_messages = ChatMessages(messages=raw_messages)

        _gen = dict(gen_kwargs or {})
        _gen.setdefault("max_new_tokens", 4096)
        _gen.setdefault("temperature", 0)
        _gen.setdefault("top_p", 0.95)

        params = {
            "temperature": _gen["temperature"],
            "max_tokens": _gen["max_new_tokens"],
            "top_p": _gen["top_p"],
        }

        video_kwargs = self._build_video_kwargs()

        messages = chat_messages.to_hf_messages(video_kwargs=video_kwargs)
        images, videos, audios = chat_messages.extract_media()

        # Resolve media
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
                    env_true("VLLM_MROPE_PATCH"),
                    self.processor.video_processor.temporal_patch_size,
                )

        scales = scale_entry.get("scales", None)
        scale_mask = scale_entry.get("scale_mask", None)
        scale_means = float(scale_entry.get("scale_means", 1.0))
        temporal_patch_size = getattr(self.processor.video_processor, "temporal_patch_size", 1)

        if self.enable_baseline_scale and self.predictor_path is None:
            baseline_scales = _build_baseline_scales(
                images,
                packed_videos,
                temporal_patch_size,
                self.baseline_scale_factor,
            )
            if baseline_scales:
                scales = torch.tensor(baseline_scales, dtype=torch.float32)
                scale_mask = None
                scale_means = float(self.baseline_scale_factor)

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

        rescaled_mm_data = None
        if scales is not None:
            mm_data = {"images": images, "videos": packed_videos}
            mm_data = apply_adaptive_scaling(
                multi_modal_data=[mm_data],
                scales=scales,
                new_scale_mask=scale_mask,
                processor=self.processor,
                patch_size=self.patch_size,
                image_factor=self.image_factor,
                temporal_patch_size=temporal_patch_size,
            )[0]

            images = mm_data.get("images", None)
            videos_scaled_flat = mm_data.get("videos", None)
            rescaled_mm_data = {
                "images": images,
                "videos": videos_scaled_flat,
            }

            if videos_scaled_flat is not None:
                video_inputs = [v[0] for v in videos_scaled_flat]
                video_metadatas = [v[1] for v in videos_scaled_flat]
            else:
                video_inputs = None
                video_metadatas = None

        elif self.predictor_path is not None:
            # Predictor enabled but scales missing: keep original media
            print(f"[VLLMGenerateCustom] Warning: scales missing for request {task}::{doc_id}, keeping original media")

        if self.add_sys:
            messages.insert(0, {"role": "system", "content": COT_SYSTEM_PROMPT})
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if video_metadatas is not None and len(video_metadatas) > 0 and "video_timestamps" in video_metadatas[0]:
            if "Qwen3VL" in type(self.processor).__name__:
                text = expand_video_prompt_blocks(text, video_inputs, video_metadatas)
            else:
                text = maybe_expand_video_prompt(
                    raw_prompt=text,
                    videos=video_inputs,
                    video_metadatas=video_metadatas,
                    temporal_patch_size=self.processor.video_processor.temporal_patch_size,
                )

        if video_inputs and env_true("VIDEO2IMAGE") and env_true("REMOVEPAD"):
            text = expand_image_prompt(text, video_inputs)

        # Build vLLM input
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

            # set once (not inside loop)
            vllm_inputs["mm_processor_kwargs"] = {**mm_kwargs}

        visual_len = self._compute_visual_len(text, images, video_inputs, video_metadatas) if self.log_visual_len else None

        # Convert GPU tensors to CPU to prevent OOM during distributed gather
        rescaled_mm_data = self._to_cpu_mm_data(rescaled_mm_data)
        return vllm_inputs, params, scale_means, visual_len, rescaled_mm_data
    

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: List[str] = []
        self.load_cache()
        res, requests = self.get_response_from_cache(requests)

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        e2e_latency = 0.0

        scale_preprocess_s = 0.0
        all_scales_dict: Dict[str, Dict[str, Any]] = {}
        
        # Check if distributed is available and initialized
        try:
            import torch
            dist_available = torch.distributed.is_available() and torch.distributed.is_initialized()
        except Exception:
            dist_available = False
        
        # Only process scales via predictor
        use_scale = (self.predictor_path is not None)
        if len(requests) > 0 and use_scale:
            if self._is_main_rank:
                # Rank 0: Gather requests from all ranks, run predictor pool, and broadcast scales
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
                        print(f"[VLLMGenerateCustom] [Rank 0] Warning: gather_object failed: {e}")
                        all_payloads = local_payloads
                        all_keys = local_keys
                else:
                    all_payloads = local_payloads
                    all_keys = local_keys

                # 2. Run predictor pool on all gathered payloads
                try:
                    t0 = time.time()
                    all_scales_dict = self._submit_payloads_with_retries(all_payloads, all_keys)
                    scale_preprocess_s = time.time() - t0
                    print(f"[VLLMGenerateCustom] [Rank 0] Scale preprocessing took {scale_preprocess_s:.2f}s for {len(all_keys)} gathered items")
                except Exception as e:
                    print(f"[VLLMGenerateCustom] [Rank 0] Warning: scale preprocessing failed: {e}")
                    import traceback
                    traceback.print_exc()
                    all_scales_dict = {k: default_scale_entry() for k in all_keys}
                
                # Unload predictor pool to free memory before vLLM generation
                self._unload_predictor()
                
                # Convert tensors to CPU-serializable format for broadcast
                all_scales_dict = self._scales_dict_to_cpu(all_scales_dict)
            
            else:
                # Non-Rank 0: Send local payloads to Rank 0
                local_payloads = []
                local_keys = []
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
                        print(f"[VLLMGenerateCustom] [Rank {os.getenv('RANK', '?')}] Warning: gather_object failed: {e}")

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
                    print(f"[VLLMGenerateCustom] Warning: broadcast failed: {e}")
                    if not self._is_main_rank:
                        # Fallback: create default scales
                        all_scales_dict = {self.make_compound_key_from_req(req): default_scale_entry() for req in requests}

        # Sync all processes before vLLM initialization
        if dist_available:
            try:
                torch.distributed.barrier()
            except Exception:
                pass

        # Init vLLM
        self._ensure_vllm_client()

        # -------------------------
        # Generate batch by batch
        # -------------------------
        for batch_requests in batched_requests:
            scales_dict, _missing = self.build_scales_dict_for_batch(
                batch_requests=batch_requests,
                all_scales_dict=all_scales_dict,
                warn_on_missing=True,
                warn_only_if_dict_nonempty=True,
            )

            batched_vllm_inputs: List[dict] = []
            scales_means: List[float] = []
            visual_lens: List[Optional[int]] = []
            rescaled_mms: List[Optional[Dict[str, Any]]] = []

            with ThreadPoolExecutor(max_workers=min(self.workers, len(batch_requests))) as executor:
                futures = [
                    executor.submit(
                        self._make_one_request_with_scale,
                        r,
                        scales_dict.get(self.make_compound_key_from_req(r), default_scale_entry()),
                    )
                    for r in batch_requests
                ]
                # keep order stable
                for fut in futures:
                    vllm_inputs, sampling_params_dict, scale_mean, visual_len, rescaled_mm_data = fut.result()
                    batched_vllm_inputs.append(vllm_inputs)
                    scales_means.append(scale_mean)
                    visual_lens.append(visual_len)
                    rescaled_mms.append(rescaled_mm_data)

            sampling_params = SamplingParams(**sampling_params_dict)

            start_time = time.time()
            response = self.client.generate(batched_vllm_inputs, sampling_params)
            end_time = time.time()
            e2e_latency += (end_time - start_time)

            response_text = [o.outputs[0].text for o in response]

            for req, text, smean, vlen, rescaled_mm in zip(batch_requests, response_text, scales_means, visual_lens, rescaled_mms):
                self.add_request_response_to_cache(req, text)
                req.scale = smean  # attach for later reporting if needed
                if rescaled_mm is not None:
                    req.rescaled_mm_data = rescaled_mm
                key = self.make_compound_key_from_req(req)
                scale_entry = scales_dict.get(key, default_scale_entry())
                req.scale_stats = self._scale_entry_stats(scale_entry)
                req.scales = self._normalize_scales(scale_entry.get("scales", None))
                req.has_predictor_scale = scale_entry.get("scales", None) is not None
                req.visual_len = vlen

            res.extend(response_text)
            pbar.update(len(batch_requests))

            per_req_gen_s = (end_time - start_time) / max(1, len(batch_requests))
            for req in batch_requests:
                req.generation_s = per_req_gen_s

        if use_scale:
            per_req_scale_preprocess_s = scale_preprocess_s / max(1, len(requests))
            for req in requests:
                req.scale_preprocess_s = per_req_scale_preprocess_s
        else:
            for req in requests:
                req.scale_preprocess_s = 0.0

        if not self.disable_log_stats:
            metrics = self.get_format_metrics()
            total_tokens = metrics["generation_tokens"]
            avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0.0
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

        # Failure report
        if self.scale_preprocess_failures:
            print(f"\n{'='*60}")
            print("[VLLMGenerateCustom] Scale Preprocessing Failures Report")
            print(f"{'='*60}")
            print(f"Total failures: {len(self.scale_preprocess_failures)}")
            for compound, error_msg in self.scale_preprocess_failures[:10]:
                print(f"  - {compound}: {error_msg}")
            if len(self.scale_preprocess_failures) > 10:
                print(f"  ... and {len(self.scale_preprocess_failures) - 10} more")
            print(f"{'='*60}\n")

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise AssertionError("GPT4V not support")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")

    def get_format_metrics(self):
        metrics = self.client.get_metrics()
        ttft = 0.0
        tpot = 0.0
        generation_tokens = 0.0

        for metric in metrics:
            name = metric.name
            if "time_to_first_token" in name:
                ttft = metric.sum / metric.count
            if "time_per_output_token_seconds" in name:
                tpot = metric.sum / metric.count
            if name == "vllm:generation_tokens":
                generation_tokens = metric.value

        return {"ttft": ttft, "tpot": tpot, "generation_tokens": generation_tokens}
