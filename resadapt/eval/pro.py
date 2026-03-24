import argparse
import os
import statistics
import time
from typing import Any, Dict, Optional, Tuple
 
from transformers import AutoProcessor
 
from lmms_eval.protocol import ChatMessages
from resadapt.utils.utils import apply_adaptive_scaling
from resadapt.utils.utils import expand_video_prompt
from resadapt.eval.multi_model_limit_async import MultiGPUInferPool
 
 
def _build_chat(video_path: str, prompt: str, *, max_pixels: int, min_pixels: int, max_frames: int):
    raw_messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "url": video_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    chat_messages = ChatMessages(messages=raw_messages)
    hf_messages = chat_messages.to_hf_messages(
        video_kwargs={
            "max_pixels": int(max_pixels),
            "min_pixels": int(min_pixels),
            "max_frames": int(max_frames),
        }
    )
    return chat_messages, hf_messages
 
 
def _decode_video(video_path: str, *, max_pixels: int, min_pixels: int, max_frames: int):
    from qwen_vl_utils import fetch_video
 
    video_dict = {
        "type": "video",
        "video": video_path,
        "max_pixels": int(max_pixels),
        "min_pixels": int(min_pixels),
        "max_frames": int(max_frames),
    }
    final_video, fps = fetch_video(
        video_dict,
        return_video_metadata=True,
        return_video_sample_fps=True,
    )
    frames, video_metadata = final_video
    mm_kwargs = {"fps": fps, "do_sample_frames": False}
    return frames, video_metadata, mm_kwargs
 
 
def _build_vllm_input(
    processor: Any,
    prompt_text: str,
    video_inputs: list,
    video_metadatas: Optional[list],
    mm_kwargs: Optional[dict],
) -> Dict[str, Any]:
    vllm_inputs: Dict[str, Any] = {"prompt": prompt_text, "multi_modal_data": {}}
    if "Qwen3VL" in type(processor).__name__:
        if video_metadatas is not None and len(video_metadatas) == len(video_inputs):
            vllm_inputs["multi_modal_data"]["video"] = list(zip(video_inputs, video_metadatas))
        else:
            vllm_inputs["multi_modal_data"]["video"] = list(video_inputs)
    else:
        vllm_inputs["multi_modal_data"]["video"] = list(video_inputs)
    if mm_kwargs:
        vllm_inputs["mm_processor_kwargs"] = dict(mm_kwargs)
    return vllm_inputs
 
 
def _drop_video_timestamps(video_metadata: Optional[dict]) -> Optional[dict]:
    if not isinstance(video_metadata, dict):
        return video_metadata
    if "video_timestamps" not in video_metadata:
        return video_metadata
    out = dict(video_metadata)
    out.pop("video_timestamps", None)
    return out
 
 
def _expand_video_prompt_blocks(prompt: str, video_inputs) -> str:
    video_block = "<|vision_start|><|video_pad|><|vision_end|>"
    parts = prompt.split(video_block)
    if len(parts) - 1 != len(video_inputs):
        raise ValueError(
            f"The prompt contains {len(parts)-1} video placeholders, but {len(video_inputs)} video inputs are provided."
        )
    new_prompt = parts[0]
    for i, video in enumerate(video_inputs):
        count_n = len(video) if isinstance(video, list) else 1
        new_prompt += (video_block * count_n) + parts[i + 1]
    return new_prompt
 
 
def _expand_video_prompt_for_scaled(
    processor: Any,
    prompt_text: str,
    *,
    per_video_chunks: list,
    temporal_patch_size: int,
) -> str:
    if "Qwen3VL" in type(processor).__name__:
        return _expand_video_prompt_blocks(prompt_text, per_video_chunks)
    return expand_video_prompt(prompt_text, per_video_chunks, int(temporal_patch_size))
 
 
def _run_one_generate(llm: Any, sampling_params: Any, vllm_inputs: Dict[str, Any]) -> Tuple[float, str]:
    t0 = time.time()
    outs = llm.generate([vllm_inputs], sampling_params)
    t1 = time.time()
    text = outs[0].outputs[0].text
    return (t1 - t0), text
 
 
def _run_one_with_allocator(
    *,
    pool: MultiGPUInferPool,
    processor: Any,
    patch_size: int,
    image_factor: int,
    temporal_patch_size: int,
    hf_messages: list,
    frames: Any,
    video_metadata: Optional[dict],
    prompt_text: str,
    mm_kwargs: dict,
    llm: Any,
    sampling_params: Any,
    idx: int,
) -> Tuple[float, float, float, str]:
    payload = {
        "messages": [hf_messages],
        "eval_mode": True,
        "return_mm_data": False,
        "_task": "bench",
        "_doc_id": idx,
        "_compound_id": f"bench::{idx}",
    }
 
    t0 = time.time()
    out = pool.submit_sync(payload)
    scales = out.get("scales", None)
    scale_mask = out.get("scale_mask", None)
 
    mm_data = {"images": None, "videos": [(frames, video_metadata)]}
    t1 = time.time()
    scaled = apply_adaptive_scaling(
        multi_modal_data=[mm_data],
        scales=scales,
        new_scale_mask=scale_mask,
        processor=processor,
        patch_size=int(patch_size),
        image_factor=int(image_factor),
        temporal_patch_size=int(temporal_patch_size),
    )[0]
    t2 = time.time()
 
    videos_scaled = scaled.get("videos", None)
    if videos_scaled is not None and isinstance(videos_scaled, (list, tuple)) and len(videos_scaled) > 0:
        frames_scaled = videos_scaled[0][0]
        meta_scaled = videos_scaled[0][1]
    else:
        frames_scaled = frames
        meta_scaled = video_metadata
 
    meta_scaled = _drop_video_timestamps(meta_scaled)
 
    vllm_inputs = _build_vllm_input(processor, prompt_text, frames_scaled, meta_scaled, mm_kwargs)
    gen_s, text = _run_one_generate(llm, sampling_params, vllm_inputs)
    t3 = time.time()
 
    allocator_s = t1 - t0
    scaling_s = t2 - t1
    total_s = t3 - t0
    return allocator_s, scaling_s, total_s, text
 
 
def _make_pool(args) -> MultiGPUInferPool:
    return MultiGPUInferPool(
        model_path=args.allocator_path,
        num_gpus=int(args.pred_num_gpus),
        enable_batch=True,
        microbatch_ms=int(args.pred_microbatch_ms),
        microbatch_max=int(args.pred_microbatch_max),
        max_inflight_per_gpu=int(args.pred_max_inflight_per_gpu),
        max_queue_per_gpu=int(args.pred_max_queue_per_gpu),
        max_total_inflight=int(args.pred_max_inflight_per_gpu) * int(args.pred_num_gpus),
        submit_threads=int(args.pred_submit_threads),
        max_frames=int(args.max_frames),
        schedule_policy=os.getenv("ALLOCATOR_SCHED_POLICY", "least_inflight"),
    )
 
 
def _predict_scales_batch(pool: MultiGPUInferPool, hf_messages: list, *, count: int) -> Tuple[float, list]:
    payloads = [
        {
            "messages": [hf_messages],
            "eval_mode": True,
            "return_mm_data": False,
            "_task": "bench",
            "_doc_id": i,
            "_compound_id": f"bench::{i}",
        }
        for i in range(int(count))
    ]
    t0 = time.time()
    outs = pool.submit_many_sync(payloads)
    t1 = time.time()
    pairs = []
    for out in outs:
        if isinstance(out, Exception) or not isinstance(out, dict) or not out.get("ok", True):
            pairs.append((None, None))
        else:
            pairs.append((out.get("scales", None), out.get("scale_mask", None)))
    return (t1 - t0), pairs
 
 
def _run_one_with_precomputed_scales(
    *,
    processor: Any,
    patch_size: int,
    image_factor: int,
    temporal_patch_size: int,
    frames: Any,
    video_metadata: Optional[dict],
    prompt_text: str,
    mm_kwargs: dict,
    llm: Any,
    sampling_params: Any,
    scales: Any,
    scale_mask: Any,
) -> Tuple[float, float, str]:
    t0 = time.time()
    mm_data = {"images": None, "videos": [(frames, video_metadata)]}
    scaled = apply_adaptive_scaling(
        multi_modal_data=[mm_data],
        scales=scales,
        new_scale_mask=scale_mask,
        processor=processor,
        patch_size=int(patch_size),
        image_factor=int(image_factor),
        temporal_patch_size=int(temporal_patch_size),
    )[0]
    t1 = time.time()
 
    videos_scaled = scaled.get("videos", None)
    video_inputs = []
    video_metadatas = []
    if videos_scaled is not None and isinstance(videos_scaled, (list, tuple)) and len(videos_scaled) > 0:
        for item in videos_scaled:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                v, m = item
                video_inputs.append(v)
                video_metadatas.append(_drop_video_timestamps(m))
            else:
                video_inputs.append(item)
        if not video_metadatas:
            video_metadatas = []
    else:
        video_inputs = [frames]
        video_metadatas = [_drop_video_timestamps(video_metadata)]
 
    per_video_chunks = [video_inputs]
    prompt_scaled = (
        _expand_video_prompt_for_scaled(
            processor,
            prompt_text,
            per_video_chunks=per_video_chunks,
            temporal_patch_size=int(temporal_patch_size),
        )
        if len(video_inputs) > 1
        else prompt_text
    )
    vllm_inputs = _build_vllm_input(
        processor,
        prompt_scaled,
        video_inputs,
        video_metadatas if video_metadatas else None,
        mm_kwargs,
    )
    gen_s, text = _run_one_generate(llm, sampling_params, vllm_inputs)
    scaling_s = t1 - t0
    return scaling_s, gen_s, text
 
import vllm.model_executor.models.qwen3_vl
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from resadapt.allocator.vllm_patch_3 import get_mrope_input_positions00
vllm.model_executor.models.qwen3_vl.Qwen3VLForConditionalGeneration.get_mrope_input_positions = get_mrope_input_positions00

from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from resadapt.allocator.vllm_patch import get_mrope_input_positions
vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.get_mrope_input_positions = get_mrope_input_positions
# vllm.model_executor.models.qwen3_vl.Qwen3VLForConditionalGeneration.iter_mm_grid_hw = iter_mm_grid_hw
# vllm.model_executor.models.qwen3_vl.Qwen3VLMultiModalProcessor._get_prompt_updates = _get_prompt_updates
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--video", type=str, default=os.getenv("VIDEO_PATH", ""))
    parser.add_argument("--prompt", type=str, default=os.getenv("VIDEO_PROMPT", "Describe the video briefly."))
    parser.add_argument("--allocator-path", type=str, default=os.getenv("ALLOCATOR_PATH", ""))
    parser.add_argument("--runs", type=int, default=int(os.getenv("RUNS", "20")))
    parser.add_argument("--warmup", type=int, default=int(os.getenv("WARMUP", "2")))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", "64")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMP", "0")))
    parser.add_argument("--top-p", type=float, default=float(os.getenv("TOP_P", "0.95")))
 
    parser.add_argument("--tp", type=int, default=int(os.getenv("TP", "1")))
    parser.add_argument("--gpu-mem-util", type=float, default=float(os.getenv("GPU_MEM_UTIL", "0.8")))
 
    parser.add_argument("--max-frames", type=int, default=int(os.getenv("MAX_FRAMES", "128")))
    parser.add_argument("--max-pixels", type=int, default=int(os.getenv("MAX_PIXELS", "1605632")))
    parser.add_argument("--min-pixels", type=int, default=int(os.getenv("MIN_PIXELS", "28")))
 
    parser.add_argument("--pred-num-gpus", type=int, default=int(os.getenv("PRED_NUM_GPUS", "1")))
    parser.add_argument("--pred-microbatch-ms", type=int, default=int(os.getenv("MICRO_BATCH_MS", "10")))
    parser.add_argument("--pred-microbatch-max", type=int, default=int(os.getenv("MICRO_BATCH", "32")))
    parser.add_argument("--pred-max-inflight-per-gpu", type=int, default=int(os.getenv("max_inflight_per_gpu", "4")))
    parser.add_argument("--pred-max-queue-per-gpu", type=int, default=int(os.getenv("MAX_QUEUE_PER_GPU", "8")))
    parser.add_argument("--pred-submit-threads", type=int, default=int(os.getenv("WORKERS", "64")))
    args = parser.parse_args()
 
    if not args.video:
        raise SystemExit("missing --video (or set VIDEO_PATH)")
 
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    patch_size = int(getattr(processor.image_processor, "patch_size"))
    merge_size = int(getattr(processor.video_processor, "merge_size", 1))
    image_factor = int(merge_size * patch_size)
    temporal_patch_size = int(getattr(processor.video_processor, "temporal_patch_size", 1))
 
    chat_messages, hf_messages = _build_chat(
        args.video,
        args.prompt,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        max_frames=args.max_frames,
    )
    prompt_text = processor.apply_chat_template(hf_messages, tokenize=False, add_generation_prompt=True)
 
    frames, video_metadata, mm_kwargs = _decode_video(
        args.video,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        max_frames=args.max_frames,
    )
    video_metadata = _drop_video_timestamps(video_metadata)
 
    from vllm import LLM, SamplingParams
 
    allocator_preprocess_s = 0.0
    scale_pairs = None
    if args.allocator_path:
        pool = _make_pool(args)
        pool.start()
        allocator_preprocess_s, scale_pairs = _predict_scales_batch(
            pool,
            hf_messages,
            count=int(args.warmup) + int(args.runs),
        )
        pool.close()
        pool = None
    
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    llm = LLM(
        model=args.model,
        tensor_parallel_size=int(args.tp),
        gpu_memory_utilization=float(args.gpu_mem_util),
        trust_remote_code=True,
        max_model_len=32768,
    )
    sampling_params = SamplingParams(
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
    )
 
    baseline_gen_s = []
    pred_scaling_s = []
    pred_gen_s = []
 
    baseline_inputs = _build_vllm_input(processor, prompt_text, [frames], [video_metadata], mm_kwargs)
 
    for _ in range(max(0, int(args.warmup))):
        _run_one_generate(llm, sampling_params, baseline_inputs)
        if scale_pairs is not None:
            scales, scale_mask = scale_pairs.pop(0)
            _run_one_with_precomputed_scales(
                processor=processor,
                patch_size=patch_size,
                image_factor=image_factor,
                temporal_patch_size=temporal_patch_size,
                frames=frames,
                video_metadata=video_metadata,
                prompt_text=prompt_text,
                mm_kwargs=mm_kwargs,
                llm=llm,
                sampling_params=sampling_params,
                scales=scales,
                scale_mask=scale_mask,
            )
 
    for i in range(int(args.runs)):
        gen_s, _ = _run_one_generate(llm, sampling_params, baseline_inputs)
        baseline_gen_s.append(gen_s)
 
        if scale_pairs is not None:
            scales, scale_mask = scale_pairs[i]
            scaling_s2, gen_s2, _ = _run_one_with_precomputed_scales(
                processor=processor,
                patch_size=patch_size,
                image_factor=image_factor,
                temporal_patch_size=temporal_patch_size,
                frames=frames,
                video_metadata=video_metadata,
                prompt_text=prompt_text,
                mm_kwargs=mm_kwargs,
                llm=llm,
                sampling_params=sampling_params,
                scales=scales,
                scale_mask=scale_mask,
            )
            pred_scaling_s.append(scaling_s2)
            pred_gen_s.append(gen_s2)
 
    def _mean_std(xs):
        if not xs:
            return None, None
        if len(xs) == 1:
            return float(xs[0]), 0.0
        return float(statistics.mean(xs)), float(statistics.stdev(xs))
 
    b_mean, b_std = _mean_std(baseline_gen_s)
    print(f"[baseline] runs={len(baseline_gen_s)} gen_s_mean={b_mean:.4f} gen_s_std={b_std:.4f}")
 
    if args.allocator_path:
        s_mean, s_std = _mean_std(pred_scaling_s)
        g_mean, g_std = _mean_std(pred_gen_s)
        per_sample_pre = allocator_preprocess_s / max(1, int(args.warmup) + int(args.runs))
        print(f"[allocator] preprocess_batch_s={allocator_preprocess_s:.4f} preprocess_per_sample_s={per_sample_pre:.4f}")
        print(f"[allocator] gen_scaled_s_mean={g_mean:.4f} gen_scaled_s_std={g_std:.4f}")
        print(f"[allocator] scaling_s_mean={s_mean:.4f} scaling_s_std={s_std:.4f}")
        print(f"[allocator] total_per_sample_est={per_sample_pre + (s_mean or 0.0) + (g_mean or 0.0):.4f}")
 
 
if __name__ == "__main__":
    main()
