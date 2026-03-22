import argparse
import gc
import json
import os
import statistics
import time
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

from lmms_eval.protocol import ChatMessages
from visionthink.adaptive.utils import apply_adaptive_scaling, compute_scales_and_sample_means_cpu, expand_video_prompt


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


def _mean_std(xs):
    if not xs:
        return None, None
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(statistics.mean(xs)), float(statistics.stdev(xs))


def _safe_json(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        try:
            return str(x)
        except Exception:
            return "<unprintable>"


def _get_profiler_totals(prof) -> Tuple[float, Optional[float], Optional[float]]:
    total_wall_s = 0.0
    total_flops = 0.0
    saw_flops = False
    try:
        events = prof.key_averages()
        for e in events:
            try:
                total_wall_s += float(getattr(e, "cpu_time_total", 0.0)) / 1e6
            except Exception:
                pass
            f = getattr(e, "flops", None)
            if f is not None:
                try:
                    total_flops += float(f)
                    saw_flops = True
                except Exception:
                    pass
    except Exception:
        pass
    flops_val = total_flops if saw_flops else None
    tflops_val = (flops_val / 1e12 / total_wall_s) if (flops_val is not None and total_wall_s > 0) else None
    return total_wall_s, flops_val, tflops_val


def _profile_torch_callable(tag: str, fn, *, topk: int = 20):
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception:
        print(f"[profile] tag={tag} torch_profiler_unavailable=1")
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    t0 = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        fn()
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    t1 = time.time()
    total_wall_s, total_flops, total_tflops = _get_profiler_totals(prof)
    out = {
        "tag": tag,
        "wall_s": float(t1 - t0),
        "cpu_sum_s": float(total_wall_s),
        "flops": total_flops,
        "tflops_est": total_tflops,
    }
    print(f"[profile] {_safe_json(out)}")
    try:
        table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=int(topk))
        print(f"[profile_table] tag={tag}\n{table}")
    except Exception:
        pass


def _maybe_print_vllm_metrics(llm: Any, *, tag: str):
    get_metrics = getattr(llm, "get_metrics", None)
    if get_metrics is None:
        print(f"[vllm_metrics] tag={tag} unavailable=1")
        return None
    try:
        metrics = get_metrics()
    except Exception as e:
        print(f"[vllm_metrics] tag={tag} error={type(e).__name__} msg={_safe_json(str(e))}")
        return None
    out = []
    try:
        for m in metrics:
            name = getattr(m, "name", None)
            item = {
                "name": name,
                "value": getattr(m, "value", None),
                "count": getattr(m, "count", None),
                "sum": getattr(m, "sum", None),
                "max": getattr(m, "max", None),
                "min": getattr(m, "min", None),
            }
            out.append(item)
    except Exception:
        out = metrics
    print(f"[vllm_metrics] tag={tag} metrics={_safe_json(out)}")
    return out


def _get_vllm_format_metrics(llm: Any) -> Optional[Dict[str, Any]]:
    get_metrics = getattr(llm, "get_metrics", None)
    if get_metrics is None:
        return None
    try:
        metrics = get_metrics()
    except Exception:
        return None

    ttft = 0.0
    tpot = 0.0
    generation_tokens = None
    prompt_tokens_total = None
    kv_cache_usage_perc = None

    for m in metrics:
        name = getattr(m, "name", "")
        if "time_to_first_token" in name:
            try:
                ttft = float(m.sum) / float(m.count) if float(m.count) > 0 else 0.0
            except Exception:
                pass
        if ("time_per_output_token_seconds" in name) or ("inter_token_latency_seconds" in name):
            try:
                tpot = float(m.sum) / float(m.count) if float(m.count) > 0 else 0.0
            except Exception:
                pass
        if name in ("vllm:generation_tokens", "vllm:generation_tokens_total"):
            try:
                generation_tokens = float(getattr(m, "value"))
            except Exception:
                pass
        if name in ("vllm:prompt_tokens", "vllm:prompt_tokens_total"):
            try:
                prompt_tokens_total = float(getattr(m, "value"))
            except Exception:
                pass
        if name in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"):
            try:
                kv_cache_usage_perc = float(getattr(m, "value"))
            except Exception:
                pass

    out = {"ttft": float(ttft), "tpot": float(tpot)}
    if generation_tokens is not None:
        out["generation_tokens"] = float(generation_tokens)
    if prompt_tokens_total is not None:
        out["prompt_tokens_total"] = float(prompt_tokens_total)
    if kv_cache_usage_perc is not None:
        out["kv_cache_usage_perc"] = float(kv_cache_usage_perc)
    return out


def _infer_transformer_dims(cfg: Any) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None) or getattr(cfg, "num_layers", None)
    d = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or getattr(cfg, "dim", None)
    ffn = getattr(cfg, "intermediate_size", None) or getattr(cfg, "ffn_hidden_size", None) or getattr(cfg, "n_inner", None)
    try:
        n_layers = int(n_layers) if n_layers is not None else None
    except Exception:
        n_layers = None
    try:
        d = int(d) if d is not None else None
    except Exception:
        d = None
    try:
        ffn = int(ffn) if ffn is not None else None
    except Exception:
        ffn = None
    return n_layers, d, ffn


def _estimate_prompt_tokens_from_text(processor: Any, prompt_text: str) -> Optional[int]:
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        return None
    try:
        enc = tok(prompt_text, add_special_tokens=False)
        ids = enc.get("input_ids", None) if isinstance(enc, dict) else getattr(enc, "input_ids", None)
        return int(len(ids)) if ids is not None else None
    except Exception:
        return None


def _estimate_decode_tflops(
    *,
    n_layers: int,
    hidden_size: int,
    intermediate_size: int,
    prompt_tokens: int,
    output_tokens_total: int,
    total_decode_time_s: float,
    tp: int,
) -> Dict[str, Any]:
    if total_decode_time_s <= 0 or output_tokens_total <= 0:
        return {"ok": False, "reason": "no_decode_time_or_tokens"}
    d = float(hidden_size)
    ffn = float(intermediate_size)
    l0 = float(max(1, int(prompt_tokens)))
    n_out = float(int(output_tokens_total))
    l_avg = l0 + (n_out - 1.0) / 2.0
    proj_flops_per_layer_per_tok = 8.0 * d * d
    mlp_flops_per_layer_per_tok = 4.0 * d * ffn
    attn_kv_flops_per_layer_per_tok = 4.0 * l_avg * d
    flops_per_tok = float(n_layers) * (proj_flops_per_layer_per_tok + mlp_flops_per_layer_per_tok + attn_kv_flops_per_layer_per_tok)
    total_flops = flops_per_tok * n_out
    tflops_total = total_flops / total_decode_time_s / 1e12
    tp_val = max(1, int(tp))
    tflops_per_gpu = tflops_total / tp_val
    return {
        "ok": True,
        "assumption": "decoder_only_forward_decode_only",
        "prompt_tokens": int(prompt_tokens),
        "output_tokens_total": int(output_tokens_total),
        "tp": tp_val,
        "total_decode_time_s": float(total_decode_time_s),
        "tflops_total_est": float(tflops_total),
        "tflops_per_gpu_est": float(tflops_per_gpu),
        "flops_total_est": float(total_flops),
    }


def _build_vllm_inputs_for_video(
    processor: Any,
    prompt_text: str,
    *,
    video_inputs: list,
    video_metadatas: Optional[list],
    mm_kwargs: dict,
) -> Dict[str, Any]:
    if "Qwen3VL" in type(processor).__name__:
        if video_metadatas is not None and len(video_metadatas) == len(video_inputs):
            mm_video = list(zip(video_inputs, video_metadatas))
        else:
            mm_video = list(video_inputs)
    else:
        mm_video = list(video_inputs)
    return {"prompt": prompt_text, "multi_modal_data": {"video": mm_video}, "mm_processor_kwargs": dict(mm_kwargs)}


def _apply_vllm_patches():
    import vllm.model_executor.models.qwen2_5_vl
    import vllm.model_executor.models.qwen3_vl

    from visionthink.predictor.vllm_patch import get_mrope_input_positions
    from visionthink.predictor.vllm_patch_3 import get_mrope_input_positions00

    vllm.model_executor.models.qwen3_vl.Qwen3VLForConditionalGeneration.get_mrope_input_positions = get_mrope_input_positions00
    vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.get_mrope_input_positions = get_mrope_input_positions


def _load_predictor_on_gpu0(
    predictor_path: str,
    *,
    max_frames: Optional[int],
    max_scale_override: Optional[float],
    min_scale_override: Optional[float],
) -> Any:
    try:
        torch.cuda.set_device(0)
    except Exception:
        pass
    config = AutoConfig.from_pretrained(predictor_path, trust_remote_code=True)
    if max_frames is not None:
        try:
            max_frames_val = int(max_frames)
            if max_frames_val > 0:
                setattr(config, "max_frames", max_frames_val)
        except Exception:
            pass
    if max_scale_override is not None:
        try:
            setattr(config, "max_scale", float(max_scale_override))
        except Exception:
            pass
    if min_scale_override is not None:
        try:
            setattr(config, "min_scale", float(min_scale_override))
        except Exception:
            pass
    predictor = AutoModel.from_pretrained(
        predictor_path,
        config=config,
        dtype="auto",
        device_map={"": 0},
        trust_remote_code=True,
    )
    predictor.eval()
    return predictor


def _run_one_generate(llm: Any, sampling_params: Any, vllm_inputs: Dict[str, Any]) -> Tuple[float, str, Optional[int], Optional[int]]:
    t0 = time.time()
    outs = llm.generate([vllm_inputs], sampling_params)
    t1 = time.time()
    req = outs[0]
    out0 = req.outputs[0]
    text = out0.text
    token_ids = getattr(out0, "token_ids", None)
    n_out = int(len(token_ids)) if token_ids is not None else None
    prompt_token_ids = getattr(req, "prompt_token_ids", None)
    n_prompt = int(len(prompt_token_ids)) if prompt_token_ids is not None else None
    return (t1 - t0), text, n_prompt, n_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--video", type=str, default=os.getenv("VIDEO_PATH", ""))
    parser.add_argument("--prompt", type=str, default=os.getenv("VIDEO_PROMPT", "Describe the video briefly."))
    parser.add_argument("--predictor-path", type=str, default=os.getenv("PREDICTOR_PATH", ""))
    parser.add_argument("--runs", type=int, default=int(os.getenv("RUNS", "20")))
    parser.add_argument("--warmup", type=int, default=int(os.getenv("WARMUP", "2")))

    parser.add_argument("--max-frames", type=int, default=int(os.getenv("MAX_FRAMES", "128")))
    parser.add_argument("--max-pixels", type=int, default=int(os.getenv("MAX_PIXELS", "1605632")))
    parser.add_argument("--min-pixels", type=int, default=int(os.getenv("MIN_PIXELS", "28")))
    parser.add_argument("--max-scale", type=float, default=None)
    parser.add_argument("--min-scale", type=float, default=None)

    parser.add_argument("--tp", type=int, default=int(os.getenv("TP", "4")))
    parser.add_argument("--gpu-mem-util", type=float, default=float(os.getenv("GPU_MEM_UTIL", "0.8")))
    parser.add_argument("--max-model-len", type=int, default=int(os.getenv("MAX_MODEL_LEN", "32768")))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", "64")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMP", "0")))
    parser.add_argument("--top-p", type=float, default=float(os.getenv("TOP_P", "0.95")))
    parser.add_argument("--extra-log", action="store_true", default=bool(int(os.getenv("EXTRA_LOG", "0"))))
    parser.add_argument("--profile", action="store_true", default=bool(int(os.getenv("PROFILE", "0"))))
    parser.add_argument("--profile-topk", type=int, default=int(os.getenv("PROFILE_TOPK", "20")))
    args = parser.parse_args()

    if not args.video:
        raise SystemExit("missing --video (or set VIDEO_PATH)")
    if int(args.runs) <= 0:
        raise SystemExit("--runs must be >= 1")

    _, hf_messages = _build_chat(
        args.video,
        args.prompt,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        max_frames=args.max_frames,
    )

    scales_list = None
    masks_list = None
    predictor_load_s = None
    scale_s_list = None

    if args.predictor_path:
        t_load0 = time.time()
        predictor = _load_predictor_on_gpu0(
            args.predictor_path,
            max_frames=args.max_frames,
            max_scale_override=args.max_scale,
            min_scale_override=args.min_scale,
        )
        t_load1 = time.time()
        predictor_load_s = t_load1 - t_load0

        max_scale = float(getattr(predictor.config, "max_scale", 2.0))
        min_scale = float(getattr(predictor.config, "min_scale", 0.25))
        use_discrete_action = bool(getattr(predictor.config, "use_discrete_action", False))
        discrete_step = float(getattr(predictor.config, "discrete_step", 0.25))

        for _ in range(max(0, int(args.warmup))):
            with torch.inference_mode():
                _ = predictor(messages=[hf_messages], eval_mode=True, return_mm_data=False)

        if bool(args.profile):
            _profile_torch_callable(
                "predictor_forward",
                lambda: predictor(messages=[hf_messages], eval_mode=True, return_mm_data=False),
                topk=int(args.profile_topk),
            )

        scales_list = []
        masks_list = []
        scale_s_list = []
        for _ in range(int(args.runs)):
            t0 = time.time()
            with torch.inference_mode():
                out = predictor(messages=[hf_messages], eval_mode=True, return_mm_data=False)
            scales_cpu, mask_cpu, _ = compute_scales_and_sample_means_cpu(
                out,
                max_scale=max_scale,
                min_scale=min_scale,
                use_discrete_action=use_discrete_action,
                default_step=discrete_step,
            )
            t1 = time.time()
            scale_s_list.append(t1 - t0)
            scales_list.append(scales_cpu[0])
            masks_list.append(None if mask_cpu is None else mask_cpu[0])

        s_mean, s_std = _mean_std(scale_s_list)
        print(
            f"[predictor] load_s={predictor_load_s:.4f} "
            f"scale_s_mean={s_mean:.4f} scale_s_std={s_std:.4f} runs={len(scale_s_list)}"
        )

        def _mean_scale(scales, mask):
            try:
                s = torch.as_tensor(scales).float()
            except Exception:
                return 0.0
            if mask is not None:
                try:
                    m = torch.as_tensor(mask).bool()
                    if m.ndim == 1 and s.ndim >= 1 and s.shape[0] == m.shape[0]:
                        s = s[m]
                except Exception:
                    pass
            return float(s.mean().item()) if s.numel() > 0 else 0.0

        mean_scales = [_mean_scale(scales_list[i], masks_list[i]) for i in range(len(scales_list))]
        ms_mean, ms_std = _mean_std(mean_scales)
        print(f"[predictor_scale] mean_scale_mean={ms_mean:.4f} mean_scale_std={ms_std:.4f} runs={len(mean_scales)}")

        del predictor
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    _apply_vllm_patches()

    from vllm import LLM, SamplingParams

    t_vllm_load0 = time.time()
    try:
        llm = LLM(
            model=args.model,
            tensor_parallel_size=int(args.tp),
            gpu_memory_utilization=float(args.gpu_mem_util),
            trust_remote_code=True,
            max_model_len=int(args.max_model_len),
            disable_log_stats=False,
        )
    except TypeError:
        llm = LLM(
            model=args.model,
            tensor_parallel_size=int(args.tp),
            gpu_memory_utilization=float(args.gpu_mem_util),
            trust_remote_code=True,
            max_model_len=int(args.max_model_len),
        )
    sampling_params = SamplingParams(
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
    )
    t_vllm_load1 = time.time()

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    patch_size = int(getattr(processor.image_processor, "patch_size"))
    merge_size = int(getattr(processor.video_processor, "merge_size", 1))
    image_factor = int(merge_size * patch_size)
    temporal_patch_size = int(getattr(processor.video_processor, "temporal_patch_size", 1))

    prompt_text = processor.apply_chat_template(hf_messages, tokenize=False, add_generation_prompt=True)

    t_decode0 = time.time()
    frames, video_metadata, mm_kwargs = _decode_video(
        args.video,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        max_frames=args.max_frames,
    )
    video_metadata = _drop_video_timestamps(video_metadata)
    t_decode1 = time.time()

    if bool(args.extra_log):
        try:
            n_frames = int(len(frames))
        except Exception:
            n_frames = None
        video_info = {
            "video": args.video,
            "max_frames": int(args.max_frames),
            "decoded_frames": n_frames,
            "max_pixels": int(args.max_pixels),
            "min_pixels": int(args.min_pixels),
            "mm_kwargs": mm_kwargs,
            "video_metadata": video_metadata,
        }
        try:
            if torch.cuda.is_available():
                video_info["cuda_device_0"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
        print(f"[video_info] {_safe_json(video_info)}")

    baseline_gen_s = []
    baseline_out_tokens = []
    m0 = _get_vllm_format_metrics(llm) if bool(args.extra_log) else None
    for _ in range(max(0, int(args.warmup))):
        vllm_inputs = _build_vllm_inputs_for_video(
            processor,
            prompt_text,
            video_inputs=[frames],
            video_metadatas=[video_metadata],
            mm_kwargs=mm_kwargs,
        )
        _run_one_generate(llm, sampling_params, vllm_inputs)

    if bool(args.profile):
        vllm_inputs = _build_vllm_inputs_for_video(
            processor,
            prompt_text,
            video_inputs=[frames],
            video_metadatas=[video_metadata],
            mm_kwargs=mm_kwargs,
        )
        _profile_torch_callable(
            "vllm_generate",
            lambda: llm.generate([vllm_inputs], sampling_params),
            topk=int(args.profile_topk),
        )

    for _ in range(int(args.runs)):
        vllm_inputs = _build_vllm_inputs_for_video(
            processor,
            prompt_text,
            video_inputs=[frames],
            video_metadatas=[video_metadata],
            mm_kwargs=mm_kwargs,
        )
        gen_s, _, _, n_out = _run_one_generate(llm, sampling_params, vllm_inputs)
        baseline_gen_s.append(gen_s)
        baseline_out_tokens.append(n_out)

    b_mean, b_std = _mean_std(baseline_gen_s)
    print(
        f"[baseline] vllm_load_s={(t_vllm_load1 - t_vllm_load0):.4f} "
        f"decode_s={(t_decode1 - t_decode0):.4f} "
        f"gen_s_mean={b_mean:.4f} gen_s_std={b_std:.4f} runs={len(baseline_gen_s)}"
    )

    if bool(args.extra_log):
        _maybe_print_vllm_metrics(llm, tag="baseline_after_runs")
        m1 = _get_vllm_format_metrics(llm)
        if m1 is not None:
            print(f"[vllm_metric_summary] tag=baseline {_safe_json(m1)}")
        valid_out_tokens = [x for x in baseline_out_tokens if isinstance(x, int)]
        total_time = float(sum(baseline_gen_s))
        total_tokens = int(sum(valid_out_tokens)) if valid_out_tokens and len(valid_out_tokens) == len(baseline_gen_s) else None
        if total_tokens is None and (m0 is not None) and (m1 is not None):
            if ("generation_tokens" in m0) and ("generation_tokens" in m1):
                try:
                    total_tokens = int(float(m1["generation_tokens"]) - float(m0["generation_tokens"]))
                except Exception:
                    total_tokens = None
        if total_tokens is not None:
            avg_speed = float(total_tokens / total_time) if total_time > 0 else None
            print(f"[efficiency] tag=baseline total_tokens={total_tokens} tokens_per_s={avg_speed}")
            try:
                cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
                n_layers, d, ffn = _infer_transformer_dims(cfg)
            except Exception:
                n_layers, d, ffn = None, None, None
            if n_layers and d and ffn:
                prompt_tok = _estimate_prompt_tokens_from_text(processor, prompt_text)
                tflops = _estimate_decode_tflops(
                    n_layers=int(n_layers),
                    hidden_size=int(d),
                    intermediate_size=int(ffn),
                    prompt_tokens=int(prompt_tok or 0),
                    output_tokens_total=int(total_tokens),
                    total_decode_time_s=float(total_time),
                    tp=int(args.tp),
                )
                print(f"[tflops_est] tag=baseline {_safe_json(tflops)}")
            else:
                print(f"[tflops_est] tag=baseline ok=0 reason=model_config_dims_unavailable")

    if scales_list is None or masks_list is None:
        return

    scaling_s_list = []
    prompt_expand_s_list = []
    scaled_gen_s_list = []
    scaled_out_tokens = []
    m2 = _get_vllm_format_metrics(llm) if bool(args.extra_log) else None
    for i in range(int(args.runs)):
        scales = scales_list[i]
        scale_mask = masks_list[i]

        t0 = time.time()
        scaled = apply_adaptive_scaling(
            multi_modal_data=[{"images": None, "videos": [(frames, video_metadata)]}],
            scales=scales,
            new_scale_mask=scale_mask,
            processor=processor,
            patch_size=int(patch_size),
            image_factor=int(image_factor),
            temporal_patch_size=int(temporal_patch_size),
        )[0]
        t1 = time.time()
        scaling_s_list.append(t1 - t0)

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
        else:
            video_inputs = [frames]
            video_metadatas = [_drop_video_timestamps(video_metadata)]

        per_video_chunks = [video_inputs]
        t2 = time.time()
        if len(video_inputs) > 1:
            if "Qwen3VL" in type(processor).__name__:
                prompt_scaled = _expand_video_prompt_blocks(prompt_text, per_video_chunks)
            else:
                prompt_scaled = expand_video_prompt(prompt_text, per_video_chunks, int(temporal_patch_size))
        else:
            prompt_scaled = prompt_text
        t3 = time.time()
        prompt_expand_s_list.append(t3 - t2)

        vllm_inputs = _build_vllm_inputs_for_video(
            processor,
            prompt_scaled,
            video_inputs=video_inputs,
            video_metadatas=video_metadatas if video_metadatas else None,
            mm_kwargs=mm_kwargs,
        )
        gen_s, _, _, n_out = _run_one_generate(llm, sampling_params, vllm_inputs)
        scaled_gen_s_list.append(gen_s)
        scaled_out_tokens.append(n_out)

    sc_mean, sc_std = _mean_std(scaling_s_list)
    pe_mean, pe_std = _mean_std(prompt_expand_s_list)
    g_mean, g_std = _mean_std(scaled_gen_s_list)
    print(f"[scaled] scaling_s_mean={sc_mean:.4f} scaling_s_std={sc_std:.4f}")
    print(f"[scaled] prompt_expand_s_mean={pe_mean:.4f} prompt_expand_s_std={pe_std:.4f}")
    print(f"[scaled] gen_s_mean={g_mean:.4f} gen_s_std={g_std:.4f} runs={len(scaled_gen_s_list)}")
    if scale_s_list:
        s_mean, _ = _mean_std(scale_s_list)
        print(f"[scaled] total_per_sample_est={(s_mean or 0.0) + (sc_mean or 0.0) + (pe_mean or 0.0) + (g_mean or 0.0):.4f}")

    if bool(args.extra_log):
        _maybe_print_vllm_metrics(llm, tag="scaled_after_runs")
        m3 = _get_vllm_format_metrics(llm)
        if m3 is not None:
            print(f"[vllm_metric_summary] tag=scaled {_safe_json(m3)}")
        valid_out_tokens = [x for x in scaled_out_tokens if isinstance(x, int)]
        total_time = float(sum(scaled_gen_s_list))
        total_tokens = int(sum(valid_out_tokens)) if valid_out_tokens and len(valid_out_tokens) == len(scaled_gen_s_list) else None
        if total_tokens is None and (m2 is not None) and (m3 is not None):
            if ("generation_tokens" in m2) and ("generation_tokens" in m3):
                try:
                    total_tokens = int(float(m3["generation_tokens"]) - float(m2["generation_tokens"]))
                except Exception:
                    total_tokens = None
        if total_tokens is not None:
            avg_speed = float(total_tokens / total_time) if total_time > 0 else None
            print(f"[efficiency] tag=scaled total_tokens={total_tokens} tokens_per_s={avg_speed}")
            try:
                cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
                n_layers, d, ffn = _infer_transformer_dims(cfg)
            except Exception:
                n_layers, d, ffn = None, None, None
            if n_layers and d and ffn:
                prompt_tok = _estimate_prompt_tokens_from_text(processor, prompt_text)
                tflops = _estimate_decode_tflops(
                    n_layers=int(n_layers),
                    hidden_size=int(d),
                    intermediate_size=int(ffn),
                    prompt_tokens=int(prompt_tok or 0),
                    output_tokens_total=int(total_tokens),
                    total_decode_time_s=float(total_time),
                    tp=int(args.tp),
                )
                print(f"[tflops_est] tag=scaled {_safe_json(tflops)}")
            else:
                print(f"[tflops_est] tag=scaled ok=0 reason=model_config_dims_unavailable")


if __name__ == "__main__":
    main()
