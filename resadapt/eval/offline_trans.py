import argparse
import gc
import json
import math
import os
import statistics
import time
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

from lmms_eval.protocol import ChatMessages
from resadapt.utils.utils import apply_adaptive_scaling, compute_scales_and_sample_means_cpu, expand_video_prompt


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
            "nframes": int(max_frames),
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
        "nframes": int(max_frames),
    }
    try:
        result = fetch_video(video_dict, return_video_metadata=True, return_video_sample_fps=True)
    except Exception:
        result = fetch_video(video_dict, return_video_sample_fps=True)
    frames = None
    fps = None
    frame_time = None
    video_metadata = None
    if isinstance(result, tuple):
        if len(result) == 2:
            frames_or_pair, fps = result
            if isinstance(frames_or_pair, tuple) and len(frames_or_pair) == 2:
                frames, video_metadata = frames_or_pair
            else:
                frames = frames_or_pair
        elif len(result) == 3:
            first, fps, frame_time = result
            if isinstance(first, tuple) and len(first) == 2:
                frames, video_metadata = first
            else:
                frames = first
    else:
        frames = result
    return frames, video_metadata, fps, frame_time


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
    acts = (
        [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        if bool(torch.cuda.is_available())
        else [ProfilerActivity.CPU]
    )
    with profile(
        activities=acts,
        record_shapes=True,
        profile_memory=False,
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


def _build_video_kwargs(num_videos: int, fps: Optional[float], frame_time: Optional[float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if fps is not None:
        out["fps"] = [float(fps)] * int(num_videos)
    if frame_time is not None:
        out["frame_time"] = [float(frame_time)] * int(num_videos)
    return out


def _select_model_cls(model_name: str):
    if "Qwen3" in model_name or "qwen3" in model_name:
        return Qwen3VLForConditionalGeneration
    return Qwen2_5_VLForConditionalGeneration


def _flops_thop(model: Any, inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        from thop import profile as thop_profile
    except Exception:
        return None
    keys = [k for k, v in inputs.items() if torch.is_tensor(v)]
    keys = sorted(keys)
    if not keys:
        return None
    args = tuple(inputs[k] for k in keys)

    class _Wrapped(torch.nn.Module):
        def __init__(self, m, ks):
            super().__init__()
            self.m = m
            self.ks = ks

        def forward(self, *xs):
            kwargs = {k: x for k, x in zip(self.ks, xs)}
            out = self.m(**kwargs)
            if isinstance(out, dict):
                return out.get("logits", None)
            return getattr(out, "logits", out)

    wrapped = _Wrapped(model, keys)
    try:
        flops, params = thop_profile(wrapped, inputs=args, verbose=False)
    except Exception:
        return None
    return {"flops": float(flops), "params": float(params), "keys": list(keys)}


def _flops_thop_allocator(allocator: Any, hf_messages: Any) -> Optional[Dict[str, Any]]:
    try:
        from thop import profile as thop_profile
    except Exception:
        return None

    class _Wrapped(torch.nn.Module):
        def __init__(self, m, msg):
            super().__init__()
            self.m = m
            self.msg = msg

        def forward(self, _dummy):
            return self.m(messages=[self.msg], eval_mode=True, return_mm_data=False)

    wrapped = _Wrapped(allocator, hf_messages)
    dummy = torch.zeros(1, device="cuda" if torch.cuda.is_available() else "cpu")
    try:
        flops, params = thop_profile(wrapped, inputs=(dummy,), verbose=False)
    except Exception:
        return None
    return {"flops": float(flops), "params": float(params)}


def _flops_thop_generate(model: Any, inputs: Dict[str, Any], *, max_new_tokens: int, temperature: float, top_p: float) -> Optional[Dict[str, Any]]:
    try:
        from thop import profile as thop_profile
    except Exception:
        return None
    keys = [k for k, v in inputs.items() if torch.is_tensor(v)]
    keys = sorted(keys)
    if not keys:
        return None
    args = tuple(inputs[k] for k in keys)

    class _Wrapped(torch.nn.Module):
        def __init__(self, m, ks, n_tok, temp, tp):
            super().__init__()
            self.m = m
            self.ks = ks
            self.n_tok = int(n_tok)
            self.temp = float(temp)
            self.tp = float(tp)

        def forward(self, *xs):
            kwargs = {k: x for k, x in zip(self.ks, xs)}
            return self.m.generate(
                **kwargs,
                max_new_tokens=int(self.n_tok),
                do_sample=bool(float(self.temp) > 0),
                temperature=float(self.temp) if float(self.temp) > 0 else None,
                top_p=float(self.tp) if float(self.temp) > 0 else None,
                use_cache=True,
            )

    wrapped = _Wrapped(model, keys, max_new_tokens, temperature, top_p)
    try:
        flops, params = thop_profile(wrapped, inputs=args, verbose=False)
    except Exception:
        return None
    return {"flops": float(flops), "params": float(params), "keys": list(keys)}


def _flops_fvcore(model: Any, inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception:
        return None
    keys = [k for k, v in inputs.items() if torch.is_tensor(v)]
    keys = sorted(keys)
    if not keys:
        return None
    args = tuple(inputs[k] for k in keys)

    class _Wrapped(torch.nn.Module):
        def __init__(self, m, ks):
            super().__init__()
            self.m = m
            self.ks = ks

        def forward(self, *xs):
            kwargs = {k: x for k, x in zip(self.ks, xs)}
            out = self.m(**kwargs)
            if isinstance(out, dict):
                return out.get("logits", None)
            return getattr(out, "logits", out)

    wrapped = _Wrapped(model, keys)
    try:
        flops = float(FlopCountAnalysis(wrapped, args).total())
    except Exception:
        return None
    return {"flops": float(flops), "keys": list(keys)}


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


def _estimate_decode_flops_total(
    *,
    n_layers: int,
    hidden_size: int,
    intermediate_size: int,
    prompt_tokens: int,
    output_tokens_total: int,
) -> Dict[str, Any]:
    if output_tokens_total <= 0 or prompt_tokens <= 0:
        return {"ok": False, "reason": "no_tokens"}
    d = float(hidden_size)
    ffn = float(intermediate_size)
    l0 = float(max(1, int(prompt_tokens)))
    n_out = float(int(output_tokens_total))
    l_avg = l0 + (n_out - 1.0) / 2.0
    proj_flops_per_layer_per_tok = 8.0 * d * d
    mlp_flops_per_layer_per_tok = 4.0 * d * ffn
    attn_kv_flops_per_layer_per_tok = 4.0 * l_avg * d
    flops_per_tok = float(n_layers) * (proj_flops_per_layer_per_tok + mlp_flops_per_layer_per_tok + attn_kv_flops_per_layer_per_tok)
    total_flops = float(flops_per_tok * n_out)
    return {
        "ok": True,
        "assumption": "decoder_only_forward_decode_only",
        "prompt_tokens": int(prompt_tokens),
        "output_tokens_total": int(output_tokens_total),
        "flops_total_est": float(total_flops),
        "tflops_total_est": float(total_flops / 1e12),
    }


def _estimate_generate_flops_via_run(
    *,
    tag: str,
    model: Any,
    inputs: Dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    t0 = time.time()
    gen = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(float(temperature) > 0),
        temperature=float(temperature) if float(temperature) > 0 else None,
        top_p=float(top_p) if float(temperature) > 0 else None,
        use_cache=True,
    )
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    t1 = time.time()
    in_ids = inputs.get("input_ids", None)
    try:
        prompt_tokens = int(in_ids.shape[1]) if in_ids is not None else 0
    except Exception:
        prompt_tokens = 0
    try:
        out_total = int(gen.shape[1]) if gen is not None else 0
    except Exception:
        out_total = 0
    out_tokens = max(0, out_total - prompt_tokens)
    n_layers, d, ffn = _infer_transformer_dims(getattr(model, "config", None))
    if n_layers is None or d is None or ffn is None:
        out = {
            "tag": tag,
            "mode": "estimate",
            "wall_s": float(t1 - t0),
            "ok": False,
            "reason": "model_config_dims_unavailable",
            "prompt_tokens": int(prompt_tokens),
            "output_tokens_total": int(out_tokens),
        }
        print(f"[profile] {_safe_json(out)}")
        return
    est = _estimate_decode_flops_total(
        n_layers=int(n_layers),
        hidden_size=int(d),
        intermediate_size=int(ffn),
        prompt_tokens=int(prompt_tokens),
        output_tokens_total=int(out_tokens),
    )
    out = {
        "tag": tag,
        "mode": "estimate",
        "wall_s": float(t1 - t0),
        "cpu_sum_s": None,
        "flops": float(est["flops_total_est"]) if bool(est.get("ok")) else None,
        "tflops_total_est": float(est["tflops_total_est"]) if bool(est.get("ok")) else None,
        "assumption": est.get("assumption"),
        "prompt_tokens": est.get("prompt_tokens"),
        "output_tokens_total": est.get("output_tokens_total"),
    }
    print(f"[profile] {_safe_json(out)}")


def _normalize_retention_ratio(retention: Optional[float], *, max_frames: int, uniform_config: str) -> float:
    if retention is None:
        if str(uniform_config) == "max1":
            table = {
                16: 0.28858384,
                32: 0.28185481,
                64: 0.27825625,
                128: 0.28174864,
            }
        else:
            table = {
                16: 0.76265289,
                32: 0.74356129,
                64: 0.73153809,
                128: 0.74218225,
            }
        if int(max_frames) not in table:
            raise ValueError(
                f"--uniform-scale requires --uniform-retention or max_frames in {sorted(table.keys())}, got max_frames={max_frames} (uniform_config={uniform_config})"
            )
        return float(table[int(max_frames)])
    r = float(retention)
    if r > 1.0:
        r = r / 100.0
    if not (0.0 < r <= 1.0):
        raise ValueError(f"invalid retention ratio: {retention} (expect 0<r<=1 or percentage)")
    return float(r)


def _frame_hw(frame) -> Tuple[int, int]:
    if isinstance(frame, torch.Tensor):
        if frame.ndim == 4:
            if int(frame.shape[-1]) in (1, 3, 4):
                return int(frame.shape[1]), int(frame.shape[2])
            return int(frame.shape[2]), int(frame.shape[3])
        if frame.ndim == 3:
            if int(frame.shape[-1]) in (1, 3, 4):
                return int(frame.shape[0]), int(frame.shape[1])
            return int(frame.shape[1]), int(frame.shape[2])
    if hasattr(frame, "size") and not callable(getattr(frame, "size")):
        try:
            w, h = frame.size
            return int(h), int(w)
        except Exception:
            pass
    if hasattr(frame, "shape"):
        try:
            h, w = frame.shape[0], frame.shape[1]
            return int(h), int(w)
        except Exception:
            pass
    raise TypeError(f"unsupported frame type for size: {type(frame)}")


def _uniform_scale_video(frames, *, scale: float, image_factor: int):
    if isinstance(frames, torch.Tensor):
        import torch.nn.functional as F

        x = frames
        if x.ndim == 4:
            if int(x.shape[-1]) in (1, 3, 4):
                x = x.permute(0, 3, 1, 2).contiguous()
            t, c, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
            new_h = max(int(image_factor), int(round(h * float(scale))))
            new_w = max(int(image_factor), int(round(w * float(scale))))
            new_h = max(int(image_factor), int(round(new_h / int(image_factor))) * int(image_factor))
            new_w = max(int(image_factor), int(round(new_w / int(image_factor))) * int(image_factor))
            y = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            if int(frames.shape[-1]) in (1, 3, 4):
                y = y.permute(0, 2, 3, 1).contiguous()
            return y
        raise TypeError(f"unsupported tensor video shape: {tuple(frames.shape)}")

    if isinstance(frames, (list, tuple)):
        if not frames:
            return frames
        h0, w0 = _frame_hw(frames[0])
        new_h = max(int(image_factor), int(round(h0 * float(scale))))
        new_w = max(int(image_factor), int(round(w0 * float(scale))))
        new_h = max(int(image_factor), int(round(new_h / int(image_factor))) * int(image_factor))
        new_w = max(int(image_factor), int(round(new_w / int(image_factor))) * int(image_factor))
        try:
            from PIL import Image
        except Exception:
            Image = None

        resample = None
        if Image is not None:
            try:
                resample = Image.Resampling.BICUBIC
            except Exception:
                try:
                    resample = Image.BICUBIC
                except Exception:
                    resample = None

        out = []
        for f in frames:
            if Image is not None and hasattr(f, "resize"):
                try:
                    out.append(f.resize((int(new_w), int(new_h)), resample=resample))
                    continue
                except Exception:
                    pass
            if hasattr(f, "shape"):
                if Image is None:
                    raise RuntimeError("PIL is required to resize numpy frames for --uniform-scale")
                img = Image.fromarray(f)
                img2 = img.resize((int(new_w), int(new_h)), resample=resample)
                import numpy as np

                out.append(np.asarray(img2))
                continue
            raise TypeError(f"unsupported frame type for resize: {type(f)}")
        return out

    raise TypeError(f"unsupported frames type for --uniform-scale: {type(frames)}")


def _load_allocator_on_gpu0(
    allocator_path: str,
    *,
    max_frames: Optional[int],
    max_scale_override: Optional[float],
    min_scale_override: Optional[float],
) -> Any:
    try:
        torch.cuda.set_device(0)
    except Exception:
        pass
    config = AutoConfig.from_pretrained(allocator_path, trust_remote_code=True)
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
    allocator = AutoModel.from_pretrained(
        allocator_path,
        config=config,
        dtype="auto",
        device_map={"": 0},
        trust_remote_code=True,
    )
    allocator.eval()
    return allocator


def _prepare_inputs(
    processor: Any,
    prompt_text: str,
    *,
    video_inputs: list,
    fps: Optional[float],
    frame_time: Optional[float],
    device: str,
) -> Dict[str, Any]:
    video_kwargs = _build_video_kwargs(len(video_inputs), fps, frame_time)
    inputs = processor(
        text=prompt_text,
        images=None,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    return inputs.to(device)


def _run_one_generate(
    model: Any,
    processor: Any,
    inputs: Dict[str, Any],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[float, str, int]:
    t0 = time.time()
    gen = model.generate(
        **inputs,
        max_new_tokens=int(max_tokens),
        do_sample=bool(temperature and float(temperature) > 0),
        temperature=float(temperature) if float(temperature) > 0 else None,
        top_p=float(top_p) if float(temperature) > 0 else None,
        use_cache=True,
    )
    t1 = time.time()
    input_ids = inputs.get("input_ids", None)
    if input_ids is not None:
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, gen)]
    else:
        trimmed = gen
    try:
        text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    except Exception:
        text = ""
    return (t1 - t0), text, int(sum(len(x) for x in trimmed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.getenv("HF_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--video", type=str, default=os.getenv("VIDEO_PATH", ""))
    parser.add_argument("--prompt", type=str, default=os.getenv("VIDEO_PROMPT", "Describe the video briefly."))
    parser.add_argument("--allocator-path", type=str, default=os.getenv("ALLOCATOR_PATH", ""))
    parser.add_argument("--runs", type=int, default=int(os.getenv("RUNS", "20")))
    parser.add_argument("--warmup", type=int, default=int(os.getenv("WARMUP", "2")))

    parser.add_argument("--max-frames", type=int, default=int(os.getenv("MAX_FRAMES", "128")))
    parser.add_argument("--max-pixels", type=int, default=int(os.getenv("MAX_PIXELS", "1605632")))
    parser.add_argument("--min-pixels", type=int, default=int(os.getenv("MIN_PIXELS", "28")))
    parser.add_argument("--max-scale", type=float, default=None)
    parser.add_argument("--min-scale", type=float, default=None)

    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", "64")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMP", "0")))
    parser.add_argument("--top-p", type=float, default=float(os.getenv("TOP_P", "0.95")))
    parser.add_argument("--extra-log", action="store_true", default=bool(int(os.getenv("EXTRA_LOG", "0"))))
    parser.add_argument("--profile", action="store_true", default=bool(int(os.getenv("PROFILE", "0"))))
    parser.add_argument("--profile-topk", type=int, default=int(os.getenv("PROFILE_TOPK", "20")))
    parser.add_argument(
        "--profile-mode",
        type=str,
        choices=["profiler", "estimate", "thop", "fvcore"],
        default=os.getenv("PROFILE_MODE", "estimate"),
    )
    parser.add_argument("--profile-max-tokens", type=int, default=int(os.getenv("PROFILE_MAX_TOKENS", "16")))
    parser.add_argument("--uniform-scale", action="store_true", default=bool(int(os.getenv("UNIFORM_SCALE", "0"))))
    parser.add_argument("--uniform-retention", type=float, default=None)
    parser.add_argument(
        "--uniform-config",
        type=str,
        choices=["ours", "max1"],
        default=os.getenv("UNIFORM_CONFIG", "ours"),
    )
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
    allocator_load_s = None
    scale_s_list = None

    if args.allocator_path:
        t_load0 = time.time()
        allocator = _load_allocator_on_gpu0(
            args.allocator_path,
            max_frames=args.max_frames,
            max_scale_override=args.max_scale,
            min_scale_override=args.min_scale,
        )
        t_load1 = time.time()
        allocator_load_s = t_load1 - t_load0

        max_scale = float(getattr(allocator.config, "max_scale", 2.0))
        min_scale = float(getattr(allocator.config, "min_scale", 0.25))
        use_discrete_action = bool(getattr(allocator.config, "use_discrete_action", False))
        discrete_step = float(getattr(allocator.config, "discrete_step", 0.25))

        for _ in range(max(0, int(args.warmup))):
            with torch.inference_mode():
                _ = allocator(messages=[hf_messages], eval_mode=True, return_mm_data=False)

        if bool(args.profile) and str(args.profile_mode) == "profiler":
            _profile_torch_callable(
                "allocator_forward",
                lambda: allocator(messages=[hf_messages], eval_mode=True, return_mm_data=False),
                topk=int(args.profile_topk),
            )
        if bool(args.profile) and str(args.profile_mode) == "thop":
            res = _flops_thop_allocator(allocator, hf_messages)
            if res is None:
                out = {"tag": "allocator_forward", "mode": "thop_forward", "ok": False, "reason": "thop_unavailable_or_failed"}
            else:
                out = {
                    "tag": "allocator_forward",
                    "mode": "thop_forward",
                    "flops": float(res["flops"]),
                    "tflops_total_est": float(res["flops"]) / 1e12,
                    "params": float(res.get("params")) if res.get("params") is not None else None,
                }
            print(f"[profile] {_safe_json(out)}")

        scales_list = []
        masks_list = []
        scale_s_list = []
        for _ in range(int(args.runs)):
            t0 = time.time()
            with torch.inference_mode():
                out = allocator(messages=[hf_messages], eval_mode=True, return_mm_data=False)
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
            f"[allocator] load_s={allocator_load_s:.4f} "
            f"scale_s_mean={s_mean:.4f} scale_s_std={s_std:.4f} runs={len(scale_s_list)}"
        )

        del allocator
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    model_cls = _select_model_cls(args.model)
    try:
        model = model_cls.from_pretrained(
            args.model,
            dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
    except Exception:
        model = model_cls.from_pretrained(
            args.model,
            dtype="auto",
            device_map="auto",
        )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompt_text = processor.apply_chat_template(hf_messages, tokenize=False, add_generation_prompt=True)

    t_decode0 = time.time()
    frames, video_metadata, fps, frame_time = _decode_video(
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
            "fps": fps,
            "frame_time": frame_time,
        }
        print(f"[video_info] {_safe_json(video_info)}")

    baseline_gen_s = []
    baseline_out_tokens = []
    for _ in range(max(0, int(args.warmup))):
        inputs = _prepare_inputs(
            processor,
            prompt_text,
            video_inputs=[frames],
            fps=fps,
            frame_time=frame_time,
            device=device,
        )
        _run_one_generate(model, processor, inputs, args.max_tokens, args.temperature, args.top_p)

    for _ in range(int(args.runs)):
        inputs = _prepare_inputs(
            processor,
            prompt_text,
            video_inputs=[frames],
            fps=fps,
            frame_time=frame_time,
            device=device,
        )
        gen_s, _, out_tokens = _run_one_generate(model, processor, inputs, args.max_tokens, args.temperature, args.top_p)
        baseline_gen_s.append(gen_s)
        baseline_out_tokens.append(out_tokens)

    b_mean, b_std = _mean_std(baseline_gen_s)
    print(
        f"[baseline] decode_s={(t_decode1 - t_decode0):.4f} "
        f"gen_s_mean={b_mean:.4f} gen_s_std={b_std:.4f} runs={len(baseline_gen_s)}"
    )
    if bool(args.extra_log) and baseline_out_tokens:
        total_tokens = int(sum(baseline_out_tokens))
        total_time = float(sum(baseline_gen_s))
        tokens_per_s = float(total_tokens / total_time) if total_time > 0 else None
        print(f"[efficiency] tag=baseline total_tokens={total_tokens} tokens_per_s={tokens_per_s}")

    patch_size = int(getattr(processor.image_processor, "patch_size"))
    merge_size = int(getattr(processor.video_processor, "merge_size", 1))
    image_factor = int(merge_size * patch_size)
    temporal_patch_size = int(getattr(processor.video_processor, "temporal_patch_size", 1))

    uniform_retention = None
    uniform_scale = None
    if bool(args.uniform_scale):
        uniform_retention = _normalize_retention_ratio(
            args.uniform_retention, max_frames=int(args.max_frames), uniform_config=str(args.uniform_config)
        )
        uniform_scale = float(math.sqrt(float(uniform_retention)))
        if bool(args.extra_log):
            print(
                f"[uniform_scale] retention_ratio={uniform_retention} scale={uniform_scale} "
                f"max_frames={int(args.max_frames)}"
            )

    if bool(args.profile):
        if bool(args.uniform_scale):
            frames_profile = _uniform_scale_video(frames, scale=float(uniform_scale), image_factor=int(image_factor))
            prompt_profile = prompt_text
            video_inputs_profile = [frames_profile]
        elif scales_list is not None and masks_list is not None:
            scales = scales_list[0]
            scale_mask = masks_list[0]
            scaled = apply_adaptive_scaling(
                multi_modal_data=[{"images": None, "videos": [(frames, video_metadata)]}],
                scales=scales,
                new_scale_mask=scale_mask,
                processor=processor,
                patch_size=int(patch_size),
                image_factor=int(image_factor),
                temporal_patch_size=int(temporal_patch_size),
            )[0]
            videos_scaled = scaled.get("videos", None)
            video_inputs_profile = []
            if videos_scaled is not None and isinstance(videos_scaled, (list, tuple)) and len(videos_scaled) > 0:
                for item in videos_scaled:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        v, m = item
                        video_inputs_profile.append(v)
                    else:
                        video_inputs_profile.append(item)
            else:
                video_inputs_profile = [frames]
            per_video_chunks = [video_inputs_profile]
            if len(video_inputs_profile) > 1:
                if "Qwen3VL" in type(processor).__name__:
                    prompt_profile = _expand_video_prompt_blocks(prompt_text, per_video_chunks)
                else:
                    prompt_profile = expand_video_prompt(prompt_text, per_video_chunks, int(temporal_patch_size))
            else:
                prompt_profile = prompt_text
        else:
            prompt_profile = prompt_text
            video_inputs_profile = [frames]
        inputs = _prepare_inputs(
            processor,
            prompt_profile,
            video_inputs=video_inputs_profile,
            fps=fps,
            frame_time=frame_time,
            device=device,
        )
        max_new_tokens_profile = int(min(int(args.max_tokens), int(args.profile_max_tokens)))
        if str(args.profile_mode) == "profiler":
            _profile_torch_callable(
                "transformers_generate",
                lambda: model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens_profile),
                    do_sample=bool(args.temperature and float(args.temperature) > 0),
                    temperature=float(args.temperature) if float(args.temperature) > 0 else None,
                    top_p=float(args.top_p) if float(args.temperature) > 0 else None,
                    use_cache=True,
                ),
                topk=int(args.profile_topk),
            )
        elif str(args.profile_mode) == "thop":
            res = _flops_thop_generate(
                model,
                inputs,
                max_new_tokens=int(max_new_tokens_profile),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
            )
            if res is None:
                _estimate_generate_flops_via_run(
                    tag="transformers_generate",
                    model=model,
                    inputs=inputs,
                    max_new_tokens=int(max_new_tokens_profile),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                )
            else:
                out = {
                    "tag": "transformers_generate",
                    "mode": "thop_generate",
                    "flops": float(res["flops"]),
                    "tflops_total_est": float(res["flops"]) / 1e12,
                    "params": float(res.get("params")) if res.get("params") is not None else None,
                    "input_keys": list(res.get("keys") or []),
                    "max_new_tokens": int(max_new_tokens_profile),
                }
                print(f"[profile] {_safe_json(out)}")
        elif str(args.profile_mode) == "fvcore":
            res = _flops_fvcore(model, inputs)
            if res is None:
                _estimate_generate_flops_via_run(
                    tag="transformers_generate",
                    model=model,
                    inputs=inputs,
                    max_new_tokens=int(max_new_tokens_profile),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                )
            else:
                out = {
                    "tag": "transformers_generate",
                    "mode": "fvcore_forward",
                    "flops": float(res["flops"]),
                    "tflops_total_est": float(res["flops"]) / 1e12,
                    "input_keys": list(res.get("keys") or []),
                }
                print(f"[profile] {_safe_json(out)}")
        else:
            _estimate_generate_flops_via_run(
                tag="transformers_generate",
                model=model,
                inputs=inputs,
                max_new_tokens=int(max_new_tokens_profile),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
            )

    if scales_list is None or masks_list is None:
        return

    scaling_s_list = []
    prompt_expand_s_list = []
    scaled_gen_s_list = []
    scaled_out_tokens = []
    for i in range(int(args.runs)):
        scales = scales_list[i]
        scale_mask = masks_list[i]

        if bool(args.uniform_scale):
            t0 = time.time()
            frames_scaled = _uniform_scale_video(frames, scale=float(uniform_scale), image_factor=int(image_factor))
            t1 = time.time()
            scaling_s_list.append(t1 - t0)
            prompt_scaled = prompt_text
            prompt_expand_s_list.append(0.0)
            video_inputs = [frames_scaled]
        else:
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
            if videos_scaled is not None and isinstance(videos_scaled, (list, tuple)) and len(videos_scaled) > 0:
                for item in videos_scaled:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        v, m = item
                        video_inputs.append(v)
                    else:
                        video_inputs.append(item)
            else:
                video_inputs = [frames]

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

        inputs = _prepare_inputs(
            processor,
            prompt_scaled,
            video_inputs=video_inputs,
            fps=fps,
            frame_time=frame_time,
            device=device,
        )
        gen_s, _, out_tokens = _run_one_generate(model, processor, inputs, args.max_tokens, args.temperature, args.top_p)
        scaled_gen_s_list.append(gen_s)
        scaled_out_tokens.append(out_tokens)

    sc_mean, sc_std = _mean_std(scaling_s_list)
    pe_mean, pe_std = _mean_std(prompt_expand_s_list)
    g_mean, g_std = _mean_std(scaled_gen_s_list)
    print(f"[scaled] scaling_s_mean={sc_mean:.4f} scaling_s_std={sc_std:.4f}")
    print(f"[scaled] prompt_expand_s_mean={pe_mean:.4f} prompt_expand_s_std={pe_std:.4f}")
    print(f"[scaled] gen_s_mean={g_mean:.4f} gen_s_std={g_std:.4f} runs={len(scaled_gen_s_list)}")
    if scale_s_list:
        s_mean, _ = _mean_std(scale_s_list)
        print(f"[scaled] total_per_sample_est={(s_mean or 0.0) + (sc_mean or 0.0) + (pe_mean or 0.0) + (g_mean or 0.0):.4f}")

    if bool(args.extra_log) and scaled_out_tokens:
        total_tokens = int(sum(scaled_out_tokens))
        total_time = float(sum(scaled_gen_s_list))
        tokens_per_s = float(total_tokens / total_time) if total_time > 0 else None
        print(f"[efficiency] tag=scaled total_tokens={total_tokens} tokens_per_s={tokens_per_s}")


if __name__ == "__main__":
    main()
