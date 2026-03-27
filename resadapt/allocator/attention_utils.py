# Copyright 2025 the ResAdapt authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared attention helpers: HuggingFace backbone dtype/attn resolution, SDPA fallbacks.

Some CUDA stacks (e.g. PyTorch 2.10+cu128) raise ``CUBLAS_STATUS_INVALID_VALUE`` in:

- ``scaled_dot_product_attention`` (SDPA), or
- ``F.linear`` / ``cublasGemmEx`` on **BF16** weights (eager attention does not fix this).

Mitigations:

- Prefer ``flash_attention_2`` when ``flash_attn`` is installed (SmolVLM / Llama text tower).
- If flash is unavailable, default to ``eager`` for HF backbones (avoids brittle SDPA).
- On persistent cuBLAS errors, the text tower forward can run with **float32 weights** briefly
  (see ``ALLOCATOR_TEXT_ENCODE_FP32``) or automatic fallback in ``forward_hf_text_model_safe``.
- For custom ``F.scaled_dot_product_attention`` call sites (aznet / importance), retry with
  math-only SDPA kernels, then re-raise.

Env:

- ``ALLOCATOR_ATTN_IMPLEMENTATION`` — unset: **prefer** ``flash_attention_2`` if ``flash_attn`` is
  installed (overrides saved config ``sdpa`` / ``eager``). Set to ``sdpa`` or ``eager`` to force.
- ``ALLOCATOR_TEXT_ENCODE_FP32`` = ``1`` — always run the HF text tower forward in FP32 weights
  (slower; avoids BF16 cuBLAS GEMM issues).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn.functional as F


def _env_truthy(name: str) -> bool:
    v = os.getenv(name, "").strip().lower()
    return v in ("1", "true", "yes", "y", "on", "t")


def _hf_set_attn_implementation(module: Any, value: str = "eager") -> bool:
    """Call HuggingFace ``set_attn_implementation`` if present; return True on success."""
    if module is None:
        return False
    fn = getattr(module, "set_attn_implementation", None)
    if not callable(fn):
        return False
    try:
        fn(value, check_supported=False)
        return True
    except TypeError:
        try:
            fn(value)
            return True
        except Exception:
            return False
    except Exception:
        return False


def _hf_force_eager_attention_stack(smol_parent: Any, text_model: Any) -> None:
    """Try eager attention on root multimodal model, inner ``.model``, and text tower."""
    candidates: list[Any] = []
    for m in (smol_parent, text_model, getattr(smol_parent, "model", None)):
        if m is not None and m not in candidates:
            candidates.append(m)
    for m in candidates:
        _hf_set_attn_implementation(m, "eager")


def torch_dtype_for_hf_pretrained(config: Any) -> Any:
    """Map ``SmolAllocatorConfig.torch_dtype`` (or legacy ``dtype``) to a HF ``torch_dtype`` value."""
    td: Any = getattr(config, "torch_dtype", None)
    if td is None:
        td = getattr(config, "dtype", None)
    if td is None or td == "auto":
        return "auto"
    if isinstance(td, str):
        tdl = td.lower()
        if tdl in ("bfloat16", "bf16"):
            return torch.bfloat16
        if tdl in ("float16", "fp16"):
            return torch.float16
        if tdl in ("float32", "fp32"):
            return torch.float32
        return "auto"
    return td


def resolve_pretrained_attn_implementation(
    attn: Optional[str],
    *,
    prefer_flash: bool = True,
) -> str:
    """Resolve HuggingFace ``_attn_implementation`` for SmolVLM / transformer backbones.

    **Priority:** use FlashAttention-2 whenever ``flash_attn`` is importable, unless the environment
    explicitly forces ``sdpa`` or ``eager`` (``ALLOCATOR_ATTN_IMPLEMENTATION``). This overrides a
    saved config that still says ``sdpa`` / ``eager``.

    If ``flash_attn`` is not installed and the resolved choice was flash, fall back to ``eager``
    when ``prefer_flash`` is True (avoids brittle SDPA/cuBLAS on some drivers), else ``sdpa``.
    """
    env = os.getenv("ALLOCATOR_ATTN_IMPLEMENTATION", "").strip()
    a_env = env.lower().replace("-", "_") if env else ""

    # Only these env values bypass the flash-first policy.
    if a_env == "eager":
        return "eager"
    if a_env in ("sdpa", "sdp"):
        return "sdpa"

    if prefer_flash:
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except ImportError:
            pass

    attn = env if env else attn
    a = (attn or "flash_attention_2").lower().replace("-", "_")
    if a in ("flash_attention_2", "flash_attn"):
        return "eager" if prefer_flash else "sdpa"
    if a in ("sdpa", "sdp"):
        return "sdpa"
    if a == "eager":
        return "eager"
    return "eager"


def _forward_text_model_fp32_weights(
    text_model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Any:
    """Run text tower with FP32 weights; avoids BF16 ``cublasGemmEx`` failures on some drivers."""
    p = next(text_model.parameters(), None)
    if p is None or p.dtype in (torch.float32, torch.float64):
        return text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
    orig_dtype = p.dtype
    text_model.to(dtype=torch.float32)
    try:
        return text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
    finally:
        text_model.to(dtype=orig_dtype)


def forward_hf_text_model_safe(
    text_model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    smol_parent: Optional[Any] = None,
) -> torch.Tensor:
    """Run Llama/text tower forward with SDPA/cuBLAS fallbacks, eager, then FP32 weights if needed."""

    def _call() -> Any:
        return text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )

    def _is_cuda_blas_runtime_error(e: BaseException) -> bool:
        err = str(e).lower()
        return "cuda" in err or "cublas" in err

    if _env_truthy("ALLOCATOR_TEXT_ENCODE_FP32"):
        outputs = _forward_text_model_fp32_weights(
            text_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hs = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        return hs

    try:
        outputs = _call()
    except RuntimeError as e:
        if not _is_cuda_blas_runtime_error(e):
            raise
        # Eager attention: fixes brittle SDPA; does not fix BF16 ``cublasGemmEx`` in ``F.linear``.
        if smol_parent is not None:
            _hf_force_eager_attention_stack(smol_parent, text_model)
        else:
            _hf_set_attn_implementation(text_model, "eager")
        try:
            outputs = _call()
        except RuntimeError as e2:
            if not _is_cuda_blas_runtime_error(e2):
                raise
            try:
                outputs = _forward_text_model_fp32_weights(
                    text_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            except RuntimeError as e3:
                if not _is_cuda_blas_runtime_error(e3):
                    raise
                from torch.backends.cuda import sdp_kernel

                with sdp_kernel(
                    enable_flash=False,
                    enable_mem_efficient=False,
                    enable_math=True,
                    enable_cudnn=False,
                ):
                    outputs = _call()
    hs = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
    return hs


def sdpa_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """``F.scaled_dot_product_attention`` with CUDA fallbacks (aznet / importance allocators)."""
    kwargs = {
        "attn_mask": attn_mask,
        "dropout_p": dropout_p,
        "is_causal": is_causal,
    }
    if scale is not None:
        kwargs["scale"] = scale
    try:
        return F.scaled_dot_product_attention(query, key, value, **kwargs)
    except RuntimeError as e:
        err = str(e).lower()
        if "cuda" not in err and "cublas" not in err:
            raise
        from torch.backends.cuda import sdp_kernel

        with sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True,
            enable_cudnn=False,
        ):
            return F.scaled_dot_product_attention(query, key, value, **kwargs)
