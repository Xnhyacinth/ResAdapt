# Copyright 2025 the ResAdapt authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Attention / dtype helpers for Smol allocator and aznet modules.

- **HF backbones:** prefer ``flash_attention_2`` when ``flash_attn`` is importable; otherwise
  ``sdpa``. Env ``ALLOCATOR_ATTN_IMPLEMENTATION`` may force ``flash_attention_2``, ``sdpa``, or ``eager``.
- **Custom SDPA (aznet / importance):** ``sdpa_scaled_dot_product_attention`` calls
  ``F.scaled_dot_product_attention`` with CUDA SDPA backends **preferring Flash** where supported.
- **flash_attn varlen:** ``flash_attn_varlen_qkv_dtype`` casts Q/K/V to fp16/bf16 for
  ``flash_attn.flash_attn_varlen_func`` (kernel requirement).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn.functional as F


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


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


def _attn_impl_flash_or_sdp() -> str:
    return "flash_attention_2" if _flash_attn_available() else "sdpa"


def resolve_pretrained_attn_implementation(
    attn: Optional[str],
    *,
    prefer_flash: bool = True,
) -> str:
    """Resolve HuggingFace ``_attn_implementation``: flash when possible, else sdpa.

    Override with env ``ALLOCATOR_ATTN_IMPLEMENTATION`` (``flash_attention_2``, ``sdpa``/``sdp``,
    or ``eager``). Saved ``attn`` is used when env is unset.
    """
    env = os.getenv("ALLOCATOR_ATTN_IMPLEMENTATION", "").strip().lower().replace("-", "_")
    if env == "eager":
        return "eager"
    if env in ("sdpa", "sdp"):
        return "sdpa"
    if env in ("flash_attention_2", "flash_attn"):
        return _attn_impl_flash_or_sdp()

    if prefer_flash and _flash_attn_available():
        return "flash_attention_2"

    if attn:
        a = attn.lower().replace("-", "_")
        if a == "eager":
            return "eager"
        if a in ("sdpa", "sdp"):
            return "sdpa"
        if a in ("flash_attention_2", "flash_attn"):
            return _attn_impl_flash_or_sdp()

    return _attn_impl_flash_or_sdp()


def forward_hf_text_model_safe(
    text_model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    smol_parent: Optional[Any] = None,
) -> torch.Tensor:
    """Run the text tower forward; ``smol_parent`` is unused (API compatibility)."""
    del smol_parent
    outputs = text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        return_dict=True,
    )
    return outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]


def flash_attn_varlen_qkv_dtype(x: torch.Tensor) -> torch.Tensor:
    """Cast Q/K/V to fp16 or bf16 for ``flash_attn.flash_attn_varlen_func`` (kernel dtype requirement)."""
    if x.dtype in (torch.float16, torch.bfloat16):
        return x
    if x.is_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return x.to(torch.bfloat16)
    return x.to(torch.float16)


def sdpa_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """``F.scaled_dot_product_attention``; on CUDA, prefer Flash / mem-efficient SDPA backends."""
    kwargs: dict[str, Any] = {
        "attn_mask": attn_mask,
        "dropout_p": dropout_p,
        "is_causal": is_causal,
    }
    if scale is not None:
        kwargs["scale"] = scale

    def _call() -> torch.Tensor:
        return F.scaled_dot_product_attention(query, key, value, **kwargs)

    if not query.is_cuda:
        return _call()

    try:
        from torch.backends.cuda import sdp_kernel
    except ImportError:
        return _call()

    try:
        with sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
            return _call()
    except (RuntimeError, ValueError, TypeError):
        return _call()
