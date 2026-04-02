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
import warnings
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F


def torch_dtype_for_hf_pretrained(config: Any) -> Any:
    """Map ``SmolAllocatorConfig.dtype`` / ``torch_dtype`` to a HF ``dtype`` value."""
    td: Any = getattr(config, "dtype", None)
    if td is None:
        td = getattr(config, "torch_dtype", None)
    if td is None or td == "auto":
        return None
    if isinstance(td, str):
        tdl = td.lower()
        if tdl in ("bfloat16", "bf16"):
            return torch.bfloat16
        if tdl in ("float16", "fp16"):
            return torch.float16
        if tdl in ("float32", "fp32"):
            return torch.float32
        return None
    return td


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


def _attn_impl_flash_or_sdp() -> str:
    return "flash_attention_2" if _flash_attn_available() else "sdpa"


def _flash_attn_compatible_with_weight_dtype(weight_dtype: Any) -> bool:
    """FlashAttention 2 kernels require activations/weights in fp16 or bf16 (not fp32)."""
    if weight_dtype is None:
        return True
    if isinstance(weight_dtype, str):
        tdl = weight_dtype.lower()
        return tdl in ("bfloat16", "bf16", "float16", "fp16")
    if isinstance(weight_dtype, torch.dtype):
        return weight_dtype in (torch.float16, torch.bfloat16)
    return False


def resolve_pretrained_attn_implementation(
    attn: Optional[str],
    *,
    prefer_flash: bool = True,
    weight_dtype: Any = None,
) -> str:
    """Resolve HuggingFace ``attn_implementation``: flash when possible, else sdpa.

    Override with env ``ALLOCATOR_ATTN_IMPLEMENTATION`` (``flash_attention_2``, ``sdpa``/``sdp``,
    or ``eager``). Saved ``attn`` is used when env is unset.

    If the backbone is loaded in fp32 (including ``dtype="auto"`` often resolving to fp32),
    ``flash_attention_2`` is **not** used — falls back to ``sdpa`` to avoid runtime errors.
    """
    env = os.getenv("ALLOCATOR_ATTN_IMPLEMENTATION", "").strip().lower().replace("-", "_")
    if env == "eager":
        return "eager"
    if env in ("sdpa", "sdp"):
        return "sdpa"
    if env in ("flash_attention_2", "flash_attn"):
        resolved = _attn_impl_flash_or_sdp()
        if resolved == "flash_attention_2" and not _flash_attn_compatible_with_weight_dtype(weight_dtype):
            return "sdpa"
        return resolved

    if prefer_flash and _flash_attn_available():
        resolved = "flash_attention_2"
        if not _flash_attn_compatible_with_weight_dtype(weight_dtype):
            return "sdpa"
        return resolved

    if attn:
        a = attn.lower().replace("-", "_")
        if a == "eager":
            return "eager"
        if a in ("sdpa", "sdp"):
            return "sdpa"
        if a in ("flash_attention_2", "flash_attn"):
            resolved = _attn_impl_flash_or_sdp()
            if resolved == "flash_attention_2" and not _flash_attn_compatible_with_weight_dtype(weight_dtype):
                return "sdpa"
            return resolved

    resolved = _attn_impl_flash_or_sdp()
    if resolved == "flash_attention_2" and not _flash_attn_compatible_with_weight_dtype(weight_dtype):
        return "sdpa"
    return resolved


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
        from torch.nn.attention import SDPBackend
        from torch.nn.attention import sdpa_kernel

        backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
        if hasattr(SDPBackend, "CUDNN_ATTENTION"):
            backends.append(SDPBackend.CUDNN_ATTENTION)
        with sdpa_kernel(backends=backends):
            return _call()
    except ImportError:
        pass
    except (RuntimeError, ValueError, TypeError):
        return _call()

    try:
        from torch.backends.cuda import sdp_kernel
    except ImportError:
        return _call()

    # Legacy API; PyTorch emits FutureWarning — prefer ``torch.nn.attention.sdpa_kernel`` (above).
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*sdp_kernel\(\) is deprecated.*",
                category=FutureWarning,
            )
            with sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
                return _call()
    except (RuntimeError, ValueError, TypeError):
        return _call()


def _regularized_beta_cdf(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Regularized incomplete Beta :math:`I_x(a,b)` (Beta CDF at ``x``)."""
    betainc = getattr(torch.special, "betainc", None)
    if betainc is not None:
        return betainc(x, a, b)
    try:
        import scipy.special as scp
    except ImportError as err:
        raise RuntimeError(
            "Beta quantile needs torch.special.betainc (PyTorch) or scipy.special.betainc; "
            "upgrade PyTorch or install scipy."
        ) from err
    x_np = x.detach().float().cpu().numpy()
    a_np = a.detach().float().cpu().numpy()
    b_np = b.detach().float().cpu().numpy()
    cdf = scp.betainc(a_np, b_np, x_np)
    return torch.as_tensor(np.asarray(cdf), device=x.device, dtype=torch.float32)


def beta_regularized_icdf(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    q: float,
    *,
    eps: float = 1e-6,
    max_iter: int = 64,
) -> torch.Tensor:
    """Solve :math:`x` such that :math:`I_x(\\alpha,\\beta)=q` (regularized incomplete Beta).

    PyTorch's ``torch.distributions.Beta.icdf`` raises ``NotImplementedError``; this uses
    bisection on the Beta CDF (``torch.special.betainc`` when available, else SciPy) and
    matches ``scipy.stats.beta.ppf`` up to numerical tolerance.
    """
    qf = float(min(max(q, eps), 1.0 - eps))
    dtype = alpha.dtype
    a = alpha.float()
    b = beta.float()
    qv = torch.full_like(a, qf)
    lo = torch.full_like(a, eps)
    hi = torch.full_like(a, 1.0 - eps)
    for _ in range(max_iter):
        mid = (lo + hi) * 0.5
        cdf = _regularized_beta_cdf(mid, a, b)
        go_low = cdf < qv
        lo = torch.where(go_low, mid, lo)
        hi = torch.where(go_low, hi, mid)
    out = (lo + hi) * 0.5
    return out.to(dtype)
