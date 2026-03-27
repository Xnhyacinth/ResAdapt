from __future__ import annotations

from collections.abc import Mapping, MutableMapping

import numpy as np
import torch

FRAME_METRICS_PREFIX = "frame_metrics:"


def encode_frame_metrics(frame_metrics: Mapping[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
    """
    Encodes a dictionary of frame metrics by adding a specific prefix to each key.
    
    Args:
        frame_metrics (Mapping[str, torch.Tensor] | None): A mapping of frame metric names to their tensor values.
        
    Returns:
        dict[str, torch.Tensor]: A new dictionary with prefixed keys.
    """
    if frame_metrics is None or len(frame_metrics) == 0:
        return {}
    return {
        f"{FRAME_METRICS_PREFIX}{key}": value
        for key, value in frame_metrics.items()
        if isinstance(value, torch.Tensor)
    }


def decode_frame_metrics(batch_tensors: Mapping[str, object] | None) -> dict[str, torch.Tensor]:
    """
    Decodes frame metrics from a batch dictionary by identifying keys with the correct prefix.
    
    Args:
        batch_tensors (Mapping[str, object] | None): A mapping of batch tensor names to values.
        
    Returns:
        dict[str, torch.Tensor]: A dictionary of decoded frame metrics with the prefix removed from the keys.
    """
    if batch_tensors is None or len(batch_tensors) == 0:
        return {}
    return {
        metric_name: value
        for key, value in batch_tensors.items()
        if key.startswith(FRAME_METRICS_PREFIX)
        and isinstance(value, torch.Tensor)
        and (metric_name := key.split(FRAME_METRICS_PREFIX, 1)[1])
    }


def align_frame_metrics_to_batch(
    *,
    original_sids: np.ndarray,
    updated_sids: np.ndarray,
    updated_frame_metrics: dict[str, torch.Tensor],
    fill_value: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Broadcast per-sample frame metrics from a filtered allocator batch back to the full batch order.

    Rows whose ``sid`` was not in the filtered update (allocator skipped) are filled with
    ``fill_value`` (default 0), analogous to keeping neutral values when aligning log-probs.

    Args:
        original_sids: Sequence IDs for the **full** training batch (e.g. ``batch_size`` prompts).
        updated_sids: Sequence IDs for rows that actually ran through the allocator (filtered).
        updated_frame_metrics: Metric tensors with batch dimension ``len(updated_sids)``.
        fill_value: Fill for samples with no allocator update.

    Returns:
        Metric tensors with batch dimension ``len(original_sids)``.
    """
    if not updated_frame_metrics:
        return {}
    n_u = len(updated_sids)
    sid_to_idx = {sid: idx for idx, sid in enumerate(updated_sids.tolist())}
    out: dict[str, torch.Tensor] = {}
    for key, tensor in updated_frame_metrics.items():
        if tensor.shape[0] != n_u:
            raise ValueError(
                f"frame metric {key!r} leading dim {tensor.shape[0]} != len(updated_sids)={n_u}"
            )
        device = tensor.device
        dtype = tensor.dtype
        tail = tensor.shape[1:]
        rows: list[torch.Tensor] = []
        for sid in original_sids.tolist():
            j = sid_to_idx.get(sid)
            if j is not None:
                rows.append(tensor[j].clone())
            else:
                rows.append(torch.full(tail, fill_value, dtype=dtype, device=device))
        out[key] = torch.stack(rows, dim=0)
    return out


def sync_frame_metrics(
    target_batch: MutableMapping[str, object] | None,
    source_batch_tensors: Mapping[str, object] | None,
    *,
    original_sids: np.ndarray | None = None,
    filtered_sids: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    """
    Synchronizes frame metrics from a source batch to a target batch.

    When ``use_filter_sid`` + ``scale_n > 1``, the allocator runs on a **filtered** subset; metric
    tensors are then shorter than ``target_batch``. Pass ``original_sids`` (full batch) and
    ``filtered_sids`` (allocator batch) so metrics are expanded by ``sid`` to match ``target_batch``.
    """
    frame_metrics = decode_frame_metrics(source_batch_tensors)
    if (
        frame_metrics
        and original_sids is not None
        and filtered_sids is not None
        and len(original_sids) != len(filtered_sids)
    ):
        frame_metrics = align_frame_metrics_to_batch(
            original_sids=original_sids,
            updated_sids=filtered_sids,
            updated_frame_metrics=frame_metrics,
        )
    if target_batch is not None:
        if frame_metrics:
            target_batch["frame_metrics"] = frame_metrics
        else:
            target_batch.pop("frame_metrics", None)
    return frame_metrics
