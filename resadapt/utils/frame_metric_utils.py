from __future__ import annotations

from collections.abc import Mapping, MutableMapping

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


def sync_frame_metrics(
    target_batch: MutableMapping[str, object] | None,
    source_batch_tensors: Mapping[str, object] | None,
) -> dict[str, torch.Tensor]:
    """
    Synchronizes frame metrics from a source batch to a target batch.
    
    Args:
        target_batch (MutableMapping[str, object] | None): The batch dictionary to update.
        source_batch_tensors (Mapping[str, object] | None): The source batch dictionary containing encoded frame metrics.
        
    Returns:
        dict[str, torch.Tensor]: The decoded frame metrics dictionary.
    """
    frame_metrics = decode_frame_metrics(source_batch_tensors)
    if target_batch is not None:
        if frame_metrics:
            target_batch["frame_metrics"] = frame_metrics
        else:
            target_batch.pop("frame_metrics", None)
    return frame_metrics
