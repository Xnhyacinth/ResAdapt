from __future__ import annotations

from collections.abc import Mapping, MutableMapping

import torch

FRAME_METRICS_PREFIX = "frame_metrics:"


def encode_frame_metrics(frame_metrics: Mapping[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
    if frame_metrics is None or len(frame_metrics) == 0:
        return {}
    return {
        f"{FRAME_METRICS_PREFIX}{key}": value
        for key, value in frame_metrics.items()
        if isinstance(value, torch.Tensor)
    }


def decode_frame_metrics(batch_tensors: Mapping[str, object] | None) -> dict[str, torch.Tensor]:
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
    frame_metrics = decode_frame_metrics(source_batch_tensors)
    if target_batch is not None:
        if frame_metrics:
            target_batch["frame_metrics"] = frame_metrics
        else:
            target_batch.pop("frame_metrics", None)
    return frame_metrics
