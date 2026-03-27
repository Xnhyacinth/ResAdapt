# Copyright 2025 the ResAdapt authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Video decode fallback when Hugging Face defaults to a broken ``torchcodec``.

``transformers`` may set ``backend="torchcodec"`` in ``VideoProcessor.fetch_videos`` whenever
``torchcodec`` is importable, but loading can still fail at runtime (missing ``libavutil``,
FFmpeg mismatch, or ``torchcodec`` built against a different PyTorch ABI).

Set ``ALLOCATOR_VIDEO_BACKEND=pyav`` (or ``torchvision``, ``opencv``, ``decord``) to force a
single backend. Otherwise we try, in order: PyAV → torchvision → opencv → decord → torchcodec.
"""

from __future__ import annotations

import os
import types
from typing import Any


def _video_backend_order() -> list[str]:
    env = os.getenv("ALLOCATOR_VIDEO_BACKEND", "").strip().lower()
    if env:
        return [env]
    # PyAV is usually available via ``pip install av``; torchcodec last.
    return ["pyav", "torchvision", "opencv", "decord", "torchcodec"]


def patch_video_processor_fetch_videos(video_processor: Any) -> None:
    """Replace ``video_processor.fetch_videos`` with multi-backend retries."""
    if video_processor is None:
        return
    if getattr(video_processor, "_resadapt_fetch_videos_patched", False):
        return

    from transformers.video_utils import load_video

    def fetch_videos(self, video_url_or_urls, sample_indices_fn=None):
        if isinstance(video_url_or_urls, list):
            return list(
                zip(
                    *[
                        self.fetch_videos(x, sample_indices_fn=sample_indices_fn)
                        for x in video_url_or_urls
                    ]
                )
            )
        last_err: BaseException | None = None
        for backend in _video_backend_order():
            try:
                return load_video(
                    video_url_or_urls,
                    backend=backend,
                    sample_indices_fn=sample_indices_fn,
                )
            except BaseException as e:
                last_err = e
                continue
        assert last_err is not None
        raise last_err

    video_processor.fetch_videos = types.MethodType(fetch_videos, video_processor)
    video_processor._resadapt_fetch_videos_patched = True
