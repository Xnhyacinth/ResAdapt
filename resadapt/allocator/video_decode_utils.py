# Copyright 2025 the ResAdapt authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Video decode with backend retries around ``transformers.video_utils.load_video``.

Default order prefers **torchcodec** first (HF’s native path when available), then falls back on
failure: PyAV → torchvision → opencv → decord. Use this when ``torchcodec`` is importable but
sometimes fails at runtime (FFmpeg/ABI), or when you want a consistent retry policy.

Set ``ALLOCATOR_VIDEO_BACKEND=pyav`` (or ``torchvision``, ``opencv``, ``decord``, ``torchcodec``)
to **force** a single backend (no retries).
"""

from __future__ import annotations

import os
import types
from typing import Any


def _video_backend_order() -> list[str]:
    env = os.getenv("ALLOCATOR_VIDEO_BACKEND", "").strip().lower()
    if env:
        return [env]
    # Prefer torchcodec when it works; on any error try other backends.
    return ["torchcodec", "pyav", "torchvision", "opencv", "decord"]


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
