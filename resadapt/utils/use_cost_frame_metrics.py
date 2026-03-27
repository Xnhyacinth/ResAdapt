# Copyright 2025 the ResAdapt authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Gate ``compute_frame_metrics`` in allocator workers from ``algorithm.use_cost``.

Must stay aligned with ``resadapt.reward_fn.advantage.compute_allocator_advantage``:
only tags whose advantage path **requires** ``frame_metrics`` return True here.
``capo`` is scalar-only by default. To **compute** ``frame_metrics`` for the optional
capo frame bonus (``framecoef``, ``wkeep``, …), either:

- set ``use_cost`` to include ``capo_frame`` (e.g. ``capo_frame_acc`` via
  ``SCALE_USE_COST=capo_frame`` in ``main.sh``), or
- add substring ``aw`` to ``scale_multi_modal_data`` (existing OR gate in workers).

Other tags that need frames: ``saliency_share_v1``, ``framepair_v1``, …

The ``scale_multi_modal_data`` substring ``aw`` is handled separately in the worker
(OR with this predicate): when ``aw`` is present, frame-aware paths always compute metrics.

See ``resadapt.utils.scale_multi_modal_tags`` for the full composite tag string and other substrings
(``ispred``, ``actor_frozen``, ``cost*``, …).
"""

from __future__ import annotations

from typing import Optional

# Substrings in ``use_cost`` whose advantage path may consume ``frame_metrics``.
# Bare ``capo`` is not listed (scalar-first); ``capo_frame`` opts into allocator
# ``compute_frame_metrics`` for the capo optional frame bonus block in advantage.py.
_USE_COST_FRAME_METRIC_TOKENS = (
    "capo_frame",
    "saliency_share_v1",
    "framepair_v1",
    "frame_rank",
    "frame_ideal",
    "frameaware",
    "frame_new",
)


def use_cost_implies_compute_frame_metrics(use_cost: Optional[str]) -> bool:
    """Return True if allocator forward should compute frame metrics for this cost tag."""
    if not use_cost:
        return False
    u = str(use_cost).lower()
    # piecewise_v2: frame auxiliary advantages are on by default; ``noframeaux`` disables.
    if "piecewise_v2" in u:
        return "noframeaux" not in u
    # piecewise_v1: frame aux only when ``frameaux`` appears in the tag (advantage.py).
    if "piecewise_v1" in u and "frameaux" in u:
        return True
    return any(tok in u for tok in _USE_COST_FRAME_METRIC_TOKENS)
