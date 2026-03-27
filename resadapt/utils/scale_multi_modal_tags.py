# Copyright 2025 the ResAdapt authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Composite Hydra tag ``scale_multi_modal_data``: semantics and resolution.

This string is assembled in ``scripts/main.sh`` (sections 4–6) from ``SCALE_BASE``,
feature toggles, optional ``ray<NNODES>-`` prefix, and passed identically to
``algorithm``, ``allocator``, and ``actor_rollout_ref`` when ``SCALE_ENABLE_SCALE=1``.

**Substrings produced by main.sh (underscore-separated)**

- ``sep``: separate allocator resource pool (``allocator.enable_resource_pool``).
- ``filter``: ``algorithm.use_filter_sid``; also gates predictor log-prob alignment when combined with ``ispred``.
- ``ccen``: concentration / scale regularization coefficients (allocator ``scale.concentration_coef``).
- ``sim``: contrastive + sim-scale losses (allocator overrides).
- ``cost<name>``: cost / advantage mode stem from ``SCALE_USE_COST`` (e.g. ``costcapo``); Hydra sets ``algorithm.use_cost=<name>_acc``.
- ``notest``: skip validation (``test_freq``, ``val_before_train``).
- ``actor_frozen``: freeze full actor (``fsdp_workers`` actor role; vision + LM).
- ``allocator_frozen``: freeze allocator weights where applicable (worker config).

**Substrings often added manually or in custom launchers (not always in main.sh)**

- ``aw``: frame-aware path; OR’d with ``use_cost``-based gating so workers compute frame metrics (``fsdp_workers``).
- ``ispred``: predictor / filtered-batch log-prob alignment in ``ray_trainer._update_allocator`` (with ``use_filter_sid``).
- ``hadw``: with rollout tag, enables GRPO ``use_cost`` rewrite path (``ray_trainer``; see ``use_hadw_in_grpo``).

**Resolution order**

:func:`resolve_scale_multi_modal_data_tag` matches ``ray_trainer`` heuristics: first non-empty among
``algorithm.scale_multi_modal_data``, ``allocator.scale_multi_modal_data``,
``actor_rollout_ref.scale_multi_modal_data``.

**Related modules**

- ``resadapt.utils.use_cost_frame_metrics`` — when allocator forward must compute ``frame_metrics`` from ``use_cost``.
- ``resadapt.reward_fn.advantage`` — which cost tags consume frame metrics or cost terms.
"""

from __future__ import annotations

from typing import Any


def resolve_scale_multi_modal_data_tag(full_cfg: Any) -> str:
    """Return the first non-empty ``scale_multi_modal_data`` from algorithm → allocator → rollout.

    Empty string if all sections are unset or explicitly null/none (case-insensitive).
    Call sites that need Hydra’s historical default of ``\"scale\"`` should apply
    ``tag or full_cfg.allocator.get(\"scale_multi_modal_data\") or \"scale\"`` as in
    ``ray_trainer.RayPPOTrainer._update_allocator``.
    """
    for section in ("algorithm", "allocator", "actor_rollout_ref"):
        sub = full_cfg.get(section)
        if sub is None:
            continue
        tag = sub.get("scale_multi_modal_data") if hasattr(sub, "get") else None
        if tag is None:
            continue
        s = str(tag).strip().lower()
        if s and s not in ("none", "null", ""):
            return str(tag)
    return ""
