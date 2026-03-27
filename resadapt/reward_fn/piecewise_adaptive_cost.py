"""
Piecewise cost–accuracy shaping for allocator RL.

Tag ``piecewise_v2`` (or ``piecewise_v1``) in ``use_cost`` selects knobs below; optional
``wwaste(x)`` etc. overrides defaults. Raw reward includes a waste penalty on wrong+high-cost.
"""

from __future__ import annotations

import re
from typing import Tuple

import torch


def parse_piecewise_knobs(use_cost_l: str) -> dict:
    """Parse optional tokens: pwright(1.5) prwrong(1.0) wunder(0.35) wover(0.9) cexp(1.5) wwaste(0.12)"""
    s = use_cost_l.lower()

    def _f(name: str, default: float) -> float:
        m = re.search(rf"{name}\(([0-9]*\.?[0-9]+)\)", s)
        return float(m.group(1)) if m else default

    v2 = "piecewise_v2" in s
    return {
        "piecewise_v2": v2,
        "pwright": _f("pwright", 1.15 if v2 else 1.0),
        "prwrong": _f("prwrong", 1.0),
        "w_right": _f("wright", 1.0),
        "w_right_pen": _f("wrightpen", 0.4 if v2 else 0.35),
        "w_under": _f("wunder", 0.42 if v2 else 0.45),
        "w_over": _f("wover", 0.95 if v2 else 0.85),
        "cexp": _f("cexp", 1.55 if v2 else 1.5),
        "w_waste": _f("wwaste", 0.1 if v2 else 0.08),
    }


def piecewise_sid_raw_reward(
    acc: torch.Tensor,
    cost_ratio: torch.Tensor,
    knobs: dict,
) -> torch.Tensor:
    """
    acc: (M,) in [0, 1]
    cost_ratio: (M,) in [0, 1], normalized mean scale
    """
    c = cost_ratio.clamp(0.0, 1.0)
    a = acc.clamp(0.0, 1.0)
    p_r = float(knobs["pwright"])
    p_w = float(knobs["prwrong"])
    w_r = float(knobs["w_right"])
    wrp = float(knobs["w_right_pen"])
    w_u = float(knobs["w_under"])
    w_o = float(knobs["w_over"])
    cexp = float(knobs["cexp"])
    w_waste = float(knobs.get("w_waste", 0.08))

    # Correct: high reward when cost low; subtract for high cost even if correct
    r_right = w_r * (1.0 - c).pow(cexp) - wrp * c.pow(p_r)

    # Wrong: encourage raising scale when under-resolved; punish high cost failures harder
    one_m_a = (1.0 - a).clamp(0.0, 1.0)
    r_wrong = one_m_a * (w_u * (1.0 - c).pow(p_w) - w_o * c.pow(p_w))

    raw = a * r_right + (1.0 - a) * r_wrong
    # Wasted compute when wrong and cost already high: monotonic extra penalty (anti hack)
    waste = (1.0 - a) * w_waste * (c * c)
    return raw - waste


def piecewise_group_advantage(
    current_accs: torch.Tensor,
    cost_ratio: torch.Tensor,
    use_cost_l: str,
    *,
    epsilon: float,
) -> Tuple[torch.Tensor, dict]:
    """
    Map sid-level acc and cost to advantages via within-group z-score of raw reward.
    """
    knobs = parse_piecewise_knobs(use_cost_l)
    raw = piecewise_sid_raw_reward(current_accs, cost_ratio, knobs)
    mean = raw.mean()
    std = raw.std(unbiased=False)
    if float(std.item()) < 1e-4:
        adv = torch.zeros_like(raw)
    else:
        adv = (raw - mean) / (std + epsilon)
    return adv, knobs
