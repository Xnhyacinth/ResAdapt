import torch
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict
import re

_HADW_STATE: Dict[str, float] = {}
_FRAMEPAIR_V1_WARNED = False


def _hadw_reweight_advantages(
    advs: torch.Tensor,
    difficulty: torch.Tensor,
    *,
    use_cost_l: str,
    epsilon: float,
) -> torch.Tensor:
    if "hadw" not in use_cost_l:
        return advs

    beta_match = re.search(r"hadwbeta([0-9]*\.?[0-9]+)", use_cost_l)
    if beta_match:
        beta = float(beta_match.group(1))
    else:
        beta_match = re.search(r"hadw([0-9]*\.?[0-9]+)", use_cost_l)
        beta = float(beta_match.group(1)) if beta_match else 1.0

    ema_match = re.search(r"hadwema([0-9]*\.?[0-9]+)", use_cost_l)
    ema = float(ema_match.group(1)) if ema_match else 0.05
    ema = max(0.0, min(1.0, ema))

    clip_match = re.search(r"hadwclip([0-9]*\.?[0-9]+)", use_cost_l)
    clip = float(clip_match.group(1)) if clip_match else 3.0
    clip = max(0.0, clip)

    with torch.no_grad():
        group_difficulty = float(difficulty.mean().clamp(0.0, 1.0).item())
        prev = _HADW_STATE.get("difficulty_anchor")
        if prev is None:
            anchor = group_difficulty
        else:
            anchor = (1.0 - ema) * float(prev) + ema * group_difficulty
        _HADW_STATE["difficulty_anchor"] = anchor

        anchor_t = torch.tensor(anchor, device=difficulty.device, dtype=difficulty.dtype)
        logw = beta * (difficulty - anchor_t)
        if clip > 0.0:
            logw = logw.clamp(min=-clip, max=clip)
        w = torch.exp(logw)

    return advs * w.to(dtype=advs.dtype)


def _as_py_key(x: Any) -> Any:
    """Convert numpy scalars / torch scalars to python hashable keys."""
    if isinstance(x, np.generic):
        return x.item()
    if torch.is_tensor(x) and x.numel() == 1:
        return x.item()
    return x


def _group_zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-score within a group; safe for near-constant vectors."""
    mean = x.mean()
    std = x.std(unbiased=False)
    if std < 1e-4:
        return torch.zeros_like(x)
    return (x - mean) / (std + eps)


def _batch_zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Batch-wise z-score; safe for near-constant vectors."""
    mean = x.mean()
    std = x.std(unbiased=False)
    if std < 1e-4:
        return torch.zeros_like(x)
    return (x - mean) / (std + eps)


def _safe_tensor_from_list(
    vals: List[float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(vals, device=device, dtype=dtype)


def _compute_sid_metric_avg(
    metric: Optional[torch.Tensor],  # (B, T)
    scale_mask: torch.Tensor,        # (B, T) 
    sid: np.ndarray,                 # (B,)
    sids_list: List,                 # List of sids in current group
) -> torch.Tensor:
    """
    Aggregate frame-level metric to sid-level average.
    
    Args:
        metric: (B, T) frame-level metric values
        scale_mask: (B, T) valid frame mask
        sid: (B,) sample IDs
        sids_list: List of unique sids in current group
    
    Returns:
        (M,) tensor of average metric per sid in sids_list order
    """
    if metric is None:
        return torch.zeros(len(sids_list), device=scale_mask.device, dtype=scale_mask.dtype)
    
    device = metric.device
    dtype = metric.dtype
    bsz = metric.shape[0]
    
    # Compute per-sample average (masked mean)
    valid_counts = scale_mask.float().sum(dim=-1).clamp(min=1.0)
    sample_avg = (metric * scale_mask.float()).sum(dim=-1) / valid_counts  # (B,)
    
    # Aggregate to sid-level
    sid2vals = defaultdict(list)
    for i in range(bsz):
        s_key = _as_py_key(sid[i])
        sid2vals[s_key].append(sample_avg[i])
    
    result = []
    for s_key in sids_list:
        vals = sid2vals.get(s_key, [torch.tensor(0.0, device=device, dtype=dtype)])
        result.append(torch.stack(vals).mean())
    
    return torch.stack(result).to(device=device, dtype=dtype)


def _compute_sid_frame_valid_mask(scale_mask: torch.Tensor, idxs: List[int]) -> torch.Tensor:
    """Return the union valid-frame mask for all rows that share the same sid."""
    if not idxs:
        return torch.zeros((scale_mask.shape[1],), device=scale_mask.device, dtype=torch.bool)
    return scale_mask[idxs].to(torch.bool).any(dim=0)


def _aggregate_sid_frame_values(
    values: Optional[torch.Tensor],
    scale_mask: torch.Tensor,
    idxs: List[int],
    *,
    default: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Average per-frame values across all rows for the same sid using valid frames only."""
    max_len = int(scale_mask.shape[1])
    default_vec = torch.full((max_len,), default, device=device, dtype=dtype)
    if values is None or not idxs:
        return default_vec

    valid = scale_mask[idxs].to(torch.bool)
    valid_f = valid.to(dtype=dtype)
    counts = valid_f.sum(dim=0)
    rows = values[idxs].to(device=device, dtype=dtype)
    avg = (rows * valid_f).sum(dim=0) / counts.clamp(min=1.0)
    return torch.where(counts > 0, avg, default_vec)


def _masked_zscore(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-score over valid entries only and keep padded positions at zero."""
    mask_f = mask.to(dtype=x.dtype)
    count = mask_f.sum().clamp(min=1.0)
    mean = (x * mask_f).sum() / count
    centered = torch.where(mask, x - mean, torch.zeros_like(x))
    var = ((centered * centered) * mask_f).sum() / count
    std = var.sqrt()
    if float(std.item()) < 1e-4:
        return torch.zeros_like(x)
    return torch.where(mask, centered / (std + eps), torch.zeros_like(x))


def _frame_priority_from_metrics(
    redundancy: torch.Tensor,
    uniqueness: torch.Tensor,
    relevance_raw: torch.Tensor,
    info_score: torch.Tensor,
    valid_row: torch.Tensor,
    *,
    keep_weight: float,
    unique_weight: float,
    relevance_weight: float,
    info_weight: float,
) -> torch.Tensor:
    """Convert frame metrics into a bounded priority score in [0, 1]."""
    redundancy = redundancy.clamp(0.0, 1.0)
    uniqueness = uniqueness.clamp(0.0, 1.0)
    relevance = ((relevance_raw + 1.0) * 0.5).clamp(0.0, 1.0)
    info_score = info_score.clamp(0.0, 1.0)
    denom = keep_weight + unique_weight + relevance_weight + info_weight + 1e-6
    priority = (
        keep_weight * (1.0 - redundancy)
        + unique_weight * uniqueness
        + relevance_weight * relevance
        + info_weight * info_score
    ) / denom
    return torch.where(valid_row, priority.clamp(0.0, 1.0), torch.zeros_like(priority))


def _compute_active_frame_importance(
    redundancy: torch.Tensor,
    relevance_raw: torch.Tensor,
    detail_score: Optional[torch.Tensor],
    valid_row: torch.Tensor,
) -> torch.Tensor:
    """Typed active-path importance: semantic relevance + novelty (+ optional detail)."""
    redundancy = redundancy.clamp(0.0, 1.0)
    novelty = 1.0 - redundancy
    relevance = ((relevance_raw + 1.0) * 0.5).clamp(0.0, 1.0)

    if detail_score is None:
        detail = torch.zeros_like(relevance)
    else:
        detail = detail_score.clamp(0.0, 1.0)

    importance = 0.6 * relevance + 0.3 * novelty + 0.1 * detail
    return torch.where(valid_row, importance.clamp(0.0, 1.0), torch.zeros_like(importance))


def _compute_active_frame_bonus(
    importance: torch.Tensor,
    scales01: torch.Tensor,
    valid_row: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Reward high scale on important frames and low scale on unimportant ones."""
    allocation_reward = importance * scales01 + (1.0 - importance) * (1.0 - scales01)
    allocation_reward = torch.where(valid_row, allocation_reward, torch.zeros_like(allocation_reward))
    return _masked_zscore(allocation_reward, valid_row, eps=eps)


def _compute_framepair_bonus(
    importance: torch.Tensor,
    scales01: torch.Tensor,
    valid_row: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Redistribution bonus: move scale share from weak frames to important frames."""
    importance = torch.where(valid_row, importance, torch.zeros_like(importance))
    scales01 = torch.where(valid_row, scales01, torch.zeros_like(scales01))
    if int(valid_row.sum().item()) <= 1:
        return torch.zeros_like(scales01)

    target_share = importance / importance.sum().clamp(min=eps)
    actual_share = scales01 / scales01.sum().clamp(min=eps)
    redistribution_reward = target_share - actual_share
    redistribution_reward = torch.where(valid_row, redistribution_reward, torch.zeros_like(redistribution_reward))
    return _masked_zscore(redistribution_reward, valid_row, eps=eps)


def _masked_normalized_share(
    values: torch.Tensor,
    valid_row: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize valid entries into a probability share, fallback to uniform when degenerate."""
    masked = torch.where(valid_row, values, torch.zeros_like(values))
    total = masked.sum()
    uniform = valid_row.to(dtype=values.dtype)
    uniform = uniform / uniform.sum().clamp(min=1.0).to(dtype=values.dtype)
    normalized = masked / total.clamp(min=eps)
    return torch.where(total > eps, normalized, uniform)


def _compute_saliency_target_share(
    *,
    text_relevance: Optional[torch.Tensor],
    temporal_surprise: Optional[torch.Tensor],
    detail_score: Optional[torch.Tensor],
    saliency_anchor: Optional[torch.Tensor],
    valid_row: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Quality-first target share using typed saliency signals only."""
    if saliency_anchor is not None:
        anchor = saliency_anchor.clamp(min=0.0)
        anchor = anchor + valid_row.to(dtype=anchor.dtype) * eps
        return _masked_normalized_share(anchor, valid_row, eps=eps)

    base_dtype = torch.float32
    if text_relevance is not None:
        base_dtype = text_relevance.dtype
    elif temporal_surprise is not None:
        base_dtype = temporal_surprise.dtype
    elif detail_score is not None:
        base_dtype = detail_score.dtype

    zeros = torch.zeros_like(valid_row, dtype=base_dtype)
    relevance01 = (
        ((text_relevance + 1.0) * 0.5).clamp(0.0, 1.0)
        if text_relevance is not None
        else zeros
    )
    surprise = temporal_surprise.clamp(0.0, 1.0) if temporal_surprise is not None else zeros
    detail = detail_score.clamp(0.0, 1.0) if detail_score is not None else zeros
    raw = 0.55 * relevance01 + 0.30 * surprise + 0.15 * detail
    raw = raw + valid_row.to(dtype=raw.dtype) * eps
    return _masked_normalized_share(raw, valid_row, eps=eps)


def compute_predictor_advantage(
    scores: torch.Tensor,                # (B,) or (B, T)
    uid: np.ndarray,                     # (B,)
    sid: np.ndarray,                     # (B,)
    scales: Optional[torch.Tensor] = None,
    actions: Optional[torch.Tensor] = None,
    scale_mask: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
    norm_adv: bool = True,
    batch_norm_adv: bool = False,  
    filter_invalid_sid: bool = False,
    use_cost: Optional[str] = None,
    use_discrete_action: bool = False,
    penalty_coef: float = 0.05,
    encouragement_coef: float = 0.02,
    min_scale: float = 0.25,
    max_scale: float = 2.0,
    rewards: Optional[list] = None,
    centered_term: float = 0.5,
    frame_metrics: Optional[Dict[str, torch.Tensor]] = None,  # Frame metrics from predictor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute predictor advantages.

    Pipeline (GDPO-consistent):
      - Group by uid (prompt-group)
      - For each uid, compute sid-level avg score(s)
      - Compute group-wise advantage per uid
      - Optionally apply batch-wise normalization (Eq.6) over unique sid advantages
      - Expand to per-object advantage via scale_mask if provided

    Returns:
      predictor_advantages: (B, Tobj) if scale_mask else (B,)
      predictor_update_mask: (B,) bool
    """
    device = scores.device
    dtype = scores.dtype

    scores_1d = scores.sum(dim=-1) if scores.dim() > 1 else scores
    bsz = int(scores_1d.shape[0])

    sid2scores = defaultdict(list)
    for i in range(bsz):
        s_key = _as_py_key(sid[i])
        sid2scores[s_key].append(scores_1d[i])
    sid2avg = {s_key: torch.stack(vs).mean() for s_key, vs in sid2scores.items()}

    uid2sid_scores = defaultdict(list)   # uid -> [avg_score(sid1), ...]
    uid_sid_map = defaultdict(list)      # uid -> [sid1, ...]
    processed_sids = set()

    for i in range(bsz):
        s_key = _as_py_key(sid[i])
        u_key = _as_py_key(uid[i])
        if s_key in processed_sids:
            continue
        processed_sids.add(s_key)
        uid2sid_scores[u_key].append(sid2avg[s_key])
        uid_sid_map[u_key].append(s_key)

    sid2acc_avg: Dict[Any, torch.Tensor] = {}
    sid2cost: Dict[Any, float] = {}

    use_cost_l = use_cost.lower() if use_cost else None
    if use_cost:

        acc_reward_tensor = scores_1d
        if rewards is not None and len(rewards) > 0:
            if isinstance(rewards, (np.ndarray, torch.Tensor)):
                rewards = rewards.tolist()
            try:
                acc_list = [float(r.get("acc_reward", 0.0)) if isinstance(r, dict) else 0.0 for r in rewards]
                if len(acc_list) == bsz:
                    acc_reward_tensor = torch.tensor(acc_list, device=device, dtype=dtype)
            except Exception as e:
                print(f"[Warning] Failed to extract acc_reward from rewards list: {e}")

        # sid -> avg acc
        sid2acc_scores = defaultdict(list)
        for i in range(bsz):
            s_key = _as_py_key(sid[i])
            sid2acc_scores[s_key].append(acc_reward_tensor[i])
        sid2acc_avg = {s_key: torch.stack(vs).mean() for s_key, vs in sid2acc_scores.items()}

        out = {
            "scales": scales,
            "actions": actions,
            "scale_mask": scale_mask,
        }

        out = {"scales": scales, "actions": actions, "scale_mask": scale_mask}

        _, _, sample_mean_scales = compute_scales_and_sample_means_cpu(
            out,
            max_scale=max_scale,
            min_scale=min_scale,
            use_discrete_action=use_discrete_action,
            default_step=0.25,
        )

        sample_costs = torch.tensor(sample_mean_scales, device=device, dtype=torch.float32)

        # map mean(scale) -> [0, 1] ratio
        sample_costs = (sample_costs - min_scale) / (max_scale - min_scale + 1e-6)
        sample_costs = sample_costs.clamp(0.0, 1.0)

        sid2cost_values = defaultdict(list)
        for i in range(bsz):
            s_key = _as_py_key(sid[i])
            sid2cost_values[s_key].append(sample_costs[i])
        sid2cost = {s_key: float(torch.stack(vs).mean().item()) for s_key, vs in sid2cost_values.items()}

    sid2advantage: Dict[Any, torch.Tensor] = {}
    per_sid_frame_adv_all: Dict[Any, torch.Tensor] | None = (
        {}
        if use_cost_l is not None
        and any(key in use_cost_l for key in ("frame_ideal", "frame_rank", "newtie", "framepair_v1", "saliency_share_v1"))
        and scale_mask is not None
        and scales is not None
        else None
    )

    for u_key, s_scores in uid2sid_scores.items():
        sids_list = uid_sid_map[u_key]
        M = len(s_scores)

        if M <= 1:
            sid2advantage[sids_list[0]] = torch.tensor(0.0, device=device, dtype=dtype)
            continue

        s_scores_tensor = torch.stack(s_scores).to(device=device, dtype=dtype)  # (M,)
        custom_adv_calculated = False

        if use_cost:
            use_cost_l = use_cost.lower()
            costs = _safe_tensor_from_list([sid2cost[sid_] for sid_ in sids_list],
                                           device=device, dtype=torch.float32)  # (M,)
            cost_ratio = costs.clamp(min=0.0, max=1.0)
            eff_rewards = (-cost_ratio)  # higher is better (lower cost)

            current_accs = _safe_tensor_from_list([float(sid2acc_avg[sid_].item()) for sid_ in sids_list],
                                                  device=device, dtype=torch.float32)
            if "acc" in use_cost_l:
                is_correct_mask = (current_accs > 0.35)
            else:
                is_correct_mask = (s_scores_tensor > 0.35)

            # reused by tie/multiply
            centered_ratio = (1.0 - cost_ratio) - centered_term if "cen" in use_cost_l else (1.0 - cost_ratio)
            efficiency_bonus = encouragement_coef * current_accs * centered_ratio  # (M,)

            if "saliency_share_v1" in use_cost_l:
                base_signal = current_accs if "acc" in use_cost_l else s_scores_tensor.float()
                base_adv = _group_zscore(base_signal, eps=epsilon)

                gas_match = re.search(r"gas([0-9]*\.?[0-9]+)", use_cost_l)
                gas_tax = float(gas_match.group(1)) if gas_match else 0.05
                gas_tax = max(0.0, gas_tax)

                advs = base_adv - gas_tax * cost_ratio.to(base_adv.dtype)

                if frame_metrics is not None and scale_mask is not None and scales is not None and per_sid_frame_adv_all is not None:
                    sid_index_map: Dict[Any, List[int]] = defaultdict(list)
                    for idx_b in range(bsz):
                        sid_index_map[_as_py_key(sid[idx_b])].append(idx_b)

                    max_len = int(scale_mask.shape[1])
                    fm_rel = frame_metrics.get("text_relevance", None)
                    fm_surprise = frame_metrics.get("temporal_surprise", None)
                    fm_detail = frame_metrics.get("detail_score", None)
                    fm_anchor = frame_metrics.get("saliency_anchor", None)
                    fm_redundancy = frame_metrics.get("redundancy", None)

                    per_sid_frame_advs: Dict[Any, torch.Tensor] = {}
                    for k, s_key in enumerate(sids_list):
                        idxs = sid_index_map.get(s_key, [])
                        if not idxs:
                            per_sid_frame_advs[s_key] = torch.zeros((max_len,), device=device, dtype=dtype)
                            continue

                        valid_row = _compute_sid_frame_valid_mask(scale_mask, idxs)
                        scales_row = _aggregate_sid_frame_values(
                            scales,
                            scale_mask,
                            idxs,
                            default=min_scale,
                            device=device,
                            dtype=dtype,
                        )
                        scales01 = ((scales_row - min_scale) / (max_scale - min_scale + 1e-6)).clamp(0.0, 1.0)
                        actual_share = _masked_normalized_share(scales01, valid_row, eps=epsilon)

                        relevance_row = None
                        if fm_rel is not None:
                            relevance_row = _aggregate_sid_frame_values(
                                fm_rel,
                                scale_mask,
                                idxs,
                                default=0.0,
                                device=device,
                                dtype=scales01.dtype,
                            )

                        surprise_row = None
                        if fm_surprise is not None:
                            surprise_row = _aggregate_sid_frame_values(
                                fm_surprise,
                                scale_mask,
                                idxs,
                                default=0.0,
                                device=device,
                                dtype=scales01.dtype,
                            )
                        elif fm_redundancy is not None:
                            redundancy_row = _aggregate_sid_frame_values(
                                fm_redundancy,
                                scale_mask,
                                idxs,
                                default=1.0,
                                device=device,
                                dtype=scales01.dtype,
                            )
                            surprise_row = (1.0 - redundancy_row).clamp(0.0, 1.0)

                        detail_row = None
                        if fm_detail is not None:
                            detail_row = _aggregate_sid_frame_values(
                                fm_detail,
                                scale_mask,
                                idxs,
                                default=0.0,
                                device=device,
                                dtype=scales01.dtype,
                            )

                        anchor_row = None
                        if fm_anchor is not None:
                            anchor_row = _aggregate_sid_frame_values(
                                fm_anchor,
                                scale_mask,
                                idxs,
                                default=0.0,
                                device=device,
                                dtype=scales01.dtype,
                            )

                        target_share = _compute_saliency_target_share(
                            text_relevance=relevance_row,
                            temporal_surprise=surprise_row,
                            detail_score=detail_row,
                            saliency_anchor=anchor_row,
                            valid_row=valid_row,
                            eps=epsilon,
                        )
                        share_bonus = _masked_zscore(target_share - actual_share, valid_row, eps=epsilon)

                        adv_vec = torch.where(
                            valid_row,
                            advs[k] + share_bonus.to(dtype),
                            torch.zeros_like(share_bonus, dtype=dtype),
                        )
                        per_sid_frame_advs[s_key] = adv_vec.to(device=device, dtype=dtype)
                        per_sid_frame_adv_all[s_key] = adv_vec.to(device=device, dtype=dtype)

                    frame_adv = torch.stack([per_sid_frame_advs[s_key] for s_key in sids_list], dim=0)
                    valid_mask_m = torch.stack(
                        [
                            _compute_sid_frame_valid_mask(scale_mask, sid_index_map.get(s_key, []))
                            for s_key in sids_list
                        ],
                        dim=0,
                    )
                    denom = valid_mask_m.float().sum(dim=-1).clamp(min=1.0)
                    advs = (frame_adv * valid_mask_m.float()).sum(dim=-1) / denom

                custom_adv_calculated = True

            elif "framepair_v1" in use_cost_l:
                global _FRAMEPAIR_V1_WARNED
                deprecated_tokens = (
                    "alpha",
                    "tau",
                    "asym",
                    "wrongscale",
                    "rightscale",
                    "accpow",
                    "accfloor",
                    "scalecap",
                    "framecoef",
                    "framecap",
                    "wkeep",
                    "wu",
                    "wrel",
                    "winf",
                )
                if (not _FRAMEPAIR_V1_WARNED) and any(token in use_cost_l for token in deprecated_tokens):
                    warnings.warn(
                        "`framepair_v1` only uses the typed adaptive path (plus optional `gas`/`noclamp`) "
                        "and ignores legacy `newtie` knobs.",
                        stacklevel=2,
                    )
                    _FRAMEPAIR_V1_WARNED = True

                base_signal = current_accs if "acc" in use_cost_l else s_scores_tensor.float()
                base_adv = _group_zscore(base_signal, eps=epsilon)
                difficulty = (1.0 - current_accs).clamp(0.0, 1.0)
                base_adv = _hadw_reweight_advantages(base_adv, difficulty, use_cost_l=use_cost_l, epsilon=epsilon)

                gas_match = re.search(r"gas([0-9]*\.?[0-9]+)", use_cost_l)
                gas_tax = float(gas_match.group(1)) if gas_match else 0.05
                gas_tax = max(0.0, gas_tax)

                advs = base_adv - gas_tax * cost_ratio.to(base_adv.dtype)

                if frame_metrics is not None and scale_mask is not None and scales is not None and per_sid_frame_adv_all is not None:
                    sid_index_map: Dict[Any, List[int]] = defaultdict(list)
                    for idx_b in range(bsz):
                        sid_index_map[_as_py_key(sid[idx_b])].append(idx_b)

                    max_len = int(scale_mask.shape[1])
                    fm_r = frame_metrics.get("redundancy", None)
                    fm_rel = frame_metrics.get("text_relevance", None)
                    fm_detail = frame_metrics.get("detail_score", None)

                    per_sid_frame_advs: Dict[Any, torch.Tensor] = {}
                    for k, s_key in enumerate(sids_list):
                        idxs = sid_index_map.get(s_key, [])
                        if not idxs:
                            per_sid_frame_advs[s_key] = torch.zeros((max_len,), device=device, dtype=dtype)
                            continue

                        valid_row = _compute_sid_frame_valid_mask(scale_mask, idxs)
                        scales_row = _aggregate_sid_frame_values(
                            scales,
                            scale_mask,
                            idxs,
                            default=min_scale,
                            device=device,
                            dtype=dtype,
                        )
                        scales01 = ((scales_row - min_scale) / (max_scale - min_scale + 1e-6)).clamp(0.0, 1.0)
                        redundancy_row = _aggregate_sid_frame_values(
                            fm_r,
                            scale_mask,
                            idxs,
                            default=0.0,
                            device=device,
                            dtype=scales01.dtype,
                        )
                        relevance_row = _aggregate_sid_frame_values(
                            fm_rel,
                            scale_mask,
                            idxs,
                            default=0.0,
                            device=device,
                            dtype=scales01.dtype,
                        )
                        detail_row = None
                        if fm_detail is not None:
                            detail_row = _aggregate_sid_frame_values(
                                fm_detail,
                                scale_mask,
                                idxs,
                                default=0.0,
                                device=device,
                                dtype=scales01.dtype,
                            )

                        importance = _compute_active_frame_importance(
                            redundancy=redundancy_row,
                            relevance_raw=relevance_row,
                            detail_score=detail_row,
                            valid_row=valid_row,
                        )
                        framepair_bonus = _compute_framepair_bonus(
                            importance=importance,
                            scales01=scales01,
                            valid_row=valid_row,
                            eps=epsilon,
                        )

                        adv_vec = torch.where(
                            valid_row,
                            advs[k] + framepair_bonus.to(dtype),
                            torch.zeros_like(framepair_bonus, dtype=dtype),
                        )
                        per_sid_frame_advs[s_key] = adv_vec.to(device=device, dtype=dtype)
                        per_sid_frame_adv_all[s_key] = adv_vec.to(device=device, dtype=dtype)

                    frame_adv = torch.stack([per_sid_frame_advs[s_key] for s_key in sids_list], dim=0)
                    valid_mask_m = torch.stack(
                        [
                            _compute_sid_frame_valid_mask(scale_mask, sid_index_map.get(s_key, []))
                            for s_key in sids_list
                        ],
                        dim=0,
                    )
                    denom = valid_mask_m.float().sum(dim=-1).clamp(min=1.0)
                    advs = (frame_adv * valid_mask_m.float()).sum(dim=-1) / denom

                if is_correct_mask.any() and "noclamp" not in use_cost_l:
                    advs = torch.where(is_correct_mask, advs.clamp(min=0.001), advs)

                custom_adv_calculated = True

            # --- TIE: base(normed) + bonus(abs) ---
            elif "tie" in use_cost_l:
                base_signal = current_accs if "acc" in use_cost_l else s_scores_tensor.float()
                base_adv = _group_zscore(base_signal, eps=epsilon)
                difficulty = (1.0 - current_accs).clamp(0.0, 1.0)
                base_adv = _hadw_reweight_advantages(base_adv, difficulty, use_cost_l=use_cost_l, epsilon=epsilon)

                if "newtie" in use_cost_l:
                    alpha_match = re.search(r"alpha([0-9]*\.?[0-9]+)", use_cost_l)
                    alpha = float(alpha_match.group(1)) if alpha_match else 0.5
                    alpha = max(0.0, min(1.0, alpha))

                    tau_match = re.search(r"tau([0-9]*\.?[0-9]+)", use_cost_l)
                    tau = float(tau_match.group(1)) if tau_match else 0.15
                    tau = max(tau, epsilon)

                    gas_match = re.search(r"gas([0-9]*\.?[0-9]+)", use_cost_l)
                    gas_tax = float(gas_match.group(1)) if gas_match else 0.05
                    gas_tax = max(0.0, gas_tax)

                    accpow_match = re.search(r"accpow([0-9]*\.?[0-9]+)", use_cost_l)
                    acc_pow = float(accpow_match.group(1)) if accpow_match else 0.5
                    acc_pow = max(0.0, acc_pow)

                    accfloor_match = re.search(r"accfloor([0-9]*\.?[0-9]+)", use_cost_l)
                    acc_floor = float(accfloor_match.group(1)) if accfloor_match else 0.05
                    acc_floor = max(0.0, min(1.0, acc_floor))

                    wrongscale_match = re.search(r"wrongscale([0-9]*\.?[0-9]+)", use_cost_l)
                    wrong_scale = float(wrongscale_match.group(1)) if wrongscale_match else 1.0
                    wrong_scale = max(0.0, wrong_scale)

                    rightscale_match = re.search(r"rightscale([0-9]*\.?[0-9]+)", use_cost_l)
                    right_scale = float(rightscale_match.group(1)) if rightscale_match else 1.0
                    right_scale = max(0.0, right_scale)

                    scalecap_match = re.search(r"scalecap([0-9]*\.?[0-9]+)", use_cost_l)
                    scale_cap = float(scalecap_match.group(1)) if scalecap_match else 0.0
                    scale_cap = max(0.0, scale_cap)

                    acc_weight = current_accs.clamp(0.0, 1.0).pow(acc_pow).clamp(min=acc_floor)
                    correct_bonus = right_scale * acc_weight * (1.0 - cost_ratio)
                    wrong_bonus = -wrong_scale * difficulty * cost_ratio
                    mixed_bonus = torch.where(is_correct_mask, correct_bonus, wrong_bonus)
                    mixed_bonus = torch.tanh(mixed_bonus / tau)
                    if scale_cap > 0.0:
                        mixed_bonus = mixed_bonus.clamp(min=-scale_cap, max=scale_cap)

                    advs = base_adv + alpha * mixed_bonus.to(base_adv.dtype) - gas_tax * cost_ratio.to(base_adv.dtype)

                    if frame_metrics is not None and scale_mask is not None and scales is not None and per_sid_frame_adv_all is not None:
                        framecoef_match = re.search(r"framecoef([0-9]*\.?[0-9]+)", use_cost_l)
                        frame_coef = float(framecoef_match.group(1)) if framecoef_match else 0.25
                        frame_coef = max(0.0, frame_coef)

                        framecap_match = re.search(r"framecap([0-9]*\.?[0-9]+)", use_cost_l)
                        frame_cap = float(framecap_match.group(1)) if framecap_match else 0.0
                        frame_cap = max(0.0, frame_cap)

                        wkeep_match = re.search(r"wkeep([0-9]*\.?[0-9]+)", use_cost_l)
                        keep_weight = float(wkeep_match.group(1)) if wkeep_match else 0.45
                        keep_weight = max(0.0, keep_weight)

                        wu_match = re.search(r"wu([0-9]*\.?[0-9]+)", use_cost_l)
                        unique_weight = float(wu_match.group(1)) if wu_match else 0.15
                        unique_weight = max(0.0, unique_weight)

                        wrel_match = re.search(r"wrel([0-9]*\.?[0-9]+)", use_cost_l)
                        relevance_weight = float(wrel_match.group(1)) if wrel_match else 0.30
                        relevance_weight = max(0.0, relevance_weight)

                        winf_match = re.search(r"winf([0-9]*\.?[0-9]+)", use_cost_l)
                        info_weight = float(winf_match.group(1)) if winf_match else 0.10
                        info_weight = max(0.0, info_weight)

                        sid_index_map: Dict[Any, List[int]] = defaultdict(list)
                        for idx_b in range(bsz):
                            sid_index_map[_as_py_key(sid[idx_b])].append(idx_b)

                        max_len = int(scale_mask.shape[1])
                        fm_r = frame_metrics.get("redundancy", None)
                        fm_u = frame_metrics.get("uniqueness", None)
                        fm_rel = frame_metrics.get("text_relevance", None)
                        fm_info = frame_metrics.get("info_score", None)

                        per_sid_frame_advs: Dict[Any, torch.Tensor] = {}
                        for k, s_key in enumerate(sids_list):
                            idxs = sid_index_map.get(s_key, [])
                            if not idxs:
                                per_sid_frame_advs[s_key] = torch.zeros((max_len,), device=device, dtype=dtype)
                                continue

                            valid_row = _compute_sid_frame_valid_mask(scale_mask, idxs)
                            scales_row = _aggregate_sid_frame_values(
                                scales,
                                scale_mask,
                                idxs,
                                default=min_scale,
                                device=device,
                                dtype=dtype,
                            )
                            scales01 = ((scales_row - min_scale) / (max_scale - min_scale + 1e-6)).clamp(0.0, 1.0)
                            redundancy_row = _aggregate_sid_frame_values(
                                fm_r,
                                scale_mask,
                                idxs,
                                default=0.0,
                                device=device,
                                dtype=scales01.dtype,
                            )
                            uniqueness_row = _aggregate_sid_frame_values(
                                fm_u,
                                scale_mask,
                                idxs,
                                default=0.0,
                                device=device,
                                dtype=scales01.dtype,
                            )
                            relevance_row = _aggregate_sid_frame_values(
                                fm_rel,
                                scale_mask,
                                idxs,
                                default=0.0,
                                device=device,
                                dtype=scales01.dtype,
                            )
                            info_row = _aggregate_sid_frame_values(
                                fm_info,
                                scale_mask,
                                idxs,
                                default=0.0,
                                device=device,
                                dtype=scales01.dtype,
                            )

                            priority = _frame_priority_from_metrics(
                                redundancy=redundancy_row,
                                uniqueness=uniqueness_row,
                                relevance_raw=relevance_row,
                                info_score=info_row,
                                valid_row=valid_row,
                                keep_weight=keep_weight,
                                unique_weight=unique_weight,
                                relevance_weight=relevance_weight,
                                info_weight=info_weight,
                            )
                            allocation_reward = priority * scales01 + (1.0 - priority) * (1.0 - scales01)
                            allocation_bonus = frame_coef * _masked_zscore(allocation_reward, valid_row, eps=epsilon)
                            if frame_cap > 0.0:
                                allocation_bonus = allocation_bonus.clamp(min=-frame_cap, max=frame_cap)
                                allocation_bonus = allocation_bonus - allocation_bonus[valid_row].mean()
                            allocation_bonus = torch.where(valid_row, allocation_bonus, torch.zeros_like(allocation_bonus))

                            adv_vec = torch.where(
                                valid_row,
                                advs[k] + allocation_bonus.to(dtype),
                                torch.zeros_like(allocation_bonus, dtype=dtype),
                            )
                            per_sid_frame_advs[s_key] = adv_vec.to(device=device, dtype=dtype)
                            per_sid_frame_adv_all[s_key] = adv_vec.to(device=device, dtype=dtype)

                        frame_adv = torch.stack([per_sid_frame_advs[s_key] for s_key in sids_list], dim=0)
                        valid_mask_m = torch.stack(
                            [
                                _compute_sid_frame_valid_mask(scale_mask, sid_index_map.get(s_key, []))
                                for s_key in sids_list
                            ],
                            dim=0,
                        )
                        denom = valid_mask_m.float().sum(dim=-1).clamp(min=1.0)
                        advs = (frame_adv * valid_mask_m.float()).sum(dim=-1) / denom
                else:
                    advs = base_adv + efficiency_bonus.to(base_adv.dtype)

                if "newtie" in use_cost_l and is_correct_mask.any() and "noclamp" not in use_cost_l:
                    advs = torch.where(is_correct_mask, advs.clamp(min=0.001), advs)

                custom_adv_calculated = True

            elif "simp" in use_cost_l:
                base_signal = current_accs if "acc" in use_cost_l else s_scores_tensor.float()
                base_adv = _group_zscore(base_signal, eps=epsilon)

                accpow_match = re.search(r"accpow([0-9]*\.?[0-9]+)", use_cost_l)
                acc_pow = float(accpow_match.group(1)) if accpow_match else 0.5
                acc_pow = max(0.0, acc_pow)

                accfloor_match = re.search(r"accfloor([0-9]*\.?[0-9]+)", use_cost_l)
                acc_floor = float(accfloor_match.group(1)) if accfloor_match else 0.05
                acc_floor = max(0.0, min(1.0, acc_floor))

                acc_weight = current_accs.clamp(0.0, 1.0).pow(acc_pow).clamp(min=acc_floor)
                difficulty = (1.0 - current_accs).clamp(0.0, 1.0)

                wrongpow_match = re.search(r"wrongpow([0-9]*\.?[0-9]+)", use_cost_l)
                wrong_pow = float(wrongpow_match.group(1)) if wrongpow_match else 1.0
                wrong_pow = max(0.0, wrong_pow)
                wrong_weight = difficulty.pow(wrong_pow)

                ct_match = re.search(r"ct([0-9]*\.?[0-9]+)", use_cost_l)
                correct_target = float(ct_match.group(1)) if ct_match else float(centered_term)
                correct_target = max(0.0, min(1.0, correct_target))

                wt_match = re.search(r"wt([0-9]*\.?[0-9]+)", use_cost_l)
                wrong_target = float(wt_match.group(1)) if wt_match else 0.7
                wrong_target = max(0.0, min(1.0, wrong_target))

                kcorr_match = re.search(r"kcorr([0-9]*\.?[0-9]+)", use_cost_l)
                k_corr = float(kcorr_match.group(1)) if kcorr_match else 1.0
                k_corr = max(0.0, k_corr)

                kwrong_match = re.search(r"kwrong([0-9]*\.?[0-9]+)", use_cost_l)
                k_wrong = float(kwrong_match.group(1)) if kwrong_match else 1.0
                k_wrong = max(0.0, k_wrong)

                correct_term = -k_corr * acc_weight * (cost_ratio - correct_target).clamp(min=0.0)
                wrong_term = k_wrong * wrong_weight * (wrong_target - cost_ratio).clamp(min=0.0)
                scale_shape = torch.where(is_correct_mask, correct_term, wrong_term)

                scalecap_match = re.search(r"scalecap([0-9]*\.?[0-9]+)", use_cost_l)
                if scalecap_match:
                    scale_cap = float(scalecap_match.group(1))
                    scale_cap = max(0.0, scale_cap)
                    if scale_cap > 0.0:
                        scale_shape = scale_shape.clamp(min=-scale_cap, max=scale_cap)

                gas_match = re.search(r"gas([0-9]*\\.?[0-9]+)", use_cost_l)
                gas_tax = float(gas_match.group(1)) if gas_match else 0.05
                gas_tax = max(0.0, gas_tax)

                advs = base_adv + (encouragement_coef * scale_shape).to(base_adv.dtype) - gas_tax * cost_ratio.to(base_adv.dtype)
                advs = _hadw_reweight_advantages(advs, difficulty, use_cost_l=use_cost_l, epsilon=epsilon)
                custom_adv_calculated = True

            # --- GDPO (paper-consistent): decoupled group-wise norm, then sum ---
            elif "gdpo" in use_cost_l:
                acc_signal = current_accs if "acc" in use_cost_l else s_scores_tensor.float()

                adv_acc = _group_zscore(acc_signal, eps=epsilon)
                adv_eff = _group_zscore(eff_rewards.to(acc_signal.dtype), eps=epsilon)

                if "mygdpo" in use_cost_l:
                    adv_eff = adv_eff * is_correct_mask.float()

                advs = adv_acc + encouragement_coef * adv_eff.to(adv_acc.dtype)
                difficulty = (1.0 - current_accs).clamp(0.0, 1.0)
                advs = _hadw_reweight_advantages(advs, difficulty, use_cost_l=use_cost_l, epsilon=epsilon)
                custom_adv_calculated = True

            # --- Frame-Aware Advantage: uses predictor frame metrics ---
            elif "frameaware" in use_cost_l:
                base_signal = current_accs if "acc" in use_cost_l else s_scores_tensor.float()
                base_adv = _group_zscore(base_signal, eps=epsilon)
                difficulty = (1.0 - current_accs).clamp(0.0, 1.0)
                base_adv = _hadw_reweight_advantages(base_adv, difficulty, use_cost_l=use_cost_l, epsilon=epsilon)

                # Parse hyperparameters
                alpha_match = re.search(r"alpha([0-9]*\.?[0-9]+)", use_cost_l)
                alpha_redundancy = float(alpha_match.group(1)) if alpha_match else 0.3
                
                beta_match = re.search(r"beta([0-9]*\.?[0-9]+)", use_cost_l)
                beta_uniqueness = float(beta_match.group(1)) if beta_match else 0.2
                
                gamma_match = re.search(r"gamma([0-9]*\.?[0-9]+)", use_cost_l)
                gamma_relevance = float(gamma_match.group(1)) if gamma_match else 0.2

                # Get frame metrics (sid-level aggregation)
                if frame_metrics is not None and scale_mask is not None:
                    # print("frame_metrics", frame_metrics)
                    # Aggregate frame metrics to sid-level
                    redundancy_avg = _compute_sid_metric_avg(frame_metrics.get("redundancy"), scale_mask, sid, sids_list)
                    uniqueness_avg = _compute_sid_metric_avg(frame_metrics.get("uniqueness"), scale_mask, sid, sids_list)
                    relevance_avg = _compute_sid_metric_avg(frame_metrics.get("text_relevance"), scale_mask, sid, sids_list)
                else:
                    # Fallback: no frame metrics available
                    redundancy_avg = torch.zeros(M, device=device, dtype=dtype)
                    uniqueness_avg = torch.full((M,), 0.5, device=device, dtype=dtype)
                    relevance_avg = torch.zeros(M, device=device, dtype=dtype)

                # Normalize scale to [0, 1]
                scales_norm = cost_ratio.clamp(0.0, 1.0)

                # === Frame-Aware Bonus ===
                # When CORRECT: reward efficiency (small scale on redundant frames)
                #   bonus = α * redundancy * (1 - scale) - small penalty for large scale on redundant
                # When WRONG: encourage exploration (larger scale on unique/relevant frames)
                #   bonus = β * uniqueness * scale + γ * relevance * scale

                correct_bonus = alpha_redundancy * redundancy_avg * (1.0 - scales_norm)
                wrong_bonus = beta_uniqueness * uniqueness_avg * scales_norm + gamma_relevance * relevance_avg.clamp(0, 1) * scales_norm
                
                frame_aware_bonus = torch.where(is_correct_mask, correct_bonus, wrong_bonus)
                
                advs = base_adv + (encouragement_coef * frame_aware_bonus).to(base_adv.dtype)
                
                # Optional gas tax (penalize scale cost)
                gas_match = re.search(r"gas([0-9]*\\.?[0-9]+)", use_cost_l)
                gas_tax = float(gas_match.group(1)) if gas_match else 0.02
                advs = advs - gas_tax * cost_ratio.to(base_adv.dtype)
                
                # Ensure correct answers have non-negative advantage
                if is_correct_mask.any() and "noclamp" not in use_cost_l:
                    advs = torch.where(is_correct_mask, advs.clamp(min=0.001), advs)
                
                custom_adv_calculated = True

            # --- Frame-New Advantage: improved frame-aware design with symmetric incentives ---
            elif "frame_new" in use_cost_l:
                base_signal = current_accs if "acc" in use_cost_l else s_scores_tensor.float()
                base_adv = _group_zscore(base_signal, eps=epsilon)

                alpha_match = re.search(r"alpha([0-9]*\.?[0-9]+)", use_cost_l)
                beta_match = re.search(r"beta([0-9]*\.?[0-9]+)", use_cost_l)
                gamma_match = re.search(r"gamma([0-9]*\.?[0-9]+)", use_cost_l)
                delta_match = re.search(r"delta([0-9]*\.?[0-9]+)", use_cost_l)
                alpha_redundancy = float(alpha_match.group(1)) if alpha_match else 0.3
                beta_uniqueness = float(beta_match.group(1)) if beta_match else 0.5
                gamma_relevance = float(gamma_match.group(1)) if gamma_match else 0.3
                delta_info = float(delta_match.group(1)) if delta_match else 0.2

                if frame_metrics is not None and scale_mask is not None:
                    redundancy = frame_metrics.get("redundancy")
                    uniqueness = frame_metrics.get("uniqueness")
                    relevance = frame_metrics.get("text_relevance")
                    info = frame_metrics.get("info_score")
                else:
                    redundancy = None
                    uniqueness = None
                    relevance = None
                    info = None

                def _to01(x, default_val):
                    if x is None:
                        return torch.full_like(cost_ratio, default_val)
                    return x.clamp(0.0, 1.0)

                r = _to01(redundancy, 0.0)
                u = _to01(uniqueness, 0.5)
                rel = _to01(((relevance + 1.0) * 0.5) if relevance is not None else None, 0.5)
                inf = _to01(info, 0.5)

                scales_norm = cost_ratio.clamp(0.0, 1.0)
                min_s = 0.25
                max_s = 1.0
                score = (alpha_redundancy * (1.0 - r) + beta_uniqueness * u + gamma_relevance * rel + delta_info * inf)
                denom = (alpha_redundancy + beta_uniqueness + gamma_relevance + delta_info + 1e-6)
                score = score / denom
                ideal = min_s + score * (max_s - min_s)
                ideal = ideal.clamp(min_s, max_s)

                delta_h = 0.1
                diff = scales_norm - ideal
                abs_diff = diff.abs()
                huber = torch.where(abs_diff <= delta_h, 0.5 * abs_diff * abs_diff / delta_h, abs_diff - 0.5 * delta_h)
                bonus = -huber

                is_correct = (s_scores_tensor > 0.35)
                bonus = torch.where(is_correct, bonus, 0.1 * bonus)

                gas_match = re.search(r"gas([0-9]*\\.?[0-9]+)", use_cost_l)
                gas_tax = float(gas_match.group(1)) if gas_match else 0.05

                total_cost_penalty = torch.zeros_like(cost_ratio)
                if "budget" in use_cost_l:
                    budget_match = re.search(r"budget([0-9]*\.?[0-9]+)", use_cost_l)
                    budget_target = float(budget_match.group(1)) if budget_match else 1.0
                    total_cost = scales_norm.mean(dim=-1) if scales_norm.dim() > 0 else scales_norm
                    total_cost_penalty = F.relu(total_cost - budget_target) * 0.5
                    if total_cost_penalty.dim() < bonus.dim():
                        total_cost_penalty = total_cost_penalty.unsqueeze(-1).expand_as(bonus)

                advs = base_adv + (encouragement_coef * bonus).to(base_adv.dtype)
                advs = advs - gas_tax * cost_ratio.to(base_adv.dtype) - total_cost_penalty.to(base_adv.dtype)
                advs = advs.clamp(min=-2.0, max=2.0)
                custom_adv_calculated = True
            elif "frame_rank" in use_cost_l and frame_metrics is not None and scale_mask is not None and scales is not None:
                base_signal = current_accs if "acc" in use_cost_l else s_scores_tensor.float()
                wu_match = re.search(r"wu([0-9]*\.?[0-9]+)", use_cost_l)
                wr_match = re.search(r"wr([0-9]*\.?[0-9]+)", use_cost_l)
                wrel_match = re.search(r"wrel([0-9]*\.?[0-9]+)", use_cost_l)
                winf_match = re.search(r"winf([0-9]*\.?[0-9]+)", use_cost_l)
                lam_match = re.search(r"lam([0-9]*\.?[0-9]+)", use_cost_l)
                rho_match = re.search(r"rho([0-9]*\.?[0-9]+)", use_cost_l)
                eta_match = re.search(r"eta([0-9]*\.?[0-9]+)", use_cost_l)
                smin_match = re.search(r"smin([0-9]*\.?[0-9]+)", use_cost_l)
                smax_match = re.search(r"smax([0-9]*\.?[0-9]+)", use_cost_l)
                deltah_match = re.search(r"delta([0-9]*\.?[0-9]+)", use_cost_l)
                margin_match = re.search(r"margin([0-9]*\.?[0-9]+)", use_cost_l)
                krank_match = re.search(r"krank([0-9]+)", use_cost_l)
                correct_match = re.search(r"correct([0-9]*\.?[0-9]+)", use_cost_l)
                wrong_eta_match = re.search(r"wrongeta([0-9]*\.?[0-9]+)", use_cost_l)
                wrongexp_match = re.search(r"wrongexp([0-9]*\.?[0-9]+)", use_cost_l)
                wu = float(wu_match.group(1)) if wu_match else 1.0
                wr = float(wr_match.group(1)) if wr_match else 1.0
                wrel = float(wrel_match.group(1)) if wrel_match else 1.0
                winf = float(winf_match.group(1)) if winf_match else 0.5
                lam = float(lam_match.group(1)) if lam_match else 0.5
                rho = float(rho_match.group(1)) if rho_match else 0.3
                eta = float(eta_match.group(1)) if eta_match else 0.1
                smin = float(smin_match.group(1)) if smin_match else 0.25
                smax = float(smax_match.group(1)) if smax_match else 1.0
                delta_h = float(deltah_match.group(1)) if deltah_match else 0.1
                margin = float(margin_match.group(1)) if margin_match else 0.1
                k_rank = int(krank_match.group(1)) if krank_match else 4
                correct_th = float(correct_match.group(1)) if correct_match else 0.35
                wrong_eta = float(wrong_eta_match.group(1)) if wrong_eta_match else 0.3
                wrong_exp = float(wrongexp_match.group(1)) if wrongexp_match else 0.2
                sid_index_map: Dict[Any, List[int]] = defaultdict(list)
                for idx_b in range(bsz):
                    s_key_b = _as_py_key(sid[idx_b])
                    sid_index_map[s_key_b].append(idx_b)
                base_signal_M = base_signal.to(device=device, dtype=base_signal.dtype)
                base_adv = _group_zscore(base_signal_M, eps=epsilon)
                per_sid_frame_advs: Dict[Any, torch.Tensor] = {}
                max_len = int(scale_mask.shape[1])
                for k, s_key in enumerate(sids_list):
                    idxs = sid_index_map.get(s_key, [])
                    if not idxs:
                        per_sid_frame_advs[s_key] = torch.zeros((max_len,), device=device, dtype=dtype)
                        continue
                    valid_row = _compute_sid_frame_valid_mask(scale_mask, idxs)
                    fm_r = frame_metrics.get("redundancy", None)
                    fm_u = frame_metrics.get("uniqueness", None)
                    fm_rel = frame_metrics.get("text_relevance", None)
                    fm_inf = frame_metrics.get("info_score", None)
                    r = _aggregate_sid_frame_values(
                        fm_r,
                        scale_mask,
                        idxs,
                        default=0.0,
                        device=device,
                        dtype=dtype,
                    ) if fm_r is not None else torch.zeros((max_len,), device=device, dtype=dtype)
                    u = _aggregate_sid_frame_values(
                        fm_u,
                        scale_mask,
                        idxs,
                        default=0.5,
                        device=device,
                        dtype=dtype,
                    ) if fm_u is not None else torch.full((max_len,), 0.5, device=device, dtype=dtype)
                    rel_raw = _aggregate_sid_frame_values(
                        fm_rel,
                        scale_mask,
                        idxs,
                        default=0.0,
                        device=device,
                        dtype=dtype,
                    ) if fm_rel is not None else torch.zeros((max_len,), device=device, dtype=dtype)
                    inf = _aggregate_sid_frame_values(
                        fm_inf,
                        scale_mask,
                        idxs,
                        default=0.5,
                        device=device,
                        dtype=dtype,
                    ) if fm_inf is not None else torch.full((max_len,), 0.5, device=device, dtype=dtype)
                    r = r.clamp(0.0, 1.0)
                    u = u.clamp(0.0, 1.0)
                    rel = ((rel_raw + 1.0) * 0.5).clamp(0.0, 1.0)
                    inf = inf.clamp(0.0, 1.0)
                    scales_row = _aggregate_sid_frame_values(
                        scales,
                        scale_mask,
                        idxs,
                        default=min_scale,
                        device=device,
                        dtype=dtype,
                    )
                    scales01 = ((scales_row - min_scale) / (max_scale - min_scale + 1e-6)).clamp(0.0, 1.0)
                    imp = wu * u + wrel * rel + winf * inf - wr * r
                    denom = wu + wrel + winf + wr + 1e-6
                    imp = (imp / denom).clamp(0.0, 1.0)
                    ideal01 = (smin + imp * (smax - smin)).clamp(smin, smax)
                    diff = scales01 - ideal01
                    abs_diff = diff.abs()
                    huber = torch.where(abs_diff <= delta_h, 0.5 * abs_diff * abs_diff / delta_h, abs_diff - 0.5 * delta_h)
                    shape_bonus = -huber
                    sid_score = current_accs[k] if "acc" in use_cost_l else s_scores_tensor[k]
                    is_correct = (sid_score > correct_th)
                    if not is_correct:
                        shape_bonus = 0.1 * shape_bonus + wrong_exp * imp
                    shaping = lam * shape_bonus - (eta if is_correct else (eta * wrong_eta)) * scales01
                    shaping = torch.where(valid_row, shaping, torch.zeros_like(shaping))
                    count = valid_row.float().sum().clamp(min=1.0)
                    mean = (shaping * valid_row.float()).sum() / count
                    var = (((shaping - mean) * valid_row.float()) ** 2).sum() / count
                    std = (var + 1e-6).sqrt()
                    shaping = torch.where(valid_row, (shaping - mean) / std, shaping)
                    rank_vec = torch.zeros_like(scales01)
                    if k_rank > 0:
                        valid_idx = torch.nonzero(valid_row, as_tuple=False).flatten()
                        if valid_idx.numel() >= 2:
                            perm = valid_idx[torch.randperm(valid_idx.numel(), device=device)]
                            count_pairs = min(k_rank, perm.numel() // 2)
                            for j in range(count_pairs):
                                t1 = perm[j]
                                t2 = perm[(j + 1) % perm.numel()]
                                imp_diff = imp[t1] - imp[t2]
                                sign = torch.sign(imp_diff)
                                sdiff = scales01[t1] - scales01[t2]
                                term = F.softplus(margin - sign * sdiff)
                                if sign > 0:
                                    rank_vec[t1] = rank_vec[t1] + term
                                    rank_vec[t2] = rank_vec[t2] - term
                                elif sign < 0:
                                    rank_vec[t1] = rank_vec[t1] - term
                                    rank_vec[t2] = rank_vec[t2] + term
                            if count_pairs > 0:
                                rank_vec = rank_vec / float(count_pairs)
                    adv_vec = torch.where(valid_row, base_adv[k] + shaping - rho * rank_vec, torch.zeros_like(shaping))
                    per_sid_frame_advs[s_key] = adv_vec
                    if per_sid_frame_adv_all is not None:
                        per_sid_frame_adv_all[s_key] = adv_vec
                frame_adv = torch.stack([per_sid_frame_advs[s_key] for s_key in sids_list], dim=0)
                valid_mask_M = torch.stack(
                    [
                        _compute_sid_frame_valid_mask(scale_mask, sid_index_map.get(s_key, []))
                        if sid_index_map.get(s_key)
                        else torch.zeros((max_len,), device=device, dtype=torch.bool)
                        for s_key in sids_list
                    ],
                    dim=0,
                )
                denom = valid_mask_M.float().sum(dim=-1).clamp(min=1.0)
                advs = (frame_adv * valid_mask_M.float()).sum(dim=-1) / denom
                custom_adv_calculated = True
            elif "frame_ideal" in use_cost_l and frame_metrics is not None and scale_mask is not None and scales is not None:
                base_signal = current_accs if "acc" in use_cost_l else s_scores_tensor.float()
                wu_match = re.search(r"wu([0-9]*\.?[0-9]+)", use_cost_l)
                wr_match = re.search(r"wr([0-9]*\.?[0-9]+)", use_cost_l)
                wrel_match = re.search(r"wrel([0-9]*\.?[0-9]+)", use_cost_l)
                winf_match = re.search(r"winf([0-9]*\.?[0-9]+)", use_cost_l)
                lam_match = re.search(r"lam([0-9]*\.?[0-9]+)", use_cost_l)
                eta_match = re.search(r"eta([0-9]*\.?[0-9]+)", use_cost_l)
                smin_match = re.search(r"smin([0-9]*\.?[0-9]+)", use_cost_l)
                smax_match = re.search(r"smax([0-9]*\.?[0-9]+)", use_cost_l)
                deltah_match = re.search(r"delta([0-9]*\.?[0-9]+)", use_cost_l)
                correct_match = re.search(r"correct([0-9]*\.?[0-9]+)", use_cost_l)
                wu = float(wu_match.group(1)) if wu_match else 1.0
                wr = float(wr_match.group(1)) if wr_match else 1.0
                wrel = float(wrel_match.group(1)) if wrel_match else 1.0
                winf = float(winf_match.group(1)) if winf_match else 0.5
                lam = float(lam_match.group(1)) if lam_match else 0.5
                eta = float(eta_match.group(1)) if eta_match else 0.1
                smin = float(smin_match.group(1)) if smin_match else 0.25
                smax = float(smax_match.group(1)) if smax_match else 1.0
                delta_h = float(deltah_match.group(1)) if deltah_match else 0.1
                correct_th = float(correct_match.group(1)) if correct_match else 0.35
                sid_index_map: Dict[Any, List[int]] = defaultdict(list)
                for idx_b in range(bsz):
                    s_key_b = _as_py_key(sid[idx_b])
                    sid_index_map[s_key_b].append(idx_b)
                base_signal_M = base_signal.to(device=device, dtype=base_signal.dtype)
                base_adv = _group_zscore(base_signal_M, eps=epsilon)
                per_sid_frame_advs: Dict[Any, torch.Tensor] = {}
                max_len = int(scale_mask.shape[1])
                for k, s_key in enumerate(sids_list):
                    idxs = sid_index_map.get(s_key, [])
                    if not idxs:
                        per_sid_frame_advs[s_key] = torch.zeros((max_len,), device=device, dtype=dtype)
                        continue
                    valid_row = _compute_sid_frame_valid_mask(scale_mask, idxs)
                    fm_r = frame_metrics.get("redundancy", None)
                    fm_u = frame_metrics.get("uniqueness", None)
                    fm_rel = frame_metrics.get("text_relevance", None)
                    fm_inf = frame_metrics.get("info_score", None)
                    r = _aggregate_sid_frame_values(
                        fm_r,
                        scale_mask,
                        idxs,
                        default=0.0,
                        device=device,
                        dtype=dtype,
                    ) if fm_r is not None else torch.zeros((max_len,), device=device, dtype=dtype)
                    u = _aggregate_sid_frame_values(
                        fm_u,
                        scale_mask,
                        idxs,
                        default=0.5,
                        device=device,
                        dtype=dtype,
                    ) if fm_u is not None else torch.full((max_len,), 0.5, device=device, dtype=dtype)
                    rel_raw = _aggregate_sid_frame_values(
                        fm_rel,
                        scale_mask,
                        idxs,
                        default=0.0,
                        device=device,
                        dtype=dtype,
                    ) if fm_rel is not None else torch.zeros((max_len,), device=device, dtype=dtype)
                    inf = _aggregate_sid_frame_values(
                        fm_inf,
                        scale_mask,
                        idxs,
                        default=0.5,
                        device=device,
                        dtype=dtype,
                    ) if fm_inf is not None else torch.full((max_len,), 0.5, device=device, dtype=dtype)
                    # Normalize to [0,1]
                    r = r.clamp(0.0, 1.0)
                    u = u.clamp(0.0, 1.0)
                    rel = ((rel_raw + 1.0) * 0.5).clamp(0.0, 1.0)
                    inf = inf.clamp(0.0, 1.0)
                    scales_row = _aggregate_sid_frame_values(
                        scales,
                        scale_mask,
                        idxs,
                        default=min_scale,
                        device=device,
                        dtype=dtype,
                    )
                    scales01 = ((scales_row - min_scale) / (max_scale - min_scale + 1e-6)).clamp(0.0, 1.0)
                    # Importance and ideal scale
                    imp = wu * u + wrel * rel + winf * inf - wr * r
                    denom = wu + wrel + winf + wr + 1e-6
                    imp = (imp / denom).clamp(0.0, 1.0)
                    ideal01 = (smin + imp * (smax - smin)).clamp(smin, smax)
                    # Huber shaping
                    diff = scales01 - ideal01
                    abs_diff = diff.abs()
                    huber = torch.where(abs_diff <= delta_h, 0.5 * abs_diff * abs_diff / delta_h, abs_diff - 0.5 * delta_h)
                    shape_bonus = -huber
                    sid_score = current_accs[k] if "acc" in use_cost_l else s_scores_tensor[k]
                    is_correct = (sid_score > correct_th)
                    shape_bonus = shape_bonus if is_correct else (0.1 * shape_bonus)
                    shaping = lam * shape_bonus - eta * scales01
                    shaping = torch.where(valid_row, shaping, torch.zeros_like(shaping))
                    count = valid_row.float().sum().clamp(min=1.0)
                    mean = (shaping * valid_row.float()).sum() / count
                    var = (((shaping - mean) * valid_row.float()) ** 2).sum() / count
                    std = (var + 1e-6).sqrt()
                    shaping = torch.where(valid_row, (shaping - mean) / std, shaping)
                    adv_vec = torch.where(valid_row, base_adv[k] + shaping, torch.zeros_like(shaping))
                    per_sid_frame_advs[s_key] = adv_vec
                    if per_sid_frame_adv_all is not None:
                        per_sid_frame_adv_all[s_key] = adv_vec
                frame_adv = torch.stack([per_sid_frame_advs[s_key] for s_key in sids_list], dim=0)
                valid_mask_M = torch.stack(
                    [
                        _compute_sid_frame_valid_mask(scale_mask, sid_index_map.get(s_key, []))
                        if sid_index_map.get(s_key)
                        else torch.zeros((max_len,), device=device, dtype=torch.bool)
                        for s_key in sids_list
                    ],
                    dim=0,
                )
                denom = valid_mask_M.float().sum(dim=-1).clamp(min=1.0)
                advs = (frame_adv * valid_mask_M.float()).sum(dim=-1) / denom
                custom_adv_calculated = True

            elif "multiply" in use_cost_l:
                s_scores_tensor = s_scores_tensor + efficiency_bonus.to(s_scores_tensor.dtype)

            elif "absolute" in use_cost_l:
                denom = (max_scale - min_scale) + 1e-6
                waste_ratio = ((costs - min_scale) / denom).clamp(min=0.0, max=1.0)
                penalty_term = penalty_coef * waste_ratio
                deduction = current_accs * penalty_term
                s_scores_tensor = s_scores_tensor - deduction.to(s_scores_tensor.dtype)

            elif "penalty" in use_cost_l:
                if is_correct_mask.any():
                    min_effective_cost = costs[is_correct_mask].min()
                    is_wasteful_mask = (costs > min_effective_cost)
                    if is_wasteful_mask.any():
                        waste_amount = (costs - min_effective_cost).clamp(min=0.0)
                        penalty = penalty_coef * waste_amount
                        s_scores_tensor = s_scores_tensor - (penalty.to(s_scores_tensor.dtype) * is_wasteful_mask.float())

                    is_insufficient_mask = (costs < min_effective_cost)
                    if is_insufficient_mask.any():
                        bonus = encouragement_coef * costs
                        s_scores_tensor = s_scores_tensor + (bonus.to(s_scores_tensor.dtype) * is_insufficient_mask.float())
                else:
                    bonus = encouragement_coef * costs
                    s_scores_tensor = s_scores_tensor + bonus.to(s_scores_tensor.dtype)

        if custom_adv_calculated and use_cost_l is not None and "norm" in use_cost_l:
            advs = _group_zscore(advs, eps=epsilon)

        if not custom_adv_calculated:
            if norm_adv and M > 2:
                advs = _group_zscore(s_scores_tensor, eps=epsilon)
            else:
                advs = s_scores_tensor - s_scores_tensor.mean()

        for k, val in enumerate(advs):
            sid2advantage[sids_list[k]] = val.to(device=device, dtype=dtype)

    adv_base = torch.zeros((bsz,), dtype=dtype, device=device)
    predictor_update_mask = torch.zeros((bsz,), dtype=torch.bool, device=device)

    seen_sids = set()
    for i in range(bsz):
        s_key = _as_py_key(sid[i])
        adv_val = sid2advantage.get(s_key, torch.tensor(0.0, device=device, dtype=dtype))
        adv_base[i] = adv_val

        should_update = True
        if filter_invalid_sid:
            if s_key in seen_sids:
                should_update = False
            else:
                seen_sids.add(s_key)
        predictor_update_mask[i] = bool(should_update)

    sid2norm: Dict[Any, torch.Tensor] = {}

    # ---- Batch-wise normalization (Eq.6), over unique sid by default ----
    if batch_norm_adv:
        if filter_invalid_sid:
            idx = predictor_update_mask.nonzero(as_tuple=False).view(-1)
            if idx.numel() > 0:
                norm_vals = _batch_zscore(adv_base[idx].float(), eps=epsilon).to(dtype)
                adv_base[idx] = norm_vals
                # propagate normalized value to duplicates (same sid)
                # build sid -> normalized adv using the first occurrence
                for j in idx.tolist():
                    sid2norm[_as_py_key(sid[j])] = adv_base[j]
                for i in range(bsz):
                    s_key = _as_py_key(sid[i])
                    if s_key in sid2norm:
                        adv_base[i] = sid2norm[s_key]
        else:
            # Build unique sid list and normalize on those, then broadcast back
            unique_sids = list(sid2advantage.keys())
            if len(unique_sids) > 0:
                uniq_vals = torch.stack([sid2advantage[s].to(device=device, dtype=torch.float32) for s in unique_sids])
                uniq_vals_norm = _batch_zscore(uniq_vals, eps=epsilon).to(dtype)
                sid2norm = {s: uniq_vals_norm[k] for k, s in enumerate(unique_sids)}
                for i in range(bsz):
                    s_key = _as_py_key(sid[i])
                    adv_base[i] = sid2norm.get(s_key, torch.tensor(0.0, device=device, dtype=dtype))

        if per_sid_frame_adv_all:
            for s_key, adv_vec in list(per_sid_frame_adv_all.items()):
                raw_sid_adv = sid2advantage.get(s_key)
                norm_sid_adv = sid2norm.get(s_key)
                if raw_sid_adv is None or norm_sid_adv is None:
                    continue
                adv_vec = adv_vec.to(device=device, dtype=dtype)
                centered = adv_vec - raw_sid_adv.to(device=device, dtype=dtype)
                per_sid_frame_adv_all[s_key] = centered + norm_sid_adv.to(device=device, dtype=dtype)

    if scale_mask is not None:
        if per_sid_frame_adv_all:
            predictor_advantages = torch.zeros_like(scale_mask, dtype=dtype)
            for i in range(bsz):
                s_key = _as_py_key(sid[i])
                adv_vec = per_sid_frame_adv_all.get(s_key)
                if adv_vec is None or adv_vec.numel() != predictor_advantages.shape[1]:
                    predictor_advantages[i] = adv_base[i] * scale_mask[i].float()
                else:
                    valid_mask = scale_mask[i].to(torch.bool)
                    predictor_advantages[i] = torch.where(
                        valid_mask,
                        adv_vec.to(device=device, dtype=dtype),
                        torch.zeros_like(adv_vec, dtype=dtype),
                    )
        else:
            predictor_advantages = adv_base.unsqueeze(-1) * scale_mask.float()
        predictor_advantages = predictor_advantages.to(dtype=dtype)
    else:
        predictor_advantages = adv_base

    return predictor_advantages, predictor_update_mask