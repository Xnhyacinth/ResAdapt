import torch
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict
import os
import re
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLMultiModalProcessor

from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

import base64
from io import BytesIO
import copy
import math

from torchvision.transforms import InterpolationMode
from transformers.image_transforms import resize
from transformers.image_utils import (
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    to_numpy_array,
    ChannelDimension,
    SizeDict
)

import torch.nn.functional as F

TensorLike = Union[torch.Tensor, np.ndarray, list]

IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768
MAX_PIXELS = 14 * 14 * 4 * 16384

_HADW_STATE: Dict[str, float] = {}


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

        sample_costs = torch.tensor(sample_mean_scales, device=actions.device, dtype=torch.float32)

        # map mean(scale) -> [0, 1] ratio
        sample_costs = (sample_costs - min_scale) / (max_scale - min_scale + 1e-6)
        sample_costs = sample_costs.clamp(0.0, 1.0)

        for i in range(bsz):
            s_key = _as_py_key(sid[i])
            if s_key not in sid2cost:
                sid2cost[s_key] = float(sample_costs[i].item())

    sid2advantage: Dict[Any, torch.Tensor] = {}
    per_sid_frame_adv_all: Dict[Any, torch.Tensor] | None = (
        {} if use_cost_l is not None and any(key in use_cost_l for key in ("frame_ideal", "frame_rank")) and scale_mask is not None and scales is not None else None
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
            is_correct_mask = (s_scores_tensor > 0.35)

            current_accs = _safe_tensor_from_list([float(sid2acc_avg[sid_].item()) for sid_ in sids_list],
                                                  device=device, dtype=torch.float32)

            # reused by tie/multiply
            centered_ratio = (1.0 - cost_ratio) - centered_term if "cen" in use_cost_l else (1.0 - cost_ratio)
            efficiency_bonus = encouragement_coef * current_accs * centered_ratio  # (M,)

            # --- TIE: base(normed) + bonus(abs) ---
            if "tie" in use_cost_l:
                base_signal = current_accs if "acc" in use_cost_l else s_scores_tensor.float()
                base_adv = _group_zscore(base_signal, eps=epsilon)
                difficulty = (1.0 - current_accs).clamp(0.0, 1.0)
                base_adv = _hadw_reweight_advantages(base_adv, difficulty, use_cost_l=use_cost_l, epsilon=epsilon)

                if "newtie" in use_cost_l:
                    alpha_match = re.search(r"alpha([0-9]*\.?[0-9]+)", use_cost_l)
                    alpha = float(alpha_match.group(1)) if alpha_match else 0.5
                    alpha = max(0.0, min(1.0, alpha))

                    mean_cost = cost_ratio.mean().detach()
                    fixed_target = torch.tensor(float(centered_term), device=cost_ratio.device, dtype=cost_ratio.dtype)
                    pivot = alpha * mean_cost + (1.0 - alpha) * fixed_target

                    tau_match = re.search(r"tau([0-9]*\.?[0-9]+)", use_cost_l)
                    tau = float(tau_match.group(1)) if tau_match else 0.15
                    tau = max(1e-3, tau)

                    reward_small = torch.sigmoid((pivot - cost_ratio) / tau) - 0.5
                    reward_big = torch.sigmoid((cost_ratio - pivot) / tau) - 0.5
                    
                    # scale_shape = torch.where(
                    #     is_correct_mask,
                    #     current_accs * reward_small,
                    #     difficulty * reward_big,
                    # )

                    accpow_match = re.search(r"accpow([0-9]*\.?[0-9]+)", use_cost_l)
                    acc_pow = float(accpow_match.group(1)) if accpow_match else 0.5
                    acc_pow = max(0.0, acc_pow)

                    accfloor_match = re.search(r"accfloor([0-9]*\.?[0-9]+)", use_cost_l)
                    acc_floor = float(accfloor_match.group(1)) if accfloor_match else 0.05
                    acc_floor = max(0.0, min(1.0, acc_floor))

                    acc_weight = current_accs.clamp(0.0, 1.0).pow(acc_pow).clamp(min=acc_floor)

                    if "asym" in use_cost_l:
                        wrongpow_match = re.search(r"wrongpow([0-9]*\.?[0-9]+)", use_cost_l)
                        wrong_pow = float(wrongpow_match.group(1)) if wrongpow_match else 1.0
                        wrong_pow = max(0.0, wrong_pow)

                        rightscale_match = re.search(r"rightscale([0-9]*\.?[0-9]+)", use_cost_l)
                        right_scale = float(rightscale_match.group(1)) if rightscale_match else 0.7
                        right_scale = max(0.0, right_scale)

                        wrongscale_match = re.search(r"wrongscale([0-9]*\.?[0-9]+)", use_cost_l)
                        wrong_scale = float(wrongscale_match.group(1)) if wrongscale_match else 1.3
                        wrong_scale = max(0.0, wrong_scale)

                        wrong_weight = difficulty.pow(wrong_pow)
                        scale_shape = torch.where(
                            is_correct_mask,
                            right_scale * acc_weight * reward_small,
                            wrong_scale * wrong_weight * reward_big,
                        )
                    else:
                        scale_shape = torch.where(
                            is_correct_mask,
                            acc_weight * reward_small,
                            difficulty * reward_big,
                        )

                    pull_match = re.search(r"pull([0-9]*\.?[0-9]+)", use_cost_l)
                    pull_coef = float(pull_match.group(1)) if pull_match else 0.0
                    pull_coef = max(0.0, pull_coef)
                    if pull_coef > 0.0:
                        scale_shape = scale_shape - pull_coef * (cost_ratio - centered_term)

                    scalecap_match = re.search(r"scalecap([0-9]*\.?[0-9]+)", use_cost_l)
                    if scalecap_match:
                        scale_cap = float(scalecap_match.group(1))
                        scale_cap = max(0.0, scale_cap)
                        if scale_cap > 0.0:
                            scale_shape = scale_shape.clamp(min=-scale_cap, max=scale_cap)
                    
                    gas_match = re.search(r"gas([0-9]*\.?[0-9]+)", use_cost_l)
                    gas_tax = float(gas_match.group(1)) if gas_match else 0.05
                    gas_tax = max(0.0, gas_tax)

                    advs = base_adv + (encouragement_coef * scale_shape).to(base_adv.dtype) - gas_tax * cost_ratio.to(base_adv.dtype)

                    if is_correct_mask.any() and "noclamp" not in use_cost_l:
                        advs = torch.where(is_correct_mask, advs.clamp(min=0.001), advs)
                else:
                    advs = base_adv + efficiency_bonus.to(base_adv.dtype)

                # advs = _hadw_reweight_advantages(advs, difficulty, use_cost_l=use_cost_l, epsilon=epsilon)
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
                base_signal_M = torch.stack(
                    [
                        base_signal[sid_index_map[s_key][0]] if sid_index_map.get(s_key) else torch.tensor(0.0, device=device, dtype=base_signal.dtype)
                        for s_key in sids_list
                    ]
                ).to(device)
                base_adv = _group_zscore(base_signal_M, eps=epsilon)
                per_sid_frame_advs: Dict[Any, torch.Tensor] = {}
                max_len = int(scale_mask.shape[1])
                for k, s_key in enumerate(sids_list):
                    idxs = sid_index_map.get(s_key, [])
                    if not idxs:
                        per_sid_frame_advs[s_key] = torch.zeros((max_len,), device=device, dtype=dtype)
                        continue
                    idx0 = idxs[0]
                    valid_row = scale_mask[idx0].to(torch.bool)
                    fm_r = frame_metrics.get("redundancy", None)
                    fm_u = frame_metrics.get("uniqueness", None)
                    fm_rel = frame_metrics.get("text_relevance", None)
                    fm_inf = frame_metrics.get("info_score", None)
                    r = fm_r[idx0] if fm_r is not None else torch.zeros((max_len,), device=device, dtype=dtype)
                    u = fm_u[idx0] if fm_u is not None else torch.full((max_len,), 0.5, device=device, dtype=dtype)
                    rel_raw = fm_rel[idx0] if fm_rel is not None else torch.zeros((max_len,), device=device, dtype=dtype)
                    inf = fm_inf[idx0] if fm_inf is not None else torch.full((max_len,), 0.5, device=device, dtype=dtype)
                    r = r.clamp(0.0, 1.0)
                    u = u.clamp(0.0, 1.0)
                    rel = ((rel_raw + 1.0) * 0.5).clamp(0.0, 1.0)
                    inf = inf.clamp(0.0, 1.0)
                    scales_row = scales[idx0]
                    scales01 = ((scales_row - min_scale) / (max_scale - min_scale + 1e-6)).clamp(0.0, 1.0)
                    imp = wu * u + wrel * rel + winf * inf - wr * r
                    denom = wu + wrel + winf + wr + 1e-6
                    imp = (imp / denom).clamp(0.0, 1.0)
                    ideal01 = (smin + imp * (smax - smin)).clamp(smin, smax)
                    diff = scales01 - ideal01
                    abs_diff = diff.abs()
                    huber = torch.where(abs_diff <= delta_h, 0.5 * abs_diff * abs_diff / delta_h, abs_diff - 0.5 * delta_h)
                    shape_bonus = -huber
                    is_correct = (scores_1d[idx0] > correct_th)
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
                    [scale_mask[sid_index_map[s_key][0]] if sid_index_map.get(s_key) else torch.zeros((max_len,), device=device, dtype=torch.bool) for s_key in sids_list],
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
                base_signal_M = torch.stack(
                    [
                        base_signal[sid_index_map[s_key][0]] if sid_index_map.get(s_key) else torch.tensor(0.0, device=device, dtype=base_signal.dtype)
                        for s_key in sids_list
                    ]
                ).to(device)
                base_adv = _group_zscore(base_signal_M, eps=epsilon)
                per_sid_frame_advs: Dict[Any, torch.Tensor] = {}
                max_len = int(scale_mask.shape[1])
                # Per-sid loop (use first occurrence)
                for k, s_key in enumerate(sids_list):
                    idxs = sid_index_map.get(s_key, [])
                    if not idxs:
                        per_sid_frame_advs[s_key] = torch.zeros((max_len,), device=device, dtype=dtype)
                        continue
                    idx0 = idxs[0]
                    valid_row = scale_mask[idx0].to(torch.bool)  # (T)
                    fm_r = frame_metrics.get("redundancy", None)
                    fm_u = frame_metrics.get("uniqueness", None)
                    fm_rel = frame_metrics.get("text_relevance", None)
                    fm_inf = frame_metrics.get("info_score", None)
                    r = fm_r[idx0] if fm_r is not None else torch.zeros((max_len,), device=device, dtype=dtype)
                    u = fm_u[idx0] if fm_u is not None else torch.full((max_len,), 0.5, device=device, dtype=dtype)
                    rel_raw = fm_rel[idx0] if fm_rel is not None else torch.zeros((max_len,), device=device, dtype=dtype)
                    inf = fm_inf[idx0] if fm_inf is not None else torch.full((max_len,), 0.5, device=device, dtype=dtype)
                    # Normalize to [0,1]
                    r = r.clamp(0.0, 1.0)
                    u = u.clamp(0.0, 1.0)
                    rel = ((rel_raw + 1.0) * 0.5).clamp(0.0, 1.0)
                    inf = inf.clamp(0.0, 1.0)
                    scales_row = scales[idx0]
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
                    is_correct = (scores_1d[idx0] > correct_th)
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
                    [scale_mask[sid_index_map[s_key][0]] if sid_index_map.get(s_key) else torch.zeros((max_len,), device=device, dtype=torch.bool) for s_key in sids_list],
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

    # ---- Batch-wise normalization (Eq.6), over unique sid by default ----
    if batch_norm_adv:
        if filter_invalid_sid:
            idx = predictor_update_mask.nonzero(as_tuple=False).view(-1)
            if idx.numel() > 0:
                norm_vals = _batch_zscore(adv_base[idx].float(), eps=epsilon).to(dtype)
                adv_base[idx] = norm_vals
                # propagate normalized value to duplicates (same sid)
                # build sid -> normalized adv using the first occurrence
                sid2norm = {}
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

    if scale_mask is not None:
        if per_sid_frame_adv_all:
            predictor_advantages = torch.zeros_like(scale_mask, dtype=dtype)
            for i in range(bsz):
                s_key = _as_py_key(sid[i])
                adv_vec = per_sid_frame_adv_all.get(s_key)
                if adv_vec is None or adv_vec.numel() != predictor_advantages.shape[1]:
                    predictor_advantages[i] = adv_base[i] * scale_mask[i].float()
                else:
                    predictor_advantages[i] = adv_vec
        else:
            predictor_advantages = adv_base.unsqueeze(-1) * scale_mask.float()
        predictor_advantages = predictor_advantages.to(dtype=dtype)
    else:
        predictor_advantages = adv_base

    return predictor_advantages, predictor_update_mask


def get_images_per_sample(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    merge_len: int,
    image_token_index: int
) -> torch.Tensor:
    tokens_per_image_list = (image_grid_thw.prod(-1) // merge_len).tolist()
    tokens_per_sample_list = (input_ids == image_token_index).sum(dim=1).tolist()
    
    video_per_sample = []
    img_cursor = 0
    
    for sample_idx, total_tokens_in_sample in enumerate(tokens_per_sample_list):
        if total_tokens_in_sample == 0:
            video_per_sample.append(0)
            continue
        
        current_sample_img_count = 0
        current_sample_token_acc = 0
        
        while current_sample_token_acc < total_tokens_in_sample:
            if img_cursor >= len(tokens_per_image_list):
                raise ValueError(f"Mismatch: Input IDs imply more image tokens than vision_grid provides.")
            
            token_len_of_next_img = tokens_per_image_list[img_cursor]
            current_sample_token_acc += token_len_of_next_img
            img_cursor += 1
            current_sample_img_count += 1
        
        if current_sample_token_acc != total_tokens_in_sample:
                raise ValueError(f"Mismatch in Sample {sample_idx}: input_ids image tokens ({total_tokens_in_sample}) "
                                f"do not align with image_grid_thw sizes.")
                                
        video_per_sample.append(current_sample_img_count)
    return video_per_sample


def get_visual_objects_per_sample(
    input_ids: torch.Tensor, 
    vision_start_token_id: int, 
    video_token_id: int, 
    image_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        input_ids: (Batch, Seq_Len)
    Returns:
        image_counts: (Batch,) 
        video_counts: (Batch,) 
    """
    is_start = (input_ids[:, :-1] == vision_start_token_id)
    next_tokens = input_ids[:, 1:]
    
    is_video = is_start & (next_tokens == video_token_id)
    is_image = is_start & (next_tokens == image_token_id)
    
    video_counts = is_video.sum(dim=1)
    image_counts = is_image.sum(dim=1)
    
    return image_counts, video_counts

def regroup_modal_data(
    multi_modal_data: dict, 
    image_counts: torch.Tensor | list, 
    video_counts: torch.Tensor | list
) -> list[dict]:
    all_images = multi_modal_data.get("images")
    if all_images is None: all_images = []
    
    all_videos = multi_modal_data.get("videos")
    if all_videos is None: all_videos = []

    if isinstance(image_counts, torch.Tensor):
        image_counts = image_counts.tolist()
    if isinstance(video_counts, torch.Tensor):
        video_counts = video_counts.tolist()

    batch_modal_list = []
    img_cursor = 0
    vid_cursor = 0

    for n_img, n_vid in zip(image_counts, video_counts):
        n_img, n_vid = int(n_img), int(n_vid)
        sample_dict = {}

        if n_img > 0:
            sample_dict["image"] = all_images[img_cursor : img_cursor + n_img]
            img_cursor += n_img
        else:
            sample_dict["image"] = None 

        if n_vid > 0:
            sample_dict["video"] = all_videos[vid_cursor : vid_cursor + n_vid]
            vid_cursor += n_vid
        else:
            sample_dict["video"] = None

        batch_modal_list.append(sample_dict)

    assert img_cursor == len(all_images)
    assert vid_cursor == len(all_videos)

    return batch_modal_list


_original_init = Qwen2_5_VLMultiModalProcessor.__init__
def __init__(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)

    predictor_path = os.getenv("PREDICTOR_PATH", None)
    print("predictor_path", predictor_path)

    if predictor_path is None:
        # raise ValueError("PREDICTOR_PATH is not set")
        print("PREDICTOR_PATH is not set, use ENABLE_BASELINE_SCALE.")
        if os.environ.get("ENABLE_BASELINE_SCALE", None) is None:
            raise ValueError("ENABLE_BASELINE_SCALE is not set")
        
    else:
        from visionthink.predictor.modeling_predictor import PredictorForConditionalGeneration
        self.predictor = PredictorForConditionalGeneration.from_pretrained(
            predictor_path,
            dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
        self.predictor.eval()


def _apply_hf_processor_main(
    self,
    prompt,
    mm_items,
    hf_processor_mm_kwargs,
    tokenization_kwargs,
    *,
    enable_hf_prompt_update: bool,
):
    """
    Apply the HF processor on the prompt text and multi-modal data.

    In addition, return whether prompt updates have been applied
    (for most HF processors, this should be `True`).

    Note:
        If `enable_hf_prompt_update=False`, we use HF processor
        to perform prompt updates if available; HF processor requires
        that the prompt corresponds to multi-modal items.
    """
    processor_data, _ = self._get_hf_mm_data(mm_items)
    images = processor_data.get("images", None)
    videos = processor_data.get("videos", None)
    if images is not None or videos is not None and hf_processor_mm_kwargs.pop("scale", False):
        mm_data = {}
        if os.environ.get("ENABLE_BASELINE_SCALE", None):
            fixed_scale = float(os.environ.get("BASELINE_SCALE_FACTOR", "1.5"))
            print(f"[Baseline Mode] Scaling all images by factor: {fixed_scale}")

            scaled_images = []
            for img in images:
                w, h = img.size
                new_w = int(w * fixed_scale)
                new_h = int(h * fixed_scale)
                
                new_w = max(14, new_w)
                new_h = max(14, new_h)

                resized_h, resized_w = smart_resize(new_h, new_w, 14 * 2, max_pixels=MAX_PIXELS)

                resized_img = img.resize((resized_w, resized_h), resample=Image.BICUBIC)
                scaled_images.append(resized_img)
                # print("before", img)
                # print("after", resized_img)
                
        else:
            inputs = self.predictor.processor(
                text=[prompt],
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
            ).to(self.predictor.device)
            
            inputs.update({"multi_modal_data": [{"image": images, "video": videos}], "text": [prompt], "eval_mode": True})

            scaled_multi_modal_data = self.predictor(**inputs)["multi_modal_data"]

            if images is not None:
                scaled_images = [img.cpu() if img.is_cuda else img for mm_item in scaled_multi_modal_data for img in mm_item["image"]]
                # prompt = expand_image_prompt
                # print("images", images)
                mm_data["image"] = scaled_images
                # print("scaled_images", scaled_images)

            if videos is not None:
                scaled_videos = [video[0] for mm_item in scaled_multi_modal_data for video in mm_item["video"]]
                video_metadata = [video[1] for mm_item in scaled_multi_modal_data for video in mm_item["video"]]

                if video_metadata is not None and "video_timestamps" in video_metadata[0].keys():
                    grouped_videos, current_videos = [], []

                    for video, meta in zip(scaled_videos, video_metadata):
                        ts = meta.get("video_timestamps")
                        if ts == 0 and len(current_videos) > 0:
                            grouped_videos.append(current_videos)
                            current_videos = []
                        current_videos.append(video)

                    if len(current_videos) > 0:
                        grouped_videos.append(current_videos)

                    temporal_patch_size = self.processor.video_processor.temporal_patch_size
                    print("before expand_video_prompt", prompt)
                    prompt = expand_video_prompt(prompt, grouped_videos, temporal_patch_size)
                    print("after expand_video_prompt", prompt)
                
                mm_data["video"] = scaled_videos
        
        mm_items = self._to_mm_items(mm_data)

        # # print("images", images)
        # images = [convert_to_rgb(image) for image in images]
        # inputs = self.predictor.processor(
        #     text=[prompt],
        #     images=images,
        #     videos=videos,
        #     padding=True,
        #     return_tensors="pt",
        # ).to(self.predictor.device)
        # inputs.update({"multi_modal_data": [{"image": images}], "text": [prompt], "eval_mode": True})

        # scaled_multi_modal_data = self.predictor(**inputs)["multi_modal_data"]
        # scaled_images = [img for mm_item in scaled_multi_modal_data for img in mm_item["image"]]
        
        # mm_items = self._to_mm_items({"image": scaled_images})

    if isinstance(prompt, str):
        if enable_hf_prompt_update:
            return self._apply_hf_processor_text_mm(
                prompt_text=prompt,
                mm_items=mm_items,
                hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                tokenization_kwargs=tokenization_kwargs,
            )

        prompt_ids = self._apply_hf_processor_text_only(prompt, tokenization_kwargs)
    else:
        prompt_ids = self._apply_hf_processor_tokens_only(prompt)

    mm_processed_data = self._apply_hf_processor_mm_only(
        mm_items=mm_items,
        hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        tokenization_kwargs=tokenization_kwargs,
    )

    return prompt_ids, mm_processed_data, False


def tensor_to_pil_list(frames):
    pil_images = []
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()
    
    for frame in frames:
        if frame.shape[0] in [1, 3]:  
            frame = np.transpose(frame, (1, 2, 0))
        
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        pil_images.append(Image.fromarray(frame))
    return pil_images


def tensor_to_tensor_list(frames):
    return list(torch.unbind(frames, dim=0))


def tensor_to_temporal_stack_list(frames, temporal_patch_size):
    """
    Args:
        frames: shape [T, C, H, W] 或 [T, H, W, C]
        temporal_patch_size: int
    Returns:
        List[Tensor]
    """
    return list(torch.split(frames, split_size_or_sections=temporal_patch_size, dim=0))


def split_video_metadata(metadata: list[dict] | dict, temporal_patch_size: int) -> list[dict]:
    if isinstance(metadata, dict):
        metadata_list = [metadata]
    else:
        metadata_list = metadata

    splitted_metadatas = []

    for original_meta in metadata_list:
        frames_indices = original_meta.get("frames_indices", [])
        
        for i in range(0, len(frames_indices), temporal_patch_size):
            chunk_indices = frames_indices[i : i + temporal_patch_size]
            
            new_meta = original_meta.copy()
            new_meta["frames_indices"] = chunk_indices

            # Video A Chunk 0 -> (0.0, 1.0) -> Set Anchor A
            # Video A Chunk 1 -> (2.0, 3.0) -> Reuse Anchor A
            # Video B Chunk 0 -> (0.0, 1.0) -> Set Anchor B (Reset)
            # curr_chunk_len = len(chunk_indices)
            # chunk_timestamps = tuple(float(i + j) for j in range(curr_chunk_len))
            # chunk_timestamps = i
            
            # new_meta["video_timestamps"] = chunk_timestamps
            # new_meta["second_per_grid_ts"] = chunk_timestamps
            
            splitted_metadatas.append(new_meta)
            
    return splitted_metadatas


def expand_image_prompt(prompt, video_inputs):
    video_block_pattern = "<|vision_start|><|video_pad|><|vision_end|>"
    parts = prompt.split(video_block_pattern)
    
    if len(parts) - 1 != len(video_inputs):
        raise ValueError(f"The prompt contains {len(parts)-1} video placeholders, but {len(video_inputs)} video inputs are provided.")
    
    new_prompt = parts[0]
    
    for i, video in enumerate(video_inputs):
        if isinstance(video, list):
            count_n = len(video)
        else:
            count_n = video.shape[0]
        expanded_block = "<|vision_start|>" + "<|image_pad|>" * count_n + "<|vision_end|>"
        new_prompt += expanded_block + parts[i + 1]
        
    return new_prompt


def expand_video_prompt(prompt, video_inputs, temporal_patch_size):
    video_block_pattern = "<|vision_start|><|video_pad|><|vision_end|>"
    parts = prompt.split(video_block_pattern)
    
    if len(parts) - 1 != len(video_inputs):
        raise ValueError(f"The prompt contains {len(parts)-1} video placeholders, but {len(video_inputs)} video inputs are provided.")
    
    new_prompt = parts[0]
    
    for i, video in enumerate(video_inputs):
        if isinstance(video, list):
            count_n = len(video)
        else:
            count_n = video.shape[0] // temporal_patch_size
        expanded_block = "<|vision_start|>" + "<|video_pad|>" * count_n + "<|vision_end|>"
        # expanded_block = "<|vision_start|><|video_pad|><|vision_end|>" * count_n
        new_prompt += expanded_block + parts[i + 1]
        
    return new_prompt


def group_videos_by_timestamps(
    videos: Sequence[Any],
    video_metadatas: Sequence[Dict[str, Any]],
    *,
    ts_key: str = "video_timestamps",
    reset_value: int = 0,
) -> List[List[Any]]:
    """
    Group videos into segments based on a timestamp reset rule.

    Rule:
      - Iterate (video, metadata) in order.
      - Start a new group whenever metadata[ts_key] == reset_value and current group is non-empty.
      - Always append the current group at the end.

    This matches your original behavior where ts==0 indicates the first chunk of a new video.
    """
    grouped: List[List[Any]] = []
    current: List[Any] = []

    # Safety: zip to avoid index mismatch
    for video, meta in zip(videos, video_metadatas):
        ts = meta.get(ts_key, reset_value)
        # Make it robust if ts is numpy/int/str
        try:
            ts_int = int(ts)
        except Exception:
            ts_int = reset_value

        if ts_int == reset_value and len(current) > 0:
            grouped.append(current)
            current = []

        current.append(video)

    if len(current) > 0:
        grouped.append(current)

    return grouped


def maybe_expand_video_prompt(
    raw_prompt: str,
    videos: Sequence[Any],
    video_metadatas: Optional[Sequence[Dict[str, Any]]],
    *,
    temporal_patch_size: int,
    ts_key: str = "video_timestamps",
) -> Tuple[str, Optional[List[List[Any]]]]:
    """
    If video_metadatas contains `ts_key`, group videos by timestamps and expand the prompt.

    Returns:
      - new_prompt: expanded prompt if applicable, else original raw_prompt
      - grouped_videos: the grouped structure if expanded, else None
    """
    if not video_metadatas:
        return raw_prompt
    if not isinstance(video_metadatas[0], dict) or ts_key not in video_metadatas[0]:
        return raw_prompt

    grouped_videos = group_videos_by_timestamps(
        videos,
        video_metadatas,
        ts_key=ts_key,
        reset_value=0,
    )

    new_prompt = expand_video_prompt(raw_prompt, grouped_videos, temporal_patch_size)
    return new_prompt


def expand_video_prompt_blocks(prompt, video_inputs, video_metadatas=None, ts_key="video_timestamps"):
    video_block = "<|vision_start|><|video_pad|><|vision_end|>"
    if video_metadatas and isinstance(video_metadatas[0], dict) and ts_key in video_metadatas[0]:
        video_inputs = group_videos_by_timestamps(
            video_inputs,
            video_metadatas,
            ts_key=ts_key,
            reset_value=0,
        )
    parts = prompt.split(video_block)
    if len(parts) - 1 != len(video_inputs):
        raise ValueError(
            f"The prompt contains {len(parts)-1} video placeholders, but {len(video_inputs)} video inputs are provided."
        )
    new_prompt = parts[0]
    for i, video in enumerate(video_inputs):
        count_n = len(video) if isinstance(video, list) else 1
        new_prompt += (video_block * count_n) + parts[i + 1]
    return new_prompt


def replace_vision_tokens(original_text, new_vision_str):
    start_tag = "<|vision_start|>"
    end_tag = "<|vision_end|>"
    
    first_idx = original_text.find(start_tag)
    last_idx = original_text.rfind(end_tag)
    
    if first_idx == -1 or last_idx == -1:
        return original_text
        
    prefix = original_text[:first_idx]
    suffix = original_text[last_idx + len(end_tag):]
    
    return prefix + new_vision_str + suffix


def video2images(messages, videos, images):
    videos, video_metadata = zip(*videos, strict=False)
    videos, video_metadata = list(videos), list(video_metadata)

    if images is None:
        images = []
    for video in videos:
        # images_tensors = tensor_to_tensor_list(video)
        # images.extend([image.numpy() for image in images_tensors])
        images.extend(tensor_to_tensor_list(video))

    new_messages = []
    for msg in messages:
        new_msg = msg.copy()
        if msg['role'] == 'user' and isinstance(msg.get('content'), list):
            new_content = []
            for content_item in msg['content']:
                if content_item.get('type') == 'video':
                    for image in images:
                        new_content.append({"type": "image", "image": torch.tensor([0], dtype=torch.uint8)})
                else:
                    new_content.append(content_item)
            new_msg['content'] = new_content
        new_messages.append(new_msg)
    
    return new_messages, images


def video2list(messages, videos, mrope_patch = False, temporal_patch_size = 2):
    videos, video_metadata = zip(*videos, strict=False)
    videos, video_metadata = list(videos), list(video_metadata)

    new_videos = []

    for video_tensor, meta in zip(videos, video_metadata):
        chunks = tensor_to_temporal_stack_list(video_tensor, temporal_patch_size)
        metas = split_video_metadata(meta, temporal_patch_size)

        new_videos.extend([(chunk, meta) for chunk, meta in zip(chunks, metas)])

    if not mrope_patch:
        new_messages = []
        for msg in messages:
            new_msg = msg.copy()
            if msg['role'] == 'user' and isinstance(msg.get('content'), list):
                new_content = []
                for content_item in msg['content']:
                    if content_item.get('type') == 'video':
                        for video in new_videos:
                            new_content.append({"type": "video", "video": torch.tensor([0], dtype=torch.uint8)}) # video
                    else:
                        new_content.append(content_item)
                new_msg['content'] = new_content
            new_messages.append(new_msg)
    else:
        new_messages = messages

    return new_messages, new_videos


def reconstruct_struct_videos_from_flat(
    flat_chunks: Sequence[Tuple[Any, Dict[str, Any]]],
    *,
    ts_key: str = "video_timestamps",
    reset_value: int = 0,
) -> List[List[Tuple[Any, Dict[str, Any]]]]:
    """
    Reconstruct per-original-video chunk groups from flattened chunks using metadata timestamps.

    flat_chunks element format: (chunk_frames, metadata)
    metadata should contain ts_key, where reset_value indicates a new original video starts.
    """
    if not flat_chunks:
        return []

    videos = [x[0] for x in flat_chunks]
    metas = [x[1] for x in flat_chunks]

    grouped_videos = group_videos_by_timestamps(
        videos, metas, ts_key=ts_key, reset_value=reset_value
    )

    # Convert grouped raw videos back to grouped (video, meta) tuples
    # We need to iterate in order and pack them back.
    out = []
    idx = 0
    for group in grouped_videos:
        g = []
        for _ in group:
            g.append((videos[idx], metas[idx]))
            idx += 1
        out.append(g)
    return out


def scale_messages_from_mmdata(
    messages: Optional[List[Any]],
    scaled_multi_modal_data: List[dict],
    video2image: bool = False,
):
    """
    Fill message placeholders (image/video) using scaled_multi_modal_data.

    Requirements:
      - mm_data["images"] is a list of scaled images
      - mm_data["videos"] is FLAT chunks: List[(chunk_frames, metadata)]
      - metadata contains "video_timestamps" so we can reconstruct per-original-video groups
    """
    if messages is None or video2image:
        return messages

    scaled_messages = []
    for i, conversation in enumerate(messages):
        mm_data = scaled_multi_modal_data[i]

        sample_images = mm_data.get("images") or []
        # IMPORTANT: videos is flat_chunks, e.g. [(frames, meta), ...]
        # sample_videos_flat = mm_data.get("videos") or []

        # Reconstruct struct videos from flat using timestamps
        # struct_videos: List[List[(frames, meta)]]
        # struct_videos = reconstruct_struct_videos_from_flat(sample_videos_flat)

        # img_it = iter(sample_images)
        # vid_it = iter(struct_videos)  # consume per-original-video

        used_images = 0
        used_videos = 0

        new_conversation = []
        for message in conversation:
            content = message.get("content", [])

            if isinstance(content, str):
                new_conversation.append(message)
                continue

            new_content_list = []
            for item in content:
                itype = item.get("type")

                if itype == "image":
                    try:
                        # img = next(img_it)
                        img = torch.tensor([0], dtype=torch.uint8)
                        used_images += 1
                    except StopIteration:
                        print(f"[Warning] Sample {i}: Image count mismatch! Expected more images.")
                        img = torch.tensor([0], dtype=torch.uint8)

                    new_item = dict(item)
                    new_item["image"] = img
                    new_content_list.append(new_item)

                elif itype == "video":
                    try:
                        # vid_chunks is a list of (frames, meta) for ONE original video
                        # vid_chunks = next(vid_it)
                        vid_chunks = [torch.tensor([0], dtype=torch.uint8)]
                        used_videos += 1
                    except StopIteration:
                        print(f"[Warning] Sample {i}: Video count mismatch! Expected more videos.")
                        vid_chunks = []

                    # What to store into "video" field depends on your downstream.
                    # Option A (recommended): keep structure (chunks + metadata) for prompt expansion
                    new_item = dict(item)
                    new_item["video"] = vid_chunks
                    new_content_list.append(new_item)

                else:
                    new_content_list.append(item if not isinstance(item, dict) else dict(item))

            new_msg = dict(message)
            new_msg["content"] = new_content_list
            new_conversation.append(new_msg)

        # Optional sanity checks
        if used_images != len(sample_images):
            print(f"[Error] Sample {i}: Used {used_images} images, but have {len(sample_images)} scaled images.")

        scaled_messages.append(new_conversation)

    return scaled_messages


def get_target_resolution(
    height: int,
    width: int,
    scale_factor: float,
    patch_size: int,
    image_factor: int,
    max_pixels: int,
) -> Tuple[int, int]:
    """
    Compute target resolution given scale factor with boundary checks and smart resizing.
    """
    sf = float(scale_factor)

    new_h = max(patch_size, int(height * sf))
    new_w = max(patch_size, int(width * sf))

    resized_h, resized_w = smart_resize(
        new_h,
        new_w,
        factor=image_factor,
        max_pixels=max_pixels,
    )
    return resized_h, resized_w


def assert_valid_scale_index(scale_row: torch.Tensor, idx: int, context: str) -> None:
    """
    Ensure scale index is within valid range.
    """
    if idx < 0 or idx >= scale_row.numel():
        raise IndexError(
            f"[adaptive_scaling] scale index out of range: idx={idx}, "
            f"num_scales={scale_row.numel()}, context={context}"
        )


def check_scale_mask(mask_row: Optional[torch.Tensor], idx: int, context: str) -> None:
    """
    Ensure that padding positions are never used for real image/video scaling.
    """
    if mask_row is not None:
        if not bool(mask_row[idx].item()):
            raise AssertionError(
                f"[adaptive_scaling] padding scale used! idx={idx}, context={context}"
            )


def process_image_list(
    images: List[Any],
    scales_row: torch.Tensor,
    start_obj_idx: int,
    scale_mask_row: Optional[torch.Tensor],
    patch_size: int,
    image_factor: int,
    max_token_num: int,
) -> Tuple[List[torch.Tensor], int]:
    """
    Process all images in a single sample.

    Returns:
        scaled_images: list of CPU tensors
        consumed_count: number of scale values consumed
    """
    imgs_np = [to_numpy_array(img) for img in images]

    scaled_images: List[torch.Tensor] = []
    consumed = 0

    for local_idx, img_np in enumerate(imgs_np):
        global_idx = start_obj_idx + local_idx

        assert_valid_scale_index(scales_row, global_idx, context="image")
        check_scale_mask(scale_mask_row, global_idx, context="image")

        scale_factor = scales_row[global_idx].item()

        input_data_format = infer_channel_dimension_format(img_np)
        height, width = get_image_size(img_np, channel_dim=input_data_format)

        resized_h, resized_w = get_target_resolution(
            height, width, scale_factor,
            patch_size, image_factor, max_token_num,
        )

        image_resized = resize(
            img_np,
            size=(resized_h, resized_w),
            resample=PILImageResampling.BICUBIC,
            input_data_format=input_data_format,
        )

        # Always return CPU contiguous tensor
        scaled_images.append(torch.from_numpy(image_resized).contiguous().cpu())
        consumed += 1

    return scaled_images, consumed


def process_video_list(
    videos: List[Any],
    scales_row: torch.Tensor,
    start_obj_idx: int,
    scale_mask_row: Optional[torch.Tensor],
    video_processor: Any,
    patch_size: int,
    image_factor: int,
    temporal_patch_size: int,
    max_token_num: int,
    **kwargs,
) -> Tuple[
    List[Tuple[torch.Tensor, Dict[str, Any]]],
    List[List[Tuple[torch.Tensor, Dict[str, Any]]]],
    int,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    Process all videos in a single sample (including temporal chunking).

    Returns:
        flat_chunks: flattened list of all chunks
        structured_chunks: chunks grouped by original video
        consumed_count: number of scale values consumed
        selected_chunk_scales: updated 1D scale tensor when a chunk-keep mode is active
        selected_chunk_scale_mask: updated 1D mask tensor aligned with selected_chunk_scales
    """
    keep_topk_chunks = kwargs.get("keep_topk_chunks", None)
    keep_chunk_threshold = kwargs.get("keep_chunk_threshold", None)
    resize_kept_chunks = bool(kwargs.get("resize_kept_chunks", True))

    topk_enabled = isinstance(keep_topk_chunks, int) and keep_topk_chunks > 0
    threshold_enabled = isinstance(keep_chunk_threshold, (int, float))
    if topk_enabled and threshold_enabled:
        raise ValueError("keep_topk_chunks and keep_chunk_threshold are mutually exclusive")

    selection_mode = "topk" if topk_enabled else "threshold" if threshold_enabled else None

    def _prepare_video(video_item: Any) -> Tuple[Any, List[Dict[str, Any]]]:
        if isinstance(video_item, tuple):
            video_frames_local, video_metadata_local = video_item
            video_metadata_local = split_video_metadata(video_metadata_local, temporal_patch_size)
            if not kwargs.get("video2list", False):
                for j in range(len(video_metadata_local)):
                    video_metadata_local[j]["video_timestamps"] = j * temporal_patch_size
            return video_frames_local, video_metadata_local

        video_frames_local = video_item
        if not kwargs.get("video2list", False):
            video_metadata_local = [
                {"video_timestamps": j}
                for j in range(0, video_frames_local.shape[0], temporal_patch_size)
            ]
        else:
            video_metadata_local = [{} for _ in range(math.ceil(video_frames_local.shape[0] / temporal_patch_size))]
        return video_frames_local, video_metadata_local

    def _infer_global_scale_mode() -> Optional[str]:
        if scale_mask_row is not None:
            remaining_scales = int(scale_mask_row[start_obj_idx:].detach().bool().sum().item())
        else:
            remaining_scales = max(int(scales_row.numel() - start_obj_idx), 0)
        total_frames = 0
        total_chunks = 0
        for video_item in videos:
            frames = video_item[0] if isinstance(video_item, tuple) else video_item
            num_frames_local = int(frames.shape[0])
            total_frames += num_frames_local
            if num_frames_local > 0:
                total_chunks += math.ceil(num_frames_local / temporal_patch_size)
        if total_frames > 0 and remaining_scales == total_frames:
            return "per_frame"
        if total_chunks > 0 and remaining_scales == total_chunks:
            return "per_chunk"
        return None

    def _chunk_compare_scale(
        *,
        video_idx_local: int,
        chunk_idx_local: int,
        chunk_scale_offset_local: int,
        start_local: int,
        end_local: int,
        num_frames_local: int,
        forced_mode: Optional[str],
    ) -> Tuple[float, bool]:
        if scale_mask_row is not None:
            remaining_scales = int(
                scale_mask_row[start_obj_idx + chunk_scale_offset_local :].detach().bool().sum().item()
            )
        else:
            remaining_scales = int(scales_row.numel() - (start_obj_idx + chunk_scale_offset_local))
        if forced_mode == "per_frame":
            use_per_frame_scales_local = True
        elif forced_mode == "per_chunk":
            use_per_frame_scales_local = False
        else:
            use_per_frame_scales_local = remaining_scales == num_frames_local

        if use_per_frame_scales_local:
            scale_start = start_obj_idx + chunk_scale_offset_local + start_local
            scale_end = start_obj_idx + chunk_scale_offset_local + end_local
            assert_valid_scale_index(
                scales_row, scale_end - 1, context=f"video[{video_idx_local}] chunk[{chunk_idx_local}]"
            )
            scale_slice = scales_row[scale_start:scale_end]
            if scale_mask_row is not None:
                mask_slice = scale_mask_row[scale_start:scale_end].to(scale_slice.dtype)
                denom = mask_slice.sum().clamp_min(1.0)
                scale_factor = (scale_slice * mask_slice).sum() / denom
            else:
                scale_factor = scale_slice.mean()
            return float(scale_factor.item()), True

        scale_idx = start_obj_idx + chunk_scale_offset_local + chunk_idx_local
        assert_valid_scale_index(scales_row, scale_idx, context=f"video[{video_idx_local}] chunk[{chunk_idx_local}]")
        check_scale_mask(scale_mask_row, scale_idx, context=f"video[{video_idx_local}] chunk[{chunk_idx_local}]")
        return float(scales_row[scale_idx].item()), False

    def _resize_chunk(chunk_frames_local: Any, scale_factor_local: float) -> Any:
        if not resize_kept_chunks:
            return chunk_frames_local.cpu() if hasattr(chunk_frames_local, "cpu") else chunk_frames_local
        height_local, width_local = get_image_size(chunk_frames_local[0], channel_dim=ChannelDimension.FIRST)
        resized_h_local, resized_w_local = get_target_resolution(
            height_local,
            width_local,
            scale_factor_local,
            patch_size,
            image_factor,
            max_token_num,
        )
        processed_frames_local = video_processor.resize(
            image=chunk_frames_local,
            size=SizeDict(height=resized_h_local, width=resized_w_local),
            interpolation=InterpolationMode.BICUBIC,
        )
        return processed_frames_local.cpu() if hasattr(processed_frames_local, "cpu") else processed_frames_local

    def _extract_original_video_and_metadata(video_item: Any) -> Tuple[Any, Optional[Any]]:
        if isinstance(video_item, tuple):
            return video_item[0], video_item[1]
        return video_item, None

    def _build_selected_video_metadata(
        original_metadata_local: Optional[Any],
        *,
        num_frames_local: int,
        selected_frame_indices_local: List[int],
    ) -> Dict[str, Any]:
        if isinstance(original_metadata_local, dict):
            selected_metadata_local = dict(original_metadata_local)
        elif isinstance(original_metadata_local, list) and len(original_metadata_local) > 0 and isinstance(original_metadata_local[0], dict):
            selected_metadata_local = dict(original_metadata_local[0])
        else:
            selected_metadata_local = {}

        selected_metadata_local.pop("video_timestamps", None)
        frame_indices_local = selected_metadata_local.get("frames_indices", None)
        if isinstance(frame_indices_local, (list, tuple)) and len(frame_indices_local) >= num_frames_local:
            frame_indices_local = list(frame_indices_local)
            selected_metadata_local["frames_indices"] = [frame_indices_local[idx] for idx in selected_frame_indices_local]
        else:
            selected_metadata_local["frames_indices"] = [int(idx) for idx in selected_frame_indices_local]
        return selected_metadata_local

    def _select_frame_indices_noresize(
        *,
        frame_scores_local: torch.Tensor,
        frame_mask_local: torch.Tensor,
    ) -> List[int]:
        # In per-frame noresize mode:
        # - topk is a frame budget
        # - threshold first keeps surviving frames, then backfills the highest remaining valid
        #   frames so the final retained length rounds up to a multiple of ``temporal_patch_size``
        valid_indices_local = [idx for idx in range(frame_scores_local.numel()) if bool(frame_mask_local[idx].item())]
        if not valid_indices_local:
            return []

        ranked_indices_local = sorted(
            valid_indices_local,
            key=lambda idx: (-float(frame_scores_local[idx].item()), idx),
        )

        if selection_mode == "topk":
            target_count_local = min(len(valid_indices_local), int(keep_topk_chunks))
            if target_count_local <= 0:
                return []
            rounded_count_local = min(
                len(valid_indices_local),
                int(math.ceil(target_count_local / temporal_patch_size) * temporal_patch_size),
            )
            chosen_indices_local = ranked_indices_local[:rounded_count_local]
        else:
            chosen_indices_local = [
                idx for idx in valid_indices_local if float(frame_scores_local[idx].item()) >= float(keep_chunk_threshold)
            ]
            if not chosen_indices_local:
                fallback_count_local = min(len(valid_indices_local), temporal_patch_size)
                chosen_indices_local = ranked_indices_local[:fallback_count_local]
            else:
                rounded_count_local = min(
                    len(valid_indices_local),
                    int(math.ceil(len(chosen_indices_local) / temporal_patch_size) * temporal_patch_size),
                )
                chosen_indices_local = ranked_indices_local[:rounded_count_local]

        return sorted(chosen_indices_local)

    if selection_mode is not None:
        scale_mode = _infer_global_scale_mode()

        if not resize_kept_chunks and scale_mode == "per_frame":
            flat_chunks: List[Tuple[torch.Tensor, Dict[str, Any]]] = []
            structured_chunks: List[List[Tuple[torch.Tensor, Dict[str, Any]]]] = []
            chunk_scale_offset = 0
            full_segments: List[torch.Tensor] = []
            mask_segments: List[torch.Tensor] = []

            for video_idx, video in enumerate(videos):
                video_frames, original_metadata = _extract_original_video_and_metadata(video)
                num_frames = int(video_frames.shape[0])
                if num_frames == 0:
                    structured_chunks.append([])
                    full_segments.append(torch.tensor([], dtype=torch.float32))
                    mask_segments.append(torch.tensor([], dtype=torch.bool))
                    continue

                scale_start = start_obj_idx + chunk_scale_offset
                scale_end = scale_start + num_frames
                assert_valid_scale_index(scales_row, scale_end - 1, context=f"noresize_full_video[{video_idx}]")

                frame_scores = scales_row[scale_start:scale_end].detach().to(torch.float32).clone()
                if scale_mask_row is not None:
                    frame_mask = scale_mask_row[scale_start:scale_end].detach().bool().clone()
                    frame_scores = torch.where(frame_mask, frame_scores, torch.zeros_like(frame_scores))
                else:
                    frame_mask = torch.ones(num_frames, dtype=torch.bool, device=frame_scores.device)

                selected_frame_indices = _select_frame_indices_noresize(
                    frame_scores_local=frame_scores,
                    frame_mask_local=frame_mask,
                )
                selected_index_tensor = torch.tensor(selected_frame_indices, dtype=torch.long, device=video_frames.device)
                selected_video_frames = (
                    video_frames.index_select(0, selected_index_tensor).cpu()
                    if selected_frame_indices
                    else video_frames[:0].cpu()
                )
                selected_metadata = _build_selected_video_metadata(
                    original_metadata,
                    num_frames_local=num_frames,
                    selected_frame_indices_local=selected_frame_indices,
                )
                selected_video = (selected_video_frames, selected_metadata)
                flat_chunks.append(selected_video)
                structured_chunks.append([selected_video])

                full = torch.zeros(num_frames, dtype=torch.float32)
                if selected_frame_indices:
                    full_cpu_idx = torch.tensor(selected_frame_indices, dtype=torch.long)
                    full[full_cpu_idx] = 1.0
                mask_full = frame_mask.cpu() if frame_mask.device.type != "cpu" else frame_mask.clone()
                full = torch.where(mask_full, full, torch.zeros_like(full))
                full_segments.append(full)
                mask_segments.append(mask_full)

                chunk_scale_offset += num_frames

            selected_scales_tensor = torch.cat(full_segments)
            selected_mask_tensor = torch.cat(mask_segments)
            return flat_chunks, structured_chunks, chunk_scale_offset, selected_scales_tensor, selected_mask_tensor

        flat_chunks: List[Tuple[torch.Tensor, Dict[str, Any]]] = []
        structured_chunks: List[List[Tuple[torch.Tensor, Dict[str, Any]]]] = []
        descriptors: List[Dict[str, Any]] = []
        chunk_scale_offset = 0
        video_infos: List[Dict[str, Any]] = []

        for video_idx, video in enumerate(videos):
            video_frames, video_metadata = _prepare_video(video)
            num_frames = int(video_frames.shape[0])
            if num_frames == 0:
                structured_chunks.append([])
                video_infos.append({"empty": True})
                continue

            num_chunks = math.ceil(num_frames / temporal_patch_size)
            uses_per_frame_for_video = False
            scale_base_start = chunk_scale_offset

            for chunk_idx in range(num_chunks):
                start = chunk_idx * temporal_patch_size
                end = min(num_frames, start + temporal_patch_size)
                chunk_frames = video_frames[start:end]
                compare_scale, use_per_frame_scales = _chunk_compare_scale(
                    video_idx_local=video_idx,
                    chunk_idx_local=chunk_idx,
                    chunk_scale_offset_local=chunk_scale_offset,
                    start_local=start,
                    end_local=end,
                    num_frames_local=num_frames,
                    forced_mode=scale_mode,
                )
                uses_per_frame_for_video = use_per_frame_scales

                if chunk_idx >= len(video_metadata):
                    raise IndexError(
                        f"[adaptive_scaling] video_metadata too short: "
                        f"len={len(video_metadata)}, chunk_idx={chunk_idx}, video_idx={video_idx}"
                    )

                descriptors.append(
                    {
                        "global_order": len(descriptors),
                        "video_idx": video_idx,
                        "chunk_idx": chunk_idx,
                        "chunk_frames": chunk_frames,
                        "metadata": dict(video_metadata[chunk_idx]),
                        "compare_scale": compare_scale,
                    }
                )

            chunk_scale_offset += num_frames if uses_per_frame_for_video else num_chunks
            video_infos.append(
                {
                    "empty": False,
                    "scale_base_start": scale_base_start,
                    "num_frames": num_frames,
                    "num_chunks": num_chunks,
                    "per_frame": uses_per_frame_for_video,
                }
            )

        if not descriptors:
            return flat_chunks, structured_chunks, chunk_scale_offset, None, None

        descriptors_by_video: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for desc in descriptors:
            descriptors_by_video[desc["video_idx"]].append(desc)

        selected_orders: set[int]
        if selection_mode == "threshold":
            selected_orders = {desc["global_order"] for desc in descriptors if desc["compare_scale"] >= float(keep_chunk_threshold)}
        else:
            selected_orders = set()
            if scale_mode == "per_frame":
                chunk_budget = max(1, int(math.ceil(int(keep_topk_chunks) / temporal_patch_size)))
            else:
                chunk_budget = int(keep_topk_chunks)
            for group in descriptors_by_video.values():
                ranked = sorted(group, key=lambda d: (-d["compare_scale"], d["global_order"]))
                selected_orders.update(desc["global_order"] for desc in ranked[:chunk_budget])

        for group in descriptors_by_video.values():
            if any(desc["global_order"] in selected_orders for desc in group):
                continue
            best_desc = max(group, key=lambda d: (d["compare_scale"], -d["global_order"]))
            selected_orders.add(best_desc["global_order"])

        selected_descriptors = [desc for desc in descriptors if desc["global_order"] in selected_orders]

        for video_idx in range(len(videos)):
            current_group = [desc for desc in selected_descriptors if desc["video_idx"] == video_idx]
            if not current_group:
                if video_idx < len(structured_chunks):
                    continue
                structured_chunks.append([])
                continue

            rebuilt_group: List[Tuple[torch.Tensor, Dict[str, Any]]] = []
            for kept_idx, desc in enumerate(current_group):
                chunk_meta = dict(desc["metadata"])
                chunk_meta["video_timestamps"] = kept_idx * temporal_patch_size
                processed_frames = _resize_chunk(desc["chunk_frames"], desc["compare_scale"])
                rebuilt_group.append((processed_frames, chunk_meta))
                flat_chunks.append((processed_frames, chunk_meta))
            structured_chunks.append(rebuilt_group)

        # Align scales with the original predictor layout (per-frame or per-chunk slots).
        # - ``scale_mask_row`` is NOT rewritten in chunk-selection mode: copy the slice for this video
        #   into ``mask_full`` unchanged.
        # - ``full`` starts as the same slice of ``scales_row``, then: (1) force 0 where mask is False
        #   (padding) for fair comparison; (2) set 0 only on **dropped** chunk indices (selection).
        # - If ``resize_kept_chunks`` is False, selected indices are rewritten to 1.0 because the kept
        #   frames/chunks are passed through at original resolution, so the effective applied scale is 1.
        desc_by_key = {(d["video_idx"], d["chunk_idx"]): d for d in descriptors}
        full_segments: List[torch.Tensor] = []
        mask_segments: List[torch.Tensor] = []

        for video_idx, info in enumerate(video_infos):
            if info.get("empty"):
                full_segments.append(torch.tensor([], dtype=torch.float32))
                mask_segments.append(torch.tensor([], dtype=torch.bool))
                continue

            scale_base = int(info["scale_base_start"])
            n_frames = int(info["num_frames"])
            n_chunks = int(info["num_chunks"])
            per_frame = bool(info["per_frame"])
            slot_len = n_frames if per_frame else n_chunks
            ss0 = start_obj_idx + scale_base
            ee0 = ss0 + slot_len
            assert_valid_scale_index(scales_row, ee0 - 1, context=f"aligned_scales v{video_idx} segment")
            full = scales_row[ss0:ee0].detach().to(torch.float32).clone()
            if scale_mask_row is not None:
                mask_full = scale_mask_row[ss0:ee0].detach().bool().clone()
                full = torch.where(mask_full, full, torch.zeros_like(full))
            else:
                mask_full = torch.ones(slot_len, dtype=torch.bool, device=full.device)

            for chunk_idx in range(n_chunks):
                start = chunk_idx * temporal_patch_size
                end = min(n_frames, start + temporal_patch_size)
                desc = desc_by_key[(video_idx, chunk_idx)]
                if desc["global_order"] in selected_orders:
                    if not resize_kept_chunks:
                        if per_frame:
                            full[start:end] = torch.where(
                                mask_full[start:end],
                                torch.ones_like(full[start:end]),
                                torch.zeros_like(full[start:end]),
                            )
                        else:
                            full[chunk_idx] = 1.0 if bool(mask_full[chunk_idx].item()) else 0.0
                    continue
                if per_frame:
                    full[start:end] = 0.0
                else:
                    full[chunk_idx] = 0.0

            full_segments.append(full)
            mask_segments.append(mask_full)

        selected_scales_tensor = torch.cat(full_segments)
        selected_mask_tensor = torch.cat(mask_segments)
        return flat_chunks, structured_chunks, chunk_scale_offset, selected_scales_tensor, selected_mask_tensor

    flat_chunks = []
    structured_chunks = []
    chunk_scale_offset = 0
    global_scale_mode = _infer_global_scale_mode()

    for video_idx, video in enumerate(videos):
        # Parse frames and metadata
        if isinstance(video, tuple):
            video_frames, video_metadata = video
            video_metadata = split_video_metadata(video_metadata, temporal_patch_size)

            if not kwargs.get("video2list", False):
                for j in range(len(video_metadata)):
                    video_metadata[j]["video_timestamps"] = j * temporal_patch_size
        else:
            video_frames = video
            if not kwargs.get("video2list", False):
                video_metadata = [
                    {"video_timestamps": j}
                    for j in range(0, video_frames.shape[0], temporal_patch_size)
                ]

        num_frames = int(video_frames.shape[0])
        if num_frames == 0:
            structured_chunks.append([])
            continue

        height, width = get_image_size(video_frames[0], channel_dim=ChannelDimension.FIRST)

        num_chunks = math.ceil(num_frames / temporal_patch_size)
        remaining_scales = int(scales_row.numel() - (start_obj_idx + chunk_scale_offset))
        if global_scale_mode == "per_frame":
            use_per_frame_scales = True
        elif global_scale_mode == "per_chunk":
            use_per_frame_scales = False
        else:
            use_per_frame_scales = remaining_scales == num_frames
        current_video_chunks = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * temporal_patch_size
            end = min(num_frames, start + temporal_patch_size)
            chunk_frames = video_frames[start:end]

            if use_per_frame_scales:
                scale_start = start_obj_idx + chunk_scale_offset + start
                scale_end = start_obj_idx + chunk_scale_offset + end
                assert_valid_scale_index(
                    scales_row, scale_end - 1, context=f"video[{video_idx}] chunk[{chunk_idx}]"
                )
                scale_slice = scales_row[scale_start:scale_end]
                if scale_mask_row is not None:
                    mask_slice = scale_mask_row[scale_start:scale_end].to(scale_slice.dtype)
                    denom = mask_slice.sum().clamp_min(1.0)
                    scale_factor = (scale_slice * mask_slice).sum() / denom
                else:
                    scale_factor = scale_slice.mean()
                scale_factor = scale_factor.item()
            else:
                scale_idx = start_obj_idx + chunk_scale_offset + chunk_idx
                assert_valid_scale_index(scales_row, scale_idx, context=f"video[{video_idx}] chunk[{chunk_idx}]")
                check_scale_mask(scale_mask_row, scale_idx, context=f"video[{video_idx}] chunk[{chunk_idx}]")
                scale_factor = scales_row[scale_idx].item()

            resized_h, resized_w = get_target_resolution(
                height, width, scale_factor,
                patch_size, image_factor, max_token_num,
            )

            processed_frames = video_processor.resize(
                image=chunk_frames,
                size=SizeDict(height=resized_h, width=resized_w),
                interpolation=InterpolationMode.BICUBIC,
            )

            if chunk_idx >= len(video_metadata):
                raise IndexError(
                    f"[adaptive_scaling] video_metadata too short: "
                    f"len={len(video_metadata)}, chunk_idx={chunk_idx}, video_idx={video_idx}"
                )

            current_video_chunks.append(
                (processed_frames.cpu(), video_metadata[chunk_idx])
            )

        flat_chunks.extend(current_video_chunks)
        structured_chunks.append(current_video_chunks)
        chunk_scale_offset += num_frames if use_per_frame_scales else num_chunks

    return flat_chunks, structured_chunks, chunk_scale_offset, None, None


def to_cpu_tensor(x: Any, *, dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
    """Convert tensor/ndarray/list to a CPU torch.Tensor. Copy only if numpy array is not writeable."""
    if x is None:
        return None

    if torch.is_tensor(x):
        t = x.detach()
        if t.is_cuda:
            t = t.cpu()
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        return t

    if isinstance(x, np.ndarray):
        # Fix: torch.from_numpy requires writeable numpy for safe behavior
        if not x.flags.writeable:
            x = x.copy()
        t = torch.from_numpy(x)
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        return t

    # list / scalar
    t = torch.tensor(x)
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    return t


def apply_adaptive_scaling(
    multi_modal_data: List[Dict[str, Any]],
    scales: Optional[TensorLike],
    new_scale_mask: Optional[TensorLike],
    processor: Any,
    patch_size: int,
    image_factor: int,
    temporal_patch_size: int,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Apply adaptive scaling to multi-modal batch data.
    """
    if scales is None:
        return multi_modal_data

    image_max_token_num=IMAGE_MAX_TOKEN_NUM * image_factor ** 2
    video_max_token_num=VIDEO_MAX_TOKEN_NUM * image_factor ** 2

    # Move to CPU once to avoid repeated synchronization
    scales_cpu = to_cpu_tensor(scales, dtype=torch.float32)
    mask_cpu = to_cpu_tensor(new_scale_mask, dtype=torch.bool) if new_scale_mask is not None else None

    if scales_cpu is not None and scales_cpu.dim() == 1:
        scales_cpu = scales_cpu.unsqueeze(0)
    if mask_cpu is not None and mask_cpu.dim() == 1:
        mask_cpu = mask_cpu.unsqueeze(0)

    output = []

    for i, mm_data in enumerate(multi_modal_data):
        # Copy to avoid side effects
        new_mm_data = dict(mm_data)

        scales_row = scales_cpu[i]
        mask_row = mask_cpu[i] if mask_cpu is not None else None

        obj_idx = 0
        updated_scale_segments: List[torch.Tensor] = []
        updated_mask_segments: Optional[List[torch.Tensor]] = [] if mask_row is not None else None
        selection_applied = False

        # -------- Images --------
        images = None
        image_key = None

        if "images" in mm_data and mm_data["images"] is not None:
            images = mm_data["images"]
            image_key = "images"
        elif "image" in mm_data and mm_data["image"] is not None:
            images = mm_data["image"]
            image_key = "image"

        if images is not None:
            try:
                scaled_imgs, consumed = process_image_list(
                    images,
                    scales_row,
                    obj_idx,
                    mask_row,
                    patch_size,
                    image_factor,
                    image_max_token_num,
                )
            except Exception as e:
                print(f"{i}: {scales_cpu}: {mask_cpu} : {images}")
                print(f"Error processing image list for sample {i}: {e}")
                scaled_imgs = images
                consumed = len(images)
            new_mm_data[image_key] = scaled_imgs
            if consumed > 0:
                updated_scale_segments.append(scales_row[obj_idx : obj_idx + consumed].clone())
                if updated_mask_segments is not None and mask_row is not None:
                    updated_mask_segments.append(mask_row[obj_idx : obj_idx + consumed].clone())
            obj_idx += consumed

        # -------- Videos --------
        videos = None
        video_key = None

        if "videos" in mm_data and mm_data["videos"] is not None:
            videos = mm_data["videos"]
            video_key = "videos"
        elif "video" in mm_data and mm_data["video"] is not None:
            videos = mm_data["video"]
            video_key = "video"

        if videos is not None:
            flat_chunks, _, consumed, selected_video_scales, selected_video_scale_mask = process_video_list(
                videos,
                scales_row,
                obj_idx,
                mask_row,
                processor.video_processor,
                patch_size,
                image_factor,
                temporal_patch_size,
                video_max_token_num,
                **kwargs,
            )

            # new_mm_data[f"{video_key}_s"] = struct_chunks
            new_mm_data[f"{video_key}"] = flat_chunks
            if selected_video_scales is not None:
                selection_applied = True
                updated_scale_segments.append(selected_video_scales.clone())
                if updated_mask_segments is not None:
                    if selected_video_scale_mask is not None:
                        updated_mask_segments.append(selected_video_scale_mask.clone())
                    else:
                        updated_mask_segments.append(torch.ones_like(selected_video_scales, dtype=torch.bool))
            elif consumed > 0:
                updated_scale_segments.append(scales_row[obj_idx : obj_idx + consumed].clone())
                if updated_mask_segments is not None and mask_row is not None:
                    updated_mask_segments.append(mask_row[obj_idx : obj_idx + consumed].clone())

            obj_idx += consumed

        if selection_applied and updated_scale_segments:
            rebuilt_scales = torch.cat(updated_scale_segments).unsqueeze(0)
            new_mm_data["_adaptive_selected_scales"] = rebuilt_scales
            if updated_mask_segments is not None:
                rebuilt_mask = torch.cat(updated_mask_segments).unsqueeze(0)
                new_mm_data["_adaptive_selected_scale_mask"] = rebuilt_mask
            else:
                new_mm_data["_adaptive_selected_scale_mask"] = torch.ones_like(rebuilt_scales, dtype=torch.bool)

        output.append(new_mm_data)

    return output


def encode_pil_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG") 
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def encode_numpy_to_base64(image_obj) -> str:
    if hasattr(image_obj, 'cpu'): 
        arr = image_obj.detach().cpu().numpy()
        
    elif isinstance(image_obj, Image.Image):
        arr = np.array(image_obj)
        
    elif isinstance(image_obj, np.ndarray):
        arr = image_obj
    else:
        arr = np.array(image_obj)

    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)

    buffer = BytesIO()
    np.save(buffer, arr, allow_pickle=False) 
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def make_messages_serializable(messages):
    new_messages = copy.deepcopy(messages)
    
    for msg in new_messages:
        if isinstance(msg.get('content'), list):
            for item in msg['content']:
                if item.get('type') == 'image':
                    img_obj = item.get('image')
                    if isinstance(img_obj, Image.Image):
                        # item['image'] = encode_numpy_to_base64(img_obj)
                        item['image'] = [0]
    return new_messages

def decode_base64_to_numpy(b64_str: str):
    try:
        bytes_data = base64.b64decode(b64_str)
        buffer = BytesIO(bytes_data)
        
        arr = np.load(buffer, allow_pickle=False)
        return torch.from_numpy(arr.copy())
    except Exception as e:
        print(f"error: {e}")
        import traceback
        traceback.print_exc()
        return None

def decode_base64_to_pil(base64_str):
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data)).convert("RGB")

def serialize_tensor_to_base64(tensor: torch.Tensor) -> str:
    arr = tensor.cpu().detach().numpy()
    
    buffer = BytesIO()
    np.save(buffer, arr)
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def deserialize_base64_to_tensor(b64_str: str) -> torch.Tensor:
    bytes_data = base64.b64decode(b64_str)
    
    buffer = BytesIO(bytes_data)
    
    arr = np.load(buffer)
    
    return torch.from_numpy(arr)


def to_numpy_cpu(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def _to_cpu_deep(x: Any) -> Any:
    """Recursively move torch.Tensors to CPU; keep metadata unchanged."""
    try:
        if torch.is_tensor(x):
            return x.detach().to("cpu")
    except Exception:
        pass

    if isinstance(x, dict):
        return {k: _to_cpu_deep(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_cpu_deep(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_cpu_deep(v) for v in x)
    return x


def compute_scales_and_sample_means_cpu(
    out: Dict[str, Any],
    max_scale = None,
    min_scale = None,
    *,
    use_discrete_action: bool = False,
    default_step: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prefer using `out["scales"]` directly. If absent, derive scales from `out["actions"]`.
    Then compute per-sample mean scales using `out["scale_mask"]` if provided.

    Returns:
        scales_cpu: Tensor on CPU, shape (B, T) (or similar)
        sample_means_cpu: Tensor on CPU, shape (B,)
    """
    scales = out.get("scales", None)
    scale_mask = out.get("scale_mask", None)

    scales_cpu = _to_cpu_deep(scales) if scales is not None else None
    mask_cpu = _to_cpu_deep(scale_mask) if scale_mask is not None else None

    if scales_cpu is None:
        actions = out.get("actions", None)
        if actions is None:
            raise RuntimeError("predictor output missing both 'scales' and 'actions'")

        actions_cpu = _to_cpu_deep(actions)

        if use_discrete_action:
            step_sz = float(default_step)
            steps_count = int((max_scale - min_scale) / step_sz) + 1

            scale_bins = torch.linspace(min_scale, max_scale, steps_count, device=actions_cpu.device)
            scales_cpu = scale_bins[actions_cpu.long()]
        else:
            scales_cpu = min_scale + actions_cpu.float() * (max_scale - min_scale)

    if mask_cpu is not None:
        scales_f = scales_cpu.float()
        mask_f = mask_cpu.float()
        denom = mask_f.sum(dim=-1).clamp(min=1.0)
        sample_means_cpu = (scales_f * mask_f).sum(dim=-1) / denom
    else:
        sample_means_cpu = scales_cpu.float().mean(dim=-1)

    return scales_cpu, mask_cpu, sample_means_cpu.tolist()


def env_true(name: str) -> bool:
    v = os.getenv(name, "")
    return v.lower() in ("1", "true", "yes", "y", "on")
