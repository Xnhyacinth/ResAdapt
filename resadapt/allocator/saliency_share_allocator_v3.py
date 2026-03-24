import math

import torch
import torch.nn.functional as F
from torch import nn

from resadapt.allocator.aznet_v3 import FrameWiseScaleAllocator


class SaliencyShareScaleAllocator(FrameWiseScaleAllocator):
    """Quality-first scale policy driven by per-frame saliency share."""

    def __init__(self, *args, dim: int, heads: int = 8, dropout: float = 0.0, ff_mult: int = 4, **kwargs):
        super().__init__(*args, dim=dim, heads=heads, dropout=dropout, ff_mult=ff_mult, **kwargs)
        self.signal_dim = 4
        self.signal_proj = nn.Sequential(
            nn.LayerNorm(self.signal_dim),
            nn.Linear(self.signal_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.share_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def post_init(self):
        if self.use_discrete_action:
            return super().post_init()

        budget_last = self.budget_head[-1]
        share_last = self.share_head[-1]
        residual_last = self.residual_head[-1]
        variance_last = self.variance_head[-1]

        nn.init.zeros_(budget_last.weight)
        nn.init.zeros_(share_last.weight)
        nn.init.zeros_(residual_last.weight)
        nn.init.zeros_(variance_last.weight)
        nn.init.zeros_(share_last.bias)
        nn.init.zeros_(residual_last.bias)

        target_ratio = self._initial_mean_ratio()
        bias_device = budget_last.bias.device
        bias_dtype = budget_last.bias.dtype
        budget_last.bias.data.fill_(
            torch.logit(
                torch.tensor(target_ratio, device=bias_device, dtype=bias_dtype),
                eps=1e-6,
            )
        )

        if self.continuous_dist == "logistic_normal":
            sigma = torch.tensor(
                float(self.logistic_normal_init_sigma),
                device=variance_last.bias.device,
                dtype=variance_last.bias.dtype,
            ).clamp_min(1e-3)
            variance_last.bias.data.fill_(self._softplus_inverse(sigma))
        else:
            base_concentration = 4.0 if self.beta_add_one else 2.0
            feasible_concentration = self._ensure_feasible_beta_concentration(
                torch.tensor(target_ratio, device=bias_device, dtype=bias_dtype),
                torch.tensor(float(self.init_concentration), device=bias_device, dtype=bias_dtype),
            )
            target_concentration = torch.maximum(
                feasible_concentration,
                torch.tensor(base_concentration + 1e-4, device=bias_device, dtype=bias_dtype),
            )
            variance_last.bias.data.fill_(
                self._softplus_inverse(
                    torch.tensor(
                        float(target_concentration.item()) - base_concentration,
                        device=variance_last.bias.device,
                        dtype=variance_last.bias.dtype,
                    )
                )
            )

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        logits = logits.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        probs = torch.where(mask, probs, torch.zeros_like(probs))
        denom = probs.sum(dim=-1, keepdim=True).clamp_min(eps)
        return probs / denom

    def _compute_temporal_surprise(self, frame_states: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if frame_states.shape[1] <= 1:
            return torch.zeros(frame_states.shape[:2], device=frame_states.device, dtype=frame_states.dtype)

        normed = F.normalize(frame_states.float(), dim=-1)
        left_context = torch.zeros_like(normed)
        right_context = torch.zeros_like(normed)
        left_context[:, 1:] = normed[:, :-1]
        left_context[:, 0] = normed[:, 1]
        right_context[:, :-1] = normed[:, 1:]
        right_context[:, -1] = normed[:, -2]
        context = F.normalize(0.5 * (left_context + right_context), dim=-1)
        cosine = (normed * context).sum(dim=-1)
        surprise = (1.0 - cosine).clamp(0.0, 1.0).to(frame_states.dtype)
        return torch.where(valid_mask, surprise, torch.zeros_like(surprise))

    @staticmethod
    def _compute_saliency_anchor(
        text_relevance: torch.Tensor,
        temporal_surprise: torch.Tensor,
        detail_score: torch.Tensor,
        valid_mask: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        relevance01 = ((text_relevance + 1.0) * 0.5).clamp(0.0, 1.0)
        raw = 0.55 * relevance01 + 0.30 * temporal_surprise.clamp(0.0, 1.0) + 0.15 * detail_score.clamp(0.0, 1.0)
        raw = torch.where(valid_mask, raw, torch.zeros_like(raw))
        counts = valid_mask.sum(dim=-1, keepdim=True)
        uniform = valid_mask.to(raw.dtype) / counts.clamp_min(1).to(raw.dtype)
        raw = raw + valid_mask.to(raw.dtype) * eps
        denom = raw.sum(dim=-1, keepdim=True)
        anchor = raw / denom.clamp_min(eps)
        return torch.where(counts > 0, anchor, uniform)

    def _build_signal_features(
        self,
        frame_states: torch.Tensor,
        valid_mask: torch.Tensor,
        text_states: torch.Tensor | None,
        text_mask: torch.Tensor | None,
        detail_score: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if text_states is not None:
            text_relevance = self.relevance_scorer(frame_states, text_states, text_mask)
        else:
            text_relevance = torch.zeros(frame_states.shape[:2], device=frame_states.device, dtype=frame_states.dtype)

        temporal_surprise = self._compute_temporal_surprise(frame_states, valid_mask)
        redundancy = (1.0 - temporal_surprise).clamp(0.0, 1.0)
        detail = torch.where(valid_mask, detail_score.clamp(0.0, 1.0), torch.zeros_like(detail_score))
        saliency_anchor = self._compute_saliency_anchor(
            text_relevance=text_relevance,
            temporal_surprise=temporal_surprise,
            detail_score=detail,
            valid_mask=valid_mask,
        )
        signal_features = torch.stack(
            [
                ((text_relevance + 1.0) * 0.5).clamp(0.0, 1.0),
                temporal_surprise,
                detail,
                saliency_anchor,
            ],
            dim=-1,
        )
        metrics = {
            "redundancy": redundancy.float(),
            "uniqueness": temporal_surprise.float(),
            "temporal_surprise": temporal_surprise.float(),
            "text_relevance": text_relevance.float().clamp(-1.0, 1.0),
            "info_score": detail.float(),
            "detail_score": detail.float(),
            "saliency_anchor": saliency_anchor.float(),
        }
        return signal_features, metrics

    def _build_continuous_head_outputs(
        self,
        frame_states: torch.Tensor,
        valid_mask: torch.Tensor,
        signal_features: torch.Tensor,
    ) -> torch.Tensor:
        signal_context = self.signal_proj(signal_features.to(self.signal_proj[1].weight.dtype)).to(frame_states.dtype)
        policy_states = self.frame_state_norm(frame_states + signal_context)
        clip_state = self._masked_mean(policy_states, valid_mask.unsqueeze(-1), dim=1, keepdim=False)
        budget_logit = self.budget_head(clip_state).expand(-1, policy_states.shape[1])

        anchor_share = signal_features[..., 3].clamp_min(1e-6)
        anchor_bias = anchor_share.log()
        anchor_bias = anchor_bias - self._masked_mean(anchor_bias, valid_mask, dim=1, keepdim=True)

        share_logits = self.share_head(policy_states).squeeze(-1) + anchor_bias
        predicted_share = self._masked_softmax(share_logits, valid_mask)
        share_bias = predicted_share.clamp_min(1e-6).log()
        share_bias = share_bias - self._masked_mean(share_bias, valid_mask, dim=1, keepdim=True)

        residual = self.residual_head(policy_states).squeeze(-1)
        mean_logit = (budget_logit + share_bias + residual).unsqueeze(-1)

        if self.continuous_dist == "logistic_normal":
            sigma_raw = self.variance_head(clip_state).unsqueeze(1).expand(-1, frame_states.shape[1], -1)
            return torch.cat([mean_logit, sigma_raw], dim=-1)

        base_concentration = 4.0 if self.beta_add_one else 2.0
        concentration = F.softplus(self.variance_head(clip_state)) + base_concentration
        mean01 = torch.sigmoid(mean_logit.squeeze(-1)).clamp(1e-4, 1.0 - 1e-4)
        concentration = self._ensure_feasible_beta_concentration(mean01, concentration)
        alpha_target = (mean01 * concentration).clamp_min(1e-4)
        beta_target = ((1.0 - mean01) * concentration).clamp_min(1e-4)
        offset = 1.0 if self.beta_add_one else 0.0
        alpha_raw = self._softplus_inverse((alpha_target - offset).clamp_min(1e-4)) / max(self.beta_param_scale, 1e-6)
        beta_raw = self._softplus_inverse((beta_target - offset).clamp_min(1e-4)) / max(self.beta_param_scale, 1e-6)
        return torch.stack([alpha_raw, beta_raw], dim=-1).to(frame_states.dtype)

    def _compute_scale_diagnostics(
        self,
        scales: torch.Tensor,
        scale_mask: torch.Tensor,
        frame_metrics: dict[str, torch.Tensor] | None,
    ) -> dict[str, float]:
        diagnostics = super()._compute_scale_diagnostics(scales, scale_mask, frame_metrics)
        if frame_metrics is None:
            return diagnostics

        surprise = frame_metrics.get("temporal_surprise")
        anchor = frame_metrics.get("saliency_anchor")
        if surprise is None or anchor is None:
            return diagnostics

        surprise_corrs = []
        anchor_corrs = []
        for i in range(scales.shape[0]):
            valid = scale_mask[i].to(torch.bool)
            if int(valid.sum().item()) <= 0:
                continue
            surprise_corrs.append(float(self._masked_corrcoef(scales[i], surprise[i], valid).item()))
            anchor_corrs.append(float(self._masked_corrcoef(scales[i], anchor[i], valid).item()))

        if surprise_corrs:
            diagnostics["scale_temporal_surprise_corr"] = float(sum(surprise_corrs) / len(surprise_corrs))
        if anchor_corrs:
            diagnostics["scale_saliency_anchor_corr"] = float(sum(anchor_corrs) / len(anchor_corrs))
        return diagnostics
