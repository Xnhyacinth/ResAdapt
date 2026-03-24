"""
FrameWiseScaleAllocatorLightV2: Lightweight frame-scale allocator

Key updates:
1. Grouped batch processing instead of per-sample loops
2. Frame-level temporal encoder for temporal consistency
3. RoPE caching to avoid recomputation
4. Optional text-conditioned spatial attention
5. Reduced shape churn in temporal/spatial blocks
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_utils import ModuleUtilsMixin
from typing import Dict
import math

from resadapt.allocator.aznet_v2 import RotaryEmbedding, PatchWiseTemporalAttention, SpatialAttention


def exists(val):
    return val is not None


class TemporalBlockV2(nn.Module):
    """Temporal block with batch-friendly shape handling."""
    
    def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.0, ff_mult=4):
        super().__init__()
        self.attn = PatchWiseTemporalAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim)
        )

    def forward(self, x, rope):
        """
        Args:
            x: (B, T, L, D) or (T, L, D)
            rope: RoPE tensor
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze_out = True
        else:
            squeeze_out = False
        
        res = x
        x = self.attn(x, rotary_pos_emb=rope)
        x = x + res
        
        if squeeze_out:
            x = x.squeeze(0)
        
        return x + self.ffn(x)


class SpatialBlockV2(nn.Module):
    """Spatial block with batch-friendly shape handling."""
    
    def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.0, ff_mult=4):
        super().__init__()
        self.attn = SpatialAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim)
        )

    def forward(self, x, rope):
        """
        Args:
            x: (B, T, L, D) or (T, L, D)
            rope: RoPE tensor
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze_out = True
        else:
            squeeze_out = False
        
        res = x
        x = self.attn(x, rotary_pos_emb=rope)
        x = x + res
        
        if squeeze_out:
            x = x.squeeze(0)
        
        return x + self.ffn(x)


class TextConditionedSpatialBlock(nn.Module):
    """Text-conditioned spatial attention block."""
    
    def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.0, ff_mult=4):
        super().__init__()
        self.spatial_attn = SpatialAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout)
        
        self.text_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim),
            nn.Sigmoid()
        )
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim)
        )
        
    def forward(self, x, rope, text_features=None, text_mask=None):
        """
        Args:
            x: (B, T, L, D)
            rope: RoPE tensor
            text_features: (B, N, D)
            text_mask: (B, N)
        """
        res = x
        x = self.spatial_attn(x, rotary_pos_emb=rope)
        x = x + res  # Residual connection for attention
        
        if exists(text_features):
            if exists(text_mask):
                text_mask_f = text_mask.to(text_features.dtype).unsqueeze(-1)
                text_global = (text_features * text_mask_f).sum(dim=1) / text_mask_f.sum(dim=1).clamp_min(1.0)
            else:
                text_global = text_features.mean(dim=1)

            text_gate = self.text_proj(text_global)
            x = x * text_gate.unsqueeze(1).unsqueeze(1)
        
        return x + self.ffn(x)


class FrameTemporalEncoder(nn.Module):
    """Frame-level temporal encoder for per-frame consistency."""
    
    def __init__(self, dim, depth=2, heads=8, dropout=0.0, ff_mult=4):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * ff_mult,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        
    def _sinusoidal_pos_emb(self, t, device, dtype):
        half_dim = self.dim // 2
        if half_dim <= 0:
            return torch.zeros((1, t, self.dim), device=device, dtype=dtype)
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        freq = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * exponent)
        positions = torch.arange(t, device=device, dtype=torch.float32).unsqueeze(1)
        emb = positions * freq.unsqueeze(0)
        sin = torch.sin(emb)
        cos = torch.cos(emb)
        pos = torch.cat([sin, cos], dim=-1)
        if self.dim % 2 == 1:
            pos = torch.cat([pos, torch.zeros((t, 1), device=device, dtype=pos.dtype)], dim=-1)
        return pos.unsqueeze(0).to(dtype)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, D)
            mask: (B, T) with True for valid positions
        """
        key_padding_mask = None if mask is None else ~mask
        pos = self._sinusoidal_pos_emb(x.shape[1], x.device, x.dtype)
        if mask is not None:
            pos = pos * mask.to(pos.dtype).unsqueeze(-1)
        x = x + pos
        
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
        
        return self.norm(x)


class CrossAttentionStack(nn.Module):
    def __init__(self, dim, heads, depth, dropout=0.0, ff_mult=4):
        super().__init__()
        self.attn = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True) for _ in range(depth)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * ff_mult),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * ff_mult, dim),
            )
            for _ in range(depth)
        ])

    def forward(self, x, context, context_mask=None):
        key_padding = None if context_mask is None else ~context_mask
        for attn, norm, ffn in zip(self.attn, self.norms, self.ffns):
            attn_out, _ = attn(x, context, context, key_padding_mask=key_padding, need_weights=False)
            x = norm(x + attn_out)
            x = x + ffn(x)
        return x


class FrameWiseScaleAllocatorLightV2(ModuleUtilsMixin, nn.Module):
    """
    Lightweight frame-scale allocator with grouped processing, frame-level
    temporal encoding, optional text-conditioned spatial attention, and RoPE caching.
    """
    
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        dropout=0.0,
        ff_mult=4,
        cross_attn_depth=1,
        max_frames=4,
        use_discrete_action=False,
        scale_bins=None,
        min_scale=0.25,
        max_scale=3.0,
        spatial_merge_size=2,
        beta_add_one=True,
        frame_temporal_depth=2,
        use_text_conditioned_spatial=False,
        enable_rope_cache=True,
        use_dirichlet_budget=False,
        dirichlet_budget=1.0,
        dirichlet_eps=1e-6,
        use_frame_info=False,
        use_learned_pooling=False,
        frame_info_ema_alpha=0.9,
        init_head=True,
        init_scale_mean=1.0,
        init_concentration=4.0,
        init_weight_std=1e-3,
        dirichlet_logit_noise_std=0.0,
        dirichlet_init_weight_std=None,
        force_uniform_ab=False,
        beta_param_scale=1.0,
        sim_scale_weight=0.1,
        sim_tau=0.5,
        sim_temp=0.1,
        sim_gamma=0.05,
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.max_frames = max_frames
        self.use_discrete_action = use_discrete_action
        self.beta_add_one = beta_add_one
        self.enable_rope_cache = enable_rope_cache
        self.use_dirichlet_budget = use_dirichlet_budget
        self.dirichlet_budget = float(dirichlet_budget)
        self.dirichlet_eps = float(dirichlet_eps)
        self.use_frame_info = use_frame_info
        self.use_learned_pooling = use_learned_pooling
        self.frame_info_ema_alpha = float(frame_info_ema_alpha)
        self.init_head = bool(init_head)
        self.init_scale_mean = float(init_scale_mean)
        self.init_concentration = float(init_concentration)
        self.init_weight_std = float(init_weight_std)
        self.dirichlet_logit_noise_std = float(dirichlet_logit_noise_std)
        self.dirichlet_init_weight_std = None if dirichlet_init_weight_std is None else float(dirichlet_init_weight_std)
        self.force_uniform_ab = bool(force_uniform_ab)
        self.beta_param_scale = float(beta_param_scale)
        self.sim_scale_weight = float(sim_scale_weight)
        self.sim_tau = float(sim_tau)
        self.sim_temp = float(sim_temp)
        self.sim_gamma = float(sim_gamma)
        self._last_sim_scale_loss = None
        self._last_scale_var = None
        self._last_concentration_loss = None
        self.last_entropy = None
        self._last_frame_metrics = None
        self.last_scales = None
        
        if self.use_discrete_action:
            if scale_bins is None:
                import numpy as np
                scale_bins = np.arange(min_scale, max_scale + min_scale, min_scale).tolist()
            self.register_buffer("scale_bins", torch.tensor(scale_bins))
            self.num_bins = len(scale_bins)
            output_dim = self.num_bins
        else:
            self.min_scale = min_scale
            self.max_scale = max_scale
            output_dim = 1 if self.use_dirichlet_budget else 2

        self.rope_gen = RotaryEmbedding(dim=max(32, dim_head // 2))
        self.temporal_rope_gen = RotaryEmbedding(dim=max(32, dim_head))
        self._rope_cache: Dict[str, torch.Tensor] = {}

        self.temporal_blocks = nn.ModuleList([
            TemporalBlockV2(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout, ff_mult=ff_mult)
            for _ in range(depth)
        ])
        if use_text_conditioned_spatial:
            self.spatial_blocks = nn.ModuleList([
                TextConditionedSpatialBlock(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout, ff_mult=ff_mult)
                for _ in range(depth)
            ])
        else:
            self.spatial_blocks = nn.ModuleList([
                SpatialBlockV2(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout, ff_mult=ff_mult)
                for _ in range(depth)
            ])
        
        self.use_text_conditioned_spatial = use_text_conditioned_spatial

        self.frame_temporal_encoder = FrameTemporalEncoder(
            dim=dim, depth=frame_temporal_depth, heads=heads, dropout=dropout, ff_mult=ff_mult
        )

        self.cross_stack = CrossAttentionStack(
            dim=dim,
            heads=heads,
            depth=cross_attn_depth,
            dropout=dropout,
            ff_mult=ff_mult,
        )
        
        if self.use_learned_pooling:
            self.pool_proj = nn.Linear(dim, 1, bias=False)
        else:
            self.pool_proj = None
        if self.use_frame_info:
            self.info_proj = nn.Sequential(
                nn.LayerNorm(dim + 4),
                nn.Linear(dim + 4, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
        else:
            self.info_proj = None
        self.regression_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, output_dim))

    def post_init(self):
        if bool(getattr(self, "force_uniform_ab", False)) and not self.use_discrete_action:
            if isinstance(self.regression_head, nn.Sequential) and isinstance(self.regression_head[-1], nn.Linear):
                head = self.regression_head[-1]
                scale = max(self.beta_param_scale, 1e-6)
                with torch.no_grad():
                    nn.init.zeros_(head.weight)
                    bias_val = -10.0 if self.beta_add_one else float(torch.log(torch.expm1(torch.tensor(1.0))).item())
                    bias_val = bias_val / scale
                    if head.bias is not None and head.bias.numel() >= 2:
                        head.bias.data[0] = bias_val
                        head.bias.data[1] = bias_val
            return
        if not self.init_head or self.use_discrete_action:
            return
        if not (isinstance(self.regression_head, nn.Sequential) and isinstance(self.regression_head[-1], nn.Linear)):
            return
        head = self.regression_head[-1]
        weight_std = self.init_weight_std
        if self.use_dirichlet_budget and self.dirichlet_init_weight_std is not None:
            weight_std = self.dirichlet_init_weight_std
        if head.weight is not None:
            nn.init.normal_(head.weight, mean=0.0, std=weight_std)
        scale = max(self.beta_param_scale, 1e-6)
        add_one = 1.0 if self.beta_add_one else 0.0
        if self.use_dirichlet_budget:
            target_alpha = max(self.init_concentration, add_one + 1e-4)
            bias = torch.log(torch.expm1(torch.tensor(target_alpha - add_one))) / scale
            if head.bias is not None:
                head.bias.data.fill_(bias.item())
            return
        mean_scale = min(max(self.init_scale_mean, self.min_scale), self.max_scale)
        mean_action = (mean_scale - self.min_scale) / max(self.max_scale - self.min_scale, 1e-6)
        mean_action = min(max(mean_action, 1e-4), 1.0 - 1e-4)
        target_alpha = max(mean_action * self.init_concentration, add_one + 1e-4)
        target_beta = max((1.0 - mean_action) * self.init_concentration, add_one + 1e-4)
        bias_alpha = torch.log(torch.expm1(torch.tensor(target_alpha - add_one))) / scale
        bias_beta = torch.log(torch.expm1(torch.tensor(target_beta - add_one))) / scale
        if head.bias is not None and head.bias.numel() >= 2:
            head.bias.data[0] = bias_alpha.item()
            head.bias.data[1] = bias_beta.item()

    def _get_cached_rope(self, seqlen: int, rope_gen, device, dtype):
        """Return cached RoPE to avoid recomputation."""
        if not self.enable_rope_cache:
            return rope_gen(seqlen)
        
        # Use tuple key for faster lookup instead of string formatting
        cache_key = (id(rope_gen), seqlen, str(device), str(dtype))
        if cache_key not in self._rope_cache:
            self._rope_cache[cache_key] = rope_gen(seqlen)
        return self._rope_cache[cache_key]

    def clear_rope_cache(self):
        """Clear RoPE cache."""
        self._rope_cache.clear()

    def get_frame_metrics(self) -> dict:
        return self._last_frame_metrics if self._last_frame_metrics is not None else {}

    def _get_discrete_policy(self, logits, actions=None):
        dist = torch.distributions.Categorical(logits=logits.float())
        if actions is None:
            action_indices = dist.sample()
        else:
            action_indices = actions.long().detach()
        log_prob = dist.log_prob(action_indices).float()
        scale_factors = self.scale_bins[action_indices].to(logits.dtype)
        return action_indices, scale_factors, log_prob.to(torch.float32)

    def _get_discrete_deterministic(self, logits):
        action_indices = torch.argmax(logits, dim=-1)
        scale_factors = self.scale_bins[action_indices]
        return action_indices, scale_factors, None

    def _get_continuous_policy(self, params, actions=None):
        add_one = 1.0 if self.beta_add_one else 0.0
        params_fp32 = params.float() * self.beta_param_scale
        alpha = F.softplus(params_fp32[..., 0]) + add_one + 1e-6
        beta = F.softplus(params_fp32[..., 1]) + add_one + 1e-6
        alpha = alpha.clamp(min=0.1, max=50.0)
        beta = beta.clamp(min=0.1, max=50.0)
        # alpha/beta already float from params_fp32, no need to convert again
        dist = torch.distributions.Beta(alpha, beta)
        if actions is None:
            use_reparam_action = self.sim_scale_weight > 0 and self.training
            action_0_1 = dist.rsample() if use_reparam_action else dist.sample()
        else:
            action_0_1 = actions.float().detach()
        epsilon = 1e-6
        action_0_1 = action_0_1.clamp(min=epsilon, max=1.0 - epsilon)
        log_prob = dist.log_prob(action_0_1).float()
        scale_factors = self.min_scale + action_0_1 * (self.max_scale - self.min_scale)
        return action_0_1.to(params.dtype), scale_factors.to(params.dtype), log_prob.to(torch.float32)

    def _get_continuous_deterministic(self, params):
        add_one = 1.0 if self.beta_add_one else 0.0
        params_fp32 = params.float() * self.beta_param_scale
        alpha = F.softplus(params_fp32[..., 0]) + add_one + 1e-6
        beta = F.softplus(params_fp32[..., 1]) + add_one + 1e-6
        alpha = alpha.clamp(min=0.1, max=50.0)
        beta = beta.clamp(min=0.1, max=50.0)
        # alpha/beta already float from params_fp32, no need to convert again
        dist = torch.distributions.Beta(alpha, beta)
        action_0_1 = dist.mean
        scale_factors = self.min_scale + action_0_1 * (self.max_scale - self.min_scale)
        return action_0_1.to(params.dtype), scale_factors.to(params.dtype), None

    def _get_dirichlet_policy(self, params, object_t_dims, actions=None):
        B, max_t, _ = params.shape
        device = params.device
        action_out = torch.zeros((B, max_t), device=device, dtype=params.dtype)
        scale_out = torch.zeros((B, max_t), device=device, dtype=params.dtype)
        log_prob_out = torch.zeros((B, max_t), device=device, dtype=torch.float32)
        add_one = 1.0 if self.beta_add_one else 0.0
        for i in range(B):
            t = int(object_t_dims[i].item())
            if t <= 0:
                continue
            alpha = F.softplus(params[i, :t, 0].float()) + add_one + 1e-6
            dist = torch.distributions.Dirichlet(alpha)
            if actions is None:
                simplex = dist.rsample()
            else:
                act = actions[i, :t].float().detach()
                simplex = act / (act.sum() + self.dirichlet_eps)
            logp = dist.log_prob(simplex)
            scaled = torch.clamp(self.dirichlet_budget * simplex, min=0.0, max=1.0)
            action_out[i, :t] = scaled.to(action_out.dtype)
            scale_out[i, :t] = (self.min_scale + scaled * (self.max_scale - self.min_scale)).to(scale_out.dtype)
            log_prob_out[i, :t] = logp.to(torch.float32)
        return action_out, scale_out, log_prob_out

    def _get_dirichlet_deterministic(self, params, object_t_dims):
        B, max_t, _ = params.shape
        device = params.device
        action_out = torch.zeros((B, max_t), device=device, dtype=params.dtype)
        scale_out = torch.zeros((B, max_t), device=device, dtype=params.dtype)
        add_one = 1.0 if self.beta_add_one else 0.0
        for i in range(B):
            t = int(object_t_dims[i].item())
            if t <= 0:
                continue
            alpha = F.softplus(params[i, :t, 0]) + add_one + 1e-6
            simplex = alpha / alpha.sum()
            scaled = torch.clamp(self.dirichlet_budget * simplex, min=0.0, max=1.0)
            action_out[i, :t] = scaled
            scale_out[i, :t] = self.min_scale + scaled * (self.max_scale - self.min_scale)
        return action_out, scale_out, None

    def forward(
        self,
        visual_features_batch,
        visual_grid_thw,
        text_features,
        visual_mask=None,
        text_mask=None,
        actions=None,
        visual_per_sample=None,
        eval_mode=False,
        scale_mask=None,
        compute_frame_metrics=False,
    ):
        """
        Returns:
            raw_actions: (Batch, MaxFrames)
            scales: (Batch, MaxFrames)
            log_probs: (Batch, MaxFrames)
            scale_mask_out: (Batch, MaxFrames)
        """
        B, Max_Tokens, D = visual_features_batch.shape
        device = visual_features_batch.device
        dtype = visual_features_batch.dtype
        compute_aux = actions is not None and eval_mode is False
        compute_frame_metrics = compute_frame_metrics and compute_aux
        
        if visual_per_sample is None:
            visual_per_sample = [B]
        elif isinstance(visual_per_sample, torch.Tensor):
            visual_per_sample = visual_per_sample.tolist()
            
        object_t_dims = visual_grid_thw[:, 0].to(torch.long)
        l_raw = (visual_grid_thw[:, 1] * visual_grid_thw[:, 2]).to(torch.long)
        merge_sq = int(self.spatial_merge_size) ** 2
        l_merged = (l_raw // merge_sq).clamp_min(1)
        use_raw = object_t_dims * l_raw <= Max_Tokens
        object_l_dims = torch.where(use_raw, l_raw, l_merged)
        max_t_obj = int(object_t_dims.max().item()) if object_t_dims.numel() > 0 else 0
        
        output_dim = self.regression_head[-1].out_features
        head_outputs_obj = torch.zeros((B, max_t_obj, output_dim), dtype=dtype, device=device)
        frame_features_out = torch.zeros((B, max_t_obj, D), dtype=dtype, device=device)
        if compute_frame_metrics:
            redundancy_obj = torch.zeros((B, max_t_obj), dtype=torch.float32, device=device)
            uniqueness_obj = torch.zeros((B, max_t_obj), dtype=torch.float32, device=device)
            text_relevance_obj = torch.zeros((B, max_t_obj), dtype=torch.float32, device=device)
            info_score_obj = torch.zeros((B, max_t_obj), dtype=torch.float32, device=device)
        
        padded_actions_in = None
        if actions is not None and scale_mask is not None:
            valid_actions_flat = actions[scale_mask]
            if valid_actions_flat.numel() != object_t_dims.sum():
                raise ValueError(
                    f"Number of actions ({valid_actions_flat.numel()}) does not match "
                    f"the number of valid frames ({object_t_dims.sum()})"
                )
            splits = torch.split(valid_actions_flat, object_t_dims.cpu().tolist())
            pad_val = -1 if self.use_discrete_action else 0.0
            padded_actions_in = pad_sequence(list(splits), batch_first=True, padding_value=pad_val)

        pair = torch.stack([object_t_dims, object_l_dims], dim=1)
        unique_pairs, inverse_indices = torch.unique(pair, dim=0, return_inverse=True)
        sim_sum = None
        sim_count = None
        sim_pre_sum = None
        sim_pre_count = None
        
        for group_idx, tl in enumerate(unique_pairs):
            t = int(tl[0].item())
            l = int(tl[1].item())
            
            if t <= 0 or l <= 0:
                continue
            
            obj_idx = (inverse_indices == group_idx).nonzero(as_tuple=False).squeeze(1).long()
            if obj_idx.numel() == 0:
                continue
            feats = visual_features_batch.index_select(0, obj_idx)[:, :t*l].reshape(-1, t, l, D)
            Bgroup = feats.shape[0]
            if compute_frame_metrics and t > 1:
                pre_frame = feats.mean(dim=2)
                ff_pre = F.normalize(pre_frame.float(), dim=-1)
                sim_pre = (ff_pre[:, 1:] * ff_pre[:, :-1]).sum(dim=-1)
                if sim_pre_sum is None:
                    sim_pre_sum = sim_pre.sum()
                    sim_pre_count = sim_pre.numel()
                else:
                    sim_pre_sum = sim_pre_sum + sim_pre.sum()
                    sim_pre_count += sim_pre.numel()
            enc_states = text_features.index_select(0, obj_idx) if exists(text_features) else None
            enc_mask = text_mask.index_select(0, obj_idx) if exists(text_mask) else None
            temporal_rope = self._get_cached_rope(t, self.temporal_rope_gen, device, dtype)
            spatial_rope = self._get_cached_rope(l, self.rope_gen, device, dtype)
            for block in self.temporal_blocks:
                feats = block(feats, temporal_rope)
            if self.use_text_conditioned_spatial and exists(enc_states):
                for block in self.spatial_blocks:
                    feats = block(feats, spatial_rope, enc_states, enc_mask)
            else:
                for block in self.spatial_blocks:
                    feats = block(feats, spatial_rope)
            if self.pool_proj is not None:
                pool_logits = self.pool_proj(feats).squeeze(-1)
                pool_weights = torch.softmax(pool_logits, dim=2)
                frame_feats = (feats * pool_weights.unsqueeze(-1)).sum(dim=2)
                entropy = -(pool_weights * (pool_weights + 1e-9).log()).sum(dim=2)
                entropy = entropy / torch.log(torch.tensor(float(l) + 1e-9, device=device, dtype=entropy.dtype))
            else:
                frame_feats = feats.mean(dim=2)
                entropy = feats.var(dim=2, unbiased=False).mean(dim=-1)
                entropy = torch.log1p(entropy)
            frame_mask = torch.ones(Bgroup, t, dtype=torch.bool, device=device)
            frame_feats = self.frame_temporal_encoder(frame_feats, mask=frame_mask)
            if self.info_proj is not None:
                ff = F.normalize(frame_feats.float(), dim=-1)
                sim_prev = (ff[:, 1:] * ff[:, :-1]).sum(dim=-1)
                sim_prev = F.pad(sim_prev, (1, 0), value=0.0).to(frame_feats.dtype)
                sim_prev[:, 0] = 0.0
                alpha = torch.tensor(self.frame_info_ema_alpha, device=device, dtype=sim_prev.dtype).clamp(min=1e-4, max=0.9999)
                t_idx = torch.arange(t, device=device, dtype=sim_prev.dtype)
                alpha_t = torch.pow(alpha, t_idx)
                inv_alpha_t = torch.pow(alpha, -t_idx)
                s = sim_prev * inv_alpha_t
                prefix = torch.cumsum(s, dim=1)
                ema_sim = alpha_t * sim_prev[:, 0:1] + (1.0 - alpha) * alpha_t * (prefix - s[:, 0:1])
                delta_r = torch.zeros_like(ema_sim)
                delta_r[:, 1:] = ema_sim[:, 1:] - ema_sim[:, :-1]
                info = torch.stack([sim_prev, ema_sim, delta_r, entropy], dim=-1)
                info = F.layer_norm(info, (info.shape[-1],))
                frame_feats = self.info_proj(torch.cat([frame_feats, info.to(frame_feats.dtype)], dim=-1))
            if exists(enc_states):
                frame_feats = self.cross_stack(frame_feats, enc_states, enc_mask)
            if compute_frame_metrics and t > 1:
                ff_eval = F.normalize(frame_feats.float(), dim=-1)
                sim_eval = (ff_eval[:, 1:] * ff_eval[:, :-1]).sum(dim=-1)
                if sim_sum is None:
                    sim_sum = sim_eval.sum()
                    sim_count = sim_eval.numel()
                else:
                    sim_sum = sim_sum + sim_eval.sum()
                    sim_count += sim_eval.numel()
            
            head_outputs = self.regression_head(frame_feats)  # (Bgroup, T, output_dim)
            head_outputs_obj[obj_idx, :t] = head_outputs
            frame_features_out[obj_idx, :t] = frame_feats.to(frame_features_out.dtype)
            if compute_frame_metrics:
                with torch.no_grad():
                    ff_norm = F.normalize(frame_feats.float(), dim=-1)
                    if t > 1:
                        sim_prev = (ff_norm[:, 1:] * ff_norm[:, :-1]).sum(dim=-1)
                        sim_prev = F.pad(sim_prev, (1, 0), value=0.0)
                    else:
                        sim_prev = torch.zeros((Bgroup, t), device=device, dtype=torch.float32)
                    redundancy_local = sim_prev.clamp(-1, 1)
                    if t > 1:
                        sim_matrix = torch.matmul(ff_norm, ff_norm.transpose(-1, -2))
                        eye = torch.eye(t, device=device, dtype=torch.bool).unsqueeze(0)
                        sim_matrix = sim_matrix.masked_fill(eye, 0.0)
                        mean_sim = sim_matrix.sum(dim=-1) / (t - 1)
                        uniqueness_local = (1.0 - mean_sim).clamp(0.0, 1.0)
                    else:
                        uniqueness_local = torch.ones((Bgroup, t), device=device, dtype=torch.float32)
                    if enc_states is not None:
                        if enc_mask is not None:
                            m = enc_mask.to(enc_states.dtype)
                            denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
                            text_mean = (enc_states * m.unsqueeze(-1)).sum(dim=1) / denom
                        else:
                            text_mean = enc_states.mean(dim=1)
                        text_relevance_local = F.cosine_similarity(frame_feats.float(), text_mean.float().unsqueeze(1), dim=-1).clamp(-1, 1)
                    else:
                        text_relevance_local = torch.zeros((Bgroup, t), device=device, dtype=torch.float32)
                    if self.info_proj is not None:
                        info_score_local = info.float().mean(dim=-1)
                    else:
                        info_score_local = entropy.float()
                    redundancy_obj[obj_idx, :t] = redundancy_local
                    uniqueness_obj[obj_idx, :t] = uniqueness_local
                    text_relevance_obj[obj_idx, :t] = text_relevance_local
                    info_score_obj[obj_idx, :t] = info_score_local

        splits = torch.split(object_t_dims, visual_per_sample)
        valid_lengths = torch.stack([s.sum() for s in splits]).to(device)
        pad_max = max(self.max_frames, int(valid_lengths.max().item())) if valid_lengths.numel() > 0 else self.max_frames
        scale_mask_out = torch.arange(pad_max, device=device)[None, :] < valid_lengths[:, None]
        temporal_valid = torch.arange(max_t_obj, device=device)[None, :] < object_t_dims[:, None]
        if self.use_dirichlet_budget and self.training and self.dirichlet_logit_noise_std > 0:
            head_outputs_obj = head_outputs_obj + torch.randn_like(head_outputs_obj) * self.dirichlet_logit_noise_std
        if compute_aux:
            entropy_obj = None
            if self.use_discrete_action:
                entropy_obj = torch.distributions.Categorical(logits=head_outputs_obj.float()).entropy().to(head_outputs_obj.dtype)
            elif self.use_dirichlet_budget:
                entropy_obj = torch.zeros((B, max_t_obj), device=device, dtype=head_outputs_obj.dtype)
                add_one = 1.0 if self.beta_add_one else 0.0
                for i in range(B):
                    t = int(object_t_dims[i].item())
                    if t <= 0:
                        continue
                    alpha = F.softplus(head_outputs_obj[i, :t, 0].float()) + add_one + 1e-6
                    entropy_val = torch.distributions.Dirichlet(alpha).entropy()
                    entropy_obj[i, :t] = entropy_val.to(entropy_obj.dtype)
            else:
                add_one = 1.0 if self.beta_add_one else 0.0
                alpha = F.softplus(head_outputs_obj[..., 0].float()) + add_one + 1e-6
                beta = F.softplus(head_outputs_obj[..., 1].float()) + add_one + 1e-6
                entropy_obj = torch.distributions.Beta(alpha, beta).entropy().to(head_outputs_obj.dtype)
            if entropy_obj is not None:
                entropy_out = self.restructure_sequence(entropy_obj, visual_grid_thw, scale_mask_out, 0.0, device)
                self.last_entropy = entropy_out.to(torch.float32) if entropy_out is not None else None
            else:
                self.last_entropy = None
            self._last_concentration_loss = None
            if not self.use_discrete_action:
                add_one = 1.0 if self.beta_add_one else 0.0
                if self.use_dirichlet_budget:
                    alpha = F.softplus(head_outputs_obj[..., 0]) + add_one + 1e-6
                    m = temporal_valid.to(alpha.dtype)
                    denom = m.sum().clamp_min(1.0)
                    self._last_concentration_loss = (alpha * m).sum() / denom
                else:
                    alpha = F.softplus(head_outputs_obj[..., 0]) + add_one + 1e-6
                    beta = F.softplus(head_outputs_obj[..., 1]) + add_one + 1e-6
                    concentration = alpha + beta
                    m = temporal_valid.to(concentration.dtype)
                    denom = m.sum().clamp_min(1.0)
                    self._last_concentration_loss = (concentration * m).sum() / denom
        else:
            self.last_entropy = None
            self._last_concentration_loss = None
        if compute_aux and self.sim_scale_weight > 0 and self.training and frame_features_out.shape[1] > 1 and not self.use_dirichlet_budget:
            f0 = frame_features_out[:, :-1]
            f1 = frame_features_out[:, 1:]
            sim = F.cosine_similarity(f0, f1, dim=-1)
            m = (temporal_valid[:, :-1] & temporal_valid[:, 1:]).float()
            sim = sim * m
            w = torch.sigmoid((sim - self.sim_tau) / self.sim_temp)
            w = w * m
            if self.use_discrete_action:
                probs = head_outputs_obj.softmax(dim=-1)
                soft_scales_obj = (probs * self.scale_bins).sum(dim=-1)
            else:
                add_one = 1.0 if self.beta_add_one else 0.0
                alpha = F.softplus(head_outputs_obj[..., 0].float()) + add_one + 1e-6
                beta = F.softplus(head_outputs_obj[..., 1].float()) + add_one + 1e-6
                alpha = alpha.clamp(min=0.1, max=50.0)
                beta = beta.clamp(min=0.1, max=50.0)
                mean_action = alpha / (alpha + beta)
                soft_scales_obj = self.min_scale + mean_action * (self.max_scale - self.min_scale)
            s = soft_scales_obj.clamp_min(1e-6).log()
            target = s[:, :-1] - self.sim_gamma * w
            err = F.relu(s[:, 1:] - target)
            self._last_sim_scale_loss = (err * w).sum() / (m.sum().clamp_min(1.0))
            # print(f"[AZNetv4] sim_scale_loss={self._last_sim_scale_loss:.6f}")
        else:
            self._last_sim_scale_loss = None
        padded_actions_for_policy = padded_actions_in
        if self.use_discrete_action and padded_actions_for_policy is not None:
            padded_actions_for_policy = padded_actions_for_policy.masked_fill(padded_actions_for_policy < 0, 0)
        if self.use_discrete_action:
            if eval_mode:
                raw_actions_obj, scales_obj, log_probs_obj = self._get_discrete_deterministic(head_outputs_obj)
            else:
                raw_actions_obj, scales_obj, log_probs_obj = self._get_discrete_policy(head_outputs_obj, padded_actions_for_policy)
            pad_val_action = -1
        else:
            if self.use_dirichlet_budget:
                if eval_mode:
                    raw_actions_obj, scales_obj, log_probs_obj = self._get_dirichlet_deterministic(head_outputs_obj, object_t_dims)
                else:
                    raw_actions_obj, scales_obj, log_probs_obj = self._get_dirichlet_policy(
                        head_outputs_obj, object_t_dims, padded_actions_for_policy
                    )
            else:
                if eval_mode:
                    raw_actions_obj, scales_obj, log_probs_obj = self._get_continuous_deterministic(head_outputs_obj)
                else:
                    raw_actions_obj, scales_obj, log_probs_obj = self._get_continuous_policy(head_outputs_obj, padded_actions_for_policy)
            pad_val_action = 0.0
        raw_actions_obj = raw_actions_obj.to(torch.long if self.use_discrete_action else torch.float32)
        scales_obj = scales_obj.to(torch.float32)
        if log_probs_obj is not None:
            log_probs_obj = log_probs_obj.to(torch.float32)
        
        raw_actions_obj = raw_actions_obj.masked_fill(~temporal_valid, pad_val_action)
        scales_obj = scales_obj.masked_fill(~temporal_valid, 1.0)
        if log_probs_obj is not None:
            log_probs_obj = log_probs_obj.masked_fill(~temporal_valid, 0.0)
        m_var = temporal_valid.to(scales_obj.dtype)
        denom_var = m_var.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_var = (scales_obj * m_var).sum(dim=1, keepdim=True) / denom_var
        var = ((scales_obj - mean_var) ** 2 * m_var).sum(dim=1, keepdim=True) / denom_var
        if compute_aux:
            self._last_scale_var = var.mean()
        else:
            self._last_scale_var = None
        if compute_frame_metrics:
            sim_mean = (sim_sum / max(sim_count, 1)).item() if sim_sum is not None else 0.0
            sim_pre_mean = (sim_pre_sum / max(sim_pre_count, 1)).item() if sim_pre_sum is not None else 0.0
            self._last_frame_metrics = {
                "frame_sim": sim_mean,
                "frame_sim_pre": sim_pre_mean,
                "redundancy": self.restructure_sequence(redundancy_obj, visual_grid_thw, scale_mask_out, 0.0, device),
                "uniqueness": self.restructure_sequence(uniqueness_obj, visual_grid_thw, scale_mask_out, 0.5, device),
                "text_relevance": self.restructure_sequence(text_relevance_obj, visual_grid_thw, scale_mask_out, 0.0, device),
                "info_score": self.restructure_sequence(info_score_obj, visual_grid_thw, scale_mask_out, 0.0, device),
            }
        else:
            self._last_frame_metrics = None
        if eval_mode:
            valid_scales = scales_obj[temporal_valid]
            scale_var = valid_scales.var(unbiased=False).item() if valid_scales.numel() > 0 else 0.0
            sim_mean = (sim_sum / max(sim_count, 1)).item() if sim_sum is not None else 0.0
            sim_pre_mean = (sim_pre_sum / max(sim_pre_count, 1)).item() if sim_pre_sum is not None else 0.0
            # print(f"[scale_eval] var={scale_var:.6f} frame_sim={sim_mean:.6f} frame_sim_pre={sim_pre_mean:.6f}")
        
        log_probs = self.restructure_sequence(log_probs_obj, visual_grid_thw, scale_mask_out, 0.0, device) if log_probs_obj is not None else None
        
        if actions is not None:
            return None, None, log_probs, None
        
        raw_actions = self.restructure_sequence(raw_actions_obj, visual_grid_thw, scale_mask_out, pad_val_action, device)
        scales = self.restructure_sequence(scales_obj, visual_grid_thw, scale_mask_out, 1.0, device)
        self.last_scales = scales
        
        return raw_actions, scales, log_probs, scale_mask_out

    def restructure_sequence(self, tensor, visual_grid_thw, target_mask, pad_val, device):
        """
        Args:
            tensor: (Total_Objects, Max_T, ...)
            visual_grid_thw: (Total_Objects, 3) - T, H, W
            target_mask: (Batch, Max_Frames)
            pad_val: padding value
        """
        if tensor is None:
            return None
        
        object_t_dims = visual_grid_thw[:, 0]
        max_len = tensor.shape[1]
        source_mask = torch.arange(max_len, device=device)[None, :] < object_t_dims[:, None]
        valid_values = tensor[source_mask]
        
        output = torch.full(target_mask.shape, pad_val, dtype=tensor.dtype, device=device)
        output[target_mask] = valid_values
        return output


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Testing FrameWiseScaleAllocatorLightV2")
    print("=" * 60)
    
    DIM = 512
    MAX_FRAMES = 32
    PATCHES_PER_FRAME = 64
    MAX_TEXT_LEN = 128
    BATCH_SIZE = 4
    
    print("\n--- Testing Basic Version ---")
    model = FrameWiseScaleAllocatorLightV2(
        dim=DIM,
        depth=2,
        dim_head=64,
        heads=8,
        max_frames=MAX_FRAMES,
        spatial_merge_size=2,
        use_discrete_action=False,
        min_scale=0.2,
        max_scale=2.0,
        frame_temporal_depth=2,
        use_text_conditioned_spatial=False,
        enable_rope_cache=True,
    ).to(device)
    
    visual_features_batch = torch.randn(BATCH_SIZE, 6 * PATCHES_PER_FRAME, DIM).to(device)
    visual_grid_thw = torch.tensor([
        [4, 16, 16],  # T=4, H=16, W=16
        [6, 16, 16],  # T=6
        [4, 16, 16],  # T=4
        [6, 16, 16],  # T=6
    ]).to(device)
    text_features = torch.randn(BATCH_SIZE, MAX_TEXT_LEN, DIM).to(device)
    text_mask = torch.ones(BATCH_SIZE, MAX_TEXT_LEN, dtype=torch.bool).to(device)
    
    raw_actions, scales, log_probs, scale_mask = model(
        visual_features_batch=visual_features_batch,
        visual_grid_thw=visual_grid_thw,
        text_features=text_features,
        text_mask=text_mask,
        visual_per_sample=[1, 1, 1, 1],  # 4 个样本，每个 1 个 object
    )
    
    print(f"Output shapes:")
    print(f"  raw_actions: {raw_actions.shape}")
    print(f"  scales: {scales.shape}")
    print(f"  log_probs: {log_probs.shape if log_probs is not None else None}")
    print(f"  scale_mask: {scale_mask.shape}")
    
    valid_scales = scales[scale_mask]
    print(f"\nScale statistics:")
    print(f"  Min: {valid_scales.min().item():.4f}")
    print(f"  Max: {valid_scales.max().item():.4f}")
    print(f"  Mean: {valid_scales.mean().item():.4f}")
    print(f"  In range [0.2, 2.0]: {(valid_scales >= 0.2).all() and (valid_scales <= 2.0).all()}")
    
    print(f"\nTemporal consistency check:")
    for i in range(BATCH_SIZE):
        sample_scales = scales[i, scale_mask[i]]
        if len(sample_scales) > 1:
            scale_diff = (sample_scales[1:] - sample_scales[:-1]).abs().mean()
            print(f"  Sample {i}: mean scale diff = {scale_diff.item():.4f}")
    
    # 测试文本感知版本
    print("\n--- Testing Text-Conditioned Version ---")
    model_tc = FrameWiseScaleAllocatorLightV2(
        dim=DIM,
        depth=2,
        dim_head=64,
        heads=8,
        max_frames=MAX_FRAMES,
        spatial_merge_size=2,
        use_discrete_action=False,
        min_scale=0.2,
        max_scale=2.0,
        frame_temporal_depth=2,
        use_text_conditioned_spatial=True,  # 启用文本感知
        enable_rope_cache=True,
    ).to(device)
    
    raw_actions_tc, scales_tc, log_probs_tc, scale_mask_tc = model_tc(
        visual_features_batch=visual_features_batch,
        visual_grid_thw=visual_grid_thw,
        text_features=text_features,
        text_mask=text_mask,
        visual_per_sample=[1, 1, 1, 1],
    )
    
    print(f"Text-conditioned model output:")
    print(f"  scales shape: {scales_tc.shape}")
    print(f"  mean scale: {scales_tc[scale_mask_tc].mean().item():.4f}")
    
    # 测试离散动作版本
    print("\n--- Testing Discrete Action Version ---")
    model_discrete = FrameWiseScaleAllocatorLightV2(
        dim=DIM,
        depth=2,
        dim_head=64,
        heads=8,
        max_frames=MAX_FRAMES,
        spatial_merge_size=2,
        use_discrete_action=True,
        scale_bins=[0.25, 0.5, 1.0, 1.5, 2.0],
        frame_temporal_depth=2,
    ).to(device)
    
    raw_actions_d, scales_d, log_probs_d, scale_mask_d = model_discrete(
        visual_features_batch=visual_features_batch,
        visual_grid_thw=visual_grid_thw,
        text_features=text_features,
        text_mask=text_mask,
        visual_per_sample=[1, 1, 1, 1],
    )
    
    print(f"Discrete model output:")
    print(f"  scales shape: {scales_d.shape}")
    print(f"  unique scale values: {torch.unique(scales_d[scale_mask_d]).tolist()}")
    
    # 测试梯度流
    print("\n--- Testing Gradient Flow ---")
    model.train()
    raw_actions, scales, log_probs, scale_mask = model(
        visual_features_batch=visual_features_batch,
        visual_grid_thw=visual_grid_thw,
        text_features=text_features,
        text_mask=text_mask,
        visual_per_sample=[1, 1, 1, 1],
    )
    
    # 模拟损失
    loss = scales[scale_mask].mean()
    loss.backward()
    
    # 检查梯度
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    print(f"  Parameters with grad: {has_grad}/{total_params}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
