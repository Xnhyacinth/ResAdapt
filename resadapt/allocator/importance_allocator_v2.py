"""
DifferentiableImportanceAllocator: A frame importance allocator that captures
per-frame information content and text relevance to output differentiated scales.

Key improvements over FrameWiseScaleAllocator:
1. Frame Information Encoder - evaluates per-frame information content
2. Cross-Modal Matcher - computes frame-text relevance independently
3. Sparse Gating - Gumbel-Softmax/Top-k for sharper attention
4. Dual-Path Encoder - preserves frame-independent features
5. Contrastive Differentiator - forces feature diversity across frames

Supports Flash Attention via PyTorch 2.0+ SDPA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.distributions import Beta, Normal, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform
from transformers.modeling_utils import ModuleUtilsMixin
from torch.nn.utils.rnn import pad_sequence
import math

# Flash attention support via SDPA (PyTorch 2.0+)
_SDPA_AVAILABLE = hasattr(F, 'scaled_dot_product_attention')

def exists(val):
    return val is not None


# ============================================================================
# Module 1: Frame Information Encoder
# ============================================================================

class FrameInformationEncoder(nn.Module):
    """
    Computes per-frame information metrics from patch-level features.
    
    Outputs:
        - Patch variance (how diverse are patches within frame)
        - Feature norm variance (magnitude diversity)
        - Attention entropy (how focused is self-attention)
        - Peak response (max patch activation)
    """
    
    def __init__(self, dim: int, info_dim: int = 8):
        super().__init__()
        self.info_dim = info_dim
        
        # Learnable projection from raw stats to info embedding
        # Now 5 features: patch_var, norm_var, concentration, gini, peak
        self.info_proj = nn.Sequential(
            nn.Linear(5, info_dim),
            nn.GELU(),
            nn.Linear(info_dim, info_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, L, D) patch features
        Returns:
            info: (B, T, info_dim) information scores per frame
        
        Uses O(L) proxies instead of O(L²) attention.
        Optimized: replaced O(L log L) sort with O(L) MAD approximation.
        Memory-optimized: reuses intermediate computations.
        """
        B, T, L, D = x.shape
        
        # 1. Patch variance within each frame (reuse centered data)
        x_mean = x.mean(dim=2, keepdim=True)  # (B, T, 1, D)
        x_centered = x - x_mean
        patch_var = (x_centered ** 2).mean(dim=(2, 3))  # (B, T)
        
        # 2. Feature norm variance (compute norms once)
        # Use squared norms to avoid sqrt, then take variance
        x_float = x.float()
        norms_sq = (x_float ** 2).sum(dim=-1)  # (B, T, L) - squared norms
        norms = norms_sq.sqrt()  # (B, T, L)
        norm_var = norms.var(dim=-1, unbiased=False)  # (B, T)
        
        # 3. TopK concentration (O(L) proxy for sparsity/focus)
        # Reuse norms as activation proxy instead of computing abs().mean() separately
        a = norms  # (B, T, L) - use norm as activation magnitude
        k = max(1, L // 8)
        topk_vals = a.topk(k, dim=-1).values.mean(dim=-1)  # (B, T)
        mean_a = a.mean(dim=-1) + 1e-6
        concentration = topk_vals / mean_a  # High = sparse/focused, Low = uniform
        
        # 4. Sparsity proxy via MAD (O(L) - replaces O(L log L) sort-based Gini)
        # MAD = Mean Absolute Deviation, correlates with Gini for unimodal distributions
        mad = (a - mean_a.unsqueeze(-1)).abs().mean(dim=-1)  # (B, T)
        sparsity = mad / (mean_a + 1e-6)  # Normalized MAD as sparsity proxy
        
        # 5. Peak response (max activation) - reuse norms
        peak = norms.max(dim=-1).values  # (B, T)
        
        # Stack all metrics as float32 for consistency
        raw_stats = torch.stack([
            patch_var.float(), norm_var.float(), concentration.float(), 
            sparsity.float(), peak.float()
        ], dim=-1)  # (B, T, 5)
        info = self.info_proj(raw_stats.to(self.info_proj[0].weight.dtype))  # (B, T, info_dim)
        
        return info


# ============================================================================
# Module 2: Cross-Modal Matcher
# ============================================================================

class CrossModalMatcher(nn.Module):
    """
    Computes per-frame relevance score to text query.
    Uses Flash Attention via SDPA when available.
    """
    
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_p = dropout
        
        self.norm_frame = nn.LayerNorm(dim)
        self.norm_text = nn.LayerNorm(dim)
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        
        # Relevance scoring head
        self.score_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
    
    def forward(
        self, 
        frame_features: torch.Tensor,  # (B, T, D)
        text_features: torch.Tensor,   # (B, N, D)
        text_mask: torch.Tensor = None # (B, N)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            relevance_scores: (B, T) - how relevant each frame is to text
            attended_features: (B, T, D) - text-attended frame representations
        """
        B, T, D = frame_features.shape
        _, N, _ = text_features.shape
        
        frame_norm = self.norm_frame(frame_features)
        text_norm = self.norm_text(text_features)
        
        q = self.to_q(frame_norm)  # (B, T, D)
        kv = self.to_kv(text_norm)  # (B, N, D*2)
        k, v = kv.chunk(2, dim=-1)
        
        # Reshape for multi-head attention: (B, H, T/N, D_head)
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # SDPA with flash attention support
        if _SDPA_AVAILABLE:
            # Convert mask for SDPA: (B, N) -> (B, 1, T, N) as float mask
            # SDPA expects additive mask (0 = keep, -inf = mask)
            # Optimized: avoid intermediate tensors by using log directly
            attn_mask = None
            if text_mask is not None:
                if text_mask.dim() == 2:
                    invalid_rows = text_mask.sum(dim=1) == 0
                    if invalid_rows.any():
                        text_mask = text_mask.clone()
                        text_mask[invalid_rows, 0] = True
                mask_bool = text_mask.unsqueeze(1).unsqueeze(2)
                attn_mask = torch.where(
                    mask_bool,
                    q.new_zeros(1),
                    q.new_full((1,), float("-inf"))
                ).expand(B, 1, T, N)
            
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
            )
        else:
            # Fallback to manual attention
            attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            if text_mask is not None:
                if text_mask.dim() == 2:
                    invalid_rows = text_mask.sum(dim=1) == 0
                    if invalid_rows.any():
                        text_mask = text_mask.clone()
                        text_mask[invalid_rows, 0] = True
                mask = text_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
                attn = attn.masked_fill(~mask, float('-inf'))
            attn_probs = F.softmax(attn, dim=-1)
            if self.training and self.dropout_p > 0:
                attn_probs = F.dropout(attn_probs, p=self.dropout_p)
            out = torch.matmul(attn_probs, v)
        
        out = rearrange(out, 'b h t d -> b t (h d)')
        
        # Residual connection + relevance scoring
        out_with_residual = out + frame_norm
        relevance_scores = self.score_head(out_with_residual).squeeze(-1)  # (B, T)
        
        return relevance_scores, out_with_residual


# ============================================================================
# Module 3: Sparse Gating
# ============================================================================

class SparseGatingModule(nn.Module):
    """
    Implements sparse attention gating with multiple modes:
    - gumbel_softmax: Differentiable sampling with temperature
    - top_k: Keep only top-k positions
    - sparsemax: Sparse alternative to softmax
    """
    
    def __init__(self, dim: int, mode: str = "gumbel_softmax", k_ratio: float = 0.25):
        super().__init__()
        self.mode = mode
        self.k_ratio = k_ratio
        
        # Learnable temperature (log-scale for numerical stability)
        self.log_temperature = nn.Parameter(torch.zeros(1))  # exp(0) = 1.0
        
        # Gate projection with LayerNorm for stability
        self.gate_proj = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1, bias=False),
        )
    
    def forward(
        self, 
        features: torch.Tensor,  # (B, T, L, D)
        query: torch.Tensor,     # (B, T, D) - query for gating
        hard: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            gate: (B, T, L) - sparse attention weights
            pooled: (B, T, D) - gated pooled features
        """
        B, T, L, D = features.shape
        
        # Expand query for concatenation
        q_expanded = query.unsqueeze(2).expand(B, T, L, D)
        gate_input = torch.cat([features, q_expanded], dim=-1)
        
        gate_logits = self.gate_proj(gate_input).squeeze(-1)  # (B, T, L)
        
        # Temperature from log-scale (clamp for stability: 0.1 to 2.0)
        # NOTE: tau gradient does not flow through gumbel_softmax (PyTorch limitation)
        # The temperature is learned indirectly through its effect on the logits scale
        # If you need tau gradient, implement custom gumbel_softmax with tau as tensor
        tau = self.log_temperature.exp().clamp(min=0.1, max=2.0)
        if self.mode == "gumbel_softmax":
            # Custom Gumbel-Softmax to support differentiable temperature (tensor tau)
            # PyTorch's F.gumbel_softmax requires scalar tau, so we implement the reparameterization manually
            # gumbels = -log(-log(uniform(0,1)))
            gumbels = -torch.empty_like(gate_logits).exponential_().log()  # ~Gumbel(0,1)
            gumbels = (gate_logits + gumbels) / tau  # Divide by tensor tau (differentiable)
            y_soft = gumbels.softmax(dim=-1)
            
            if hard:
                # Straight-through estimator
                index = y_soft.max(dim=-1, keepdim=True)[1]
                y_hard = torch.zeros_like(gate_logits).scatter_(-1, index, 1.0)
                gate = y_hard - y_soft.detach() + y_soft
            else:
                gate = y_soft
        elif self.mode == "top_k":
            k = max(1, int(L * self.k_ratio))
            topk_vals, topk_idx = gate_logits.topk(k, dim=-1)
            sparse_mask = torch.zeros_like(gate_logits).scatter(-1, topk_idx, 1.0)
            gate = F.softmax(gate_logits.masked_fill(sparse_mask == 0, float('-inf')), dim=-1)
        else:  # sparsemax-like via ReLU thresholding
            gate = F.softmax(gate_logits / tau, dim=-1)
            threshold = gate.mean(dim=-1, keepdim=True)
            gate = F.relu(gate - threshold * 0.5)
            gate = gate / (gate.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Weighted pooling with gate
        pooled = (gate.unsqueeze(-1) * features).sum(dim=2)  # (B, T, D)
        
        return gate, pooled


# ============================================================================
# Module 4: Dual-Path Encoder
# ============================================================================

class DualPathEncoder(nn.Module):
    """
    Encodes frame features through two parallel paths:
    1. Frame-Independent Path: No temporal interaction, preserves per-frame uniqueness
    2. Temporal-Aware Path: Lightweight temporal attention for context
    
    Uses per-frame SCALAR gate (not per-dimension) to avoid over-smoothing.
    Gate is biased towards independent path by default.
    """
    
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0, temporal_depth: int = 1):
        super().__init__()
        self.dropout = dropout
        
        # Path 1: Frame-Independent (preserves uniqueness)
        self.independent_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        
        # Path 2: Temporal-Aware (lightweight attention for context)
        depth = max(int(temporal_depth), 0)
        self.temporal_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.temporal_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            for _ in range(depth)
        ])
        self.temporal_ff = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 2, dim),
            )
            for _ in range(depth)
        ])
        self.temporal_out_dropout = nn.Dropout(dropout)
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        # Initialize bias > 0 to favor independent path initially
        nn.init.constant_(self.fusion_gate[-1].bias, 1.0)
        
        # Decorrelation projection to encourage diversity
        self.decorr_proj = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.decorr_proj.weight)
    
    def forward(
        self, 
        x: torch.Tensor,  # (B, T, D)
        temporal_mask: torch.Tensor = None  # (B, T)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused: (B, T, D) - combined features
            independent: (B, T, D) - frame-independent features (for contrastive)
            temporal: (B, T, D) - temporal-aware features
        """
        # Path 1: Independent
        independent = self.independent_proj(x)
        independent = self.decorr_proj(independent)  # Encourage orthogonality
        
        # Path 2: Temporal
        temporal = x
        key_padding_mask = None
        if temporal_mask is not None:
            key_padding_mask = ~temporal_mask
        for norm, attn, ffn in zip(self.temporal_norms, self.temporal_attn, self.temporal_ff):
            x_norm = norm(temporal)
            attn_out, _ = attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
            temporal = temporal + self.temporal_out_dropout(attn_out)
            temporal = temporal + self.temporal_out_dropout(ffn(temporal))
        
        concat = torch.cat([independent, temporal], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(concat))
        fused = gate * independent + (1 - gate) * temporal
        
        return fused, independent, temporal


# ============================================================================
# Module 5: Contrastive Differentiator
# ============================================================================

class ContrastiveDifferentiator(nn.Module):
    """
    Applies contrastive loss to encourage frame feature diversity.
    Uses logsumexp instead of max for denser gradients.
    """
    
    def __init__(self, dim: int, temperature: float = 0.1, margin: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
        # Projection head for contrastive learning
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
        )
    
    def forward(self, features: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Memory-efficient contrastive loss using neighbor pairs instead of full T×T matrix.
        
        Args:
            features: (B, T, D) frame features
            mask: (B, T) valid frame mask
        Returns:
            loss: scalar contrastive loss (minimize to increase diversity)
        """
        B, T, D = features.shape
        
        if T <= 1:
            # Return zero tensor without unnecessary computation graph attachment
            # Use zeros with requires_grad only during training to allow gradient flow if needed
            return torch.zeros(1, device=features.device, dtype=features.dtype, 
                              requires_grad=self.training)
        
        # Project and normalize
        z = self.proj(features)  # (B, T, D//2)
        z = F.normalize(z, dim=-1, eps=1e-6)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Memory-efficient: only compute neighbor similarities O(T) instead of O(T²)
        # This encourages diversity between consecutive frames
        z_prev = z[:, :-1]  # (B, T-1, D//2)
        z_next = z[:, 1:]   # (B, T-1, D//2)
        
        # Cosine similarity between consecutive frames
        sim = (z_prev * z_next).sum(dim=-1) / self.temperature  # (B, T-1)
        sim = F.relu(sim - self.margin)
        
        # Apply valid frame mask if provided
        if mask is not None:
            valid_pairs = mask[:, :-1] & mask[:, 1:]  # (B, T-1)
            sim = sim.masked_fill(~valid_pairs, 0.0)
            valid_count = valid_pairs.float().sum().clamp(min=1.0)
            loss_local = sim.sum() / valid_count
        else:
            loss_local = sim.mean()
            
        # Global diversity: Sample random non-adjacent pairs (O(T) complexity)
        # This encourages diversity beyond just neighbors, preventing global collapse
        if T > 2:
            # Random permutation for efficient pair sampling
            idx1 = torch.randperm(T, device=features.device)
            idx2 = torch.roll(idx1, shifts=1) # Offset by 1 in permutation space (essentially random pair)
            
            # Ensure we don't pick adjacent frames (which are already covered)
            # Simple heuristic: if difference is 1, skip (masking)
            diff = (idx1 - idx2).abs()
            non_adjacent_mask = (diff > 1)
            
            if non_adjacent_mask.any():
                z1 = z[:, idx1]
                z2 = z[:, idx2]
                sim_global = (z1 * z2).sum(dim=-1) / self.temperature
                sim_global = F.relu(sim_global - self.margin)
                
                if mask is not None:
                     # Check validity of both frames
                    m1 = mask[:, idx1]
                    m2 = mask[:, idx2]
                    valid_global = m1 & m2 & non_adjacent_mask.unsqueeze(0)
                    sim_global = sim_global.masked_fill(~valid_global, 0.0)
                    valid_global_count = valid_global.float().sum().clamp(min=1.0)
                    loss_global_val = sim_global.sum() / valid_global_count
                else:
                    sim_global = sim_global.masked_fill(~non_adjacent_mask.unsqueeze(0), 0.0)
                    loss_global_val = sim_global.sum() / non_adjacent_mask.float().sum().clamp(min=1.0)
                
                # Combine local and global loss (equal weight)
                loss = (loss_local + loss_global_val) * 0.5
            else:
                loss = loss_local
        else:
            loss = loss_local
        
        return loss


# ============================================================================
# Main Allocator: DifferentiableImportanceAllocator
# ============================================================================

class DifferentiableImportanceAllocator(ModuleUtilsMixin, nn.Module):
    """
    A allocator that captures per-frame importance based on:
    1. Frame information content (intrinsic)
    2. Text relevance (extrinsic)
    
    Produces differentiated scale factors for each frame.
    Compatible with discrete (Categorical) and continuous (Beta/LogisticNormal) actions.
    """
    
    def __init__(
        self,
        *,
        dim: int,
        depth: int = 2,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
        ff_mult: int = 4,
        cross_attn_depth: int = 1,
        max_frames: int = 32,
        spatial_merge_size: int = 2,
        # Action space config
        use_discrete_action: bool = False,
        scale_bins: list = None,
        min_scale: float = 0.25,
        max_scale: float = 3.0,
        # Gating config
        gate_mode: str = "gumbel_softmax",
        gate_k_ratio: float = 0.25,
        # Distribution config
        continuous_dist: str = "beta",
        continuous_eval_quantile: float = 0.5,
        beta_param_scale: float = 1.0,
        beta_add_one: bool = True,
        logistic_normal_init_sigma: float = 0.7,
        # Contrastive config
        contrastive_weight: float = 0.1,
        contrastive_temperature: float = 0.1,
        contrastive_margin: float = 0.0,
        # Similarity-based scale regularization (compress redundant frames)
        sim_scale_weight: float = 0.1,  # Weight for pairwise sim loss
        sim_tau: float = 0.5,           # Similarity threshold
        sim_temp: float = 0.1,          # Sigmoid temperature
        sim_gamma: float = 0.05,        # Scale decrease magnitude
        temporal_mixer_depth: int = 1,
        temporal_use_pos: bool = True,
        dual_path_depth: int = 1,
    ):
        super().__init__()
        
        self.dim = dim
        self.max_frames = max_frames
        self.spatial_merge_size = spatial_merge_size
        self.use_discrete_action = use_discrete_action
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.continuous_dist = continuous_dist
        self.continuous_eval_quantile = continuous_eval_quantile
        self.beta_param_scale = beta_param_scale
        self.beta_add_one = beta_add_one
        self.logistic_normal_init_sigma = logistic_normal_init_sigma
        self.sim_scale_weight = sim_scale_weight
        self.sim_tau = sim_tau
        self.sim_temp = sim_temp
        self.sim_gamma = sim_gamma
        self.temporal_use_pos = temporal_use_pos
        self.contrastive_weight = contrastive_weight
        
        # Discrete action setup
        if use_discrete_action:
            if scale_bins is None:
                import numpy as np
                scale_bins = np.arange(min_scale, max_scale + min_scale, min_scale).tolist()
            self.register_buffer('scale_bins', torch.tensor(scale_bins))
            self.num_bins = len(scale_bins)
            output_dim = self.num_bins
        else:
            output_dim = 2
        
        # ===== Core Modules =====
        
        info_dim = 8
        self.info_encoder = FrameInformationEncoder(dim=dim, info_dim=info_dim)
        self.info_to_query = nn.Linear(info_dim, dim)
        
        # 2. Cross-Modal Matcher
        self.cross_matcher = CrossModalMatcher(dim=dim, num_heads=max(1, heads // 2), dropout=dropout)
        
        # 3. Sparse Gating
        self.sparse_gate = SparseGatingModule(dim=dim, mode=gate_mode, k_ratio=gate_k_ratio)
        
        # 4. Dual-Path Encoder
        self.dual_encoder = DualPathEncoder(
            dim=dim,
            num_heads=max(1, heads // 2),
            dropout=dropout,
            temporal_depth=dual_path_depth,
        )
        self.temporal_mixer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=max(1, heads // 2),
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(max(int(temporal_mixer_depth), 0))
        ])
        
        # 5. Contrastive Differentiator
        self.contrastive = ContrastiveDifferentiator(
            dim=dim,
            temperature=contrastive_temperature,
            margin=contrastive_margin,
        )
        
        # ===== Feature Fusion =====
        
        # Combine info scores + relevance + pooled features
        self.feature_fuser = nn.Sequential(
            nn.LayerNorm(dim + info_dim + 1),  # +1 for relevance score
            nn.Linear(dim + info_dim + 1, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        
        # ===== Scale Prediction Head =====
        self.scale_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, output_dim),
        )
        
        # ===== Query for gating (learnable or text-derived) =====
        self.gate_query_proj = nn.Linear(dim, dim)
        
        self.debug_checks = False
        self._last_contrastive_loss = None  # Tensor for gradient flow
        self._last_frame_metrics = None     # Frame metrics dict for advantage computation
        self._last_concentration_loss = None
        self._last_scale_var = None
        self.last_entropy = None
    
    def post_init(self):
        """Initialize scale head bias for reasonable default outputs."""
        last_layer = self.scale_head[-1]
        if not hasattr(last_layer, "weight"):
            return
        
        nn.init.orthogonal_(last_layer.weight, gain=0.01)
        
        if last_layer.bias is None:
            return
            
        with torch.no_grad():
            if self.use_discrete_action:
                target_val = 1.0
                diff = torch.abs(self.scale_bins - target_val)
                target_idx = torch.argmin(diff).item()
                nn.init.zeros_(last_layer.bias)
                last_layer.bias[target_idx] = 0.0
            else:
                target_ratio = (1.0 - self.min_scale) / (self.max_scale - self.min_scale)
                target_ratio = max(0.01, min(0.99, target_ratio))
                
                if self.continuous_dist == "logistic_normal":
                    mu_bias = torch.logit(torch.tensor(target_ratio), eps=1e-6)
                    sigma_bias = torch.log(torch.expm1(torch.tensor(self.logistic_normal_init_sigma).clamp_min(1e-3)))
                    last_layer.bias[0] = mu_bias
                    last_layer.bias[1] = sigma_bias
                else:
                    # FIX: Beta distribution - use smaller target_sum (2.0 instead of 4.0)
                    # for higher initial entropy, allowing more exploration
                    target_sum = 2.0  # Reduced from 4.0 to increase initial variance
                    target_alpha = target_sum * target_ratio
                    target_beta = target_sum * (1.0 - target_ratio)
                    # Clamp to ensure positive values for softplus inverse
                    y0 = max(0.1, target_alpha - 1.0)
                    y1 = max(0.1, target_beta - 1.0)
                    last_layer.bias[0] = torch.log(torch.expm1(torch.tensor(y0)))
                    last_layer.bias[1] = torch.log(torch.expm1(torch.tensor(y1)))

    def _temporal_pos_emb(self, t, device, dtype):
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

    def _get_continuous_policy(self, params, actions=None):
        """Sample from continuous distribution."""
        if self.continuous_dist == "beta":
            params_fp32 = (params * self.beta_param_scale).float()
            add_one = 1.0 if self.beta_add_one else 0.0
            alpha = F.softplus(params_fp32[..., 0]) + add_one + 1e-6
            beta = F.softplus(params_fp32[..., 1]) + add_one + 1e-6
            # FIX: Allow smaller alpha/beta (min 0.1) for higher variance when needed
            # Max clamp reduced to 20 for numerical stability while allowing sharper peaks
            alpha = alpha.clamp(min=0.1, max=20.0)
            beta = beta.clamp(min=0.1, max=20.0)
            alpha32 = alpha.float().contiguous()
            beta32 = beta.float().contiguous()
            dist = Beta(alpha32, beta32)

            if actions is None:
                use_reparam_action = self.sim_scale_weight > 0 and self.training
                action_0_1 = dist.rsample() if use_reparam_action else dist.sample()
            else:
                action_0_1 = actions.float().detach()
            
            action_0_1 = action_0_1.clamp(min=1e-6, max=1.0 - 1e-6)
            log_prob = dist.log_prob(action_0_1).float()
            log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # Logistic Normal
            params_fp32 = params.float()
            mu = params_fp32[..., 0]
            sigma = F.softplus(params_fp32[..., 1]) + 1e-6
            # FIX: Clamp sigma to ensure reasonable exploration range
            # Min 0.1 prevents over-concentration, max 5.0 prevents numerical instability
            sigma = sigma.clamp(min=0.1, max=5.0)
            mu32 = mu.float().contiguous()
            sigma32 = sigma.float().contiguous()
            base = Normal(mu32, sigma32)
            dist = TransformedDistribution(base, SigmoidTransform())

            if actions is None:
                use_reparam_action = self.sim_scale_weight > 0 and self.training
                action_0_1 = dist.rsample() if use_reparam_action else dist.sample()
            else:
                action_0_1 = actions.float().detach()

            action_0_1 = action_0_1.clamp(min=1e-6, max=1.0 - 1e-6)
            log_prob = dist.log_prob(action_0_1).float()
            log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0)
        
        scale_factors = self.min_scale + action_0_1 * (self.max_scale - self.min_scale)
        return action_0_1.to(params.dtype), scale_factors.to(params.dtype), log_prob.to(torch.float32)

    def _get_continuous_deterministic(self, params):
        """Get deterministic output (mean or quantile)."""
        if self.continuous_dist == "beta":
            params_fp32 = (params * self.beta_param_scale).float()
            add_one = 1.0 if self.beta_add_one else 0.0
            alpha = F.softplus(params_fp32[..., 0]) + add_one + 1e-6
            beta = F.softplus(params_fp32[..., 1]) + add_one + 1e-6
            # FIX: Consistent clamping with _get_continuous_policy
            alpha = alpha.clamp(min=0.1, max=20.0)
            beta = beta.clamp(min=0.1, max=20.0)
            alpha32 = alpha.float()
            beta32 = beta.float()
            dist = Beta(alpha32, beta32)

            if self.continuous_eval_quantile != 0.5:
                q = torch.tensor(self.continuous_eval_quantile, device=params.device, dtype=torch.float32).clamp(1e-4, 1-1e-4)
                action_0_1 = dist.icdf(q)
            else:
                action_0_1 = dist.mean
        else:
            params_fp32 = params.float()
            mu32 = params_fp32[..., 0]
            sigma32 = F.softplus(params_fp32[..., 1]) + 1e-6
            sigma32 = sigma32.clamp(min=0.1, max=5.0)
            if self.continuous_eval_quantile != 0.5:
                q = torch.tensor(self.continuous_eval_quantile, device=params.device, dtype=torch.float32).clamp(1e-4, 1-1e-4)
                z = Normal(0.0, 1.0).icdf(q).to(torch.float32)
                action_0_1 = torch.sigmoid(mu32 + sigma32 * z)
            else:
                denom = torch.sqrt(1.0 + (torch.pi * sigma32 * sigma32 / 8.0))
                action_0_1 = torch.sigmoid(mu32 / denom)
        
        scale_factors = self.min_scale + action_0_1 * (self.max_scale - self.min_scale)
        return action_0_1.to(params.dtype), scale_factors.to(params.dtype), None

    def _get_continuous_mean_action(self, params):
        if self.continuous_dist == "beta":
            params_fp32 = (params * self.beta_param_scale).float()
            add_one = 1.0 if self.beta_add_one else 0.0
            alpha = F.softplus(params_fp32[..., 0]) + add_one + 1e-6
            beta = F.softplus(params_fp32[..., 1]) + add_one + 1e-6
            alpha = alpha.clamp(min=0.1, max=20.0)
            beta = beta.clamp(min=0.1, max=20.0)
            action_0_1 = alpha / (alpha + beta)
        else:
            params_fp32 = params.float()
            mu32 = params_fp32[..., 0]
            sigma32 = F.softplus(params_fp32[..., 1]) + 1e-6
            sigma32 = sigma32.clamp(min=0.1, max=5.0)
            denom = torch.sqrt(1.0 + (torch.pi * sigma32 * sigma32 / 8.0))
            action_0_1 = torch.sigmoid(mu32 / denom)
        return action_0_1.to(params.dtype)

    def _get_discrete_policy(self, logits, actions=None):
        """Sample from categorical distribution."""
        from torch.distributions import Categorical
        dist = Categorical(logits=logits.float())
        
        if actions is None:
            action_indices = dist.sample()
        else:
            action_indices = actions.long().detach()
        
        log_prob = dist.log_prob(action_indices).float()
        scale_factors = self.scale_bins[action_indices].to(logits.dtype)
        return action_indices, scale_factors, log_prob.to(torch.float32)

    def _get_discrete_deterministic(self, logits):
        """Get argmax for discrete actions."""
        action_indices = torch.argmax(logits, dim=-1)
        scale_factors = self.scale_bins[action_indices]
        return action_indices, scale_factors, None

    def forward(
        self,
        visual_features_batch: torch.Tensor,  # (B, Max_T*Max_L, D)
        visual_grid_thw: torch.Tensor,        # (B, 3)
        text_features: torch.Tensor = None,   # (B, N, D)
        visual_mask: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        actions: torch.Tensor = None,
        visual_per_sample: list = None,
        eval_mode: bool = False,
        scale_mask: torch.Tensor = None,
        compute_frame_metrics: bool = False,  # Whether to compute frame metrics for advantage
    ):
        """
        Forward pass producing per-frame scales.
        
        Returns:
            raw_actions: (Batch, MaxFrames) - action values
            scales: (Batch, MaxFrames) - scale factors
            log_probs: (Batch, MaxFrames) - log probabilities (None if eval)
            scale_mask_out: (Batch, MaxFrames) - valid frame mask
        """
        B, Max_Tokens, D = visual_features_batch.shape
        device = visual_features_batch.device
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
        max_tokens = int(visual_features_batch.shape[1])
        use_raw = object_t_dims * l_raw <= max_tokens
        object_l_dims = torch.where(use_raw, l_raw, l_merged)
        max_t_obj = int(object_t_dims.max().item()) if object_t_dims.numel() > 0 else 0
        
        # Prepare actions if provided
        padded_actions_in = None
        if actions is not None and scale_mask is not None:
            valid_actions = actions[scale_mask]
            if valid_actions.numel() == object_t_dims.sum():
                splits = torch.split(valid_actions, object_t_dims.cpu().tolist())
                pad_val = -1 if self.use_discrete_action else 0.0
                padded_actions_in = pad_sequence(list(splits), batch_first=True, padding_value=pad_val)
        
        # Output buffers
        output_dim = self.num_bins if self.use_discrete_action else 2
        head_outputs = torch.zeros((B, max_t_obj, output_dim), dtype=visual_features_batch.dtype, device=device)
        frame_features_out = torch.zeros((B, max_t_obj, D), dtype=visual_features_batch.dtype, device=device)
        
        # Contrastive loss accumulator
        contrastive_loss_sum = 0.0
        contrastive_count = 0
        
        # Frame metrics buffers (only if needed for advantage computation)
        if compute_frame_metrics:
            redundancy_obj = torch.zeros((B, max_t_obj), dtype=torch.float32, device=device)
            uniqueness_obj = torch.zeros((B, max_t_obj), dtype=torch.float32, device=device)
            text_relevance_obj = torch.zeros((B, max_t_obj), dtype=torch.float32, device=device)
            info_score_obj = torch.zeros((B, max_t_obj), dtype=torch.float32, device=device)
        
        # Debug stats
        gate_entropy_sum, frame_sim_sum = 0.0, 0.0
        debug_count = 0
        
        # Group by (T, L) for efficient processing
        # Optimized: use return_inverse to avoid repeated nonzero calls
        pair = torch.stack([object_t_dims, object_l_dims], dim=1)
        unique_pairs, inverse_indices = torch.unique(pair, dim=0, return_inverse=True)
        
        for group_idx, tl in enumerate(unique_pairs):
            t = int(tl[0].item())
            l = int(tl[1].item())
            
            if t <= 0 or l <= 0:
                continue
            
            # Use pre-computed inverse indices for O(B) grouping instead of O(B) nonzero per group
            idx = (inverse_indices == group_idx).nonzero(as_tuple=False).squeeze(1)
            
            if idx.numel() == 0:
                continue
            
            # Extract patch features: (Bobj, T, L, D)
            feats = visual_features_batch.index_select(0, idx)[:, :t*l].reshape(-1, t, l, D)
            Bobj = feats.shape[0]
            
            # Get text for this batch
            enc_states = text_features.index_select(0, idx) if exists(text_features) else None
            enc_mask = text_mask.index_select(0, idx) if exists(text_mask) else None
            
            # ===== Step 1: Frame Information Encoding =====
            info_scores = self.info_encoder(feats)  # (Bobj, T, info_dim)
            
            # ===== Step 2: Initial frame pooling =====
            frame_mean = feats.mean(dim=2)  # (Bobj, T, D)
            
            # ===== Step 3: Sparse Gating for patch pooling =====
            gate_query = self.gate_query_proj(frame_mean) + self.info_to_query(info_scores.to(frame_mean.dtype))
            gate, pooled = self.sparse_gate(feats, gate_query, hard=False)
            matcher_input = (frame_mean + pooled) * 0.5
            
            # ===== Step 4: Cross-Modal Matching =====
            attended = None
            if enc_states is not None:
                relevance, attended = self.cross_matcher(matcher_input, enc_states, enc_mask)
            else:
                relevance = torch.zeros(Bobj, t, device=device, dtype=feats.dtype)
            
            # ===== Step 5: Dual-Path Encoding =====
            fused, independent, temporal = self.dual_encoder(pooled)
            if attended is not None:
                fused = fused + attended
            
            # ===== Step 7: Feature Fusion =====
            relevance_expanded = relevance.unsqueeze(-1)  # (Bobj, T, 1)
            combined = torch.cat([fused, info_scores, relevance_expanded], dim=-1)
            frame_features = self.feature_fuser(combined)
            if len(self.temporal_mixer) > 0:
                if self.temporal_use_pos:
                    frame_features = frame_features + self._temporal_pos_emb(t, device, frame_features.dtype)
                for layer in self.temporal_mixer:
                    frame_features = layer(frame_features)
            
            if compute_aux and self.training and self.contrastive_weight > 0:
                c_loss = 0.5 * (self.contrastive(independent) + self.contrastive(frame_features))
                contrastive_loss_sum += c_loss
                contrastive_count += 1
            
            scale_params = self.scale_head(frame_features)  # (Bobj, T, output_dim)
            
            head_outputs[idx, :t] = scale_params
            frame_features_out[idx, :t] = frame_features.to(frame_features_out.dtype)
            
            if compute_frame_metrics:
                with torch.no_grad():
                    ff_norm = F.normalize(frame_features.float(), dim=-1, eps=1e-6)  # (Bobj, T, D)
                    ff_norm = torch.nan_to_num(ff_norm, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Redundancy: cosine similarity with previous frame
                    if t > 1:
                        sim_prev = (ff_norm[:, 1:] * ff_norm[:, :-1]).sum(dim=-1)  # (Bobj, T-1)
                        redundancy_local = F.pad(sim_prev, (1, 0), value=0.0)  # (Bobj, T)
                    else:
                        redundancy_local = torch.zeros((Bobj, t), device=device, dtype=torch.float32)
                    
                    # Uniqueness: 1 - mean similarity to neighbors (memory-efficient O(T) approximation)
                    # Instead of O(T×T) matrix, use neighbor similarities as proxy
                    if t > 1:
                        # Reuse sim_prev from redundancy computation above
                        sim_prev_padded = redundancy_local  # Already padded (Bobj, T)
                        sim_next_padded = F.pad(sim_prev, (0, 1), value=0.0)  # (Bobj, T)
                        mean_neighbor_sim = (sim_prev_padded + sim_next_padded) / 2.0
                        # Handle boundary frames (only one neighbor)
                        mean_neighbor_sim[:, 0] = sim_prev_padded[:, 1] if t > 1 else 0.0
                        mean_neighbor_sim[:, -1] = sim_next_padded[:, -2] if t > 1 else 0.0
                        uniqueness_local = (1.0 - mean_neighbor_sim).clamp(0.0, 1.0)
                    else:
                        uniqueness_local = torch.ones((Bobj, t), device=device, dtype=torch.float32)
                    
                    # Info score: mean of info_scores embedding (aggregate stat)
                    info_score_local = info_scores.float().mean(dim=-1)  # (Bobj, T)
                    
                    # Text relevance: already computed
                    text_relevance_local = relevance.float()  # (Bobj, T)
                    
                    # Store in buffers
                    redundancy_obj[idx, :t] = redundancy_local.clamp(-1, 1)
                    uniqueness_obj[idx, :t] = uniqueness_local.clamp(0, 1)
                    text_relevance_obj[idx, :t] = text_relevance_local.clamp(-1, 1)
                    info_score_obj[idx, :t] = info_score_local
            
            # Debug stats
            if eval_mode:
                gate_safe = torch.nan_to_num(gate, nan=0.0, posinf=0.0, neginf=0.0)
                gate_ent = -(gate_safe.clamp(1e-9) * gate_safe.clamp(1e-9).log()).sum(dim=-1).mean()
                gate_entropy_sum += gate_ent.item()
                if t > 1:
                    f_sim = F.cosine_similarity(frame_features[:, :-1], frame_features[:, 1:], dim=-1, eps=1e-6).mean()
                    frame_sim_sum += f_sim.item()
                debug_count += 1
        
        # Store contrastive loss TENSOR for training (gradient flow)
        if compute_aux:
            if contrastive_count > 0:
                self._last_contrastive_loss = contrastive_loss_sum / contrastive_count
            else:
                self._last_contrastive_loss = torch.tensor(0.0, device=device, requires_grad=False)
        else:
            self._last_contrastive_loss = None
        
        # Build output mask
        splits = torch.split(object_t_dims, visual_per_sample)
        valid_lengths = torch.stack([s.sum() for s in splits]).to(device)
        pad_max = max(self.max_frames, int(valid_lengths.max().item()))
        scale_mask_out = torch.arange(pad_max, device=device)[None, :] < valid_lengths[:, None]
        
        temporal_valid = torch.arange(max_t_obj, device=device)[None, :] < object_t_dims[:, None]
        
        # Get actions and scales
        padded_actions_policy = padded_actions_in
        if self.use_discrete_action and padded_actions_policy is not None:
            padded_actions_policy = padded_actions_policy.masked_fill(padded_actions_policy < 0, 0)
        
        if self.use_discrete_action:
            if eval_mode:
                raw_actions_obj, scales_obj, log_probs_obj = self._get_discrete_deterministic(head_outputs)
            else:
                raw_actions_obj, scales_obj, log_probs_obj = self._get_discrete_policy(head_outputs, padded_actions_policy)
            pad_val = -1
        else:
            if eval_mode:
                raw_actions_obj, scales_obj, log_probs_obj = self._get_continuous_deterministic(head_outputs)
            else:
                raw_actions_obj, scales_obj, log_probs_obj = self._get_continuous_policy(head_outputs, padded_actions_policy)
            pad_val = 0.0

        # Apply masks
        # Apply masks to obj-level outputs
        raw_actions_obj = raw_actions_obj.masked_fill(~temporal_valid, pad_val)
        scales_obj = scales_obj.masked_fill(~temporal_valid, 1.0)
        scale_mask_obj = temporal_valid  # (Bobj, T)
        
        if log_probs_obj is not None:
            log_probs_obj = log_probs_obj.masked_fill(~temporal_valid, 0.0)

        if compute_aux:
            entropy_obj = None
            if self.use_discrete_action:
                entropy_obj = torch.distributions.Categorical(logits=head_outputs.float()).entropy().to(head_outputs.dtype)
                self._last_concentration_loss = None
            else:
                params_fp32 = (head_outputs * self.beta_param_scale).float()
                add_one = 1.0 if self.beta_add_one else 0.0
                if self.continuous_dist == "beta":
                    alpha = F.softplus(params_fp32[..., 0]) + add_one + 1e-6
                    beta = F.softplus(params_fp32[..., 1]) + add_one + 1e-6
                    alpha = alpha.clamp(min=0.1, max=20.0)
                    beta = beta.clamp(min=0.1, max=20.0)
                    entropy_obj = Beta(alpha, beta).entropy().to(head_outputs.dtype)
                    concentration = alpha + beta
                else:
                    mu = params_fp32[..., 0]
                    sigma = F.softplus(params_fp32[..., 1]) + 1e-6
                    sigma = sigma.clamp(min=0.1, max=5.0)
                    entropy_obj = Normal(mu, sigma).entropy().to(head_outputs.dtype)
                    concentration = 1.0 / sigma
                m = temporal_valid.to(concentration.dtype)
                denom = m.sum().clamp_min(1.0)
                self._last_concentration_loss = (concentration * m).sum() / denom
            entropy = self._restructure(entropy_obj, visual_grid_thw, scale_mask_out, 0.0, device)
            self.last_entropy = entropy.to(torch.float32) if entropy is not None else None
        else:
            self._last_concentration_loss = None
            self.last_entropy = None

        if compute_aux:
            m_var = temporal_valid.to(scales_obj.dtype)
            denom_var = m_var.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_var = (scales_obj * m_var).sum(dim=1, keepdim=True) / denom_var
            var = ((scales_obj - mean_var) ** 2 * m_var).sum(dim=1, keepdim=True) / denom_var
            self._last_scale_var = var.mean()
        else:
            self._last_scale_var = None
        
        # Compute sim_scale_loss using obj-level tensors (Bobj, T) BEFORE restructure
        # This ensures shapes match even if Bobj != B
        if compute_aux and self.sim_scale_weight > 0 and self.training and frame_features_out.shape[1] > 1:
            f0 = frame_features_out[:, :-1]  # (Bobj, T-1, D)
            f1 = frame_features_out[:, 1:]   # (Bobj, T-1, D)
            sim = F.cosine_similarity(f0, f1, dim=-1)  # (Bobj, T-1)
            
            # Mask for valid pairs
            m = (scale_mask_obj[:, :-1] & scale_mask_obj[:, 1:]).float()
            sim = sim * m
            
            # Sigmoid weight: high sim -> high weight
            w = torch.sigmoid((sim - self.sim_tau) / self.sim_temp)
            w = w * m
            
            # For discrete actions, use soft scales (differentiable) for loss computation
            if self.use_discrete_action:
                probs = head_outputs.softmax(dim=-1)
                soft_scales_obj = (probs * self.scale_bins).sum(dim=-1)
                s = soft_scales_obj.clamp_min(1e-6).log()
            else:
                mean_action = self._get_continuous_mean_action(head_outputs)
                soft_scales_obj = self.min_scale + mean_action * (self.max_scale - self.min_scale)
                s = soft_scales_obj.clamp_min(1e-6).log()
            
            # Target: s[t] should be <= s[t-1] - gamma*w when frames are similar
            target = s[:, :-1] - self.sim_gamma * w
            err = F.relu(s[:, 1:] - target)  # Only penalize when scale increases too much
            
            self._last_sim_scale_loss = (err * w).sum() / (m.sum().clamp_min(1.0))
        else:
            self._last_sim_scale_loss = None
            
        # Restructure to batch format for return
        log_probs = self._restructure(log_probs_obj, visual_grid_thw, scale_mask_out, 0.0, device)
        scales = self._restructure(scales_obj, visual_grid_thw, scale_mask_out, 1.0, device)
        
        self.last_frame_features = frame_features_out
        self.last_scales = scales
        
        # Restructure frame metrics to batch format (only if computed)
        if compute_frame_metrics:
            self._last_frame_metrics = {
                "redundancy": self._restructure(redundancy_obj, visual_grid_thw, scale_mask_out, 0.0, device),
                "uniqueness": self._restructure(uniqueness_obj, visual_grid_thw, scale_mask_out, 0.5, device),
                "text_relevance": self._restructure(text_relevance_obj, visual_grid_thw, scale_mask_out, 0.0, device),
                "info_score": self._restructure(info_score_obj, visual_grid_thw, scale_mask_out, 0.0, device),
            }
        else:
            self._last_frame_metrics = None
            
        if actions is not None:
            if isinstance(scales, torch.Tensor) and scales.dtype == torch.bfloat16:
                scales = scales.float()
            if isinstance(log_probs, torch.Tensor) and log_probs.dtype == torch.bfloat16:
                log_probs = log_probs.float()
            if isinstance(scale_mask_out, torch.Tensor) and scale_mask_out.dtype == torch.bfloat16:
                scale_mask_out = scale_mask_out.float()
            return None, scales, log_probs, scale_mask_out
        
        raw_actions = self._restructure(raw_actions_obj, visual_grid_thw, scale_mask_out, pad_val, device)
        
        # Print debug info
        if eval_mode and debug_count > 0:
            mask = (scales != 0.0).float() # Simple mask proxy
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean = (scales * mask).sum(dim=1, keepdim=True) / denom
            var = ((scales - mean) ** 2 * mask).sum(dim=1, keepdim=True) / denom
            # Safe tensor-to-scalar conversion for print
            c_loss_val = self._last_contrastive_loss.item() if isinstance(self._last_contrastive_loss, torch.Tensor) else self._last_contrastive_loss
            def _fmt(v):
                return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"
            print(
                f"[DIP-debug] gate_entropy={_fmt(gate_entropy_sum/debug_count)} "
                f"frame_sim={_fmt(frame_sim_sum/debug_count)} "
                f"scale_var={var.mean().item():.6f} "
                f"contrastive_loss={_fmt(c_loss_val)}"
            )
        
        if isinstance(scales, torch.Tensor) and scales.dtype == torch.bfloat16:
            scales = scales.float()
        if isinstance(log_probs, torch.Tensor) and log_probs.dtype == torch.bfloat16:
            log_probs = log_probs.float()
        if isinstance(scale_mask_out, torch.Tensor) and scale_mask_out.dtype == torch.bfloat16:
            scale_mask_out = scale_mask_out.float()
        return raw_actions, scales, log_probs, scale_mask_out
    
    def _restructure(self, tensor, visual_grid_thw, target_mask, pad_val, device):
        """Restructure per-object tensor to batch format."""
        if tensor is None:
            return None
        
        object_t_dims = visual_grid_thw[:, 0]
        max_len = tensor.shape[1]
        source_mask = torch.arange(max_len, device=device)[None, :] < object_t_dims[:, None]
        valid_values = tensor[source_mask]
        
        output = torch.full(target_mask.shape, pad_val, dtype=tensor.dtype, device=device)
        output[target_mask] = valid_values
        return output
    
    def get_contrastive_loss(self) -> torch.Tensor:
        """Get the last computed contrastive loss tensor for training."""
        if self._last_contrastive_loss is None:
            return torch.tensor(0.0)
        return self._last_contrastive_loss
    
    def get_weighted_contrastive_loss(self) -> torch.Tensor:
        """Get contrastive loss multiplied by weight for adding to policy loss."""
        return self.get_contrastive_loss() * self.contrastive_weight

    def get_sim_scale_loss(self) -> torch.Tensor:
        """Get pairwise similarity-based scale loss (compress redundant frames)."""
        if self._last_sim_scale_loss is None:
            return torch.tensor(0.0)
        return self._last_sim_scale_loss

    def get_weighted_sim_scale_loss(self) -> torch.Tensor:
        """Get sim_scale_loss multiplied by weight for adding to policy loss."""
        return self.get_sim_scale_loss() * self.sim_scale_weight

    def get_frame_metrics(self) -> dict:
        """Get last computed frame metrics dict for advantage computation.
        
        Returns dict with keys: redundancy, uniqueness, text_relevance, info_score
        Each value is (Batch, MaxFrames) tensor.
        """
        return self._last_frame_metrics
