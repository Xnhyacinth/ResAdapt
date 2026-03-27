import torch
import torch.nn.functional as F

from resadapt.allocator.attention_utils import (
    flash_attn_varlen_qkv_dtype,
    sdpa_scaled_dot_product_attention as sdpa_attn,
)
from torch import nn
from einops import rearrange, repeat
from torch.distributions import Beta, Categorical, Normal, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform
from transformers.modeling_utils import ModuleUtilsMixin
from torch.nn.utils.rnn import pad_sequence

try:
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
    from verl.utils.attention_utils import pad_input as _pad_input
    from verl.utils.attention_utils import unpad_input as _unpad_input
except Exception:
    _flash_attn_varlen_func = None
    _pad_input = None
    _unpad_input = None

def exists(val):
    return val is not None

def FeedForward(dim, mult=4, dropout=0.):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias=False),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim, bias=False)
    )


class RotaryEmbedding(nn.Module):
    # def __init__(self, dim):
    #     super().__init__()
    #     inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    #     self.register_buffer("inv_freq", inv_freq)

    # def forward(self, max_seq_len, *, device, dtype=torch.float32):
    #     seq = torch.arange(max_seq_len, device=device, dtype=dtype)
    #     freqs = einsum("i, j -> i j", seq, self.inv_freq.to(dtype))
    #     return torch.cat((freqs, freqs), dim=-1)
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(t, pos):
    # pos: (1, 1, SeqLen, Dim) or (Batch, 1, SeqLen, Dim)
    rotate_dim = pos.shape[-1]
    # Ensure pos matches t dimensions for broadcasting
    # t: (B*L, H, T, D) -> need pos to broadcast over B*L and H
    if pos.ndim == 2:
        pos = pos.unsqueeze(0).unsqueeze(0) # (1, 1, S, D)
    
    t_pass, t_rot = t[..., rotate_dim:], t[..., :rotate_dim]
    cos = pos.cos().to(t_rot.dtype)
    sin = pos.sin().to(t_rot.dtype)
    t_rot = (t_rot * cos) + (rotate_half(t_rot) * sin)
    return torch.cat((t_rot, t_pass), dim=-1)

def _apply_rotary_pos_emb_packed(t: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    rotate_dim = pos.shape[-1]
    pos = pos.unsqueeze(1)
    t_pass, t_rot = t[..., rotate_dim:], t[..., :rotate_dim]
    cos = pos.cos().to(t_rot.dtype)
    sin = pos.sin().to(t_rot.dtype)
    t_rot = (t_rot * cos) + (rotate_half(t_rot) * sin)
    return torch.cat((t_rot, t_pass), dim=-1)

def _build_pos_ids_from_cu_seqlens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    if cu_seqlens.numel() <= 1:
        return torch.empty((0,), dtype=torch.long, device=cu_seqlens.device)
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
    total = int(cu_seqlens[-1].item())
    if total == 0:
        return torch.empty((0,), dtype=torch.long, device=cu_seqlens.device)
    starts = cu_seqlens[:-1].to(torch.long).repeat_interleave(lens)
    return torch.arange(total, device=cu_seqlens.device, dtype=torch.long) - starts

def _sdpa_mask_from_valid(valid: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros_like(valid, dtype=dtype)
    return mask.masked_fill(~valid, float("-inf"))


class PatchWiseTemporalAttention(nn.Module):
    """
    Performs temporal attention on tokens at the same spatial position across different frames.
    Optimized to use Flash Attention (SDPA) when available.
    """
    temporal_dim_scale = 0.5

    def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.):
        super().__init__()
        self.num_heads = max(1, round(heads * self.temporal_dim_scale))
        self.dim_head = dim_head
        inner_dim = self.num_heads * self.dim_head
        self.norm = nn.LayerNorm(dim)
        self.dropout_p = dropout
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, rotary_pos_emb=None, attn_mask=None):
        """
        x: (B, T, L, D) -> We want attention over T
        """
        B, T, L, D = x.shape
        x = self.norm(x)
        
        # Merge Batch and Spatial: (B*L, T, D)
        x = rearrange(x, 'b t l d -> (b l) t d')

        if _flash_attn_varlen_func is not None and _unpad_input is not None and _pad_input is not None:
            if attn_mask is None:
                attn_mask = torch.ones((B, T), device=x.device, dtype=torch.bool)
            flat_mask = repeat(attn_mask, "b t -> (b l) t", l=L)
            seqlens = flat_mask.sum(dim=-1)
            max_seqlen = int(seqlens.max().item()) if seqlens.numel() > 0 else 0

            qkv = self.to_qkv(x)
            if max_seqlen <= 1:
                v = rearrange(qkv[..., -self.num_heads * self.dim_head :], "b t (h d) -> b t h d", h=self.num_heads)
                out = rearrange(v, "b t h d -> b t (h d)")
                out = self.to_out(out)
                return rearrange(out, "(b l) t d -> b t l d", b=B, l=L)

            unpad_ret = _unpad_input(qkv, attention_mask=flat_mask)
            qkv_unpad, indices, cu_seqlens, max_seqlen = unpad_ret[:4]
            qkv_unpad = rearrange(qkv_unpad, "nnz (three h d) -> three nnz h d", three=3, h=self.num_heads)
            q, k, v = qkv_unpad[0], qkv_unpad[1], qkv_unpad[2]
            q = flash_attn_varlen_qkv_dtype(q)
            k = flash_attn_varlen_qkv_dtype(k)
            v = flash_attn_varlen_qkv_dtype(v)

            if exists(rotary_pos_emb):
                if rotary_pos_emb.ndim == 4:
                    rope = rotary_pos_emb.squeeze(0).squeeze(0)
                else:
                    rope = rotary_pos_emb
                pos_ids = _build_pos_ids_from_cu_seqlens(cu_seqlens)
                pos = rope.index_select(0, pos_ids).to(q.dtype)
                q = _apply_rotary_pos_emb_packed(q, pos)
                k = _apply_rotary_pos_emb_packed(k, pos)

            out_unpad = _flash_attn_varlen_func(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                dropout_p=self.dropout_p if self.training else 0.0,
                causal=False,
            )
            out_unpad = rearrange(out_unpad, "nnz h d -> nnz (h d)")
            out = _pad_input(hidden_states=out_unpad, indices=indices, batch=x.shape[0], seqlen=T)
            out = out.to(x.dtype)
            out = self.to_out(out)
            return rearrange(out, "(b l) t d -> b t l d", b=B, l=L)
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=self.num_heads), qkv)

        if exists(rotary_pos_emb):
            # rotary_pos_emb should be compatible with (B*L, H, T, D)
            q = apply_rotary_pos_emb(q, rotary_pos_emb)
            k = apply_rotary_pos_emb(k, rotary_pos_emb)

        sdpa_mask = None
        if exists(attn_mask):
            sdpa_mask = repeat(attn_mask.to(torch.bool), "b t -> (b l) 1 1 t", l=L)
            sdpa_mask = _sdpa_mask_from_valid(sdpa_mask, dtype=q.dtype)

        out = sdpa_attn(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout_p if self.training else 0.0
        )
        
        out = rearrange(out, 'b h t d -> b t (h d)')
        out = self.to_out(out)
        
        # Restore: (B*L, T, D) -> (B, T, L, D)
        return rearrange(out, '(b l) t d -> b t l d', b=B, l=L)


class SpatialAttention(nn.Module):
    """
    Performs spatial self-attention within each frame.
    Input is (B, T*L, D), reshaped to (B*T, L, D) for attention.
    """
    def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.norm = nn.LayerNorm(dim)
        self.dropout_p = dropout
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, rotary_pos_emb=None, attn_mask=None):
        """
        x: (B, T, L, D) -> We want attention over L
        """
        B, T, L, D = x.shape
        x = self.norm(x)
        
        # Merge Batch and Temporal: (B*T, L, D)
        x = rearrange(x, 'b t l d -> (b t) l d')

        if _flash_attn_varlen_func is not None and _unpad_input is not None and _pad_input is not None:
            if attn_mask is None:
                attn_mask = torch.ones((B, T, L), device=x.device, dtype=torch.bool)
            flat_mask = rearrange(attn_mask, "b t l -> (b t) l")
            seqlens = flat_mask.sum(dim=-1)
            max_seqlen = int(seqlens.max().item()) if seqlens.numel() > 0 else 0

            qkv = self.to_qkv(x)
            if max_seqlen <= 1:
                v = rearrange(qkv[..., -self.num_heads * self.dim_head :], "b l (h d) -> b l h d", h=self.num_heads)
                out = rearrange(v, "b l h d -> b l (h d)")
                out = self.to_out(out)
                return rearrange(out, "(b t) l d -> b t l d", b=B, t=T)

            unpad_ret = _unpad_input(qkv, attention_mask=flat_mask)
            qkv_unpad, indices, cu_seqlens, max_seqlen = unpad_ret[:4]
            qkv_unpad = rearrange(qkv_unpad, "nnz (three h d) -> three nnz h d", three=3, h=self.num_heads)
            q, k, v = qkv_unpad[0], qkv_unpad[1], qkv_unpad[2]
            q = flash_attn_varlen_qkv_dtype(q)
            k = flash_attn_varlen_qkv_dtype(k)
            v = flash_attn_varlen_qkv_dtype(v)

            if exists(rotary_pos_emb):
                if rotary_pos_emb.ndim == 4:
                    rope = rotary_pos_emb.squeeze(0).squeeze(0)
                else:
                    rope = rotary_pos_emb
                pos_ids = _build_pos_ids_from_cu_seqlens(cu_seqlens)
                pos = rope.index_select(0, pos_ids).to(q.dtype)
                q = _apply_rotary_pos_emb_packed(q, pos)
                k = _apply_rotary_pos_emb_packed(k, pos)

            out_unpad = _flash_attn_varlen_func(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                dropout_p=self.dropout_p if self.training else 0.0,
                causal=False,
            )
            out_unpad = rearrange(out_unpad, "nnz h d -> nnz (h d)")
            out = _pad_input(hidden_states=out_unpad, indices=indices, batch=x.shape[0], seqlen=L)
            out = out.to(x.dtype)
            out = self.to_out(out)
            return rearrange(out, "(b t) l d -> b t l d", b=B, t=T)
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.num_heads), qkv)

        if exists(rotary_pos_emb):
            q = apply_rotary_pos_emb(q, rotary_pos_emb)
            k = apply_rotary_pos_emb(k, rotary_pos_emb)

        sdpa_mask = None
        if exists(attn_mask):
            sdpa_mask = rearrange(attn_mask.to(torch.bool), "b t l -> (b t) 1 1 l")
            sdpa_mask = _sdpa_mask_from_valid(sdpa_mask, dtype=q.dtype)

        out = sdpa_attn(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout_p if self.training else 0.0
        )
        
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.to_out(out)
        
        # Restore: (B*T, L, D) -> (B, T, L, D)
        return rearrange(out, '(b t) l d -> b t l d', b=B, t=T)


class CrossAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        inner_dim = heads * dim_head
        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.dropout_p = dropout
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, context, context_mask=None):
        """
        x: (B, T, L, D) -> Flatten to (B, T*L, D) for Cross Attn? 
        Or treat (B, T*L) as sequence. Usually Cross Attn is done on the flattened sequence.
        """
        B, T, L, D = x.shape
        x_flat = rearrange(x, 'b t l d -> b (t l) d')
        
        x_norm = self.norm(x_flat)
        context = self.context_norm(context)
        
        q = self.to_q(x_norm)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b m (h d) -> b h m d', h=self.heads)
        v = rearrange(v, 'b m (h d) -> b h m d', h=self.heads)

        sdpa_mask = None
        if exists(context_mask):
            sdpa_mask = rearrange(context_mask.to(torch.bool), "b m -> b 1 1 m")
            sdpa_mask = _sdpa_mask_from_valid(sdpa_mask, dtype=q.dtype)

        out = sdpa_attn(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout_p if self.training else 0.0
        )
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        # Reshape back to (B, T, L, D)
        return rearrange(out, 'b (t l) d -> b t l d', t=T, l=L)

class SmallTextEncoder(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, dropout: float, ff_mult: int):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=dim * ff_mult,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask.to(torch.bool)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x


class AllocatorDecoderLayer(ModuleUtilsMixin, nn.Module):
    """
    A single unified layer that contains:
    1. Temporal Attention (optional based on depth)
    2. Spatial Attention (optional based on depth)
    3. Cross Attention (optional based on configuration)
    4. Feed Forward
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        dim = config.hidden_size
        dim_head = config.dim_head
        heads = config.num_heads
        dropout = config.dropout
        ff_mult = config.ff_mult
        self.spatial_attn = SpatialAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout)
        self.temporal_attn = PatchWiseTemporalAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout)
        self.cross_attn = CrossAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout)
        self.ffn = FeedForward(dim, mult=ff_mult, dropout=dropout)

    def forward(self, hidden_states, encoder_hidden_states=None, encoder_attention_mask=None,
                temporal_rope=None, spatial_rope=None,
                temporal_mask=None, spatial_mask=None):
        # hidden_states: (B, T, L, D)
        hidden_states = hidden_states + self.spatial_attn(
            hidden_states, rotary_pos_emb=spatial_rope, attn_mask=spatial_mask
        )
        hidden_states = hidden_states + self.temporal_attn(
            hidden_states, rotary_pos_emb=temporal_rope, attn_mask=temporal_mask
        )
        if encoder_hidden_states is not None:
            hidden_states = hidden_states + self.cross_attn(
                hidden_states, encoder_hidden_states, context_mask=encoder_attention_mask
            )
        hidden_states = hidden_states + self.ffn(hidden_states)

        return hidden_states


class AllocatorConfig:
    """Helper to pass config params cleanly"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class ProjectorBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = None, dropout: float = 0.0):
        """
        LayerNorm -> Linear -> GELU -> Dropout -> Linear
        
        Args:
            input_dim
            output_dim
            hidden_dim
            dropout: Dropout 
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            # nn.Dropout(dropout), 
            nn.Linear(hidden_dim, output_dim)
        )
        self.skip = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.net(x) + self.skip(x)

    def get_last_layer(self):
        return self.net[-1]


class TextFrameRelevanceScorer(nn.Module):
    """
    Computes explicit relevance scores between frame features and text query.
    Higher score = frame is more relevant to the text query.
    """
    def __init__(self, dim: int, use_projection: bool = True):
        super().__init__()
        self.use_projection = use_projection
        if use_projection:
            self.text_proj = nn.Linear(dim, dim)
            self.frame_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
    def forward(
        self, 
        frame_embeds: torch.Tensor,  # (B, T, D)
        text_features: torch.Tensor,  # (B, N, D)
        text_mask: torch.Tensor = None,  # (B, N)
    ) -> torch.Tensor:
        """
        Returns:
            relevance: (B, T) scores in range [-1, 1], higher = more relevant
        """
        # Global text representation (masked mean)
        target_dtype = frame_embeds.dtype
        if text_mask is not None:
            text_mask_f = text_mask.to(dtype=target_dtype).unsqueeze(-1)  # (B, N, 1)
            text_avg = (text_features * text_mask_f).sum(dim=1) / text_mask_f.sum(dim=1).clamp_min(1.0)
        else:
            text_avg = text_features.mean(dim=1)  # (B, D)
        
        text_avg = text_avg.to(dtype=target_dtype)
        frame_embeds = frame_embeds.to(dtype=target_dtype)

        # Project for compatibility
        if self.use_projection:
            text_q = self.text_proj(text_avg)  # (B, D)
            frame_k = self.frame_proj(frame_embeds)  # (B, T, D)
        else:
            text_q = text_avg
            frame_k = frame_embeds
        
        # Normalize and compute cosine similarity
        text_q = F.normalize(text_q, dim=-1).unsqueeze(1)  # (B, 1, D)
        frame_k = F.normalize(frame_k, dim=-1)  # (B, T, D)
        relevance = (text_q * frame_k).sum(dim=-1)  # (B, T)
        
        return relevance



class FrameWiseScaleAllocator(ModuleUtilsMixin, nn.Module):
    """
    Unified Allocator supporting both Discrete (Categorical) and Continuous (Beta) actions.
    Predicts per-frame scale factors from patch-level vision features with optional text conditioning.
    Returns: (actions in [0,1], scale_factors in [min_scale,max_scale], log_prob).
    """
    def __init__(
        self, *, dim, depth, dim_head=64, heads=8, dropout=0., ff_mult=4,
        max_frames=4, use_discrete_action=False,
        scale_bins=None, min_scale=0.25, max_scale=3.0, spatial_merge_size=2,
        gate_temperature=0.5, gate_query_scale=1.0, regression_head_mode="mlp",
        beta_param_scale=1.0, beta_add_one=True, beta_init_mode="uniform",
        continuous_dist="beta", continuous_eval_quantile=0.5, logistic_normal_init_sigma=0.7,
        categorical_temperature=1.0,
        pool_gate_mode="no_ln", info_fuse_mode="pooled_ln",
        sim_scale_weight=0.0, sim_tau=0.0, sim_temp=1.0, sim_gamma=0.0,
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.max_frames = max_frames
        dist_name = str(continuous_dist)
        if dist_name in {"norm", "normal", "logistic-norm", "logisticnormal"}:
            dist_name = "logistic_normal"
        if dist_name in {"cat"}:
            dist_name = "categorical"
        self.continuous_dist = dist_name
        self.use_discrete_action = bool(use_discrete_action) or (self.continuous_dist == "categorical")
        self.gate_temperature = float(gate_temperature)
        self.gate_query_scale = float(gate_query_scale)
        self.beta_param_scale = float(beta_param_scale)
        self.beta_add_one = bool(beta_add_one)
        self.beta_init_mode = str(beta_init_mode)
        self.continuous_eval_quantile = float(continuous_eval_quantile)
        self.logistic_normal_init_sigma = float(logistic_normal_init_sigma)
        self.categorical_temperature = float(categorical_temperature)
        self.pool_gate_mode = str(pool_gate_mode)
        self.info_fuse_mode = str(info_fuse_mode)
        self.sim_scale_weight = float(sim_scale_weight)
        self.sim_tau = float(sim_tau)
        self.sim_temp = float(sim_temp)
        self.sim_gamma = float(sim_gamma)

        if self.continuous_dist not in {"beta", "logistic_normal", "categorical"}:
            raise ValueError(f"Unsupported continuous_dist: {self.continuous_dist}")
        
        if self.use_discrete_action:
            if scale_bins is None:
                import numpy as np
                scale_bins = np.arange(min_scale, max_scale + min_scale, min_scale).tolist()
            self.register_buffer('scale_bins', torch.tensor(scale_bins))
            self.num_bins = len(scale_bins)
            output_dim = self.num_bins
        else:
            self.min_scale = min_scale
            self.max_scale = max_scale
            output_dim = 2

        self.rope_gen = RotaryEmbedding(dim=max(32, dim_head // 2))
        self.temporal_rope_gen = RotaryEmbedding(dim=max(32, dim_head))

        self.config = AllocatorConfig(
            hidden_size=dim, dim_head=dim_head, num_heads=heads, dropout=dropout,
            ff_mult=ff_mult,
            depth=depth,
        )
        
        self.layers = nn.ModuleList([
            AllocatorDecoderLayer(self.config, i) for i in range(depth)
        ])
        
        if self.pool_gate_mode == "patch_ln":
            self.patch_norm = nn.LayerNorm(dim)
            self.pool_gate = nn.Sequential(
                nn.Linear(dim * 2, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, 1, bias=False),
            )
        else:
            self.patch_norm = None
            self.pool_gate = nn.Sequential(
                nn.Linear(dim * 2, dim // 2, bias=True),
                nn.GELU(),
                nn.Linear(dim // 2, 1, bias=False),
            )
        
        # Text-Frame Relevance Scorer (NEW)
        self.relevance_scorer = TextFrameRelevanceScorer(dim=dim, use_projection=True)
        
        # info_dim: variance, entropy, peak, sim_prev, relevance (NEW: 4 -> 5)
        self.info_dim = 5
        if self.info_fuse_mode == "residual_gate":
            self.info_proj = nn.Linear(dim + self.info_dim, dim)
            self.info_gate = nn.Parameter(torch.zeros(1))
            self.pooled_norm = None
        elif self.info_fuse_mode == "pooled_ln":
            self.pooled_norm = nn.LayerNorm(dim)
            self.info_proj = nn.Linear(dim + self.info_dim, dim, bias=True)
        else:
            self.pooled_norm = None
            self.info_proj = nn.Sequential(
                nn.LayerNorm(dim + self.info_dim),
                nn.Linear(dim + self.info_dim, dim),
            )
        if regression_head_mode == "linear":
            self.regression_head = nn.Linear(dim, output_dim)
        elif regression_head_mode == "ln_linear":
            self.regression_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, output_dim))
        elif regression_head_mode == "identity" and dim == output_dim:
            self.regression_head = nn.Identity()
        else:
            self.regression_head = ProjectorBlock(input_dim=dim, output_dim=output_dim, hidden_dim=dim // 2)

        self._last_frame_metrics = None  # Frame metrics dict for advantage computation
        self._last_scale_var = None
        self._last_concentration_loss = None
        

    def post_init(self):
        last_layer = None
        if isinstance(self.regression_head, ProjectorBlock):
            last_layer = self.regression_head.get_last_layer()
        elif isinstance(self.regression_head, nn.Sequential):
            last_layer = self.regression_head[-1]
        elif isinstance(self.regression_head, nn.Linear):
            last_layer = self.regression_head
        if last_layer is None or not hasattr(last_layer, "weight"):
            return
        nn.init.orthogonal_(last_layer.weight, gain=0.01)
        with torch.no_grad():
            if self.use_discrete_action:
                if last_layer.bias is None:
                    return
                target_val = 1.0
                diff = torch.abs(self.scale_bins - target_val)
                target_idx = torch.argmin(diff).item()
                nn.init.zeros_(last_layer.bias)
                last_layer.bias[target_idx] = 0.0
            else:
                if last_layer.bias is None:
                    return
                bias_device = last_layer.bias.device
                bias_dtype = last_layer.bias.dtype
                target_ratio = (1.0 - self.min_scale) / (self.max_scale - self.min_scale)
                target_ratio = max(0.01, min(0.99, target_ratio))
                if self.continuous_dist == "logistic_normal":
                    mu_bias = torch.logit(
                        torch.tensor(target_ratio, device=bias_device, dtype=bias_dtype),
                        eps=1e-6
                    )
                    sigma = torch.tensor(
                        float(self.logistic_normal_init_sigma),
                        device=bias_device,
                        dtype=bias_dtype
                    ).clamp_min(1e-3)
                    sigma_bias = torch.log(torch.expm1(sigma))  # inverse softplus

                    last_layer.bias[0] = mu_bias
                    last_layer.bias[1] = sigma_bias
                else:
                    if not self.beta_add_one:
                        # Target: alpha = 1.0, beta = 1.0 (Uniform distribution)
                        # When beta_add_one=False:
                        # alpha = softplus(params * beta_param_scale) + 1e-6 = 1.0
                        # => softplus(params * beta_param_scale) = 1.0 - 1e-6 ≈ 1.0
                        # => params * beta_param_scale = softplus_inverse(1.0) = log(exp(1)-1) ≈ 0.541
                        # => bias = 0.541 / beta_param_scale
                        target_softplus_output = torch.tensor(1.0, device=bias_device, dtype=bias_dtype)
                        # softplus_inverse(x) = log(exp(x) - 1)
                        target_softplus_input = torch.log(torch.expm1(target_softplus_output))
                        if self.beta_param_scale > 0:
                            init_bias = target_softplus_input / self.beta_param_scale
                        else:
                            init_bias = target_softplus_input
                        # Also zero out weights to ensure params ≈ bias initially
                        nn.init.zeros_(last_layer.weight)
                        last_layer.bias[0] = init_bias
                        last_layer.bias[1] = init_bias
                    elif self.beta_init_mode == "uniform":
                        # Target: alpha = 1.0, beta = 1.0 (Uniform distribution)
                        # alpha = softplus(params * beta_param_scale) + 1.0 = 1.0
                        # => softplus(params * beta_param_scale) = 0
                        # => params * beta_param_scale << 0
                        # If we want params=0 (after layernorm) to produce alpha=1,
                        # we need bias such that softplus(bias * beta_param_scale) ≈ 0
                        # softplus(-10) ≈ 4.5e-5 ≈ 0, so bias * beta_param_scale = -10
                        # => bias = -10 / beta_param_scale
                        target_softplus_input = torch.tensor(-10.0, device=bias_device, dtype=bias_dtype)
                        if self.beta_param_scale > 0:
                            init_bias = target_softplus_input / self.beta_param_scale
                        else:
                            init_bias = target_softplus_input
                        # Also zero out weights to ensure params ≈ bias initially
                        nn.init.zeros_(last_layer.weight)
                        last_layer.bias[0] = init_bias
                        last_layer.bias[1] = init_bias
                    else:
                        target_sum = 2.0
                        target_alpha = target_sum * target_ratio
                        target_beta = target_sum * (1.0 - target_ratio)
                        y0 = torch.tensor(max(0.1, target_alpha - 1.0), device=bias_device, dtype=bias_dtype)
                        y1 = torch.tensor(max(0.1, target_beta - 1.0), device=bias_device, dtype=bias_dtype)
                        last_layer.bias[0] = torch.log(torch.expm1(y0))
                        last_layer.bias[1] = torch.log(torch.expm1(y1))

    def _get_discrete_policy(self, logits, actions=None):
        """
        Input: logits (B, T, num_bins)
        Output: indices (B, T), scales (B, T), log_probs (B, T)
        """
        temp = float(getattr(self, "categorical_temperature", 1.0))
        if not (temp > 0):
            temp = 1.0
        dist = Categorical(logits=(logits / temp).float())
        
        if actions is None:
            # Sample Index
            action_indices = dist.sample()
        else:
            # Use provided actions (must be LongTensor indices)
            action_indices = actions.long().detach()
            
        log_prob = dist.log_prob(action_indices).float()
        
        # Map Index -> Float Scale
        scale_factors = self.scale_bins[action_indices].to(logits.dtype)
        
        return action_indices, scale_factors, log_prob.to(torch.float32)

    def _get_discrete_deterministic(self, logits):
        action_indices = torch.argmax(logits, dim=-1)
        scale_factors = self.scale_bins[action_indices]
        return action_indices, scale_factors, None

    def _get_continuous_policy(self, params, actions=None):
        """
        Input: params (B, T, 2) -> Alpha, Beta
        Output: action_0_1 (B, T), scales (B, T), log_probs (B, T)
        """
        # Ensure positivity
        if self.continuous_dist == "beta":
            params_fp32 = (params * self.beta_param_scale).float()
            add_one = 1.0 if self.beta_add_one else 0.0
            alpha = F.softplus(params_fp32[..., 0]) + add_one + 1e-6
            beta = F.softplus(params_fp32[..., 1]) + add_one + 1e-6
            # FIX: Allow smaller alpha/beta (min 0.1) for higher variance when needed
            # Max clamp reduced to 20 for numerical stability while allowing sharper peaks
            alpha = alpha.clamp(min=0.1, max=20.0)
            beta = beta.clamp(min=0.1, max=20.0)
            alpha32 = alpha.float()
            beta32 = beta.float()
            dist0 = Beta(alpha32, beta32)
            if actions is None:
                use_reparam = (getattr(self, "sim_scale_weight", 0.0) > 0) or (getattr(self, "contrastive_weight", 0.0) > 0)
                action_0_1 = dist0.rsample() if use_reparam else dist0.sample()
            else:
                action_0_1 = actions.float().detach()
            epsilon = 1e-6
            action_0_1 = action_0_1.clamp(min=epsilon, max=1.0 - epsilon)
            log_prob = dist0.log_prob(action_0_1).float()
        else:
            params_fp32 = params.float()
            mu = params_fp32[..., 0]
            sigma = F.softplus(params_fp32[..., 1]) + 1e-6
            # FIX: Clamp sigma to ensure reasonable exploration range
            # Min 0.1 prevents over-concentration, max 5.0 prevents numerical instability
            sigma = sigma.clamp(min=0.1, max=5.0)
            mu32 = mu.float()
            sigma32 = sigma.float()
            base = Normal(mu32, sigma32)
            dist0 = TransformedDistribution(base, SigmoidTransform())
            if actions is None:
                use_reparam = (getattr(self, "sim_scale_weight", 0.0) > 0) or (getattr(self, "contrastive_weight", 0.0) > 0)
                action_0_1 = dist0.rsample() if use_reparam else dist0.sample()
            else:
                action_0_1 = actions.float().detach()
            epsilon = 1e-6
            action_0_1 = action_0_1.clamp(min=epsilon, max=1.0 - epsilon)
            log_prob = dist0.log_prob(action_0_1).float()
        scale_factors = self.min_scale + action_0_1 * (self.max_scale - self.min_scale)
        return action_0_1.to(params.dtype), scale_factors.to(params.dtype), log_prob.to(torch.float32)

    def _get_continuous_deterministic(self, params):
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
            dist0 = Beta(alpha32, beta32)
            if self.continuous_eval_quantile != 0.5:
                q = torch.tensor(self.continuous_eval_quantile, device=params.device, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
                action_0_1 = dist0.icdf(q)
            else:
                action_0_1 = dist0.mean
        else:
            params_fp32 = params.float()
            mu32 = params_fp32[..., 0]
            sigma32 = F.softplus(params_fp32[..., 1]) + 1e-6
            sigma32 = sigma32.clamp(min=0.1, max=5.0)
            if self.continuous_eval_quantile != 0.5:
                q = torch.tensor(self.continuous_eval_quantile, device=params.device, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
                z = Normal(0.0, 1.0).icdf(q).to(torch.float32)
                action_0_1 = torch.sigmoid(mu32 + sigma32 * z)
            else:
                denom = torch.sqrt(1.0 + (torch.pi * sigma32 * sigma32 / 8.0))
                action_0_1 = torch.sigmoid(mu32 / denom)
        scale_factors = self.min_scale + action_0_1 * (self.max_scale - self.min_scale)
        return action_0_1.to(params.dtype), scale_factors.to(params.dtype), None

    def forward(
        self,
        visual_features_batch,  # (B, Max_T*Max_L, D) - already padded
        visual_grid_thw,        # (B, 3)
        text_features,          # (B, N, D)
        visual_mask=None,       # (B, Max_T*Max_L) - True for valid
        text_mask=None,         # (B, N)
        actions=None,
        visual_per_sample=None,
        eval_mode=False,
        scale_mask=None,
        compute_frame_metrics: bool = False,  # Whether to compute frame metrics for advantage
    ):
        B, Max_Tokens, D = visual_features_batch.shape
        device = visual_features_batch.device
        compute_aux = actions is not None and eval_mode is False
        compute_frame_metrics = compute_frame_metrics and compute_aux

        if visual_per_sample is None:
            visual_per_sample = [B]
        elif isinstance(visual_per_sample, torch.Tensor):
            visual_per_sample = visual_per_sample.tolist()
        object_t_dims = visual_grid_thw[:, 0].to(torch.long)
        object_l_dims = ((visual_grid_thw[:, 1] * visual_grid_thw[:, 2]) // (self.spatial_merge_size**2)).to(torch.long)
        max_t_obj = int(object_t_dims.max().item())

        padded_actions_in = None
        if actions is not None:
            valid_actions_flat = actions[scale_mask]

            if valid_actions_flat.numel() != object_t_dims.sum():
                raise ValueError(f"Number of actions ({valid_actions_flat.numel()}) does not match the number of valid frames ({object_t_dims.sum()})")
            
            splits = torch.split(valid_actions_flat, object_t_dims.cpu().tolist())
            pad_val = -1 if self.use_discrete_action else 0.0
            
            # Result Shape: (Total_Objects, T_object_max) -> (4, 7)
            padded_actions_in = pad_sequence(
                list(splits), 
                batch_first=True, 
                padding_value=pad_val
            )

        if self.use_discrete_action:
            output_dim = int(self.num_bins)
        else:
            output_dim = 2
        head_outputs_obj = torch.zeros((B, max_t_obj, output_dim), dtype=visual_features_batch.dtype, device=device)
        frame_features_obj = torch.zeros((B, max_t_obj, D), dtype=visual_features_batch.dtype, device=device)

        # Frame metrics buffers (only if needed for advantage computation)
        if compute_frame_metrics:
            redundancy_obj = torch.zeros((B, max_t_obj), dtype=torch.float32, device=device)
            uniqueness_obj = torch.zeros((B, max_t_obj), dtype=torch.float32, device=device)
            text_relevance_obj = torch.zeros((B, max_t_obj), dtype=torch.float32, device=device)
            info_score_obj = torch.zeros((B, max_t_obj), dtype=torch.float32, device=device)

        # Group by (T, L) for efficient processing
        # Optimized: use return_inverse to avoid repeated nonzero calls
        pair = torch.stack([object_t_dims, object_l_dims], dim=1)
        unique_pair, inverse_indices = torch.unique(pair, dim=0, return_inverse=True)
        temporal_rope_cache: dict[int, torch.Tensor] = {}
        spatial_rope_cache: dict[int, torch.Tensor] = {}

        for group_idx, tl in enumerate(unique_pair):
            t = int(tl[0].item())
            l = int(tl[1].item())
            
            if t <= 0 or l <= 0:
                continue
            
            # Use pre-computed inverse indices for O(B) grouping instead of O(B) nonzero per group
            idx = (inverse_indices == group_idx).nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue

            feats = visual_features_batch.index_select(0, idx)[:, : t * l].reshape(-1, t, l, D)
            enc_states = text_features.index_select(0, idx) if exists(text_features) else None
            enc_mask = text_mask.index_select(0, idx) if exists(text_mask) else None

            temporal_rope = temporal_rope_cache.get(t)
            if temporal_rope is None:
                temporal_rope = self.temporal_rope_gen(t)
                temporal_rope_cache[t] = temporal_rope
            spatial_rope = spatial_rope_cache.get(l)
            if spatial_rope is None:
                spatial_rope = self.rope_gen(l)
                spatial_rope_cache[l] = spatial_rope

            hidden_states = feats
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states,
                    encoder_hidden_states=enc_states,
                    encoder_attention_mask=enc_mask,
                    temporal_rope=temporal_rope,
                    spatial_rope=spatial_rope,
                    temporal_mask=None,
                    spatial_mask=None,
                )

            Bobj, T, L, D = hidden_states.shape

            frame_token = hidden_states.mean(dim=2)
            if enc_states is not None:
                att = torch.matmul(frame_token.float(), enc_states.float().transpose(-1, -2)) / (float(D) ** 0.5)
                if enc_mask is not None:
                    att = att.masked_fill(~enc_mask[:, None, :], float("-inf"))
                w = att.softmax(dim=-1).to(frame_token.dtype)
                q_frame = torch.matmul(w, enc_states)
            else:
                q_frame = frame_token

            if self.patch_norm is not None:
                hidden_gate = self.patch_norm(hidden_states)
            else:
                hidden_gate = hidden_states
            q = (q_frame * self.gate_query_scale)[:, :, None, :].expand(Bobj, T, L, D)
            gate_in = torch.cat([hidden_gate, q], dim=-1)
            gate_logits = self.pool_gate(gate_in.to(self.pool_gate[0].weight.dtype)).squeeze(-1)
            gate = (gate_logits.float() / max(self.gate_temperature, 1e-6)).softmax(dim=-1).to(hidden_states.dtype)
            pooled = (gate.unsqueeze(-1) * hidden_states).sum(dim=2)

            mu = hidden_states.mean(dim=2, keepdim=True)
            var = ((hidden_states - mu) ** 2).mean(dim=(2, 3))
            p = gate.clamp_min(1e-9)
            entropy = -(p * p.log()).sum(dim=-1)
            peak = gate.max(dim=-1).values
            pooled32 = pooled.float()
            if int(T) > 1:
                prev = F.normalize(pooled32[:, :-1], dim=-1)
                cur = F.normalize(pooled32[:, 1:], dim=-1)
                sim_prev = (cur * prev).sum(dim=-1)
                sim_prev = F.pad(sim_prev, (1, 0), value=0.0)
            else:
                sim_prev = torch.zeros((Bobj, T), device=pooled.device, dtype=pooled.dtype)
            
            # Compute text-frame relevance (NEW)
            if enc_states is not None:
                relevance = self.relevance_scorer(pooled, enc_states, enc_mask)  # (Bobj, T)
            else:
                relevance = torch.zeros((Bobj, T), device=pooled.device, dtype=pooled.dtype)
            
            # info now includes 5 signals: var, entropy, peak, sim_prev, relevance
            info = torch.stack([var, entropy, peak, sim_prev.to(hidden_states.dtype), relevance.to(hidden_states.dtype)], dim=-1)

            if self.pooled_norm is not None:
                pooled_in = self.pooled_norm(pooled)
            else:
                pooled_in = pooled
            frame_features_in = torch.cat([pooled_in, info], dim=-1)
            if self.info_fuse_mode == "residual_gate":
                proj = self.info_proj(frame_features_in.to(self.info_proj.weight.dtype))
                frame_features = pooled + torch.tanh(self.info_gate) * proj
            else:
                if isinstance(self.info_proj, nn.Sequential):
                    dtype = self.info_proj[0].weight.dtype
                else:
                    dtype = self.info_proj.weight.dtype
                frame_features = self.info_proj(frame_features_in.to(dtype))
            head_outputs = self.regression_head(frame_features)
            head_outputs_obj[idx, :t] = head_outputs
            frame_features_obj[idx, :t] = frame_features

            # ===== Store Frame Metrics (scalar, memory-efficient) =====
            if compute_frame_metrics:
                with torch.no_grad():
                    # Redundancy: sim_prev already computed (similarity with previous frame)
                    redundancy_local = sim_prev.float().clamp(-1, 1)  # (Bobj, T)
                    
                    # Uniqueness: 1 - mean similarity to all other frames
                    ff_norm = F.normalize(frame_features.float(), dim=-1)  # (Bobj, T, D)
                    if T > 1:
                        sim_matrix = torch.matmul(ff_norm, ff_norm.transpose(-1, -2))  # (Bobj, T, T)
                        eye = torch.eye(T, device=device, dtype=torch.bool).unsqueeze(0)
                        sim_matrix = sim_matrix.masked_fill(eye, 0.0)
                        mean_sim = sim_matrix.sum(dim=-1) / (T - 1)
                        uniqueness_local = (1.0 - mean_sim).clamp(0.0, 1.0)
                    else:
                        uniqueness_local = torch.ones((Bobj, T), device=device, dtype=torch.float32)
                    
                    # Info score: aggregate of info vector (var + entropy + peak etc.)
                    info_score_local = info.float().mean(dim=-1)  # (Bobj, T)
                    
                    # Text relevance: already computed
                    text_relevance_local = relevance.float().clamp(-1, 1)  # (Bobj, T)
                    
                    # Store in buffers
                    redundancy_obj[idx, :t] = redundancy_local
                    uniqueness_obj[idx, :t] = uniqueness_local
                    text_relevance_obj[idx, :t] = text_relevance_local
                    info_score_obj[idx, :t] = info_score_local

        splits = torch.split(object_t_dims, visual_per_sample)
        valid_lengths = torch.stack([s.sum() for s in splits]).to(device)
        pad_max = max(self.max_frames, int(valid_lengths.max().item()))
        scale_mask_out = torch.arange(pad_max, device=device)[None, :] < valid_lengths[:, None]

        temporal_valid = torch.arange(max_t_obj, device=device)[None, :] < object_t_dims[:, None]

        padded_actions_for_policy = padded_actions_in
        if self.use_discrete_action and padded_actions_for_policy is not None:
            padded_actions_for_policy = padded_actions_for_policy.masked_fill(padded_actions_for_policy < 0, 0)

        if self.use_discrete_action:
            if not eval_mode:
                raw_actions_obj, scales_obj, log_probs_obj = self._get_discrete_policy(head_outputs_obj, padded_actions_for_policy)
            else:
                raw_actions_obj, scales_obj, log_probs_obj = self._get_discrete_deterministic(head_outputs_obj)
            pad_val_action = -1
        else:
            if not eval_mode:
                raw_actions_obj, scales_obj, log_probs_obj = self._get_continuous_policy(head_outputs_obj, padded_actions_for_policy)
            else:
                raw_actions_obj, scales_obj, log_probs_obj = self._get_continuous_deterministic(head_outputs_obj)
            pad_val_action = 0.0

        raw_actions_obj = raw_actions_obj.to(torch.long if self.use_discrete_action else torch.float32)
        scales_obj = scales_obj.to(torch.float32)
        log_probs_obj = log_probs_obj.to(torch.float32) if log_probs_obj is not None else None

        raw_actions_obj = raw_actions_obj.masked_fill(~temporal_valid, pad_val_action)
        scales_obj = scales_obj.masked_fill(~temporal_valid, 1.0)
        if log_probs_obj is not None:
            log_probs_obj = log_probs_obj.masked_fill(~temporal_valid, 0.0)

        if compute_aux and not self.use_discrete_action and self.training:
            m = temporal_valid.float()
            params = head_outputs_obj
            if self.continuous_dist == "beta":
                params = params * self.beta_param_scale
                add_one = 1.0 if self.beta_add_one else 0.0
                alpha = F.softplus(params[..., 0]) + add_one + 1e-6
                beta = F.softplus(params[..., 1]) + add_one + 1e-6
                concentration = alpha + beta
                self._last_concentration_loss = (concentration * m).sum() / m.sum().clamp_min(1.0)
            else:
                sigma = F.softplus(params[..., 1]) + 1e-6
                sigma = sigma.clamp(min=0.1, max=5.0)
                self._last_concentration_loss = ((1.0 / (sigma + 1e-6)) * m).sum() / m.sum().clamp_min(1.0)
        else:
            self._last_concentration_loss = None

        if not self.use_discrete_action:
            invalid_actions_obj = ~torch.isfinite(raw_actions_obj)
            if invalid_actions_obj.any():
                raw_actions_obj = raw_actions_obj.masked_fill(invalid_actions_obj, 0.4)

        invalid_scales_obj = ~torch.isfinite(scales_obj)
        if invalid_scales_obj.any():
            scales_obj = scales_obj.masked_fill(invalid_scales_obj, 1.0)

        if log_probs_obj is not None:
            invalid_log_probs_obj = ~torch.isfinite(log_probs_obj)
            if invalid_log_probs_obj.any():
                log_probs_obj = log_probs_obj.masked_fill(invalid_log_probs_obj, 0.0)
                
        # Compute sim_scale_loss: leverage redundancy to compress scales
        if compute_aux and self.sim_scale_weight > 0 and self.training and frame_features_obj.shape[1] > 1:
            f0 = frame_features_obj[:, :-1]
            f1 = frame_features_obj[:, 1:]
            sim = F.cosine_similarity(f0, f1, dim=-1)
            
            # If sim is high (close to 1), w approaches 1.
            # Target = previous_scale - gamma * w
            # This forces current scale to drop relative to previous scale if redundant
            w = torch.sigmoid((sim - self.sim_tau) / self.sim_temp)
            m = (temporal_valid[:, :-1] & temporal_valid[:, 1:]).float()
            
            # For discrete actions, use soft scales (differentiable) for loss computation
            if self.use_discrete_action:
                probs = head_outputs_obj.softmax(dim=-1)
                soft_scales_obj = (probs * self.scale_bins).sum(dim=-1)
                s = soft_scales_obj.clamp_min(1e-6).log()
            else:
                # Use distribution mean to form a differentiable soft scale proxy
                params = head_outputs_obj
                if self.continuous_dist == "beta":
                    params = params * self.beta_param_scale
                    add_one = 1.0 if self.beta_add_one else 0.0
                    alpha = F.softplus(params[..., 0]) + add_one + 1e-6
                    beta = F.softplus(params[..., 1]) + add_one + 1e-6
                    alpha = alpha.clamp(min=0.1, max=20.0)
                    beta = beta.clamp(min=0.1, max=20.0)
                    mean_action = alpha / (alpha + beta)
                else:
                    mu = params[..., 0]
                    sigma = F.softplus(params[..., 1]) + 1e-6
                    sigma = sigma.clamp(min=0.1, max=5.0)
                    denom = torch.sqrt(1.0 + (torch.pi * sigma * sigma / 8.0))
                    mean_action = torch.sigmoid(mu / denom)
                soft_scales_obj = self.min_scale + mean_action * (self.max_scale - self.min_scale)
                s = soft_scales_obj.clamp_min(1e-6).log()
            # Align shapes: w(B, T-1), m(B, T-1), s_prev(B, T-1), s_cur(B, T-1)
            w = w[:, : s.shape[1] - 1]
            s_prev = s[:, :-1]
            s_cur = s[:, 1:]
            target = s_prev - self.sim_gamma * w
            err = F.relu(s_cur - target) # Only penalize if s[t] > target
            self._last_sim_scale_loss = (err * w * m).sum() / (m.sum().clamp_min(1.0))
        else:
            self._last_sim_scale_loss = None

        log_probs = self.restructure_sequence(log_probs_obj, visual_grid_thw, scale_mask_out, 0.0, device) if log_probs_obj is not None else None
        scales = self.restructure_sequence(scales_obj, visual_grid_thw, scale_mask_out, 1.0, device)
        mask = scale_mask_out.float()
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (scales * mask).sum(dim=1, keepdim=True) / denom
        var = ((scales - mean) ** 2 * mask).sum(dim=1, keepdim=True) / denom
        if compute_aux:
            self._last_scale_var = var.mean()
        else:
            self._last_scale_var = None

        object_t_dims = visual_grid_thw[:, 0]
        max_len_temp = frame_features_obj.shape[1]
        source_mask = torch.arange(max_len_temp, device=device)[None, :] < object_t_dims[:, None]
        valid_values = frame_features_obj[source_mask]
        frame_features_seq = torch.zeros(
            (scale_mask_out.shape[0], scale_mask_out.shape[1], D),
            dtype=frame_features_obj.dtype,
            device=device,
        )
        frame_features_seq[scale_mask_out] = valid_values
        self.last_frame_features = frame_features_seq
        self.last_scales = scales
        
        # Restructure frame metrics to batch format (only if computed)
        if compute_frame_metrics:
            self._last_frame_metrics = {
                "redundancy": self.restructure_sequence(redundancy_obj, visual_grid_thw, scale_mask_out, 0.0, device),
                "uniqueness": self.restructure_sequence(uniqueness_obj, visual_grid_thw, scale_mask_out, 0.5, device),
                "text_relevance": self.restructure_sequence(text_relevance_obj, visual_grid_thw, scale_mask_out, 0.0, device),
                "info_score": self.restructure_sequence(info_score_obj, visual_grid_thw, scale_mask_out, 0.0, device),
            }
        else:
            self._last_frame_metrics = None

        if actions is not None:
            return None, scales, log_probs, scale_mask_out

        raw_actions = self.restructure_sequence(raw_actions_obj, visual_grid_thw, scale_mask_out, pad_val_action, device)
        if raw_actions is not None:
            invalid_actions = ~torch.isfinite(raw_actions)
            if invalid_actions.any():
                raw_actions = raw_actions.masked_fill(invalid_actions, 0.0)

        if scales is not None:
            invalid_scales = ~torch.isfinite(scales)
            if invalid_scales.any():
                scales = scales.masked_fill(invalid_scales, 1.0)

        return raw_actions, scales, log_probs, scale_mask_out


    def get_scale_mask(self, visual_grid_thw, visual_per_sample, max_frames, device):
        """
        Generate Scale Mask
        
        Args:
            visual_grid_thw: [Total_Objects, 3]
            visual_per_sample: List[int], e.g. [2, 1, 1]
            max_frames
        """
        # e.g., [1, 1, 1, 7]
        object_t_dims = visual_grid_thw[:, 0]
        
        # Input: tensor([1, 1, 1, 7]), split_sizes=[2, 1, 1]
        # splits -> (tensor([1, 1]), tensor([1]), tensor([7]))
        splits = torch.split(object_t_dims, visual_per_sample)
        
        # Sample 0: 1+1=2
        # Sample 1: 1
        # Sample 2: 7
        # valid_lengths -> tensor([2, 1, 7])
        valid_lengths = torch.stack([s.sum() for s in splits]).to(device)
        
        # shape: [Batch, Max_Frames]
        scale_mask = torch.arange(max_frames, device=device)[None, :] < valid_lengths[:, None]
        
        return scale_mask, valid_lengths


    def restructure_sequence(self, flat_tensor, visual_grid_thw, target_mask, pad_val, device):
        """
        Extract valid values from a flattened tensor with invalid padding and map them to a target matrix 
        according to the target mask, using temporal dimension info from visual_grid_thw.
        
        Args:
            flat_tensor: Tensor of shape (Total_Objects, Max_Len_Temp) (e.g., (4, 7)), containing invalid padding values.
            visual_grid_thw: Tensor of shape (Total_Objects, 3). The first column stores temporal dimension (T) 
                            which determines the valid length of each row in flat_tensor.
            target_mask: Boolean tensor of shape (Batch_Size, Max_Len) (e.g., (3, 7)) indicating valid positions 
                        in the target matrix (True = valid position to fill, False = padding position).
            pad_val: Scalar value to fill the padding positions in the output tensor.
        
        Returns:
            Tensor of shape (Batch_Size, Max_Len) with valid values mapped from flat_tensor and padding filled with pad_val.
        """
        if flat_tensor is None: 
            return None
        
        # visual_grid_thw[:, 0] stores temporal dimension (T) for each object (e.g., [1, 1, 1, 7])
        object_t_dims = visual_grid_thw[:, 0] 
        max_len_temp = flat_tensor.shape[1]  # 7 (temporal length of flat_tensor)
        
        # source_mask shape: [4, 7] (boolean mask)
        # Row 0 (T=1): [True, False, False, ..., False] (only first element is valid)
        # Row 3 (T=7): [True, True, True, ..., True] (all elements are valid)
        source_mask = torch.arange(max_len_temp, device=device)[None, :] < object_t_dims[:, None]
        
        valid_values = flat_tensor[source_mask]
        
        assert valid_values.numel() == target_mask.sum(), \
           f"Value count mismatch: Source has {valid_values.numel()} valid values, Target expects {target_mask.sum()}"

        output = torch.full(
            target_mask.shape, 
            pad_val, 
            dtype=flat_tensor.dtype, 
            device=device
        )
        
        output[target_mask] = valid_values
        
        return output

    def get_frame_metrics(self) -> dict:
        """Get last computed frame metrics dict for advantage computation.
        
        Returns dict with keys: redundancy, uniqueness, text_relevance, info_score
        Each value is (Batch, MaxFrames) tensor.
        """
        return self._last_frame_metrics if self._last_frame_metrics is not None else {}


class RegressionHeadAllocator(ModuleUtilsMixin, nn.Module):
    """
    Projects text features into vision feature space and delegates scale prediction
    to `FrameWiseScaleAllocator`. Provides a lightweight projector with LayerNorm -> Linear -> GELU -> Linear.
    """
    def __init__(self, vision_config):
        super().__init__()
        vocab_size = int(getattr(vision_config, "vocab_size"))
        llm_hidden_size = int(getattr(vision_config, "llm_hidden_size"))
        out_hidden_size = int(getattr(vision_config, "out_hidden_size", llm_hidden_size))
        output_dim = int(getattr(vision_config, "output_dim"))

        self.text_embed_dim = int(getattr(vision_config, "text_embed_dim", 0) or llm_hidden_size)
        self.embed_tokens = nn.Embedding(vocab_size, self.text_embed_dim)
        # self.text_to_output = (
        #     nn.Identity()
        #     if self.text_embed_dim == output_dim
        #     else nn.Sequential(nn.LayerNorm(self.text_embed_dim), nn.Linear(self.text_embed_dim, output_dim))
        # )
        enable_text_encoder = False
        if hasattr(vision_config, "use_text_encoder"):
            enable_text_encoder = bool(getattr(vision_config, "use_text_encoder"))
        elif hasattr(vision_config, "text_encoder_depth"):
            enable_text_encoder = int(getattr(vision_config, "text_encoder_depth")) > 0

        self.text_encoder = None
        if enable_text_encoder:
            text_encoder_depth = int(getattr(vision_config, "text_encoder_depth", 2))
            text_encoder_dropout = float(getattr(vision_config, "text_encoder_dropout", 0.0))
            text_encoder_ff_mult = int(getattr(vision_config, "text_encoder_ff_mult", 4))
            desired_heads = int(getattr(vision_config, "text_encoder_heads", getattr(vision_config, "num_heads", 8)))
            dim = output_dim
            num_heads = max(1, desired_heads)
            while num_heads > 1 and dim % num_heads != 0:
                num_heads -= 1
            self.text_encoder = SmallTextEncoder(
                dim=dim,
                depth=text_encoder_depth,
                num_heads=num_heads,
                dropout=text_encoder_dropout,
                ff_mult=text_encoder_ff_mult,
            )
        
        # Projectors
        mlp_mode = getattr(vision_config, "mlp_mode", "mlp")
        if mlp_mode == "linear":
            self.mlp = nn.Linear(out_hidden_size, output_dim)
        elif mlp_mode == "identity" and out_hidden_size == output_dim:
            self.mlp = nn.Identity()
        else:
            self.mlp = ProjectorBlock(
                input_dim=out_hidden_size,
                output_dim=output_dim,
                hidden_dim=output_dim,
            )

        vlp_in_dim = vision_config.hidden_size * vision_config.spatial_merge_size ** 2
        vlp_mode = getattr(vision_config, "vlp_mode", "mlp")
        if vlp_mode == "linear":
            self.vlp = nn.Linear(vlp_in_dim, output_dim)
        elif vlp_mode == "identity" and vlp_in_dim == output_dim:
            self.vlp = nn.Identity()
        else:
            self.vlp = ProjectorBlock(
                input_dim=vlp_in_dim,
                output_dim=output_dim,
                hidden_dim=output_dim,
            )

        self.spatial_merge_size = vision_config.spatial_merge_size
        
        self.scorer = FrameWiseScaleAllocator(
            dim=output_dim,
            depth=vision_config.self_depth,
            dim_head=vision_config.dim_head,
            heads=vision_config.num_heads,
            cross_attn_depth=vision_config.cross_depth,
            max_frames=vision_config.max_frames if hasattr(vision_config, "max_frames") else 4,
            spatial_merge_size=vision_config.spatial_merge_size,
            use_discrete_action=vision_config.use_discrete_action,
            scale_bins=vision_config.scale_bins if hasattr(vision_config, "scale_bins") else None,
            min_scale=vision_config.min_scale,
            max_scale=vision_config.max_scale,
            gate_temperature=getattr(vision_config, "gate_temperature", 1.0),
            gate_query_scale=getattr(vision_config, "gate_query_scale", 1.0),
            regression_head_mode=getattr(vision_config, "regression_head_mode", "mlp"),
            beta_param_scale=getattr(vision_config, "beta_param_scale", 1.0),
            beta_add_one=getattr(vision_config, "beta_add_one", True),
            beta_init_mode=getattr(vision_config, "beta_init_mode", "uniform"),
            continuous_dist=getattr(vision_config, "continuous_dist", "beta"),
            continuous_eval_quantile=getattr(vision_config, "continuous_eval_quantile", 0.5),
            logistic_normal_init_sigma=getattr(vision_config, "logistic_normal_init_sigma", 0.7),
            pool_gate_mode=getattr(vision_config, "pool_gate_mode", "no_ln"),
            info_fuse_mode=getattr(vision_config, "info_fuse_mode", "pooled_ln"),
            sim_scale_weight=getattr(vision_config, "sim_scale_weight", 0.0),
            sim_tau=getattr(vision_config, "sim_tau", 0.0),
            sim_temp=getattr(vision_config, "sim_temp", 1.0),
            sim_gamma=getattr(vision_config, "sim_gamma", 0.0),
        )
        self.out_hidden_size = out_hidden_size
        self.output_dim = output_dim

    def forward(
        self,
        visual_features,   # List[(T_i * L_i, D)]
        visual_grid_thw,   # (B, 3)
        text_features=None,  # (B, N, D)
        visual_mask=None,     # (B, T)
        text_mask=None,      # (B, N)
        actions=None,
        visual_per_sample=None,
        eval_mode=None,
        scale_mask=None,
        compute_frame_metrics: bool = False,  # Whether to compute frame metrics for advantage
    ):
        """
        Run text projector (if provided) then delegate to `FrameWiseScaleAllocator`.
        Returns (actions, scales, log_probs) from the scorer.
        """
        if text_features is not None:
            if int(text_features.shape[-1]) == self.out_hidden_size:
                text_out = self.mlp(text_features)
            elif int(text_features.shape[-1]) == self.text_embed_dim:
                text_out = self.text_to_output(text_features)
            elif int(text_features.shape[-1]) == self.output_dim:
                text_out = text_features
            else:
                raise ValueError(
                    f"Unsupported text_features dim: {int(text_features.shape[-1])}, "
                    f"expected one of [{self.out_hidden_size}, {self.text_embed_dim}, {self.output_dim}]"
                )

            if self.text_encoder is not None:
                text_out = self.text_encoder(text_out, text_mask)
            projected_text = text_out
        else:
            projected_text = None

        B_vis = len(visual_features)
        
        if projected_text is not None:
            B_txt = projected_text.shape[0]
            
            if B_vis != B_txt:
                if visual_per_sample is not None:
                    device = projected_text.device
                    if not isinstance(visual_per_sample, torch.Tensor):
                        repeats = torch.tensor(visual_per_sample, device=device)
                    else:
                        repeats = visual_per_sample.to(device)
                else:
                    assert B_vis % B_txt == 0, f"Visual batch {B_vis} not divisible by Text batch {B_txt}"
                    k = B_vis // B_txt
                    repeats = torch.full((B_txt,), k, device=projected_text.device, dtype=torch.long)

                # (B_txt, N, D) -> (B_vis, N, D)
                projected_text = projected_text.repeat_interleave(repeats, dim=0)
                
                if text_mask is not None:
                    text_mask = text_mask.repeat_interleave(repeats, dim=0)
        
        # Pad: (B, Max_Seq_Len, D)
        visual_batch = pad_sequence(visual_features, batch_first=True)
        
        # Project: (B, Max_Seq_Len, OutDim)
        # Running one big Linear layer is much faster than loop
        visual_batch_proj = self.vlp(visual_batch)

        return self.scorer(
            visual_features_batch=visual_batch_proj,
            visual_grid_thw=visual_grid_thw,
            visual_mask=visual_mask,
            text_features=projected_text,
            text_mask=text_mask,
            actions=actions,
            eval_mode=eval_mode,
            visual_per_sample=visual_per_sample,
            scale_mask=scale_mask,
            compute_frame_metrics=compute_frame_metrics,
        )
    
    # Proxy properties for consistent API with V2
    @property
    def _last_contrastive_loss(self) -> "torch.Tensor":
        """V1 doesn't have contrastive loss, return None."""
        return None
    
    @property
    def _last_sim_scale_loss(self) -> "torch.Tensor":
        """Proxy property for accessing scorer's sim_scale_loss."""
        return getattr(self.scorer, "_last_sim_scale_loss", None)
    
    @property
    def _last_concentration_loss(self) -> "torch.Tensor":
        return getattr(self.scorer, "_last_concentration_loss", None)
    
    @property
    def last_scales(self) -> "torch.Tensor":
        """Proxy property for accessing scorer's last scales."""
        return getattr(self.scorer, "last_scales", None)
    
    @property
    def last_frame_features(self) -> "torch.Tensor":
        """Proxy property for accessing scorer's last frame features."""
        return getattr(self.scorer, "last_frame_features", None)
    
    def get_frame_metrics(self) -> dict:
        """Get last computed frame metrics dict for advantage computation."""
        return self.scorer.get_frame_metrics()

# ============================================================================
# RegressionHeadAllocatorV2: Uses DifferentiableImportanceAllocator
# ============================================================================

class RegressionHeadAllocatorV2(ModuleUtilsMixin, nn.Module):
    """
    V2 Allocator using DifferentiableImportanceAllocator for improved frame differentiation.
    
    Key improvements over V1:
    - Frame Information Encoder (intrinsic info metrics)
    - Cross-Modal Matcher (per-frame text relevance)
    - Sparse Gating (Gumbel-Softmax/Top-k)
    - Dual-Path Encoder (preserves frame independence)
    - Contrastive Differentiator (forces feature diversity)
    
    Interface is identical to RegressionHeadAllocator.
    """
    
    def __init__(self, vision_config):
        super().__init__()
        from resadapt.allocator.importance_allocator_v2 import DifferentiableImportanceAllocator
        
        vocab_size = int(getattr(vision_config, "vocab_size"))
        llm_hidden_size = int(getattr(vision_config, "llm_hidden_size"))
        out_hidden_size = int(getattr(vision_config, "out_hidden_size", llm_hidden_size))
        output_dim = int(getattr(vision_config, "output_dim"))

        self.text_embed_dim = int(getattr(vision_config, "text_embed_dim", 0) or llm_hidden_size)
        self.embed_tokens = nn.Embedding(vocab_size, self.text_embed_dim)
        
        # Text encoder (optional)
        enable_text_encoder = False
        if hasattr(vision_config, "use_text_encoder"):
            enable_text_encoder = bool(getattr(vision_config, "use_text_encoder"))
        elif hasattr(vision_config, "text_encoder_depth"):
            enable_text_encoder = int(getattr(vision_config, "text_encoder_depth")) > 0

        self.text_encoder = None
        if enable_text_encoder:
            text_encoder_depth = int(getattr(vision_config, "text_encoder_depth", 2))
            text_encoder_dropout = float(getattr(vision_config, "text_encoder_dropout", 0.0))
            text_encoder_ff_mult = int(getattr(vision_config, "text_encoder_ff_mult", 4))
            desired_heads = int(getattr(vision_config, "text_encoder_heads", getattr(vision_config, "num_heads", 8)))
            dim = output_dim
            num_heads = max(1, desired_heads)
            while num_heads > 1 and dim % num_heads != 0:
                num_heads -= 1
            self.text_encoder = SmallTextEncoder(
                dim=dim,
                depth=text_encoder_depth,
                num_heads=num_heads,
                dropout=text_encoder_dropout,
                ff_mult=text_encoder_ff_mult,
            )
        
        # Text projector
        mlp_mode = getattr(vision_config, "mlp_mode", "mlp")
        if mlp_mode == "linear":
            self.mlp = nn.Linear(out_hidden_size, output_dim)
        elif mlp_mode == "identity" and out_hidden_size == output_dim:
            self.mlp = nn.Identity()
        else:
            self.mlp = ProjectorBlock(
                input_dim=out_hidden_size,
                output_dim=output_dim,
                hidden_dim=output_dim,
            )

        # Vision projector
        vlp_in_dim = vision_config.hidden_size * vision_config.spatial_merge_size ** 2
        vlp_mode = getattr(vision_config, "vlp_mode", "mlp")
        if vlp_mode == "linear":
            self.vlp = nn.Linear(vlp_in_dim, output_dim)
        elif vlp_mode == "identity" and vlp_in_dim == output_dim:
            self.vlp = nn.Identity()
        else:
            self.vlp = ProjectorBlock(
                input_dim=vlp_in_dim,
                output_dim=output_dim,
                hidden_dim=output_dim,
            )

        self.spatial_merge_size = vision_config.spatial_merge_size
        
        # V2: Use DifferentiableImportanceAllocator
        self.scorer = DifferentiableImportanceAllocator(
            dim=output_dim,
            depth=vision_config.self_depth,
            dim_head=vision_config.dim_head,
            heads=vision_config.num_heads,
            cross_attn_depth=vision_config.cross_depth,
            max_frames=vision_config.max_frames if hasattr(vision_config, "max_frames") else 4,
            spatial_merge_size=vision_config.spatial_merge_size,
            use_discrete_action=vision_config.use_discrete_action,
            scale_bins=vision_config.scale_bins if hasattr(vision_config, "scale_bins") else None,
            min_scale=vision_config.min_scale,
            max_scale=vision_config.max_scale,
            # New V2 params
            gate_mode=getattr(vision_config, "gate_mode", "gumbel_softmax"),
            gate_k_ratio=getattr(vision_config, "gate_k_ratio", 0.25),
            contrastive_weight=getattr(vision_config, "contrastive_weight", 0.1),
            contrastive_temperature=getattr(vision_config, "contrastive_temperature", 0.1),
            # Sim scale params (pass through)
            sim_scale_weight=getattr(vision_config, "sim_scale_weight", 0.0),
            sim_tau=getattr(vision_config, "sim_tau", 0.5),
            sim_temp=getattr(vision_config, "sim_temp", 0.1),
            sim_gamma=getattr(vision_config, "sim_gamma", 0.05),
            # Shared params
            continuous_dist=getattr(vision_config, "continuous_dist", "beta"),
            continuous_eval_quantile=getattr(vision_config, "continuous_eval_quantile", 0.5),
            beta_param_scale=getattr(vision_config, "beta_param_scale", 1.0),
            beta_add_one=getattr(vision_config, "beta_add_one", True),
            beta_init_mode=getattr(vision_config, "beta_init_mode", "uniform"),
            logistic_normal_init_sigma=getattr(vision_config, "logistic_normal_init_sigma", 0.7),
        )
        self.out_hidden_size = out_hidden_size
        self.output_dim = output_dim
        self.scorer.post_init()

    def forward(
        self,
        visual_features,   # List[(T_i * L_i, D)]
        visual_grid_thw,   # (B, 3)
        text_features=None,  # (B, N, D)
        visual_mask=None,     # (B, T)
        text_mask=None,      # (B, N)
        actions=None,
        visual_per_sample=None,
        eval_mode=None,
        scale_mask=None,
        compute_frame_metrics: bool = False,  # Whether to compute frame metrics for advantage
    ):
        """Forward pass - same interface as RegressionHeadAllocator."""
        if text_features is not None:
            if int(text_features.shape[-1]) == self.out_hidden_size:
                text_out = self.mlp(text_features)
            elif int(text_features.shape[-1]) == self.output_dim:
                text_out = text_features
            else:
                raise ValueError(f"Unsupported text_features dim: {int(text_features.shape[-1])}")

            if self.text_encoder is not None:
                text_out = self.text_encoder(text_out, text_mask)
            projected_text = text_out
        else:
            projected_text = None

        B_vis = len(visual_features)
        
        if projected_text is not None:
            B_txt = projected_text.shape[0]
            
            if B_vis != B_txt:
                if visual_per_sample is not None:
                    device = projected_text.device
                    if not isinstance(visual_per_sample, torch.Tensor):
                        repeats = torch.tensor(visual_per_sample, device=device)
                    else:
                        repeats = visual_per_sample.to(device)
                else:
                    assert B_vis % B_txt == 0
                    k = B_vis // B_txt
                    repeats = torch.full((B_txt,), k, device=projected_text.device, dtype=torch.long)

                projected_text = projected_text.repeat_interleave(repeats, dim=0)
                if text_mask is not None:
                    text_mask = text_mask.repeat_interleave(repeats, dim=0)
        
        # Pad and project visual features
        visual_batch = pad_sequence(visual_features, batch_first=True)
        visual_batch_proj = self.vlp(visual_batch)
        
        return self.scorer(
            visual_features_batch=visual_batch_proj,
            visual_grid_thw=visual_grid_thw,
            visual_mask=visual_mask,
            text_features=projected_text,
            text_mask=text_mask,
            actions=actions,
            eval_mode=eval_mode,
            visual_per_sample=visual_per_sample,
            scale_mask=scale_mask,
            compute_frame_metrics=compute_frame_metrics,
        )
    
    def get_contrastive_loss(self) -> "torch.Tensor":
        """Get contrastive loss tensor from the scorer for training."""
        return self.scorer.get_contrastive_loss()
    
    def get_weighted_contrastive_loss(self) -> "torch.Tensor":
        """Get weighted contrastive loss for adding to policy loss."""
        return self.scorer.get_weighted_contrastive_loss()
    
    def get_sim_scale_loss(self) -> "torch.Tensor":
        """Get pairwise similarity-based scale loss (compress redundant frames)."""
        return self.scorer.get_sim_scale_loss()
    
    def get_weighted_sim_scale_loss(self) -> "torch.Tensor":
        """Get sim_scale_loss multiplied by weight for adding to policy loss."""
        return self.scorer.get_weighted_sim_scale_loss()
    
    def get_frame_metrics(self) -> dict:
        """Get last computed frame metrics dict for advantage computation."""
        return self.scorer.get_frame_metrics()
    
    @property
    def _last_contrastive_loss(self) -> "torch.Tensor":
        """Proxy property for accessing scorer's contrastive loss."""
        return self.scorer._last_contrastive_loss
    
    @property
    def _last_sim_scale_loss(self) -> "torch.Tensor":
        """Proxy property for accessing scorer's sim_scale_loss."""
        return self.scorer._last_sim_scale_loss
    
    @property
    def _last_concentration_loss(self) -> "torch.Tensor":
        return getattr(self.scorer, "_last_concentration_loss", None)
    
    @property
    def last_scales(self) -> "torch.Tensor":
        """Proxy property for accessing scorer's last scales."""
        return getattr(self.scorer, 'last_scales', None)
    
    @property
    def last_frame_features(self) -> "torch.Tensor":
        """Proxy property for accessing scorer's last frame features."""
        return getattr(self.scorer, 'last_frame_features', None)


# --- Main Execution ---
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    MAX_FRAMES = 32
    PATCHES_PER_FRAME = 256
    MAX_TEXT_LEN = 1024
    
    model = FrameWiseScaleAllocator(
        dim=512,
        depth=2,
        dim_head=64,
        heads=8,
        cross_attn_depth=2,
        temporal_depth=2,
        max_frames=MAX_FRAMES,
        patches_per_frame=PATCHES_PER_FRAME,
        max_text_len=MAX_TEXT_LEN,
    ).to(device)

    B, T, L, N, D = 2, MAX_FRAMES, PATCHES_PER_FRAME, 128, 512
    
    visual_features = torch.randn(B, T, L, D).to(device)
    text_features = torch.randn(B, N, D).to(device)

    video_mask = torch.ones(B, T, L, dtype=torch.bool).to(device)
    text_mask = torch.ones(B, N, dtype=torch.bool).to(device)

    scale_factors, log_prob = model(
        visual_patch_features=visual_features,
        text_features=text_features,
        video_mask=video_mask,
        text_mask=text_mask
    )

    assert scale_factors.shape == (B, T)
    assert (scale_factors >= model.min_scale).all() and (scale_factors <= model.max_scale).all()

    rewards = torch.randn(scale_factors.shape[0], device=device)  # Dummy rewards, shape (B,)
    _ = -(log_prob * rewards).mean()
