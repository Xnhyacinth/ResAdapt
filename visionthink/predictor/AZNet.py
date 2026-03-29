import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from torch.distributions import Beta, Categorical
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

def _flash_attn_cast_dtype(x: torch.Tensor) -> torch.Tensor:
    if x.dtype in (torch.float16, torch.bfloat16):
        return x
    if x.is_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return x.to(torch.bfloat16)
    return x.to(torch.float16)

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
            q = _flash_attn_cast_dtype(q)
            k = _flash_attn_cast_dtype(k)
            v = _flash_attn_cast_dtype(v)

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
            sdpa_mask = repeat(attn_mask, "b t -> (b l) 1 1 t", l=L)
            sdpa_mask = _sdpa_mask_from_valid(sdpa_mask, dtype=q.dtype)

        out = F.scaled_dot_product_attention(
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
            q = _flash_attn_cast_dtype(q)
            k = _flash_attn_cast_dtype(k)
            v = _flash_attn_cast_dtype(v)

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
            sdpa_mask = rearrange(attn_mask, "b t l -> (b t) 1 1 l")
            sdpa_mask = _sdpa_mask_from_valid(sdpa_mask, dtype=q.dtype)

        out = F.scaled_dot_product_attention(
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
            sdpa_mask = rearrange(context_mask, "b m -> b 1 1 m")
            sdpa_mask = _sdpa_mask_from_valid(sdpa_mask, dtype=q.dtype)

        out = F.scaled_dot_product_attention(
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


class PredictorDecoderLayer(ModuleUtilsMixin, nn.Module):
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
        
        self.has_cross_attn = layer_idx < config.cross_attn_depth
        self.has_spatio_temporal = layer_idx >= config.cross_attn_depth and layer_idx < (config.cross_attn_depth + config.depth)

        if self.has_cross_attn:
            self.cross_attn = CrossAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout)
            self.ff_cross = FeedForward(dim, mult=ff_mult, dropout=dropout)
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))
            self.use_cross_attn_gate = bool(getattr(config, "use_cross_attn_gate", False))

        if self.has_spatio_temporal:
            self.temporal_attn = PatchWiseTemporalAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout)
            self.spatial_attn = SpatialAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout)
            self.ff_spatio_temporal = FeedForward(dim, mult=ff_mult, dropout=dropout)

    def forward(self, hidden_states, encoder_hidden_states=None, encoder_attention_mask=None,
                temporal_rope=None, spatial_rope=None,
                temporal_mask=None, spatial_mask=None):
        # hidden_states: (B, T, L, D)
        
        if self.has_cross_attn and encoder_hidden_states is not None:
            residual = hidden_states
            # Cross Attn internally handles reshape
            use_gate = bool(getattr(self, "use_cross_attn_gate", False))
            gate = torch.tanh(self.cross_attn_gate) if use_gate else 1.0
            out = self.cross_attn(hidden_states, encoder_hidden_states, context_mask=encoder_attention_mask)
            hidden_states = residual + gate * out
            residual = hidden_states
            hidden_states = residual + gate * self.ff_cross(hidden_states)

        if self.has_spatio_temporal:
            # Temporal
            residual = hidden_states
            hidden_states = hidden_states + self.temporal_attn(
                hidden_states, rotary_pos_emb=temporal_rope, attn_mask=temporal_mask
            )
            
            # Spatial
            residual = hidden_states
            hidden_states = hidden_states + self.spatial_attn(
                hidden_states, rotary_pos_emb=spatial_rope, attn_mask=spatial_mask
            )
            
            # Feed Forward
            hidden_states = hidden_states + self.ff_spatio_temporal(hidden_states)

        return hidden_states


class PredictorConfig:
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

    def forward(self, x):
        return self.net(x)

    def get_last_layer(self):
        return self.net[-1]


class FrameWiseScalePredictor(ModuleUtilsMixin, nn.Module):
    """
    Unified Predictor supporting both Discrete (Categorical) and Continuous (Beta) actions.
    Predicts per-frame scale factors from patch-level vision features with optional text conditioning.
    Returns: (actions in [0,1], scale_factors in [min_scale,max_scale], log_prob).
    """
    def __init__(
        self, *, dim, depth, dim_head=64, heads=8, dropout=0., ff_mult=4,
        cross_attn_depth=1, max_frames=4, use_discrete_action=False,
        scale_bins=None, min_scale=0.25, max_scale=3.0, spatial_merge_size=2,
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.max_frames = max_frames
        self.use_discrete_action = use_discrete_action
        
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

        self.config = PredictorConfig(
            hidden_size=dim, dim_head=dim_head, num_heads=heads, dropout=dropout,
            ff_mult=ff_mult, cross_attn_depth=cross_attn_depth, depth=depth
        )
        
        total_layers = cross_attn_depth + depth
        self.layers = nn.ModuleList([
            PredictorDecoderLayer(self.config, i) for i in range(total_layers)
        ])
        
        self.regression_head = ProjectorBlock(input_dim=dim, output_dim=output_dim, hidden_dim=dim // 2)

        # Enable optional debug checks to catch NaN/Inf during development
        self.debug_checks = False
        

    def post_init(self):
        last_layer = self.regression_head.get_last_layer()
        nn.init.orthogonal_(last_layer.weight, gain=0.01)
        with torch.no_grad():
            if self.use_discrete_action:
                target_val = 1.0
                diff = torch.abs(self.scale_bins - target_val)
                target_idx = torch.argmin(diff).item()
                nn.init.zeros_(last_layer.bias)
                last_layer.bias[target_idx] = 5.0  
            else:
                target_ratio = (1.0 - self.min_scale) / (self.max_scale - self.min_scale)
                target_ratio = max(0.01, min(0.99, target_ratio))
                target_sum = 4.0 
                target_alpha = target_sum * target_ratio
                target_beta  = target_sum * (1.0 - target_ratio)
                def inverse_softplus_plus1(val):
                    y = val - 1.0
                    if y < 1e-4: y = 1e-4 
                    return torch.log(torch.exp(torch.tensor(y)) - 1.0)
                last_layer.bias[0] = inverse_softplus_plus1(target_alpha)
                last_layer.bias[1] = inverse_softplus_plus1(target_beta)

    def _get_discrete_policy(self, logits, actions=None):
        """
        Input: logits (B, T, num_bins)
        Output: indices (B, T), scales (B, T), log_probs (B, T)
        """
        dist = Categorical(logits=logits.float())
        
        if actions is None:
            # Sample Index
            action_indices = dist.sample()
        else:
            # Use provided actions (must be LongTensor indices)
            action_indices = actions.long().detach()
            
        log_prob = dist.log_prob(action_indices).float()
        
        # Map Index -> Float Scale
        scale_factors = self.scale_bins[action_indices].to(logits.dtype)
        
        return action_indices, scale_factors, log_prob.to(logits.dtype)

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
        params_fp32 = params.float()
        alpha = F.softplus(params_fp32[..., 0]) + 1.0 + 1e-6
        beta  = F.softplus(params_fp32[..., 1]) + 1.0 + 1e-6
        
        # Clamp for numerical stability
        alpha = alpha.clamp(max=50.0).contiguous()
        beta = beta.clamp(max=50.0).contiguous()
        
        dist = Beta(alpha, beta)
        
        if actions is None:
            action_0_1 = dist.sample()
        else:
            # Assume actions provided are already normalized to 0-1
            action_0_1 = actions.detach().float()

        epsilon = 1e-6
        action_0_1 = action_0_1.clamp(min=epsilon, max=1.0 - epsilon)
        
        log_prob = dist.log_prob(action_0_1)
        
        # Map 0-1 -> [min, max]
        scale_factors = self.min_scale + action_0_1 * (self.max_scale - self.min_scale)
        
        return action_0_1.to(params.dtype), scale_factors.to(params.dtype), log_prob.to(torch.float32)

    def _get_continuous_deterministic(self, params):
        params_fp32 = params.float()
        alpha = F.softplus(params_fp32[..., 0]) + 1.0 + 1e-6
        beta  = F.softplus(params_fp32[..., 1]) + 1.0 + 1e-6
        
        dist = Beta(alpha, beta)
        action_0_1 = dist.mean
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
    ):
        B, Max_Tokens, D = visual_features_batch.shape
        device = visual_features_batch.device

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

        output_dim = int(self.regression_head.get_last_layer().out_features)
        head_outputs_obj = torch.zeros((B, max_t_obj, output_dim), dtype=visual_features_batch.dtype, device=device)

        pair = torch.stack([object_t_dims, object_l_dims], dim=1)
        unique_pair = torch.unique(pair, dim=0)
        temporal_rope_cache: dict[int, torch.Tensor] = {}
        spatial_rope_cache: dict[int, torch.Tensor] = {}

        for tl in unique_pair:
            t = int(tl[0].item())
            l = int(tl[1].item())
            idx = torch.nonzero((pair[:, 0] == t) & (pair[:, 1] == l), as_tuple=False).squeeze(1)
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

            frame_features = hidden_states.mean(dim=2)
            head_outputs = self.regression_head(frame_features)
            head_outputs_obj[idx, :t] = head_outputs

        splits = torch.split(object_t_dims, visual_per_sample)
        valid_lengths = torch.stack([s.sum() for s in splits]).to(device)
        pad_max = max(self.max_frames, int(valid_lengths.max().item()))
        scale_mask_out = torch.arange(pad_max, device=device)[None, :] < valid_lengths[:, None]

        temporal_valid = torch.arange(max_t_obj, device=device)[None, :] < object_t_dims[:, None]

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
            if eval_mode:
                raw_actions_obj, scales_obj, log_probs_obj = self._get_continuous_deterministic(head_outputs_obj)
            else:
                raw_actions_obj, scales_obj, log_probs_obj = self._get_continuous_policy(head_outputs_obj, padded_actions_for_policy)
            pad_val_action = 0.0

        raw_actions_obj = raw_actions_obj.to(torch.long if self.use_discrete_action else torch.float32)
        scales_obj = scales_obj.to(torch.float32)
        log_probs_obj = log_probs_obj.to(torch.float32) if log_probs_obj is not None else None

        raw_actions_obj = raw_actions_obj.masked_fill(~temporal_valid, pad_val_action)
        scales_obj = scales_obj.masked_fill(~temporal_valid, 1.0)
        if log_probs_obj is not None:
            log_probs_obj = log_probs_obj.masked_fill(~temporal_valid, 0.0)

        if not self.use_discrete_action:
            invalid_actions_obj = ~torch.isfinite(raw_actions_obj)
            if invalid_actions_obj.any():
                invalid_count = int(invalid_actions_obj.sum().item())
                print(f"[predictor] non-finite actions detected: {invalid_count}")
                raw_actions_obj = raw_actions_obj.masked_fill(invalid_actions_obj, 0.4)

        invalid_scales_obj = ~torch.isfinite(scales_obj)
        if invalid_scales_obj.any():
            invalid_count = int(invalid_scales_obj.sum().item())
            print(f"[predictor] non-finite scales detected: {invalid_count}")
            scales_obj = scales_obj.masked_fill(invalid_scales_obj, 1.0)

        if log_probs_obj is not None:
            invalid_log_probs_obj = ~torch.isfinite(log_probs_obj)
            if invalid_log_probs_obj.any():
                invalid_count = int(invalid_log_probs_obj.sum().item())
                print(f"[predictor] non-finite log_probs detected: {invalid_count}")
                log_probs_obj = log_probs_obj.masked_fill(invalid_log_probs_obj, 0.0)
        log_probs = self.restructure_sequence(log_probs_obj, visual_grid_thw, scale_mask_out, 0.0, device) if log_probs_obj is not None else None

        if actions is not None:
            return None, None, log_probs, None

        raw_actions = self.restructure_sequence(raw_actions_obj, visual_grid_thw, scale_mask_out, pad_val_action, device)
        scales = self.restructure_sequence(scales_obj, visual_grid_thw, scale_mask_out, 1.0, device)

        if raw_actions is not None:
            invalid_actions = ~torch.isfinite(raw_actions)
            if invalid_actions.any():
                bad_samples = torch.nonzero(invalid_actions.any(dim=1), as_tuple=False).flatten().tolist()
                print(f"[predictor] non-finite actions in samples: {bad_samples}")
                raw_actions = raw_actions.masked_fill(invalid_actions, 0.0)

        if scales is not None:
            invalid_scales = ~torch.isfinite(scales)
            if invalid_scales.any():
                bad_samples = torch.nonzero(invalid_scales.any(dim=1), as_tuple=False).flatten().tolist()
                print(f"[predictor] non-finite scales in samples: {bad_samples}")
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

    def _debug_nan(self):
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN in param: {name}")


class RegressionHeadPredictor(ModuleUtilsMixin, nn.Module):
    """
    Projects text features into vision feature space and delegates scale prediction
    to `FrameWiseScalePredictor`. Provides a lightweight projector with LayerNorm -> Linear -> GELU -> Linear.
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
        self.mlp = ProjectorBlock(
            input_dim=out_hidden_size,
            output_dim=output_dim,
            hidden_dim=output_dim,
        )
        self.vlp = ProjectorBlock(
            input_dim=vision_config.hidden_size * vision_config.spatial_merge_size ** 2,
            output_dim=output_dim,
            hidden_dim=output_dim,
        )
        
        self.scorer = FrameWiseScalePredictor(
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
    ):
        """
        Run text projector (if provided) then delegate to `FrameWiseScalePredictor`.
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
            scale_mask=scale_mask
        )

# --- Main Execution ---
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    MAX_FRAMES = 32
    PATCHES_PER_FRAME = 256
    MAX_TEXT_LEN = 1024
    
    model = FrameWiseScalePredictor(
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

    print("Output scale_factors shape:", scale_factors.shape)
    assert scale_factors.shape == (B, T)
    
    is_in_range = (scale_factors >= model.min_scale).all() and (scale_factors <= model.max_scale).all()
    print(f"Are scale factors in range? {is_in_range}")
    assert is_in_range
    
    print("\nExample scale factors for the first sample in batch:")
    print(scale_factors[0].detach().cpu().numpy())
    
    print("\nFrameWiseScalePredictor ran successfully!")

    rewards = torch.randn(scale_factors.shape[0], device=device) # Dummy rewards, shape (B,)

    # 4. Compute policy loss
    policy_loss = -(log_prob * rewards).mean()
    
    print("--- Beta Distribution Output ---")
    print("Sampled scale_factors shape:", scale_factors.shape)
    print("Log probabilities shape:", log_prob.shape)
    print("Policy Loss (example):", policy_loss.item())
