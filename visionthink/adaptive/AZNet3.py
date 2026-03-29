import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from torch.distributions import Beta, Normal
# from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionRotaryEmbedding
from transformers.modeling_utils import ModuleUtilsMixin
from typing import List, Optional, Tuple

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
        return freqs


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(t, pos):
    rotate_dim = pos.shape[-1]
    pos = pos.unsqueeze(0).unsqueeze(0)
    t_pass, t_rot = t[..., rotate_dim:], t[..., :rotate_dim]
    t_rot = (t_rot * pos.cos()) + (rotate_half(t_rot) * pos.sin())
    return torch.cat((t_rot, t_pass), dim=-1)


class PatchWiseTemporalAttention(nn.Module):
    """
    Performs temporal attention on tokens at the same spatial position across different frames.
    Optimized to use Flash Attention (SDPA) when available.
    """
    temporal_dim_scale = 0.25 

    def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.):
        super().__init__()
        self.num_heads = max(1, round(heads * self.temporal_dim_scale))
        self.dim_head = dim_head
        # For SDPA, scaling is handled internally, but good to keep for manual fallback if needed
        self.scale = dim_head ** -0.5
        inner_dim = self.num_heads * self.dim_head
        
        self.norm = nn.LayerNorm(dim)
        self.dropout_p = dropout # Store dropout probability for SDPA
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, T: int, L: int, rotary_pos_emb=None):
        """
        Args:
            x: (B, T*L, D)
        """
        B, _, D = x.shape
        x = self.norm(x)
        
        qkv = self.to_qkv(x) # (B, T*L, 3 * inner_dim)
        
        # (B, T*L, 3*H*D) -> (B, T, L, 3, H, D)
        qkv = rearrange(qkv, 'b (t l) (three h d) -> b t l three h d', t=T, l=L, three=3, h=self.num_heads)
        
        # Permute to put L in batch position and T in sequence position
        # Target for Attention: (Batch_Size, Num_Heads, Seq_Len, Dim)
        # Here: Batch_Size = B * L
        # (B, T, L, 3, H, D) -> (3, B, L, H, T, D)
        qkv = rearrange(qkv, 'b t l three h d -> three (b l) h t d')
        q, k, v = qkv[0], qkv[1], qkv[2] # Each is (B*L, H, T, D)

        if exists(rotary_pos_emb):
            q = apply_rotary_pos_emb(q, rotary_pos_emb)
            k = apply_rotary_pos_emb(k, rotary_pos_emb)

        # PyTorch SDPA automatically selects FlashAttention kernel if inputs are fp16/bf16 and on CUDA
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        
        # (B*L, H, T, D) -> (B, L, H, T, D) -> (B, L, T, H, D) -> (B, T, L, H*D)
        out = rearrange(out, '(b l) h t d -> b t l (h d)', b=B, l=L)
        
        # Flatten back to (B, T*L, InnerDim)
        out = rearrange(out, 'b t l d -> b (t l) d')
        
        return self.to_out(out)


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

    def forward(self, x, context, rotary_pos_emb=None, context_mask=None):
        x = self.norm(x)
        context = self.context_norm(context)
        
        # Projections
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        
        # Reshape to (Batch, Heads, SeqLen, Dim) for SDPA
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b m (h d) -> b h m d', h=self.heads)
        v = rearrange(v, 'b m (h d) -> b h m d', h=self.heads)

        # RoPE (if needed)
        # Note: Logic kept generic. If RoPE is needed for CrossAttn, insert here.
        # if exists(rotary_pos_emb): ...
        
        # Prepare Mask for SDPA
        attn_mask = None
        if exists(context_mask):
            # context_mask: (B, M) -> (B, 1, 1, M) to broadcast over Heads and Query_Len
            attn_mask = rearrange(context_mask, 'b m -> b 1 1 m')
            # For SDPA, bool mask (True=Keep, False=Mask) or float mask (0, -inf) works.
            # Usually bool mask is safer if converting from typical padding masks.
            # Assuming context_mask is True for Valid tokens.
            # However, SDPA expects True for values to *participate* in attention.
            # If context_mask is bool, verify if True means padding or valid. 
            # Assuming standard: True = Valid.
            pass

        # Attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0
        )
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class SpatialAttention(nn.Module):
    """
    Performs spatial self-attention within each frame.
    Input is (B, T*L, D), reshaped to (B*T, L, D) for attention.
    """
    def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        
        self.norm = nn.LayerNorm(dim)
        self.dropout_p = dropout
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, T: int, L: int, rotary_pos_emb=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T*L, D).
        """
        B, _, D = x.shape
        x = self.norm(x)
        
        # Effective Batch = B * T, Sequence Length = L
        x = rearrange(x, 'b (t l) d -> (b t) l d', t=T, l=L)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        
        # Reshape to (Batch, Heads, SeqLen, Dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Apply RoPE
        if exists(rotary_pos_emb):
            # apply_rotary_pos_emb typically expects (B, H, SeqLen, D)
            q = apply_rotary_pos_emb(q, rotary_pos_emb)
            k = apply_rotary_pos_emb(k, rotary_pos_emb)
        
        # Attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0
        )
        
        # Reshape Output
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        # Reshape back to original format (B, T*L, D)
        return rearrange(out, '(b t) l d -> b (t l) d', b=B, t=T)


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
        
        dim = config.hidden_size # Assuming config object has these attributes
        dim_head = config.dim_head
        heads = config.num_heads
        dropout = config.dropout
        ff_mult = config.ff_mult
        
        # Determine if this layer should have temporal/spatial/cross attention
        # You can customize the logic. For example:
        # - First `cross_attn_depth` layers have CrossAttention
        # - Next `depth` layers have Temporal+Spatial
        
        self.has_cross_attn = layer_idx < config.cross_attn_depth
        self.has_spatio_temporal = layer_idx >= config.cross_attn_depth and layer_idx < (config.cross_attn_depth + config.depth)

        if self.has_cross_attn:
            self.cross_attn = CrossAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout)
            self.ff_cross = FeedForward(dim, mult=ff_mult, dropout=dropout)

        if self.has_spatio_temporal:
            self.temporal_attn = PatchWiseTemporalAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout)
            self.spatial_attn = SpatialAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout)
            self.ff_spatio_temporal = FeedForward(dim, mult=ff_mult, dropout=dropout)

    def forward(
        self, 
        hidden_states, 
        grid_thw, 
        encoder_hidden_states=None, 
        encoder_attention_mask=None,
        temporal_rope=None, 
        spatial_rope=None
    ):
        # Unpack grid info for reshaping inside attention blocks
        T = grid_thw[0].item()
        # Calculate L based on total tokens and T. 
        # hidden_states is (B=1, T*L, D) in the `_fuse_one_sample` context
        L = hidden_states.shape[1] // T 

        # 1. Cross Attention Block
        if self.has_cross_attn and encoder_hidden_states is not None:
            assert encoder_hidden_states is not None, "Cross attention requires encoder_hidden_states (text features)"
            residual = hidden_states
            hidden_states = self.cross_attn(
                hidden_states, 
                context=encoder_hidden_states, 
                context_mask=encoder_attention_mask
            )
            hidden_states = residual + hidden_states
            
            residual = hidden_states
            hidden_states = self.ff_cross(hidden_states)
            hidden_states = residual + hidden_states

        # 2. Spatio-Temporal Block
        if self.has_spatio_temporal:
            # Temporal
            residual = hidden_states
            hidden_states = self.temporal_attn(
                hidden_states, 
                T=T, L=L, 
                rotary_pos_emb=temporal_rope
            )
            hidden_states = residual + hidden_states
            
            # Spatial
            residual = hidden_states
            hidden_states = self.spatial_attn(
                hidden_states, 
                T=T, L=L, 
                rotary_pos_emb=spatial_rope
            )
            hidden_states = residual + hidden_states
            
            # Feed Forward
            residual = hidden_states
            hidden_states = self.ff_spatio_temporal(hidden_states)
            hidden_states = residual + hidden_states

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
    Predicts per-frame scale factors from patch-level vision features with optional text conditioning.
    Returns: (actions in [0,1], scale_factors in [min_scale,max_scale], log_prob).
    """
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        dropout=0.,
        ff_mult=4,
        cross_attn_depth=1,
        # max_frames: int = 2048,
        # patches_per_frame: int = 16384,
        # max_text_len: int = 32768,
        min_scale: float = 0.25,
        max_scale: float = 3.0,
        spatial_merge_size: int = 2,
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        # self.max_frames = max_frames
        # self.patches_per_frame = patches_per_frame
        
        # --- Embeddings ---
        # self.temporal_pos_emb = nn.Parameter(torch.randn(1, max_frames, dim))
        # self.spatial_pos_emb = nn.Parameter(torch.randn(1, patches_per_frame, dim))
        # self.text_pos_emb = nn.Embedding(max_text_len, dim)
        
        self.rope_gen = RotaryEmbedding(dim=max(32, dim_head // 2))
        self.temporal_rope_gen = RotaryEmbedding(dim=max(32, dim_head))

        # --- Layers ---
        self.config = PredictorConfig(
            hidden_size=dim,
            dim_head=dim_head,
            num_heads=heads,
            dropout=dropout,
            ff_mult=ff_mult,
            cross_attn_depth=cross_attn_depth,
            depth=depth # depth for spatio-temporal layers
        )
        
        total_layers = cross_attn_depth + depth
        self.layers = nn.ModuleList([
            PredictorDecoderLayer(self.config, layer_idx) 
            for layer_idx in range(total_layers)
        ])
            
        # self.regression_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, 2)
        # )
        # self.regression_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, dim // 2),  
        #     nn.GELU(),                 
        #     nn.Linear(dim // 2, 2)     
        # )
        self.regression_head = ProjectorBlock(input_dim=dim, output_dim=2, hidden_dim=dim // 2)

        # Enable optional debug checks to catch NaN/Inf during development
        self.debug_checks = True
        

    def post_init(self):
        last_layer = self.regression_head.get_last_layer()
        nn.init.orthogonal_(last_layer.weight, gain=0.01)
        nn.init.zeros_(last_layer.bias)

    def get_action_and_log_prob(self, alpha, beta, action_0_1 = None):
        """
        Samples an action from the Beta distribution and computes the log_prob.
        No correction is needed as the action is naturally in [0, 1].
        """
        alpha = alpha.float().clamp(min=1.01, max=50.0)
        beta = beta.float().clamp(min=1.01, max=50.0)
        dist = Beta(alpha, beta)

        if action_0_1 is None:
            action_0_1 = dist.sample()
        else:
            action_0_1 = action_0_1.float().detach()

        epsilon = 1e-6
        action_0_1 = action_0_1.clamp(min=epsilon, max=1.0 - epsilon)

        log_prob = dist.log_prob(action_0_1).float()
        # log_prob = log_prob.sum(dim=-1) # -> (B,)
        
        scale_factors = self.min_scale + action_0_1 * (self.max_scale - self.min_scale)
        return action_0_1, scale_factors, log_prob

    def get_action_and_log_prob00(self, alpha, beta, action_0_1 = None):
        # Force float32 for distribution stability
        alpha = alpha.float().clamp(min=1.01, max=50.0)
        beta = beta.float().clamp(min=1.01, max=50.0)
        # self._debug_nan()
        # print("alpha, beta", alpha, beta)
        
        
        # Clamp to prevent explosion
        alpha = torch.clamp(alpha, min=1e-6, max=100.0) 
        beta = torch.clamp(beta, min=1e-6, max=100.0)

        dist = Beta(alpha, beta)
        
        if action_0_1 is None:
            action_0_1 = dist.sample()
        else:
            action_0_1 = action_0_1.float().detach()

        epsilon = 1e-6
        action_0_1 = action_0_1.clamp(min=epsilon, max=1.0 - epsilon)
        
        log_prob = dist.log_prob(action_0_1)
        # log_prob = log_prob.sum(dim=-1)
        
        # Cast back if necessary for downstream mixed precision
        # action_0_1 = action_0_1.to(torch.bfloat16) 
        
        scale_factors = self.min_scale + action_0_1 * (self.max_scale - self.min_scale)
        # print("action_0_1, scale_factors", action_0_1, scale_factors)
        return action_0_1, scale_factors, log_prob

    def get_deterministic_action(self, dist: Beta):
        """
        Returns the mean of the Beta distribution as the deterministic action.
        """
        # The mean of a Beta(alpha, beta) distribution is alpha / (alpha + beta)
        action_0_1 = dist.mean
        
        # Transform to the desired scale range
        scale_factors = self.min_scale + action_0_1 * (self.max_scale - self.min_scale)
        return scale_factors

    def _fuse_one_sample(self, video_patch_features, grid_thw, text_features, video_mask, text_mask):
        """
        Processes a SINGLE sample (video + text).
        This is the core logic that will be looped over the batch.
        """
        # video_patch_features: (T*L, D)
        # text_features: (N, D)
        TL, D = video_patch_features.shape
        # N = text_features.shape[0]
        
        T = grid_thw[0].item()
        L = grid_thw[1].item() * grid_thw[2].item() // self.spatial_merge_size**2
        assert TL == T * L

        # vid_feats_reshaped = rearrange(video_patch_features, '(t l) d -> t l d', t=T, l=L)
        # temporal_pos = self.temporal_pos_emb[:, :T, :].squeeze(0).unsqueeze(1) 
        # spatial_pos = self.spatial_pos_emb[:, :L, :].squeeze(0).unsqueeze(0)   
        # vid_feats_with_pos = vid_feats_reshaped + temporal_pos + spatial_pos
        # video_with_pos = rearrange(vid_feats_with_pos, 't l d -> (t l) d')
        
        # text_feats_with_pos = text_features + self.text_pos_emb(torch.arange(N, device=text_features.device))

        video_with_pos = video_patch_features
        text_feats_with_pos = text_features

        # Unsqueeze to create batch of 1
        hidden_states = video_with_pos.unsqueeze(0) # (1, T*L, D)

        if text_features is not None:
            # text_features: (N, D) -> (1, N, D)
            encoder_hidden_states = text_features.unsqueeze(0) 
            encoder_attention_mask = text_mask.unsqueeze(0) if exists(text_mask) else None
        else:
            encoder_hidden_states = None
            encoder_attention_mask = None

        # Precompute RoPE
        temporal_rope = self.temporal_rope_gen(T)
        spatial_rope = self.rope_gen(L)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, 
                grid_thw=grid_thw,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                temporal_rope=temporal_rope,
                spatial_rope=spatial_rope
            )

        # Reshape to (1, T, L, D)
        hidden_states_reshaped = rearrange(hidden_states, 'b (t l) d -> b t l d', t=T, l=L)
        if video_mask is not None:
            # video_mask: (T*L) or (T, L), boolean
            vm = video_mask.view(T, L) if video_mask.ndim == 1 else video_mask
            vm = vm.bool()
            vm_exp = vm.unsqueeze(0).unsqueeze(-1)          # (1, T, L, 1)
            masked = hidden_states_reshaped.masked_fill(~vm_exp, 0.0)
            counts = vm.sum(dim=1).clamp_min(1).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
            frame_features = masked.sum(dim=2) / counts     # (1, T, D)
        else:
            frame_features = hidden_states_reshaped.mean(dim=2)  # (1, T, D)

        return frame_features.squeeze(0)  # (T, D)

    def forward(
        self,
        video_features_list,  # List[(T_i * L_i, D)]
        video_grid_thw,
        text_features,        # (B, N, D)
        video_mask=None,      # (B, T*L) - Optional for patches
        text_mask=None,       # (B, N)
        actions=None,
        images_per_sample=None
    ):
        B = len(video_features_list)

        if images_per_sample is None:
            if text_features is not None:
                 B_text = text_features.shape[0]
                 if B_text == B:
                     images_per_sample = [1] * B
                 else:
                     images_per_sample = [1 * B]
            else:
                 images_per_sample = [1 * B]

        if text_features is not None:
            device = text_features.device
            repeats = torch.tensor(images_per_sample, device=device)
            text_features = text_features.repeat_interleave(repeats, dim=0)
            
            if exists(text_mask):
                text_mask = text_mask.repeat_interleave(repeats, dim=0)

        # Build fused per-frame features for each sample
        fused_features_list = []
        for i in range(B):
            video_feats_i = video_features_list[i]
            video_mask_i = video_mask[i] if exists(video_mask) else None

            if text_features is not None:
                text_feats_i = text_features[i]
                text_mask_i = text_mask[i] if exists(text_mask) else None
            else:
                text_feats_i = None
                text_mask_i = None

            fused_feats_i = self._fuse_one_sample(
                video_patch_features=video_feats_i,
                text_features=text_feats_i,
                video_mask=video_mask_i,
                text_mask=text_mask_i,
                grid_thw=video_grid_thw[i]
            )  # (T_i, D)
            fused_features_list.append(fused_feats_i)

        # Pad to (B, T_max, D) and build frame mask for valid frames
        padded_fused_features = nn.utils.rnn.pad_sequence(
            fused_features_list,
            batch_first=True,
            padding_value=0.0
        )  # (B, T_max, D)
        frame_lengths = torch.tensor(
            [t.shape[0] for t in fused_features_list],
            device=padded_fused_features.device
        )  # (B,)
        T_max = padded_fused_features.shape[1]
        frame_mask = torch.arange(T_max, device=padded_fused_features.device)[None, :] < frame_lengths[:, None]  # (B, T_max)

        # Regression head → distribution params
        distribution_params = self.regression_head(padded_fused_features)
        # Clamp extreme values to improve stability (optional)
        distribution_params = distribution_params.clamp(min=-20.0, max=20.0)

        alpha = F.softplus(distribution_params[..., 0]) + 1.0 + 1e-6
        beta  = F.softplus(distribution_params[..., 1]) + 1.0 + 1e-6

        # Sample actions and compute log_probs
        actions_out, scale_factors, log_prob = self.get_action_and_log_prob(alpha, beta, actions)
        # breakpoint()

        # Mask out padded frames: default scale 1.0, zero log-prob and neutral action
        if actions_out is not None:
            actions_out = actions_out.masked_fill(~frame_mask, 0.0)
        scale_factors = scale_factors.masked_fill(~frame_mask, 1.0)
        log_prob = log_prob.masked_fill(~frame_mask, 0.0)

        if self.debug_checks:
            try:
                assert torch.isfinite(distribution_params).all()
                assert torch.isfinite(alpha).all() and torch.isfinite(beta).all()
                assert torch.isfinite(log_prob).all()
                assert actions_out is None or (actions_out.min() >= 0 and actions_out.max() <= 1 and torch.isfinite(actions_out).all())
            except:
                breakpoint()

        return actions_out, scale_factors, log_prob

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

        text_hidden_size = vision_config.out_hidden_size
        vision_hidden_size = vision_config.hidden_size * vision_config.spatial_merge_size ** 2
        num_heads = vision_config.num_heads
        spatial_merge_size = vision_config.spatial_merge_size
        self.config = vision_config

        self.scorer = FrameWiseScalePredictor(
            dim=vision_hidden_size,
            depth=vision_config.self_depth,
            dim_head=vision_hidden_size // num_heads,
            heads=num_heads,
            cross_attn_depth=vision_config.cross_depth,
            spatial_merge_size=spatial_merge_size,
            min_scale=vision_config.min_scale,
            max_scale=vision_config.max_scale,
        )

        self.mlp = ProjectorBlock(
            input_dim=text_hidden_size,
            output_dim=vision_hidden_size,
            hidden_dim=vision_hidden_size,
        )

    def forward(
        self,
        video_features,   # List[(T_i * L_i, D)]
        video_grid_thw,   # (B, 3)
        text_features=None,  # (B, N, D)
        video_mask=None,     # (B, T)
        text_mask=None,      # (B, N)
        actions=None,
        images_per_sample=None
    ):
        """
        Run text projector (if provided) then delegate to `FrameWiseScalePredictor`.
        Returns (actions, scales, log_probs) from the scorer.
        """
        if text_features is not None:
            projected_text_features = self.mlp(text_features)
        else:
            projected_text_features = None

        return self.scorer(
            video_features_list=video_features,
            video_grid_thw=video_grid_thw,
            text_features=projected_text_features,
            video_mask=video_mask,
            text_mask=text_mask,
            actions=actions,
            images_per_sample=images_per_sample,
        )


        # out = torch.randn(x.shape[0], x.shape[1], self.dim, device=x.device, dtype=x.dtype)
        # out[total_mask] = self.mlp(x[total_mask])  # (batch * seq_len, vision_hidden_size)
        # assert not torch.isnan(out).any()
        # out = out.reshape(out.shape[0], -1, self.vision_hidden_size)
        # out = self.perceiver(out, prefix_mask=prefix_mask.repeat_interleave(4, dim=1))
        # assert not torch.isnan(out).any()
        # return out

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
    
    video_features = torch.randn(B, T, L, D).to(device)
    text_features = torch.randn(B, N, D).to(device)

    video_mask = torch.ones(B, T, L, dtype=torch.bool).to(device)
    text_mask = torch.ones(B, N, dtype=torch.bool).to(device)

    scale_factors, log_prob = model(
        video_patch_features=video_features,
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



# class PatchWiseTemporalAttention(nn.Module):
#     """
#     Performs temporal attention on tokens at the same spatial position across different frames.
#     This precisely mimics the logic of the provided `forward` snippet.
#     """
#     temporal_dim_scale = 0.25 # Kept from original for parameter consistency

#     def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.):
#         super().__init__()
#         self.num_heads = max(1, round(heads * self.temporal_dim_scale))
#         self.dim_head = dim_head
#         self.scale = dim_head ** -0.5
#         inner_dim = self.num_heads * self.dim_head
        
#         self.norm = nn.LayerNorm(dim)
#         self.dropout = nn.Dropout(dropout)
        
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
#         self.to_out = nn.Linear(inner_dim, dim, bias=False)

#     def forward(self, x, T: int, L: int, rotary_pos_emb=None):
#         """
#         Args:
#             x (torch.Tensor): Input tensor of shape (B, T*L, D).
#             T (int): Number of frames (time steps).
#             L (int): Number of patches per frame.
#             rotary_pos_emb (torch.Tensor, optional): RoPE for the temporal dimension (length T).
#         """
#         B, _, D = x.shape
#         # device = x.device
        
#         x = self.norm(x)
        
#         # Project to Q, K, V
#         qkv = self.to_qkv(x) # (B, T*L, 3 * inner_dim)

#         # Reshape to isolate all dimensions
#         # (B, T*L, 3*H*D_h) -> (B, T, L, 3, H, D_h)
#         qkv = rearrange(qkv, 'b (t l) (three h d) -> b t l three h d', t=T, l=L, three=3, h=self.num_heads)

#         # Permute to group tokens from the same spatial position across time.
#         # This makes 'L' (spatial position) act as a new batch dimension.
#         # (B, T, L, 3, H, D_h) -> (L, 3, B, T, H, D_h)
#         qkv = rearrange(qkv, 'b t l three h d -> l three b t h d')
        
#         # Unbind along the 'three' dimension (dim=1) to get Q, K, V
#         q, k, v = qkv.unbind(1) # -> 3 tensors of shape (L, B, T, H, D_h)
        
#         # Combine L and B to run attention in a single, large batch pass
#         # -> (L*B, H, T, D_h)
#         q = rearrange(q, 'l b t h d -> (l b) h t d')
#         k = rearrange(k, 'l b t h d -> (l b) h t d')
#         v = rearrange(v, 'l b t h d -> (l b) h t d')

#         # Apply rotary positional embeddings on the temporal dimension (T)
#         if exists(rotary_pos_emb):
#             # rotary_pos_emb has shape (T, D_rot)
#             # apply_rotary_pos_emb expects (B, H, T, D_h) and (T, D_rot)
#             q = apply_rotary_pos_emb(q, rotary_pos_emb)
#             k = apply_rotary_pos_emb(k, rotary_pos_emb)
            
#         # Compute attention scores
#         sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
#         attn = nn.functional.softmax(sim, dim=-1, dtype=torch.float32).to(q.dtype)
#         #attn = sim.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
#         attn = self.dropout(attn)
        
#         # Get output
#         out = einsum('b h i j, b h j d -> b h i d', attn, v) # (L*B, H, T, D_h)
        
#         # Rearrange back to the original input format
#         # (L*B, H, T, D_h) -> (L, B, T, H*D_h)
#         out = rearrange(out, '(l b) h t d -> l b t (h d)', l=L, b=B)
        
#         # (L, B, T, inner_dim) -> (B, T, L, inner_dim)
#         out = rearrange(out, 'l b t d -> b t l d')
        
#         # (B, T, L, inner_dim) -> (B, T*L, inner_dim)
#         out = rearrange(out, 'b t l d -> b (t l) d')

#         # print('v.dtype', v.dtype)
#         # print('attn.dtype', attn.dtype)
#         # print('x.dtype', x.dtype)
#         # print('q.dtype', q.dtype)
#         # print('out.dtype', out.dtype)
#         # print('to_out.dtype', self.to_out.weight.dtype)
        
#         return self.to_out(out)


# class CrossAttention(nn.Module):
#     def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.):
#         super().__init__()
#         self.scale = dim_head ** -0.5
#         self.heads = heads
#         inner_dim = heads * dim_head
#         self.norm = nn.LayerNorm(dim)
#         self.context_norm = nn.LayerNorm(dim)
#         self.dropout = nn.Dropout(dropout)
#         self.to_q = nn.Linear(dim, inner_dim, bias=False)
#         self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
#         self.to_out = nn.Linear(inner_dim, dim, bias=False)

#     def forward(self, x, context, rotary_pos_emb=None, context_mask=None):
#         x = self.norm(x)
#         context = self.context_norm(context)
#         q = self.to_q(x)
#         k, v = self.to_kv(context).chunk(2, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
#         q = q * self.scale
#         # if exists(rotary_pos_emb):
#         #     # q = apply_rotary_pos_emb(q, rotary_pos_emb[-q.shape[-2]:, :])
#         #     # k = apply_rotary_pos_emb(k, rotary_pos_emb[-k.shape[-2]:, :])
#         #     q = apply_rotary_pos_emb(rotary_pos_emb, q)
#         #     k = apply_rotary_pos_emb(rotary_pos_emb, k)
#         sim = einsum('b h i d, b h j d -> b h i j', q, k)
#         if exists(context_mask):
#             mask = rearrange(context_mask, 'b j -> b 1 1 j')
#             sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
#         # attn = sim.softmax(dim=-1)
#         attn = nn.functional.softmax(sim, dim=-1, dtype=torch.float32).to(q.dtype)
#         attn = self.dropout(attn)
#         out = einsum('b h i j, b h j d -> b h i d', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)


# class SpatialAttention(nn.Module):
    # """
    # Performs spatial self-attention within each frame.
    # Input is (B, T*L, D), reshaped to (B*T, L, D) for attention.
    # """
    # def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.):
    #     super().__init__()
    #     self.heads = heads
    #     self.dim_head = dim_head
    #     self.scale = dim_head ** -0.5
    #     inner_dim = heads * dim_head
        
    #     self.norm = nn.LayerNorm(dim)
    #     self.dropout = nn.Dropout(dropout)
        
    #     self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
    #     self.to_out = nn.Linear(inner_dim, dim, bias=False)

    # def forward(self, x, T: int, L: int, rotary_pos_emb=None):
    #     """
    #     Args:
    #         x (torch.Tensor): Input tensor of shape (B, T*L, D).
    #         T (int): Number of frames.
    #         L (int): Number of patches per frame.
    #         rotary_pos_emb (torch.Tensor, optional): RoPE for the spatial dimension (length L).
    #     """
    #     B, _, D = x.shape
    #     x = self.norm(x)
        
    #     # Reshape for spatial attention: treat each frame as a batch item
    #     x = rearrange(x, 'b (t l) d -> (b t) l d', t=T, l=L)
        
    #     q, k, v = self.to_qkv(x).chunk(3, dim=-1)
    #     q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

    #     # Apply standard rotary embeddings for spatial positions
    #     if exists(rotary_pos_emb):
    #         q = apply_rotary_pos_emb(q, rotary_pos_emb)
    #         k = apply_rotary_pos_emb(k, rotary_pos_emb)
        
    #     sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
    #     # attn = sim.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
    #     attn = nn.functional.softmax(sim, dim=-1, dtype=torch.float32).to(q.dtype)
    #     attn = self.dropout(attn)
        
    #     out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
    #     out = rearrange(out, 'b h n d -> b n (h d)')
        
    #     out = self.to_out(out)
        
    #     # Reshape back to original format
    #     return rearrange(out, '(b t) l d -> b (t l) d', b=B, t=T)
