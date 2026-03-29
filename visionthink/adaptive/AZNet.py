import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from torch.distributions import Beta, Normal, Categorical
from transformers.modeling_utils import ModuleUtilsMixin

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
    Unified Predictor supporting both Discrete (Categorical) and Continuous (Beta) actions.
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
        max_frames: int = 4,
        use_discrete_action: bool = False,
        scale_bins: list = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5], # Discrete Only
        min_scale: float = 0.25, # Continuous Only
        max_scale: float = 3.0,  # Continuous Only
        spatial_merge_size: int = 2,
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.max_frames = max_frames

        self.use_discrete_action = use_discrete_action
        if self.use_discrete_action:
            # Discrete: Register bins
            if scale_bins is None:
                import numpy as np
                step = min_scale
                scale_bins = np.arange(min_scale, max_scale + step, step).tolist()
            print("scale_bins", scale_bins)

            self.register_buffer('scale_bins', torch.tensor(scale_bins))
            self.num_bins = len(scale_bins)
            output_dim = self.num_bins
        else:
            # Continuous: Set min/max
            self.min_scale = min_scale
            self.max_scale = max_scale
            output_dim = 2 # Alpha, Beta

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
        self.regression_head = ProjectorBlock(input_dim=dim, output_dim=output_dim, hidden_dim=dim // 2)

        # Enable optional debug checks to catch NaN/Inf during development
        self.debug_checks = False
        

    def post_init(self):
        last_layer = self.regression_head.get_last_layer()
        nn.init.orthogonal_(last_layer.weight, gain=0.01)
        # nn.init.zeros_(last_layer.weight)
        with torch.no_grad():
            if self.use_discrete_action:
                target_val = 1.0
                diff = torch.abs(self.scale_bins - target_val)
                target_idx = torch.argmin(diff).item()
                
                nn.init.zeros_(last_layer.bias)
                # nn.init.constant_(last_layer.bias, -8.0)
                last_layer.bias[target_idx] = 5.0  
                
            else:
                target_ratio = (1.0 - self.min_scale) / (self.max_scale - self.min_scale)
                target_ratio = max(0.01, min(0.99, target_ratio))
                target_sum = 4.0 
                
                target_alpha = target_sum * target_ratio
                target_beta  = target_sum * (1.0 - target_ratio)
                
                def inverse_softplus_plus1(val):
                    # val = softplus(x) + 1 => softplus(x) = val - 1
                    y = val - 1.0
                    if y < 1e-4: y = 1e-4 
                    return torch.log(torch.exp(torch.tensor(y)) - 1.0)

                bias_alpha = inverse_softplus_plus1(target_alpha)
                bias_beta  = inverse_softplus_plus1(target_beta)
                
                last_layer.bias[0] = bias_alpha
                last_layer.bias[1] = bias_beta

    def _get_discrete_policy(self, logits, actions=None):
        """
        Input: logits (B, T, num_bins)
        Output: indices (B, T), scales (B, T), log_probs (B, T)
        """
        dist = Categorical(logits=logits)
        
        if actions is None:
            # Sample Index
            action_indices = dist.sample()
        else:
            # Use provided actions (must be LongTensor indices)
            action_indices = actions.long().detach()
            
        log_prob = dist.log_prob(action_indices)
        
        # Map Index -> Float Scale
        scale_factors = self.scale_bins[action_indices]
        
        return action_indices, scale_factors, log_prob

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
        alpha = F.softplus(params[..., 0]) + 1.0 + 1e-6
        beta  = F.softplus(params[..., 1]) + 1.0 + 1e-6
        
        # Clamp for numerical stability
        alpha = alpha.clamp(max=50.0)
        beta = beta.clamp(max=50.0)
        
        dist = Beta(alpha, beta)
        
        if actions is None:
            action_0_1 = dist.sample()
        else:
            # Assume actions provided are already normalized to 0-1
            action_0_1 = actions.float().detach()

        epsilon = 1e-6
        action_0_1 = action_0_1.clamp(min=epsilon, max=1.0 - epsilon)
        
        log_prob = dist.log_prob(action_0_1)
        
        # Map 0-1 -> [min, max]
        scale_factors = self.min_scale + action_0_1 * (self.max_scale - self.min_scale)
        
        return action_0_1, scale_factors, log_prob

    def _get_continuous_deterministic(self, params):
        alpha = F.softplus(params[..., 0]) + 1.0 + 1e-6
        beta  = F.softplus(params[..., 1]) + 1.0 + 1e-6
        
        dist = Beta(alpha, beta)
        action_0_1 = dist.mean
        scale_factors = self.min_scale + action_0_1 * (self.max_scale - self.min_scale)
        return action_0_1, scale_factors, None

    def _fuse_one_sample(self, visual_patch_features, grid_thw, text_features, visual_mask, text_mask):
        """
        Processes a SINGLE sample (visual + text).
        This is the core logic that will be looped over the batch.
        """
        # visual_patch_features: (T*L, D)
        # text_features: (N, D)
        TL, D = visual_patch_features.shape
        # N = text_features.shape[0]
        
        T = grid_thw[0].item()
        L = grid_thw[1].item() * grid_thw[2].item() // self.spatial_merge_size**2
        assert TL == T * L

        # vid_feats_reshaped = rearrange(visual_patch_features, '(t l) d -> t l d', t=T, l=L)
        # temporal_pos = self.temporal_pos_emb[:, :T, :].squeeze(0).unsqueeze(1) 
        # spatial_pos = self.spatial_pos_emb[:, :L, :].squeeze(0).unsqueeze(0)   
        # vid_feats_with_pos = vid_feats_reshaped + temporal_pos + spatial_pos
        # visual_with_pos = rearrange(vid_feats_with_pos, 't l d -> (t l) d')
        
        # text_feats_with_pos = text_features + self.text_pos_emb(torch.arange(N, device=text_features.device))

        visual_with_pos = visual_patch_features
        text_feats_with_pos = text_features

        # Unsqueeze to create batch of 1
        hidden_states = visual_with_pos.unsqueeze(0) # (1, T*L, D)

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
        if visual_mask is not None:
            # visual_mask: (T*L) or (T, L), boolean
            vm = visual_mask.view(T, L) if visual_mask.ndim == 1 else visual_mask
            vm = vm.bool()
            vm_exp = vm.unsqueeze(0).unsqueeze(-1)          # (1, T, L, 1)
            masked = hidden_states_reshaped.masked_fill(~vm_exp, 0.0)
            counts = vm.sum(dim=1).clamp_min(1).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
            frame_features = masked.sum(dim=2) / counts     # (1, T, D)
        else:
            frame_features = hidden_states_reshaped.mean(dim=2)  # (1, T, D)

        return frame_features.squeeze(0)  # (T, D)

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

    def forward(
        self,
        visual_features_list,  # List[(T_i * L_i, D)]
        visual_grid_thw,
        text_features,        # (B, N, D)
        visual_mask=None,      # (B, T*L) - Optional for patches
        text_mask=None,       # (B, N)
        actions=None,
        visual_per_sample=None,
        eval_mode=False,
        scale_mask=None,
    ):
        B = len(visual_features_list)
        device = visual_grid_thw.device

        if visual_per_sample is None:
            if text_features is not None:
                 B_text = text_features.shape[0]
                 if B_text == B:
                     visual_per_sample = [1] * B
                 else:
                     visual_per_sample = [1 * B]
            else:
                 visual_per_sample = [1 * B]

        if text_features is not None:
            # device = text_features.device
            repeats = torch.tensor(visual_per_sample, device=device)
            text_features = text_features.repeat_interleave(repeats, dim=0)
            
            if exists(text_mask):
                text_mask = text_mask.repeat_interleave(repeats, dim=0)

        # Build fused per-frame features for each sample
        fused_features_list = []
        for i in range(B):
            visual_feats_i = visual_features_list[i]
            visual_mask_i = visual_mask[i] if exists(visual_mask) else None

            if text_features is not None:
                text_feats_i = text_features[i]
                text_mask_i = text_mask[i] if exists(text_mask) else None
            else:
                text_feats_i = None
                text_mask_i = None

            fused_feats_i = self._fuse_one_sample(
                visual_patch_features=visual_feats_i,
                text_features=text_feats_i,
                visual_mask=visual_mask_i,
                text_mask=text_mask_i,
                grid_thw=visual_grid_thw[i]
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
            device=device
        )  # (B,)
        # frame_mask = torch.arange(self.max_frames, device=padded_fused_features.device)[None, :] < frame_lengths[:, None] # (B, T_max)
        T_max = padded_fused_features.shape[1]
        frame_mask = torch.arange(T_max, device=device)[None, :] < frame_lengths[:, None]  # (B, T_max)

        # Regression head → distribution params
        # Output shape: (B, T, Num_Bins) OR (B, T, 2)
        head_outputs = self.regression_head(padded_fused_features)

        padded_actions_in = None
        if actions is not None:
            object_t_dims = visual_grid_thw[:, 0] # e.g. [1, 1, 1, 7]
            valid_actions_flat = actions[scale_mask]

            if valid_actions_flat.numel() != object_t_dims.sum():
                raise ValueError(f"Number of actions ({valid_actions_flat.numel()}) does not match the number of valid frames ({object_t_dims.sum()})")
            
            splits = torch.split(valid_actions_flat, object_t_dims.cpu().tolist())
            pad_val = -1 if self.use_discrete_action else 0.0
            
            # Result Shape: (Total_Objects, T_object_max) -> (4, 7)
            padded_actions_in = nn.utils.rnn.pad_sequence(
                list(splits), 
                batch_first=True, 
                padding_value=pad_val
            )
        else:
            padded_actions_in = None

        # Policy Branching
        if self.use_discrete_action:
            if eval_mode:
                raw_actions, scales, log_probs = self._get_discrete_deterministic(head_outputs)
            else:
                raw_actions, scales, log_probs = self._get_discrete_policy(head_outputs, padded_actions_in[:, :T_max] if padded_actions_in is not None else None)
        else:
            if eval_mode:
                raw_actions, scales, log_probs = self._get_continuous_deterministic(head_outputs)
            else:
                raw_actions, scales, log_probs = self._get_continuous_policy(head_outputs, padded_actions_in[:, :T_max] if padded_actions_in is not None else None)

        pad_max = max(max(self.max_frames, T_max), max(visual_per_sample))
        scale_mask_out, _ = self.get_scale_mask(visual_grid_thw, visual_per_sample, pad_max, device)

        if log_probs is not None:
            log_probs = log_probs.masked_fill(~frame_mask, 0.0)
            log_probs = self.restructure_sequence(log_probs, visual_grid_thw, scale_mask_out, 0.0, device)

        if actions is not None:
            return None, None, log_probs, None

        # Restructure Actions & Scales
        # Raw Actions: 
        #   Discrete -> Indices (0, 1, 2...) -> fill with 0
        #   Continuous -> Floats (0.0 - 1.0) -> fill with 0.0
        pad_val_action = -1 if self.use_discrete_action else 0.0
        raw_actions = raw_actions.masked_fill(~frame_mask, pad_val_action)
        raw_actions = self.restructure_sequence(raw_actions, visual_grid_thw, scale_mask_out, pad_val_action, device)

        # Scales: Real world scale factors -> fill with 1.0 (neutral)
        scales = scales.masked_fill(~frame_mask, 1.0)
        scales = self.restructure_sequence(scales, visual_grid_thw, scale_mask_out, 1.0, device)

        return raw_actions, scales, log_probs, scale_mask_out

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
    def __init__(self, config):
        super().__init__()
        vision_config = config.vision_config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        text_hidden_size = vision_config.out_hidden_size
        vision_hidden_size = vision_config.hidden_size * vision_config.spatial_merge_size ** 2
        spatial_merge_size = vision_config.spatial_merge_size

        num_heads = vision_config.num_heads
        output_dim = vision_config.output_dim
        dim_head = vision_config.dim_head
        self.config = vision_config

        use_discrete_action = vision_config.use_discrete_action if hasattr(vision_config, "use_discrete_action") else False
        scale_bins = vision_config.scale_bins if hasattr(vision_config, "scale_bins") else None

        self.scorer = FrameWiseScalePredictor(
            dim=output_dim,
            depth=vision_config.self_depth,
            dim_head=dim_head,
            heads=num_heads,
            cross_attn_depth=vision_config.cross_depth,
            max_frames=vision_config.max_frames if hasattr(vision_config, "max_frames") else 4,
            spatial_merge_size=spatial_merge_size,
            use_discrete_action=use_discrete_action,
            scale_bins=scale_bins,
            min_scale=vision_config.min_scale,
            max_scale=vision_config.max_scale,
        )

        self.mlp = ProjectorBlock(
            input_dim=text_hidden_size,
            output_dim=output_dim,
            hidden_dim=output_dim,
        )

        self.vlp = ProjectorBlock(
            input_dim=vision_hidden_size,
            output_dim=output_dim,
            hidden_dim=output_dim,
        )

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
            projected_text_features = self.mlp(text_features)
        else:
            projected_text_features = None

        visual_features = [self.vlp(visual_feature) for visual_feature in visual_features]

        return self.scorer(
            visual_features_list=visual_features,
            visual_grid_thw=visual_grid_thw,
            text_features=projected_text_features,
            visual_mask=visual_mask,
            text_mask=text_mask,
            actions=actions,
            visual_per_sample=visual_per_sample,
            eval_mode=eval_mode,
            scale_mask=scale_mask
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




# class PatchWiseTemporalAttention(nn.Module):
#     """
#     Performs temporal attention on tokens at the same spatial position across different frames.
#     Uses Flash Attention.
#     """
#     temporal_dim_scale = 0.25

#     def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.):
#         super().__init__()
#         self.num_heads = max(1, round(heads * self.temporal_dim_scale))
#         self.dim_head = dim_head
#         self.scale = dim_head ** -0.5
#         inner_dim = self.num_heads * self.dim_head
        
#         self.norm = nn.LayerNorm(dim)
#         self.dropout_p = dropout # Store dropout probability
        
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
#         self.to_out = nn.Linear(inner_dim, dim, bias=False)

#     def forward(self, x, T: int, L: int, rotary_pos_emb=None):
#         """
#         Input: (B, T*L, D)
#         """
#         B, _, _ = x.shape
#         x = self.norm(x)
        
#         # 1. Project to QKV
#         qkv = self.to_qkv(x) # (B, T*L, 3 * inner_dim)
        
#         # (B, T*L, 3*H*D) -> (B, T, L, 3, H, D)
#         qkv = rearrange(qkv, 'b (t l) (three h d) -> b t l three h d', t=T, l=L, three=3, h=self.num_heads)
        
#         # Permute to (B, L, T, 3, H, D) and then collapse B*L -> (B*L, T, 3, H, D)
#         qkv = rearrange(qkv, 'b t l three h d -> (b l) t three h d')
        
#         # Unbind Q, K, V
#         q, k, v = qkv.unbind(2) # Each is (B*L, T, H, D)

#         if exists(rotary_pos_emb):
#             # Transpose to (Batch, Heads, SeqLen, Dim) for RoPE
#             q = rearrange(q, 'b t h d -> b h t d')
#             k = rearrange(k, 'b t h d -> b h t d')
            
#             # apply_rotary_pos_emb expects (B, H, T, D)
#             q = apply_rotary_pos_emb(q, rotary_pos_emb)
#             k = apply_rotary_pos_emb(k, rotary_pos_emb)
            
#             # Transpose back to (Batch, SeqLen, Heads, Dim) for Flash Attention
#             q = rearrange(q, 'b h t d -> b t h d')
#             k = rearrange(k, 'b h t d -> b t h d')

#         # q, k, v shape: (B*L, T, H, D)
#         out = flash_attn_func(
#             q, k, v, 
#             dropout_p=self.dropout_p if self.training else 0.0,
#             softmax_scale=self.scale
#         )
        
#         # (B*L, T, H, D) -> (B, L, T, H*D)
#         out = rearrange(out, '(b l) t h d -> b l t (h d)', b=B, l=L)
        
#         # (B, L, T, InnerDim) -> (B, T, L, InnerDim) -> (B, T*L, InnerDim)
#         out = rearrange(out, 'b l t d -> b (t l) d')
        
#         return self.to_out(out)


# class SpatialAttention(nn.Module):
#     """
#     Performs spatial self-attention within each frame.
#     Uses Flash Attention.
#     """
#     def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.):
#         super().__init__()
#         self.heads = heads
#         self.dim_head = dim_head
#         self.scale = dim_head ** -0.5
#         inner_dim = heads * dim_head
        
#         self.norm = nn.LayerNorm(dim)
#         self.dropout_p = dropout
        
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
#         self.to_out = nn.Linear(inner_dim, dim, bias=False)

#     def forward(self, x, T: int, L: int, rotary_pos_emb=None):
#         """
#         Input: (B, T*L, D)
#         """
#         B, _, D = x.shape
#         x = self.norm(x)
        
#         # (B, T*L, D) -> (B*T, L, D)
#         x = rearrange(x, 'b (t l) d -> (b t) l d', t=T, l=L)
        
#         qkv = self.to_qkv(x) # (B*T, L, 3*H*D)
        
#         # Reshape to (B*T, L, 3, H, D)
#         qkv = rearrange(qkv, 'b l (three h d) -> b l three h d', three=3, h=self.heads)
        
#         q, k, v = qkv.unbind(2) # Each is (B*T, L, H, D)

#         # 3. Apply RoPE
#         if exists(rotary_pos_emb):
#             # Transpose to (B*T, H, L, D) for RoPE
#             q = rearrange(q, 'b l h d -> b h l d')
#             k = rearrange(k, 'b l h d -> b h l d')
            
#             q = apply_rotary_pos_emb(q, rotary_pos_emb)
#             k = apply_rotary_pos_emb(k, rotary_pos_emb)
            
#             # Transpose back to (B*T, L, H, D) for Flash
#             q = rearrange(q, 'b h l d -> b l h d')
#             k = rearrange(k, 'b h l d -> b l h d')
        
#         # Flash Attention
#         out = flash_attn_func(
#             q, k, v,
#             dropout_p=self.dropout_p if self.training else 0.0,
#             softmax_scale=self.scale
#         )
        
#         # (B*T, L, H, D) -> (B*T, L, InnerDim)
#         out = rearrange(out, 'b l h d -> b l (h d)')
        
#         out = self.to_out(out)
        
#         # Reshape back to (B, T*L, D)
#         return rearrange(out, '(b t) l d -> b (t l) d', b=B, t=T)


# class CrossAttention(nn.Module):
#     """
#     Cross Attention using Flash Attention.
#     Note: Standard flash_attn_func DOES NOT support arbitrary 'context_mask' (like padding mask) 
#     efficiently without using the varlen API. If context_mask is strictly required for padding, 
#     consider using torch.nn.functional.scaled_dot_product_attention or flash_attn_varlen.
    
#     This implementation assumes dense context or ignores the mask for performance.
#     """
#     def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.):
#         super().__init__()
#         self.scale = dim_head ** -0.5
#         self.heads = heads
#         inner_dim = heads * dim_head
#         self.norm = nn.LayerNorm(dim)
#         self.context_norm = nn.LayerNorm(dim)
#         self.dropout_p = dropout
        
#         self.to_q = nn.Linear(dim, inner_dim, bias=False)
#         self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
#         self.to_out = nn.Linear(inner_dim, dim, bias=False)

#     def forward(self, x, context, rotary_pos_emb=None, context_mask=None):
#         x = self.norm(x)
#         context = self.context_norm(context)
        
#         q = self.to_q(x)
#         k, v = self.to_kv(context).chunk(2, dim=-1)
        
#         q = rearrange(q, 'b n (h d) -> b n h d', h=self.heads)
#         k = rearrange(k, 'b n (h d) -> b n h d', h=self.heads)
#         v = rearrange(v, 'b n (h d) -> b n h d', h=self.heads)

#         # Flash Attention
#         if context_mask is not None:
#             pass 

#         out = flash_attn_func(
#             q, k, v,
#             dropout_p=self.dropout_p if self.training else 0.0,
#             softmax_scale=self.scale
#         )
        
#         # Output projection
#         out = rearrange(out, 'b n h d -> b n (h d)')
#         return self.to_out(out)
