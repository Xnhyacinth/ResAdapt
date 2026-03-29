import torch
import torch.nn.functional as F
from torch import nn, einsum
import math
from einops import rearrange, repeat
from torch.distributions import Beta, Normal

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

# --- Attention Modules ---

class VisionTemporalAttention(nn.Module):
    temporal_dim_scale = 0.25
    def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.):
        super().__init__()
        self.num_heads = max(1, round(heads * self.temporal_dim_scale))
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = self.num_heads * self.dim_head
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, rotary_pos_emb=None, attention_mask=None):
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        if exists(rotary_pos_emb):
            q = apply_rotary_pos_emb(q, rotary_pos_emb[-q.shape[-2]:, :])
            k = apply_rotary_pos_emb(k, rotary_pos_emb[-k.shape[-2]:, :])
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if exists(attention_mask):
            mask = rearrange(attention_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PatchWiseTemporalAttention(nn.Module):
    """
    Performs temporal attention on tokens at the same spatial position across different frames.
    This precisely mimics the logic of the provided `forward` snippet.
    """
    temporal_dim_scale = 0.25 # Kept from original for parameter consistency

    def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.):
        super().__init__()
        self.num_heads = max(1, round(heads * self.temporal_dim_scale))
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = self.num_heads * self.dim_head
        
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, T: int, L: int, rotary_pos_emb=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T*L, D).
            T (int): Number of frames (time steps).
            L (int): Number of patches per frame.
            rotary_pos_emb (torch.Tensor, optional): RoPE for the temporal dimension (length T).
        """
        B, _, D = x.shape
        device = x.device
        
        x = self.norm(x)
        
        # Project to Q, K, V
        qkv = self.to_qkv(x) # (B, T*L, 3 * inner_dim)

        # Reshape to isolate all dimensions
        # (B, T*L, 3*H*D_h) -> (B, T, L, 3, H, D_h)
        qkv = rearrange(qkv, 'b (t l) (three h d) -> b t l three h d', t=T, l=L, three=3, h=self.num_heads)

        # Permute to group tokens from the same spatial position across time.
        # This makes 'L' (spatial position) act as a new batch dimension.
        # (B, T, L, 3, H, D_h) -> (L, 3, B, T, H, D_h)
        qkv = rearrange(qkv, 'b t l three h d -> l three b t h d')
        
        # Unbind along the 'three' dimension (dim=1) to get Q, K, V
        q, k, v = qkv.unbind(1) # -> 3 tensors of shape (L, B, T, H, D_h)
        
        # Combine L and B to run attention in a single, large batch pass
        # -> (L*B, H, T, D_h)
        q = rearrange(q, 'l b t h d -> (l b) h t d')
        k = rearrange(k, 'l b t h d -> (l b) h t d')
        v = rearrange(v, 'l b t h d -> (l b) h t d')

        # Apply rotary positional embeddings on the temporal dimension (T)
        if exists(rotary_pos_emb):
            # rotary_pos_emb has shape (T, D_rot)
            # apply_rotary_pos_emb expects (B, H, T, D_h) and (T, D_rot)
            q = apply_rotary_pos_emb(q, rotary_pos_emb)
            k = apply_rotary_pos_emb(k, rotary_pos_emb)
            
        # Compute attention scores
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        attn = sim.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
        attn = self.dropout(attn)
        
        # Get output
        out = einsum('b h i j, b h j d -> b h i d', attn, v) # (L*B, H, T, D_h)
        
        # Rearrange back to the original input format
        # (L*B, H, T, D_h) -> (L, B, T, H*D_h)
        out = rearrange(out, '(l b) h t d -> l b t (h d)', l=L, b=B)
        
        # (L, B, T, inner_dim) -> (B, T, L, inner_dim)
        out = rearrange(out, 'l b t d -> b t l d')
        
        # (B, T, L, inner_dim) -> (B, T*L, inner_dim)
        out = rearrange(out, 'b t l d -> b (t l) d')
        
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = heads * dim_head
        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, context, rotary_pos_emb=None, context_mask=None):
        x = self.norm(x)
        context = self.context_norm(context)
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        q = q * self.scale
        # if exists(rotary_pos_emb):
        #     # q = apply_rotary_pos_emb(q, rotary_pos_emb[-q.shape[-2]:, :])
        #     # k = apply_rotary_pos_emb(k, rotary_pos_emb[-k.shape[-2]:, :])
        #     q = apply_rotary_pos_emb(rotary_pos_emb, q)
        #     k = apply_rotary_pos_emb(rotary_pos_emb, k)
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        if exists(context_mask):
            mask = rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class SelfAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = heads * dim_head
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, rotary_pos_emb=None, attention_mask=None):
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        q = q * self.scale
        if exists(rotary_pos_emb):
            q = apply_rotary_pos_emb(q, rotary_pos_emb[-q.shape[-2]:, :])
            k = apply_rotary_pos_emb(k, rotary_pos_emb[-k.shape[-2]:, :])
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        if exists(attention_mask):
            mask = rearrange(attention_mask, 'b j -> b 1 1 j') * rearrange(attention_mask, 'b i -> b 1 i 1')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# --- The Final MultimodalAttentiveScorer ---

class FrameWiseScalePredictor(nn.Module):
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
        temporal_depth=1,
        max_frames: int = 2048,
        patches_per_frame: int = 16384,
        max_text_len: int = 32768,
        min_scale: float = 0.5,
        max_scale: float = 2.0,
        spatial_merge_size: int = 2,
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        self.max_frames = max_frames
        self.patches_per_frame = patches_per_frame
        
        # --- Embeddings ---
        self.temporal_pos_emb = nn.Parameter(torch.randn(1, max_frames, dim))
        self.spatial_pos_emb = nn.Parameter(torch.randn(1, patches_per_frame, dim))
        self.text_pos_emb = nn.Embedding(max_text_len, dim)
        
        self.rope_gen = RotaryEmbedding(dim=max(32, dim_head // 2))
        self.temporal_rope_gen = RotaryEmbedding(dim=max(32, dim_head))

        # --- Layers ---
        self.temporal_layers = nn.ModuleList([])
        for _ in range(temporal_depth):
            self.temporal_layers.append(nn.ModuleList([
                PatchWiseTemporalAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout),
                FeedForward(dim, mult=ff_mult, dropout=dropout)
            ]))
            
        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                CrossAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout),
                FeedForward(dim, mult=ff_mult, dropout=dropout)
            ]))

        self.fusion_layers = nn.ModuleList([])
        for _ in range(depth):
            self.fusion_layers.append(nn.ModuleList([
                SelfAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout),
                FeedForward(dim, mult=ff_mult, dropout=dropout)
            ]))
            
        self.regression_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2)
        )

    def get_action_and_log_prob(self, dist: Beta, action_0_1 = None):
        """
        Samples an action from the Beta distribution and computes the log_prob.
        No correction is needed as the action is naturally in [0, 1].
        """
        # 1. Sample an action in the [0, 1] range using reparameterization
        # Clamp to avoid extreme values 0 or 1 which can cause -inf in log_prob
        if action_0_1 is None:
            action_0_1 = dist.rsample().clamp(1e-6, 1.0 - 1e-6)
        
        # 2. Compute the log probability of this action
        # This is now much simpler, no tanh correction needed.
        log_prob = dist.log_prob(action_0_1)
        
        # Sum log_probs over the action dimensions (T) to get a single value per batch sample
        log_prob = log_prob.sum(dim=-1) # -> (B,)
        
        # 3. Transform the [0, 1] action to the desired scale range
        scale_factors = self.min_scale + action_0_1 * (self.max_scale - self.min_scale)
        
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
        N = text_features.shape[0]
        
        T = grid_thw[0].item()
        L = grid_thw[1].item() * grid_thw[2].item() // self.spatial_merge_size**2
        assert TL == T * L
        
        # --- 1. Add Positional Embeddings ---
        vid_feats_reshaped = rearrange(video_patch_features, '(t l) d -> t l d', t=T, l=L)
        temporal_pos = self.temporal_pos_emb[:, :T, :].squeeze(0).unsqueeze(1) # (T, 1, D)
        spatial_pos = self.spatial_pos_emb[:, :L, :].squeeze(0).unsqueeze(0)   # (1, L, D)
        vid_feats_with_pos = vid_feats_reshaped + temporal_pos + spatial_pos
        video_with_pos = rearrange(vid_feats_with_pos, 't l d -> (t l) d')
        
        text_feats_with_pos = text_features + self.text_pos_emb(torch.arange(N, device=text_features.device))

        # --- Unsqueeze to create a batch of 1 for attention modules ---
        video_with_pos = video_with_pos.unsqueeze(0) # (1, T*L, D)
        text_feats_with_pos = text_feats_with_pos.unsqueeze(0) # (1, N, D)
        if exists(video_mask): video_mask = video_mask.unsqueeze(0)
        if exists(text_mask): text_mask = text_mask.unsqueeze(0)

        # --- 2. Patch-wise Temporal Attention ---
        temporal_rope = self.temporal_rope_gen(T)
        x_vid = video_with_pos
        for attn, ff in self.temporal_layers:
            x_vid = attn(x_vid, T=T, L=L, rotary_pos_emb=temporal_rope) + x_vid
            x_vid = ff(x_vid) + x_vid
            
        # --- 3. Pool Patches ---
        frame_features = rearrange(x_vid, 'b (t l) d -> b t l d', t=T, l=L).mean(dim=2) # (1, T, D)
        frame_mask = None
        if exists(video_mask):
            frame_mask = rearrange(video_mask, 'b (t l) -> b t l', t=T, l=L).any(dim=2)

        # --- 4. Cross & Self Attention ---
        x_frames = frame_features
        for cross_attn, ff in self.cross_attn_layers:
            x_frames = cross_attn(x_frames, context=text_feats_with_pos, context_mask=text_mask) + x_frames
            x_frames = ff(x_frames) + x_frames
        for self_attn, ff in self.fusion_layers:
            x_frames = self_attn(x_frames, rotary_pos_emb=temporal_rope, attention_mask=frame_mask) + x_frames
            x_frames = ff(x_frames) + x_frames
            
        return x_frames.squeeze(0) # Return (T, D)

    def forward(
        self,
        video_features_list, # Shape # List of (T_i * L_i, D)
        video_grid_thw,
        text_features,        # Shape (B, N, D)
        video_mask=None,      # Shape (B, T*L) - Optional for patches
        text_mask=None        # Shape (B, N)
    ):
        B = len(video_features_list)
        
        # --- Loop over each sample in the batch ---
        fused_features_list = []
        for i in range(B):
            video_feats_i = video_features_list[i]
            text_feats_i = text_features[i]
            text_mask_i = text_mask[i] if exists(text_mask) else None
            video_mask_i = video_mask[i] if exists(video_mask) else None

            # Process one sample
            fused_feats_i = self._fuse_one_sample(
                video_patch_features=video_feats_i,
                text_features=text_feats_i,
                video_mask=video_mask_i, # For simplicity, assuming no padding within a sample
                text_mask=text_mask_i,
                grid_thw=video_grid_thw[i]
            ) # -> (T_i, D)
            fused_features_list.append(fused_feats_i)

        # --- Pad the results to create a batch tensor ---
        # `pad_sequence` requires (SeqLen, Batch, D), so we permute
        padded_fused_features = nn.utils.rnn.pad_sequence(
            [t for t in fused_features_list], 
            batch_first=True, 
            padding_value=0.0
        ) # -> (B, T_max, D)

        # --- 6. Regression Head & Distribution ---
        distribution_params = self.regression_head(padded_fused_features)
        alpha = F.softplus(distribution_params[..., 0]) + 1
        beta = F.softplus(distribution_params[..., 1]) + 1
        dist = Beta(alpha, beta)

        # Sample an action (scale_factors) and get its log_prob
        actions, scale_factors, log_prob = self.get_action_and_log_prob(dist)
            
        return actions, scale_factors, log_prob

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


class MultimodalAttentiveScorer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        max_seq_len=32768,
        dim_head=64,
        heads=8,
        dropout=0.,
        ff_mult=4,
        temporal_depth=1,
        cross_attn_depth=1,
        min_scale: float = 0.5,
        max_scale: float = 2.0,
    ):
        super().__init__()
        
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.rotary_pos_emb = RotaryEmbedding(dim = max(32, dim_head // 2))
        self.temporal_pos_emb_gen = RotaryEmbedding(dim=max(32, dim_head))

        # --- Stage 1: Temporal Layers ---
        self.temporal_layers = nn.ModuleList([])
        for _ in range(temporal_depth):
            self.temporal_layers.append(nn.ModuleList([
                VisionTemporalAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout),
                FeedForward(dim, mult=ff_mult, dropout=dropout)
            ]))

        # --- Stage 2: Cross-Attention Layers ---
        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                CrossAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout),
                FeedForward(dim, mult=ff_mult, dropout=dropout)
            ]))

        # --- Stage 3: Fusion Self-Attention Layers ---
        self.self_attn_layers = nn.ModuleList([])
        for _ in range(depth):
            self.self_attn_layers.append(nn.ModuleList([
                SelfAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout),
                FeedForward(dim, mult=ff_mult, dropout=dropout)
            ]))
            
        # --- Stage 4: Regression Head ---
        self.regression_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )

    def forward(
        self,
        video_features,   # Shape (B, T, D)
        text_features,    # Shape (B, N, D)
        video_mask=None,  # Shape (B, T)
        text_mask=None    # Shape (B, N)
    ):
        device, dtype = video_features.device, video_features.dtype
        seq_len = text_features.shape[1]
        T = video_features.shape[1]

        # 1. Temporal Self-Attention
        temporal_rotary_emb = self.temporal_pos_emb_gen(T, device=device, dtype=dtype)
        x = video_features
        for attn, ff in self.temporal_layers:
            x = attn(x, rotary_pos_emb=temporal_rotary_emb, attention_mask=video_mask) + x
            x = ff(x) + x
        
        
        text_features = text_features + self.pos_emb(torch.arange(seq_len, device = device))
        rotary_pos_emb = self.rotary_pos_emb(seq_len, device = device)
        # 2. Cross-Attention (Video queries Text)
        for cross_attn, ff in self.cross_attn_layers:
            x = cross_attn(x, context=text_features, rotary_pos_emb=rotary_pos_emb, context_mask=text_mask) + x
            x = ff(x) + x

        # 3. Deep Fusion with Self-Attention
        for self_attn, ff in self.self_attn_layers:
            x = self_attn(x, rotary_pos_emb=rotary_pos_emb, attention_mask=video_mask) + x
            x = ff(x) + x
            
        # 4. Regression Head to get scores for each frame
        # x shape: (B, T, D)
        scores = self.regression_head(x).squeeze(-1) # -> (B, T)
        
        # 5. Map scores to scale factors [min_scale, max_scale]
        scale_factors = self.min_scale + torch.sigmoid(scores) * (self.max_scale - self.min_scale)
            
        return scale_factors

class VideoScalePredictor(nn.Module): # Renamed for clarity
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
        temporal_depth=1, # Depth of the new temporal attention
        max_frames: int = 1024,
        patches_per_frame: int = 4096,
        max_text_len: int = 32684,
        min_scale: float = 0.5,
        max_scale: float = 2.0,
    ):
        super().__init__()
        
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        self.max_frames = max_frames
        self.patches_per_frame = patches_per_frame
        
        # --- Learnable Tokens and Embeddings ---
        self.scale_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_pos_emb = nn.Parameter(torch.randn(1, max_frames, dim))
        self.spatial_pos_emb = nn.Parameter(torch.randn(1, patches_per_frame, dim))
        self.text_pos_emb = nn.Embedding(max_text_len, dim)
        
        # RoPE generators for video (temporal) and text
        self.temporal_rope_gen = RotaryEmbedding(dim=max(32, dim_head // 2))
        self.text_rope_gen = RotaryEmbedding(dim=max(32, dim_head // 2))

        # --- Model Layers ---
        self.temporal_layers = nn.ModuleList([])
        for _ in range(temporal_depth):
            self.temporal_layers.append(nn.ModuleList([
                # Use the new PatchWiseTemporalAttention
                PatchWiseTemporalAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout),
                FeedForward(dim, mult=ff_mult, dropout=dropout)
            ]))
            
        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                CrossAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout),
                FeedForward(dim, mult=ff_mult, dropout=dropout)
            ]))

        # The final fusion layers can be standard SelfAttention
        self.fusion_layers = nn.ModuleList([])
        for _ in range(depth):
            self.fusion_layers.append(nn.ModuleList([
                SelfAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout),
                FeedForward(dim, mult=ff_mult, dropout=dropout)
            ]))
            
        self.regression_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )

    def forward(
        self,
        video_patch_features, # Shape (B, T*L, D)
        text_features,        # Shape (B, N, D)
        video_mask=None,      # Shape (B, T*L) - Optional
        text_mask=None        # Shape (B, N) - Optional
    ):
        device, dtype = video_patch_features.device, video_patch_features.dtype
        T, L = video_patch_features.shape[-3], video_patch_features.shape[-2]
        N = text_features.shape[1]
        
        # --- 1. Add Positional Embeddings ---
        # Add 2D pos emb to video
        # vid_feats_with_pos = rearrange(video_patch_features, 'b (t l) d -> b t l d', t=T, l=L)
        # breakpoint()
        vid_feats_with_pos = video_patch_features + self.temporal_pos_emb[:, :T].unsqueeze(2) + self.spatial_pos_emb[:, :L].unsqueeze(1)
        vid_feats_with_pos = rearrange(vid_feats_with_pos, 'b t l d -> b (t l) d')

        # Add pos emb to text
        text_feats_with_pos = text_features + self.text_pos_emb(torch.arange(N, device=device))

        # --- 2. Generate Rotary Embeddings ---
        temporal_rope = self.temporal_rope_gen(T, device=device, dtype=dtype)
        text_rope = self.text_rope_gen(N, device=device, dtype=dtype)

        # --- 3. Patch-wise Temporal Attention ---
        x_vid = vid_feats_with_pos
        for attn, ff in self.temporal_layers:
            x_vid = attn(x_vid, T=T, L=L, rotary_pos_emb=temporal_rope) + x_vid
            x_vid = ff(x_vid) + x_vid
            
        # --- 4. Cross-Attention: scale_token queries everything ---
        scale_token = self.scale_token.expand(x_vid.shape[0], -1, -1)
        
        # Create full context for cross-attention
        full_context = torch.cat((x_vid, text_feats_with_pos), dim=1)
        
        if exists(video_mask) and exists(text_mask):
            full_context_mask = torch.cat((video_mask, text_mask), dim=1)
        else:
            full_context_mask = None

        fused_token = scale_token
        for cross_attn, ff in self.cross_attn_layers:
            # Cross-attention does not use RoPE by design
            fused_token = cross_attn(fused_token, context=full_context, context_mask=full_context_mask, rotary_pos_emb=text_rope) + fused_token
            fused_token = ff(fused_token) + fused_token
            
        # --- 5. Final Self-Attention Fusion (Optional) ---
        # For simplicity, we directly use the fused_token for regression
        # If deeper fusion is needed, this part can be activated
        full_sequence = torch.cat((fused_token, full_context), dim=1)
        
        if exists(full_context_mask):
            # Prepend a True for the scale_token's mask
            full_sequence_mask = F.pad(full_context_mask, (1, 0), value=True)
        else:
            full_sequence_mask = None

        # The self-attention layers process the entire sequence
        processed_sequence = full_sequence
        for self_attn, ff in self.fusion_layers:
             # RoPE is omitted here for simplicity. If needed, a RoPE for the full length
             # `1 + T*L + N` should be generated and passed.
             processed_sequence = self_attn(processed_sequence, attention_mask=full_sequence_mask) + processed_sequence
             processed_sequence = ff(processed_sequence) + processed_sequence

        # --- 6. Regression ---
        # Extract the final state of the scale_token (which is at position 0)
        final_agg_token = processed_sequence[:, 0] # Shape: (B, D)
        
        scores = self.regression_head(final_agg_token) # -> (B, 1)
        
        # --- 7. Map to scale factors ---
        scale_factors = self.min_scale + torch.sigmoid(scores) * (self.max_scale - self.min_scale)
            
        return scale_factors.squeeze(-1) # -> (B,)


# --- Main Execution ---
# if __name__ == '__main__':
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     MAX_FRAMES = 32
#     PATCHES_PER_FRAME = 256
#     MAX_TEXT_LEN = 1024
    
#     model = VideoScalePredictor(
#         dim=512,
#         depth=2,
#         dim_head=64,
#         heads=8,
#         cross_attn_depth=2,
#         temporal_depth=2,
#         max_frames=MAX_FRAMES,
#         patches_per_frame=PATCHES_PER_FRAME,
#         max_text_len=MAX_TEXT_LEN,
#     ).to(device)

#     B, T, L, N, D = 2, MAX_FRAMES, PATCHES_PER_FRAME, 128, 512
    
#     video_features = torch.randn(B, T, L, D).to(device)
#     text_features = torch.randn(B, N, D).to(device)
#     assert N <= MAX_TEXT_LEN

#     video_mask = torch.ones(B, T * L, dtype=torch.bool).to(device)
#     text_mask = torch.ones(B, N, dtype=torch.bool).to(device)

#     scale_factors = model(
#         video_patch_features=video_features,
#         text_features=text_features,
#         video_mask=video_mask,
#         text_mask=text_mask
#     )

#     print("Output scale_factors shape:", scale_factors.shape)
#     assert scale_factors.shape == (B,)
#     print("Example scale factors:", scale_factors.detach().cpu().numpy())
#     print("\nVideoScalePredictor with PatchWiseTemporalAttention ran successfully!")

# # --- Main Execution ---
# if __name__ == '__main__':
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Model Instantiation
#     model = MultimodalAttentiveScorer(
#         dim=512,
#         depth=4,              # 4 layers of final self-attention
#         dim_head=64,
#         heads=8,
#         temporal_depth=2,     # 2 layers of temporal attention
#         cross_attn_depth=2,   # 2 layers of cross-attention
#         min_scale=0.25,       # e.g., allow shrinking to 1/4 size
#         max_scale=3.0,        # e.g., allow enlarging by 300%
#     ).to(device)

#     # Dummy Data
#     B, T, N, D = 2, 64, 128, 512 # Batch, VideoFrames, TextTokens, Dim
    
#     video_features = torch.randn(B, T, D).to(device)
#     text_features = torch.randn(B, N, D).to(device)

#     # Masks (optional, but good practice)
#     video_mask = torch.ones(B, T, dtype=torch.bool).to(device)
#     video_mask[1, -10:] = False # Second video is shorter
    
#     text_mask = torch.ones(B, N, dtype=torch.bool).to(device)
#     text_mask[0, -20:] = False # First text is shorter

#     # Forward Pass
#     scale_factors = model(
#         video_features=video_features,
#         text_features=text_features,
#         video_mask=video_mask,
#         text_mask=text_mask
#     )

#     print("Output scale_factors shape:", scale_factors.shape)
#     assert scale_factors.shape == (B, T)
    
#     # Check if the scale factors are within the desired range
#     is_in_range = (scale_factors >= model.min_scale).all() and \
#                   (scale_factors <= model.max_scale).all()
#     print(f"Are scale factors in range [{model.min_scale}, {model.max_scale}]? {is_in_range}")
#     assert is_in_range
    
#     # Print some example outputs
#     print("\nExample scale factors for the first sample in batch:")
#     print(scale_factors[0].detach().cpu().numpy())
    
#     print("\nMultimodalAttentiveScorer ran successfully!")



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class AZNet(nn.Module):
#     """
#     Attentive Zoom Network (AZ-Net)
    
#     Predicts a scaling factor for each frame of a video based on low-resolution
#     features and a text embedding query.
#     """
#     def __init__(
#         self,
#         vision_dim: int,
#         text_dim: int,
#         projection_dim: int = 128,
#         num_heads: int = 4,
#         num_layers: int = 2,
#         max_scale: float = 3.0,
#         min_scale: float = 0.2
#     ):
#         """
#         Args:
#             vision_dim (int): Dimension of input vision features (D_v).
#             text_dim (int): Dimension of input text features (D_t).
#             projection_dim (int): Internal dimension for attention.
#             num_heads (int): Number of attention heads.
#             num_layers (int): Number of cross-attention layers.
#             max_scale (float): The maximum scaling factor allowed.
#         """
#         super().__init__()
#         self.vision_dim = vision_dim
#         self.text_dim = text_dim
#         self.projection_dim = projection_dim
#         self.max_scale = max_scale
#         self.min_scale = min_scale

#         # --- Layers ---
#         # Project text embedding to be the query
#         self.text_projector = nn.Linear(text_dim, projection_dim)
        
#         # Project vision features to be the key/value
#         self.vision_projector = nn.Linear(vision_dim, projection_dim)

#         # Transformer-based cross-attention layers
#         self.transformer_layers = nn.ModuleList([
#             nn.TransformerDecoderLayer(
#                 d_model=projection_dim,
#                 nhead=num_heads,
#                 dim_feedforward=projection_dim * 4,
#                 batch_first=True,
#             ) for _ in range(num_layers)
#         ])
        
#         # Regression head to predict the scale factor
#         self.regression_head = nn.Sequential(
#             nn.LayerNorm(projection_dim),
#             nn.Linear(projection_dim, 1)
#         )

#         # Time position encoding
#         self.positional_encoding = nn.Parameter(torch.randn(1, 1024, vision_dim)) # Max 1024 frames

#     def forward(self, low_res_features: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             low_res_features (torch.Tensor): Shape (B, T, L, D_v)
#             text_embedding (torch.Tensor): Shape (B, D_t)

#         Returns:
#             torch.Tensor: Predicted scale factors, shape (B, T)
#         """
#         B, T, L, D_v = low_res_features.shape

#         # 1. Aggregate patch features for each frame
#         # (B, T, L, D_v) -> (B, T, D_v)
#         frame_features = low_res_features.mean(dim=2)

#         # 2. Add positional encoding for time dimension
#         frame_features = frame_features + self.positional_encoding[:, :T, :]

#         # 3. Project features for attention
#         # Text becomes the Query: (B, D_t) -> (B, 1, proj_dim)
#         query = self.text_projector(text_embedding).unsqueeze(1)
        
#         # Video frames become Key/Value: (B, T, D_v) -> (B, T, proj_dim)
#         memory = self.vision_projector(frame_features)
        
#         # 4. Pass through cross-attention layers
#         # In TransformerDecoderLayer, `tgt` is the query, `memory` is the key/value
#         # We use the query as the target to be updated by the memory.
#         # To get per-frame scores, we need to attend from each frame TO the text.
#         # So we'll flip the roles: Query=memory, Key/Value=query
#         # This is more efficient if T > 1. Let's use standard way for simplicity
#         # where the query attends to the memory. The output shape will be (B, 1, proj_dim).
#         # To get per-frame scores, we'll make a different choice:
#         # Let each frame feature be a query attending to the text.
        
#         # Correct approach for per-frame scores:
#         # Query: memory (B, T, proj_dim)
#         # Key/Value: query (B, 1, proj_dim)
        
#         x = memory # The features we want to refine based on text
#         text_kv = query # The text context
        
#         for layer in self.transformer_layers:
#             # Here, x is the target sequence, and text_kv provides the context.
#             x = layer(tgt=x, memory=text_kv)

#         # `x` is now text-aware frame features: (B, T, proj_dim)

#         # 5. Predict scale value from each text-aware frame feature
#         # (B, T, proj_dim) -> (B, T, 1)
#         scale_logits = self.regression_head(x)

#         # 6. Map to the desired range [1.0, max_scale]
#         # (B, T, 1) -> (B, T)
#         scale_sigmoid = torch.sigmoid(scale_logits.squeeze(-1))
#         scale_factors = scale_sigmoid * (self.max_scale - self.min_scale) + self.min_scale
        
#         return scale_factors


# class SimpleMainModel(nn.Module):
#     """A dummy main model for demonstration."""
#     def __init__(self, vision_dim=512, num_classes=10):
#         super().__init__()
#         self.vision_processor = nn.Linear(vision_dim, 256)
#         self.classifier = nn.Linear(256, num_classes)
        
#     def forward(self, high_res_features):
#         # (B, T, D_v) -> (B, D_v) (pooling time)
#         x = high_res_features.mean(dim=1) 
#         x = F.relu(self.vision_processor(x))
#         return self.classifier(x)

# class AdaptiveModelWrapper(nn.Module):
#     """
#     Wraps the AZNet and the main model to create an end-to-end trainable system.
#     """
#     def __init__(self, az_net: AZNet, main_model: nn.Module, base_res: int, cost_weight: float = 0.01):
#         super().__init__()
#         self.az_net = az_net
#         self.main_model = main_model
#         self.base_res = base_res # e.g., 112
#         self.cost_weight = cost_weight

#     def _differentiable_resize(self, frames: torch.Tensor, scale_factors: torch.Tensor) -> torch.Tensor:
#         """
#         Simulates resizing frames using a differentiable operation.
        
#         Args:
#             frames (torch.Tensor): Original high-res frames (B, T, C, H, W).
#             scale_factors (torch.Tensor): Scale factors for each frame (B, T).
        
#         Returns:
#             torch.Tensor: Resized frames (B, T, C, H_base, W_base).
#         """
#         B, T, C, H, W = frames.shape
#         device = frames.device
        
#         resized_frames = []
#         for i in range(B):
#             batch_resized = []
#             for j in range(T):
#                 scale = scale_factors[i, j]
#                 # Target size based on scale factor
#                 target_h = int(self.base_res * scale.item())
#                 target_w = int(self.base_res * scale.item())

#                 # Simulate resizing by downsampling from original high-res
#                 # This is a simplification. In reality, you'd load from disk at this res.
#                 # Here, we use grid_sample to make it differentiable.
#                 frame_to_resize = frames[i, j].unsqueeze(0) # (1, C, H, W)
                
#                 # Downsample to target size, then upsample back to base_res for batching
#                 # A more direct approach is to have a vision encoder that can handle variable sizes
#                 # For this demo, let's assume we resize to a fixed size for the main model
                
#                 # Let's simplify: resize to a fixed size, but the quality depends on scale
#                 # We can simulate this by blurring based on how much downsampling would occur
                
#                 # A more direct differentiable proxy:
#                 # We resize to the target size, then take a center crop to the base_res size
#                 # This is complex. Let's use a simpler differentiable proxy for loss.
                
#                 # Simplest proxy: Let's assume the main model gets features from a
#                 # fixed-size input, and we just use the scale factors for loss.
                
#                 # For a true E2E example, let's use grid_sample to resize to base_res
#                 # The 'quality' is implicitly encoded by what grid_sample does.
#                 resized = F.interpolate(frame_to_resize, size=(self.base_res, self.base_res), mode='bilinear', align_corners=False)
#                 batch_resized.append(resized)
            
#             resized_frames.append(torch.cat(batch_resized, dim=0))

#         # We need to return features, not frames. Let's assume a fixed feature extractor.
#         # This part is hard to simulate perfectly without a real multi-resolution feature extractor.
        
#         # Let's pivot to a more practical differentiable setup.
#         # Assume we have feature maps at different resolutions.
#         # Let's fake this for the demo.
        
#         # Faking multi-resolution features
#         # feat_low, feat_mid, feat_high
#         # scale_factors will decide the mixing ratio.
#         return frames # Let's assume this function is a placeholder

#     def forward(
#         self,
#         low_res_features: torch.Tensor,
#         text_embedding: torch.Tensor,
#         high_res_frames_for_main_model: torch.Tensor, # This would be the actual data
#         labels: torch.Tensor,
#     ):
#         # 1. AZ-Net predicts scale factors
#         scale_factors = self.az_net(low_res_features, text_embedding) # (B, T)
#         breakpoint()

#         # 2. Simulate adaptive resizing and feature extraction (Placeholder)
#         # In a real system, this step would involve loading data at different resolutions
#         # and passing it to a vision tower. For this demo, we'll just pass the
#         # high-res features to the main model to get a task loss.
#         # The key is that the gradient will flow back through `scale_factors`.
        
#         # We need a way for `scale_factors` to influence the `main_model` input/output.
#         # Differentiable Proxy for Quality:
#         # Let's use scale_factors to "gate" the high-res features.
#         # This simulates that a lower scale factor leads to worse quality features.
        
#         # (B, T, D) -> (B, T, 1) -> (B, T, D)
#         gate = scale_factors.unsqueeze(-1) / self.az_net.max_scale 
#         gated_features = high_res_frames_for_main_model * gate
        
#         # 3. Main model performs the task
#         logits = self.main_model(gated_features)

#         # 4. Calculate losses
#         # Task loss
#         task_loss = F.cross_entropy(logits, labels)
        
#         # Cost loss (encourage smaller scales)
#         # We want to minimize the scale, so the loss is the scale itself.
#         cost_loss = scale_factors.mean()
        
#         # Total loss
#         total_loss = task_loss + self.cost_weight * cost_loss
        
#         return {
#             "total_loss": total_loss,
#             "task_loss": task_loss,
#             "cost_loss": cost_loss,
#             "predicted_scales": scale_factors,
#             "predicted_logits": logits,
#         }

# def main():
#     # --- Hyperparameters & Setup ---
#     B, T, L, D_v, D_t = 4, 16, 64, 256, 128 # Batch, Time, Patches, VisionDim, TextDim
#     num_classes = 10
#     max_scale = 4.0
#     base_res = 112
#     cost_weight = 0.05
#     learning_rate = 1e-4
#     num_epochs = 50
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # --- Instantiate Models ---
#     az_net = AZNet(
#         vision_dim=D_v,
#         text_dim=D_t,
#         max_scale=max_scale
#     ).to(device)
    
#     # A dummy feature extractor for the main model
#     main_model_feature_dim = 256
#     main_model = SimpleMainModel(vision_dim=main_model_feature_dim, num_classes=num_classes).to(device)
    
#     wrapper = AdaptiveModelWrapper(
#         az_net,
#         main_model,
#         base_res=base_res,
#         cost_weight=cost_weight
#     ).to(device)
    
#     optimizer = torch.optim.Adam(wrapper.parameters(), lr=learning_rate)

#     print("--- Starting Training ---")
    
#     for epoch in range(num_epochs):
#         # --- Generate Dummy Data ---
#         low_res_features = torch.randn(B, T, L, D_v, device=device)
#         text_embedding = torch.randn(B, D_t, device=device)
        
#         # For the main model, we use a different dimension to simulate a real feature extractor
#         high_res_features = torch.randn(B, T, main_model_feature_dim, device=device)
        
#         labels = torch.randint(0, num_classes, (B,), device=device)
        
#         # --- Forward and Backward Pass ---
#         optimizer.zero_grad()
        
#         output = wrapper(
#             low_res_features=low_res_features,
#             text_embedding=text_embedding,
#             high_res_frames_for_main_model=high_res_features,
#             labels=labels,
#         )
        
#         loss = output["total_loss"]
#         loss.backward()
#         optimizer.step()
        
#         # --- Logging ---
#         if (epoch + 1) % 10 == 0:
#             avg_scale = output["predicted_scales"].mean().item()
#             task_loss = output["task_loss"].item()
#             cost_loss = output["cost_loss"].item()
            
#             # Check accuracy
#             preds = torch.argmax(output["predicted_logits"], dim=1)
#             acc = (preds == labels).float().mean().item()
            
#             print(
#                 f"Epoch {epoch+1:03d} | Total Loss: {loss.item():.4f} | "
#                 f"Task Loss: {task_loss:.4f} | Cost Loss: {cost_loss:.4f} | "
#                 f"Avg Scale: {avg_scale:.2f} | Accuracy: {acc:.2f}"
#             )

# if __name__ == "__main__":
#     main()