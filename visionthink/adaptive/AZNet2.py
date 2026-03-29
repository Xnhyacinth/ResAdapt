import torch
import torch.nn.functional as F
from torch import nn, einsum
import math
from einops import rearrange, repeat
from torch.distributions import Beta, Normal
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionRotaryEmbedding
from transformers.modeling_utils import ModuleUtilsMixin
from typing import List, Optional

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
        # device = x.device
        
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
        
        attn = nn.functional.softmax(sim, dim=-1, dtype=torch.float32).to(q.dtype)
        #attn = sim.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
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

        # print('v.dtype', v.dtype)
        # print('attn.dtype', attn.dtype)
        # print('x.dtype', x.dtype)
        # print('q.dtype', q.dtype)
        # print('out.dtype', out.dtype)
        # print('to_out.dtype', self.to_out.weight.dtype)
        
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
        # attn = sim.softmax(dim=-1)
        attn = nn.functional.softmax(sim, dim=-1, dtype=torch.float32).to(q.dtype)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
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
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, T: int, L: int, rotary_pos_emb=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T*L, D).
            T (int): Number of frames.
            L (int): Number of patches per frame.
            rotary_pos_emb (torch.Tensor, optional): RoPE for the spatial dimension (length L).
        """
        B, _, D = x.shape
        x = self.norm(x)
        
        # Reshape for spatial attention: treat each frame as a batch item
        x = rearrange(x, 'b (t l) d -> (b t) l d', t=T, l=L)
        
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        # Apply standard rotary embeddings for spatial positions
        if exists(rotary_pos_emb):
            q = apply_rotary_pos_emb(q, rotary_pos_emb)
            k = apply_rotary_pos_emb(k, rotary_pos_emb)
        
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        # attn = sim.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
        attn = nn.functional.softmax(sim, dim=-1, dtype=torch.float32).to(q.dtype)
        attn = self.dropout(attn)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        out = self.to_out(out)
        
        # Reshape back to original format
        return rearrange(out, '(b t) l d -> b (t l) d', b=B, t=T)

# --- The Final MultimodalAttentiveScorer ---

class FrameWiseScalePredictor(ModuleUtilsMixin, nn.Module):
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
        min_scale: float = 0.25,
        max_scale: float = 3.0,
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
        
        self.rope_gen = Qwen2_5_VisionRotaryEmbedding(dim=max(32, dim_head // 2))
        self.temporal_rope_gen = Qwen2_5_VisionRotaryEmbedding(dim=max(32, dim_head))

        # --- Layers ---
        self.cross_attn_blocks = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_blocks.append(nn.ModuleList([
                CrossAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout),
                FeedForward(dim, mult=ff_mult, dropout=dropout)
            ]))

        self.spatio_temporal_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.spatio_temporal_blocks.append(nn.ModuleList([
                PatchWiseTemporalAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout),
                SpatialAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout),
                FeedForward(dim, mult=ff_mult, dropout=dropout)
            ]))
            
        self.regression_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2)
        )

    def get_action_and_log_prob(self, alpha, beta, action_0_1 = None):
        """
        Samples an action from the Beta distribution and computes the log_prob.
        No correction is needed as the action is naturally in [0, 1].
        """
        dist = Beta(alpha, beta)
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
        x_vid = video_with_pos.unsqueeze(0) # (1, T*L, D)
        x_text = text_feats_with_pos.unsqueeze(0) # (1, N, D)
        if exists(video_mask): video_mask = video_mask.unsqueeze(0)
        if exists(text_mask): text_mask = text_mask.unsqueeze(0)

        # --- 2. Patch-wise Temporal Attention ---
        temporal_rope = self.temporal_rope_gen(T)
        spatial_rope = self.rope_gen(L)
        text_rope = self.rope_gen(N)

        for temp_attn, spat_attn, ff in self.spatio_temporal_blocks:
            # temporal_rope
            x_vid = temp_attn(x_vid, T=T, L=L, rotary_pos_emb=temporal_rope) + x_vid
            # spatial_rope
            x_vid = spat_attn(x_vid, T=T, L=L, rotary_pos_emb=spatial_rope) + x_vid
            # FF
            x_vid = ff(x_vid) + x_vid
            
        # --- 4. Cross-Attention Blocks ---
        for cross_attn, ff in self.cross_attn_blocks:
            x_vid = cross_attn(x_vid, context=x_text, rotary_pos_emb=text_rope, context_mask=text_mask) + x_vid
            x_vid = ff(x_vid) + x_vid

        frame_features = rearrange(x_vid, 'b (t l) d -> b t l d', t=T, l=L).mean(dim=2) # (1, T, D)
        
        return frame_features.squeeze(0) # Return (T, D)

    def forward(
        self,
        video_features_list, # Shape # List of (T_i * L_i, D)
        video_grid_thw,
        text_features,        # Shape (B, N, D)
        video_mask=None,      # Shape (B, T*L) - Optional for patches
        text_mask=None,        # Shape (B, N)
        actions=None
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
        # dist = Beta(alpha, beta)

        # Sample an action (scale_factors) and get its log_prob
        actions, scale_factors, log_prob = self.get_action_and_log_prob(alpha, beta, actions)
            
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





class SimpleFrameFactorPredictor(nn.Module):
    def __init__(
        self, 
        config, 
        min_factor: float = 0.5, 
        max_factor: float = 2.0
    ):
        """
        Args:
            input_dim: 输入特征维度 D
            hidden_dim: 中间层维度
            min_factor: 预测因子的最小值
            max_factor: 预测因子的最大值
        """
        super().__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor

        # 简单的 MLP 预测头
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(), # 或者 nn.ReLU()
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # 将输出限制在 (0, 1) 之间，方便映射到 min-max 范围
        )

    def forward(
        self, 
        video_features_list: List[torch.Tensor], 
        grid_thw_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Args:
            video_features_list: List of (T_i * L_i, D)
            grid_thw_list: List of (3,), 包含每个视频的 [T, H, W]
        
        Returns:
            List of (T_i, 1) or (T_i,), 每个视频每帧的预测因子
        """
        batch_results = []

        for features, grid_thw in zip(video_features_list, grid_thw_list):
            # 1. 解析维度
            # features: (Total_Patches, D)
            T = grid_thw[0].item()
            H = grid_thw[1].item()
            W = grid_thw[2].item()
            
            # 计算每帧的空间 Patch 数量 L
            # 注意：这里假设 features 的总长度等于 T * H * W (或者 T * H * W / patch_size^2)
            # 无论 spatial 怎么 merge，L = Total_Len // T
            total_len, D = features.shape
            L = total_len // T 

            # 2. 空间聚合 (Spatial Pooling)
            # Reshape: (T*L, D) -> (T, L, D)
            # 这样我们将同一帧的所有 patches 聚合在一起
            features_reshaped = features.view(T, L, D)
            
            # Mean Pooling: (T, L, D) -> (T, D)
            # 得到每一帧的全局特征
            frame_features = features_reshaped.mean(dim=1)

            # 3. 预测 Factor
            # (T, D) -> (T, 1)
            raw_scores = self.mlp(frame_features)

            # 4. 映射到目标范围 [min, max]
            # output = min + sigmoid_val * (max - min)
            factors = self.min_factor + raw_scores * (self.max_factor - self.min_factor)
            
            # 移除最后一个维度: (T, 1) -> (T,)
            batch_results.append(factors.squeeze(-1))

        return batch_results