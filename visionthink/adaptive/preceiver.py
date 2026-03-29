import torch
import torch.nn.functional as F
from torch import nn, einsum
import math
from einops import rearrange, repeat
# from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionRotaryEmbedding, apply_rotary_pos_emb_vision

# helper functions

def exists(val):
    return val is not None

# feedforward

def FeedForward(dim, mult = 4, dropout = 0.):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias = False),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim, bias = False)
    )

# rotary positional embedding
# https://arxiv.org/abs/2104.09864

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)).to(torch.bfloat16)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device = device, dtype = self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim = -1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(pos, t):
    seq_len, rotate_dim = t.shape[-2], pos.shape[-1]
    pos = pos[..., -seq_len:, :]
    t, t_pass = t[..., :rotate_dim], t[..., rotate_dim:]
    t = (t * pos.cos()) + (rotate_half(t) * pos.sin())
    return torch.cat((t, t_pass), dim = -1)


class VisionTemporalAttention(nn.Module):
    """
    Performs self-attention over the temporal dimension of video features.
    It takes (B, T, D) as input and models dependencies between frames.
    """
    temporal_dim_scale = 0.25 # Kept from original for parameter consistency

    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        dropout=0.
    ):
        super().__init__()
        # Use a fraction of heads as specified by the scale
        self.num_heads = max(1, round(heads * self.temporal_dim_scale))
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        inner_dim = self.num_heads * self.dim_head

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, rotary_pos_emb=None, attention_mask=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D).
            rotary_pos_emb (torch.Tensor, optional): Precomputed rotary embeddings.
            attention_mask (torch.Tensor, optional): Padding mask of shape (B, T).
        """
        x = self.norm(x)

        # Project to Q, K, V and rearrange for multi-head attention
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

        # Apply rotary positional embeddings
        if exists(rotary_pos_emb):
            # Assuming the rotary embedding functions are available
            q = apply_rotary_pos_emb(rotary_pos_emb, q)
            k = apply_rotary_pos_emb(rotary_pos_emb, k)

        # Compute attention scores
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # Apply padding mask if provided
        if exists(attention_mask):
            mask = rearrange(attention_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


# attention

class CausalAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = heads * dim_head

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, rotary_pos_emb = None):
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        q = q * self.scale

        if exists(rotary_pos_emb):
            q = apply_rotary_pos_emb(rotary_pos_emb, q)
            k = apply_rotary_pos_emb(rotary_pos_emb, k)

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device = x.device, dtype = torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)


        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CausalPrefixAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        max_heads_process = 2,
        dropout = 0.,
        cross_attn_dropout = 0.
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.max_heads_process = max_heads_process

        inner_dim = heads * dim_head

        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.cross_attn_dropout = cross_attn_dropout # they drop out a percentage of the prefix during training, shown to help prevent overfitting

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context, context_mask = None, rotary_pos_emb = None):
        batch, context_len, device = x.shape[0], context.shape[-2], x.device

        q_rotary_pos_emb = rotary_pos_emb
        k_rotary_pos_emb = rotary_pos_emb

        # take care of cross attention dropout

        if self.training and self.cross_attn_dropout > 0.:
            rand = torch.zeros((batch, context_len), device = device).uniform_()
            keep_context_len = context_len - int(context_len * self.cross_attn_dropout)
            keep_indices = rand.topk(keep_context_len, dim = -1).indices
            keep_mask = torch.zeros_like(rand).scatter_(1, keep_indices, 1).bool()

            context = rearrange(context[keep_mask], '(b n) d -> b n d', b = batch)

            if exists(context_mask):
                context_mask = rearrange(context_mask[keep_mask], '(b n) -> b n', b = batch)

            # operate on rotary position embeddings for keys

            k_rotary_pos_emb = repeat(k_rotary_pos_emb, '... -> b ...', b = batch)
            k_rotary_pos_emb_context, k_rotary_pos_emb_seq = k_rotary_pos_emb[:, :context_len], k_rotary_pos_emb[:, context_len:]
            k_rotary_pos_emb_context = rearrange(k_rotary_pos_emb_context[keep_mask], '(b n) d -> b n d', b = batch)

            k_rotary_pos_emb = torch.cat((k_rotary_pos_emb_context, k_rotary_pos_emb_seq), dim = 1)
            k_rotary_pos_emb = rearrange(k_rotary_pos_emb, 'b n d -> b 1 n d')

        # normalization

        x = self.norm(x)
        context = self.context_norm(context)

        # derive queries, keys, values

        q = self.to_q(x)

        k_input, v_input = self.to_kv(x).chunk(2, dim = -1)
        k_context, v_context = self.to_kv(context).chunk(2, dim = -1)

        k = torch.cat((k_context, k_input), dim = 1)
        v = torch.cat((v_context, v_input), dim = 1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        q = q * self.scale

        # rotate queries and keys with rotary embeddings

        if exists(rotary_pos_emb):
            q = apply_rotary_pos_emb(q_rotary_pos_emb, q)
            k = apply_rotary_pos_emb(k_rotary_pos_emb, k)

        # take care of masking

        i, j = q.shape[-2], k.shape[-2]
        mask_value = -torch.finfo(q.dtype).max

        if exists(context_mask):
            mask_len = context_mask.shape[-1]
            context_mask = F.pad(context_mask, (0, max(j - mask_len, 0)), value = True)
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')

        causal_mask = torch.ones((i, j), device = x.device, dtype = torch.bool).triu(j - i + 1)

        # process in chunks of heads

        out = []

        max_heads = self.max_heads_process

        for q_chunk, k_chunk, v_chunk in zip(q.split(max_heads, dim = 1), k.split(max_heads, dim = 1), v.split(max_heads, dim = 1)):
            sim = einsum('b h i d, b h j d -> b h i j', q_chunk, k_chunk)
            

            if exists(context_mask):
                sim = sim.masked_fill(~context_mask, mask_value)

            sim = sim.masked_fill(causal_mask, mask_value)

            attn = sim.softmax(dim = -1)
            attn = self.dropout(attn)

            out_chunk = einsum('b h i j, b h j d -> b h i d', attn, v_chunk)
            out.append(out_chunk)

        # concat all the heads together

        out = torch.cat(out, dim = 1)

        # merge heads and then combine with linear

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class PerceiverAR(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        max_seq_len,
        cross_attn_seq_len,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        cross_attn_dropout = 0.,
        ff_mult = 4,
        perceive_depth = 1,
        temporal_depth = 1,
        perceive_max_heads_process = 2 # processes the heads in the perceiver layer in chunks to lower peak memory, in the case the prefix is really long
    ):
        super().__init__()
        assert max_seq_len > cross_attn_seq_len, 'max_seq_len must be greater than cross_attn_seq_len, the length of the sequence for which to cross attend to "perceiver" style'
        self.max_seq_len = max_seq_len
        self.cross_attn_seq_len = cross_attn_seq_len

        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.rotary_pos_emb = RotaryEmbedding(dim = max(32, dim_head // 2))
        self.temporal_pos_emb = RotaryEmbedding(dim = max(64, dim_head))

        self.temporal_layers = nn.ModuleList([])
        for _ in range(temporal_depth):
            self.temporal_layers.append(nn.ModuleList([
                VisionTemporalAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout),
                FeedForward(dim, mult=ff_mult, dropout=dropout)
            ]))


        self.perceive_layers  = nn.ModuleList([])
        for _ in range(perceive_depth):
            self.perceive_layers.append(nn.ModuleList([
                CausalPrefixAttention(dim = dim, dim_head = dim_head, heads = heads, max_heads_process = perceive_max_heads_process, dropout = dropout, cross_attn_dropout = cross_attn_dropout),
                FeedForward(dim, mult = ff_mult, dropout = dropout)
            ]))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CausalAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim, mult = ff_mult, dropout = dropout),
            ]))

    def forward(
        self,
        x,                # Shape (B, TotalSeqLen, D)
        video_features,   # Shape (B, T, D) - Extracted frame features
        prefix_mask=None, # Shape (B, CrossAttnSeqLen) - Mask for the context prefix
        video_mask=None   # Shape (B, T) - Padding mask for the video frames
    ):
        seq_len, device = x.shape[1], x.device
        assert self.cross_attn_seq_len < seq_len <= self.max_seq_len

        x = x + self.pos_emb(torch.arange(seq_len, device = device))

        # rotary positional embedding

        rotary_pos_emb = self.rotary_pos_emb(seq_len, device = device)

        T = video_features.shape[1]
        temporal_rotary_emb = self.temporal_pos_emb(T, device = device)

        temp_x = video_features
        for attn, ff in self.temporal_layers:
            # The temporal attention needs its own rotary embedding
            temp_x = attn(temp_x, rotary_pos_emb=temporal_rotary_emb, attention_mask=video_mask) + temp_x
            assert torch.isnan(x).any() == False, 'x is nan after attn in temporal layer'
            temp_x = ff(temp_x) + temp_x
            assert torch.isnan(x).any() == False, 'x is nan after feedforward in temporal layer'

        # text_prefix, seq_to_decode = x[:, :self.cross_attn_seq_len], x[:, self.cross_attn_seq_len:]
        prefix, x = x[:, :self.cross_attn_seq_len], x[:, self.cross_attn_seq_len:]

        # prefix = temp_x
        # initial perceiver attention and feedforward (one cross attention)
        breakpoint()
        for cross_attn, ff in self.perceive_layers:
            x = cross_attn(x, prefix, context_mask = prefix_mask, rotary_pos_emb = rotary_pos_emb) + x
            assert torch.isnan(x).any() == False, 'x is nan after attn in perceiver layer'
            x = ff(x) + x
            assert torch.isnan(x).any() == False, 'x is nan after feedforward in perceiver layer'

        # layers

        for attn, ff in self.layers:
            x = attn(x, rotary_pos_emb = rotary_pos_emb) + x
            assert torch.isnan(x).any() == False, 'x is nan after attn in decoding layers'
            assert torch.isinf(x).any() == False, 'x is inf after attn in decoding layers'
            x = ff(x) + x
            assert torch.isnan(x).any() == False, 'x is nan after feedforward in decoding layer'

        # take care of cross entropy loss if labels are provided

        return x


if __name__ == '__main__':


    # --- Model Instantiation ---
    model = PerceiverAR(
        dim=512,
        depth=6,
        max_seq_len=2048,
        cross_attn_seq_len=1024, # Length of the text prefix
        dim_head=64,
        heads=8,
        perceive_depth=2,
        temporal_depth=2, # Use 2 layers of temporal attention
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dummy Data ---
    B = 2
    TEXT_PREFIX_LEN = 1024
    DECODE_LEN = 100
    VIDEO_FRAMES = 64
    DIM = 512

    # Text input (prefix + part to be decoded)
    text_input = torch.randn(B, TEXT_PREFIX_LEN + VIDEO_FRAMES, DIM).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Video input (frame features)
    video_features = torch.randn(B, VIDEO_FRAMES, DIM).to("cuda" if torch.cuda.is_available() else "cpu")

    # Masks
    text_prefix_mask = torch.ones(B, TEXT_PREFIX_LEN + VIDEO_FRAMES, dtype=torch.bool).to("cuda" if torch.cuda.is_available() else "cpu")
    video_mask = torch.ones(B, VIDEO_FRAMES, dtype=torch.bool).to("cuda" if torch.cuda.is_available() else "cpu")
    # Let's say the second sample in the batch has a shorter video
    video_mask[1, -10:] = False 

    # --- Forward Pass ---
    output = model(
        x=text_input, 
        video_features=video_features, 
        prefix_mask=text_prefix_mask,
        video_mask=video_mask
    )

    print("Output shape:", output.shape) # Expected: (B, DECODE_LEN, DIM)
    assert output.shape == (B, DECODE_LEN, DIM)
    print("PerceiverAR with VisionTemporalAttention ran successfully!")