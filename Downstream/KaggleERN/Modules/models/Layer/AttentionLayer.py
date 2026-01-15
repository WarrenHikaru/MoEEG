import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .utils import apply_rotary_emb

class GroupAttention(nn.Module):
    """
    对 B*N维度做跨序列同位置注意力计算
    """
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0. else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0. else nn.Identity()

    def forward(self, x):
        BN, C, D = x.shape
        x_reshaped = x.permute(1, 0, 2)  # [C, BN, D]

        # [C, BN, D] → [C, BN, 3*D]
        qkv = self.qkv(x_reshaped)  # [C, BN, 3*D]
        qkv = qkv.reshape(C, BN, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [C, num_heads, BN, head_dim]

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        attn_out = (attn_weights @ v)  # [C, num_heads, BN, head_dim]
        attn_out = attn_out.transpose(1, 2).reshape(C, BN, D)
        proj_out = self.proj(attn_out)  # [C, BN, D]
        proj_out = self.proj_drop(proj_out)
        out = proj_out.permute(1, 0, 2)  # [BN, C, D]

        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., is_causal=False, use_rope=False,
                 return_attention=False, use_gate=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope
        self.return_attention = return_attention
        self.is_causal = is_causal
        self.use_gate = use_gate

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.gate_norm = nn.LayerNorm(dim) if self.use_gate else None
        self.gate_weight = nn.Linear(dim, num_heads, bias=False) if self.use_gate else None

    def forward(self, x, freqs=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, T, head_dim)

        if self.use_rope:
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)

        if self.return_attention:
            if self.is_causal:
                attn_mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril(diagonal=0)
                attn_mask = torch.full((T, T), -float('inf'), device=x.device).masked_fill(attn_mask, 0.)
            else:
                attn_mask = None
            attn_weight = torch.softmax(
                (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) + (attn_mask if self.is_causal else 0),
                dim=-1
            )
            attn_weight = torch.nn.functional.dropout(attn_weight, p=self.attn_drop, training=self.training)
            return attn_weight

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=self.is_causal
        )  #(B, num_heads, T, head_dim)

        if self.use_gate:
            gate_input = self.gate_norm(x)  # PreNorm (B, T, C)
            gate_score = self.gate_weight(gate_input)  # (B, T, num_heads)
            gate_score = torch.sigmoid(gate_score)  # (B, T, num_heads)

            gate_score = gate_score.permute(0, 2, 1).unsqueeze(-1)  # (B, num_heads, T, 1)

            y = y * gate_score  # (B, num_heads, T, head_dim)

        x = y.transpose(1, 2).contiguous().view(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x



