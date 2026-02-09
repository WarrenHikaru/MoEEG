import torch.nn as nn
from utils import *
from .MoE import MoE
from .AttentionLayer import Attention,GroupAttention

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_path(self, x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0.,num_patches=4, act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_rope=False,
                 return_attention=False, is_group_attn=False,use_gate=True):
        super().__init__()

        self.num_patches=num_patches
        self.is_group_attn = is_group_attn
        self.return_attention = return_attention
        self.use_gate = use_gate
        self.norm1 = norm_layer(dim, eps=1e-5)

        self.time_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_rope=use_rope, return_attention=return_attention,use_gate=self.use_gate)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.is_group_attn:
            self.norm2 = norm_layer(dim, eps=1e-5)
            self.group_attn = GroupAttention(
                dim=dim, num_heads=num_heads,num_patches=self.num_patches, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
                )

        self.norm3 = norm_layer(dim, eps=1e-5)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, freqs=None):
        Residual = x
        y_time = self.drop_path((self.time_attn(self.norm1(x), freqs)))

        if self.is_group_attn:
            y_group = self.drop_path((self.group_attn(self.norm2(x))))
            x = Residual + y_time + y_group
        else:
            x = Residual + y_time

        x = self.norm3(x)
        x = x + self.drop_path(self.mlp(x))
        return x

class MoEBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0.,top_k=2, norm_layer=nn.LayerNorm, is_causal=False, use_rope=False,
                 return_attention=False):
        super().__init__()
        self.return_attention = return_attention
        self.norm1 = norm_layer(dim,eps=1e-6)
        self.time_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, is_causal=is_causal,
            use_rope=use_rope, return_attention=return_attention)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim,eps=1e-6)

        self.moe = MoE(
            dim=dim,
            num_experts=4,
            top_k=top_k,
            mlp_ratio=mlp_ratio,
            drop=drop
        )

    def forward(self, x, freqs=None):

        y = self.time_attn(self.norm1(x), freqs)
        if self.return_attention:
            return y
        x = x + self.drop_path(y)

        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + self.drop_path(moe_out)

        return x, aux_loss
