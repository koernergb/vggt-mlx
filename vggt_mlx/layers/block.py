"""Pre-norm transformer block used by VGGT alternating attention."""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from vggt_mlx.layers.attention import Attention
from vggt_mlx.layers.mlp import Mlp


class LayerScale(nn.Module):
    """Learned per-channel residual scale with upstream ``gamma`` naming."""

    def __init__(self, dim: int, init_values: float) -> None:
        super().__init__()
        self.gamma = mx.full((dim,), init_values, dtype=mx.float32)

    def __call__(self, x):
        return x * self.gamma


class Block(nn.Module):
    """VGGT transformer block matching upstream state-dict structure."""

    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = 0.01,
        drop_path: float = 0.0,
        qk_norm: bool = True,
        rope: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if drop_path:
            raise ValueError("Stochastic depth is unsupported by the inference port")

        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            rope=rope,
        )
        self.ls1 = (
            LayerScale(dim, init_values) if init_values is not None else nn.Identity()
        )
        self.drop_path1 = nn.Identity()

        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            bias=ffn_bias,
            drop=drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values) if init_values is not None else nn.Identity()
        )
        self.drop_path2 = nn.Identity()
        self.sample_drop_ratio = drop_path

    def __call__(self, x, pos=None):
        attention_residual = self.ls1(self.attn(self.norm1(x), pos=pos))
        x = x + self.drop_path1(attention_residual)
        mlp_residual = self.ls2(self.mlp(self.norm2(x)))
        return x + self.drop_path2(mlp_residual)
