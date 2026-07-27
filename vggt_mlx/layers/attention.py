"""Fused-QKV self-attention shared by VGGT frame and global blocks."""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn


class Attention(nn.Module):
    """Upstream-compatible attention with optional Q/K norm and axial RoPE.

    Frame/global scope is determined by the leading/token dimensions supplied
    by ``Aggregator``: frame streams use ``[B*S,P,C]`` while global streams use
    ``[B,S*P,C]``.
    """

    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 16,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = True,
        rope: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if attn_drop:
            raise ValueError("Fused VGGT inference attention requires attn_drop=0")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = (
            nn.LayerNorm(self.head_dim, eps=1e-5) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            nn.LayerNorm(self.head_dim, eps=1e-5) if qk_norm else nn.Identity()
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def __call__(self, x, pos=None, mask=None):
        if x.ndim != 3:
            raise ValueError(f"Attention expects [B,T,C], got {x.shape}")

        batch, token_count, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = self.q_norm(query)
        key = self.k_norm(key)

        if self.rope is not None:
            if pos is None:
                raise ValueError("2D positions are required when RoPE is enabled")
            query = self.rope(query, pos)
            key = self.rope(key, pos)

        attended = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=self.scale,
            mask=mask,
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(
            batch,
            token_count,
            channels,
        )
        return self.proj_drop(self.proj(attended))
