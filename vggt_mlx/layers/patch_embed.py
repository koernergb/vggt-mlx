"""DINOv2 ViT-L/14-with-registers backbone used by VGGT."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from vggt_mlx.layers.mlp import Mlp


def _cubic_kernel(distance: np.ndarray) -> np.ndarray:
    """PyTorch's antialiased bicubic kernel (Keys cubic with a=-0.5)."""
    distance = np.abs(distance)
    near = 1.5 * distance**3 - 2.5 * distance**2 + 1.0
    far = -0.5 * distance**3 + 2.5 * distance**2 - 4.0 * distance + 2.0
    return np.where(distance < 1.0, near, np.where(distance < 2.0, far, 0.0))


def _antialiased_bicubic_matrix(input_size: int, output_size: int):
    """Build the separable resize matrix used by torch bicubic+antialias."""
    scale = input_size / output_size
    kernel_scale = max(scale, 1.0)
    support = 2.0 * kernel_scale
    weights = np.zeros((output_size, input_size), dtype=np.float32)

    for output_index in range(output_size):
        center = (output_index + 0.5) * scale
        first = int(np.floor(center - support + 0.5))
        last = int(np.ceil(center + support - 0.5))
        indices = np.arange(first, last + 1)
        indices = indices[(indices >= 0) & (indices < input_size)]
        coefficients = _cubic_kernel(
            (indices.astype(np.float64) + 0.5 - center) / kernel_scale
        )
        coefficients /= coefficients.sum()
        weights[output_index, indices] = coefficients.astype(np.float32)

    return mx.array(weights)


class _PatchProjection(nn.Module):
    """Image-to-patch projection with upstream-compatible ``proj`` naming."""

    def __init__(self, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def __call__(self, images):
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(
                f"Patch projection expects NHWC images, received shape {images.shape}"
            )
        features = self.proj(images)
        return features.reshape(features.shape[0], -1, features.shape[-1])


class _LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float) -> None:
        super().__init__()
        self.gamma = mx.full((dim,), init_values)

    def __call__(self, x):
        return x * self.gamma


class _DinoAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"embed_dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(0.0)

    def __call__(self, x):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch,
            tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = self.q_norm(query)
        key = self.k_norm(key)

        attended = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=self.scale,
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(attended))


class _DinoBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        init_values: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _DinoAttention(dim, num_heads)
        self.ls1 = _LayerScale(dim, init_values)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
        )
        self.ls2 = _LayerScale(dim, init_values)

    def __call__(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    """Complete DINOv2 ViT-L/14 tokenizer used by the VGGT aggregator.

    Input images are NHWC and already normalized by the aggregator. Only the
    normalized patch-token stream is returned; DINO's class and register tokens
    remain internal to this backbone.
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
        init_values: float = 1.0,
    ) -> None:
        super().__init__()
        if img_size % patch_size:
            raise ValueError("img_size must be divisible by patch_size")

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens
        self.base_grid_size = img_size // patch_size

        # Names and shapes mirror upstream DinoVisionTransformer exactly.
        self.patch_embed = _PatchProjection(patch_size, embed_dim)
        self.cls_token = mx.zeros((1, 1, embed_dim))
        self.pos_embed = mx.zeros(
            (1, self.base_grid_size * self.base_grid_size + 1, embed_dim)
        )
        self.register_tokens = mx.zeros((1, num_register_tokens, embed_dim))
        self.mask_token = mx.zeros((1, embed_dim))
        self.blocks = [
            _DinoBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                init_values=init_values,
            )
            for _ in range(depth)
        ]
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Identity()

    def _interpolate_pos_encoding(self, grid_height: int, grid_width: int):
        class_position = self.pos_embed[:, :1]
        patch_positions = self.pos_embed[:, 1:]
        if (
            grid_height == self.base_grid_size
            and grid_width == self.base_grid_size
        ):
            return class_position, patch_positions

        patch_positions = patch_positions.reshape(
            1,
            self.base_grid_size,
            self.base_grid_size,
            self.embed_dim,
        )
        if grid_height != self.base_grid_size:
            height_weights = _antialiased_bicubic_matrix(
                self.base_grid_size, grid_height
            )
            patch_positions = mx.einsum(
                "yh,bhwc->bywc", height_weights, patch_positions
            )
        if grid_width != self.base_grid_size:
            width_weights = _antialiased_bicubic_matrix(
                self.base_grid_size, grid_width
            )
            patch_positions = mx.einsum(
                "xw,bywc->byxc", width_weights, patch_positions
            )
        return class_position, patch_positions.reshape(
            1,
            grid_height * grid_width,
            self.embed_dim,
        )

    def __call__(self, images):
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"PatchEmbed expects NHWC images, got {images.shape}")
        height, width = images.shape[1:3]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Image dimensions {(height, width)} must be divisible by "
                f"patch_size={self.patch_size}"
            )

        patch_tokens = self.patch_embed(images)
        class_position, patch_positions = self._interpolate_pos_encoding(
            height // self.patch_size,
            width // self.patch_size,
        )
        batch = images.shape[0]
        class_token = mx.broadcast_to(self.cls_token, (batch, 1, self.embed_dim))
        tokens = mx.concatenate(
            (
                class_token + class_position,
                patch_tokens + patch_positions,
            ),
            axis=1,
        )
        register_tokens = mx.broadcast_to(
            self.register_tokens,
            (batch, self.num_register_tokens, self.embed_dim),
        )
        tokens = mx.concatenate(
            (tokens[:, :1], register_tokens, tokens[:, 1:]),
            axis=1,
        )

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens[:, self.num_register_tokens + 1 :]
