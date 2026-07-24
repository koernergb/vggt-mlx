"""Axial 2D rotary position embeddings used by VGGT attention."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn


class PositionGetter:
    """Generate and cache flattened ``(row, column)`` patch-grid positions."""

    def __init__(self) -> None:
        self.position_cache: dict[tuple[int, int], Any] = {}

    def __call__(self, batch_size: int, height: int, width: int):
        if batch_size < 1 or height < 1 or width < 1:
            raise ValueError("batch_size, height, and width must all be positive")

        cache_key = (height, width)
        if cache_key not in self.position_cache:
            rows = mx.repeat(mx.arange(height, dtype=mx.int32), width)
            columns = mx.tile(mx.arange(width, dtype=mx.int32), height)
            self.position_cache[cache_key] = mx.stack((rows, columns), axis=-1)

        positions = self.position_cache[cache_key]
        return mx.broadcast_to(positions[None], (batch_size, height * width, 2))


class RotaryPositionEmbedding2D(nn.Module):
    """Apply independent 1D RoPE rotations to row and column feature halves."""

    def __init__(
        self,
        frequency: float = 100.0,
        scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if frequency <= 0:
            raise ValueError("frequency must be positive")
        if scaling_factor <= 0:
            raise ValueError("scaling_factor must be positive")

        self.base_frequency = float(frequency)
        self.scaling_factor = float(scaling_factor)
        self.frequency_cache: dict[tuple[int, int, str], tuple[Any, Any]] = {}

    def _compute_frequency_components(self, dim: int, seq_len: int, dtype):
        if dim % 2:
            raise ValueError(f"Per-axis feature dimension must be even, got {dim}")

        cache_key = (dim, seq_len, str(dtype))
        if cache_key not in self.frequency_cache:
            exponents = mx.arange(0, dim, 2, dtype=mx.float32) / dim
            inverse_frequency = mx.power(self.base_frequency, -exponents)
            # Upstream retains scaling_factor as metadata but does not apply it
            # in the released VGGT-1B implementation.
            positions = mx.arange(seq_len, dtype=mx.float32)
            angles = positions[:, None] * inverse_frequency[None, :]
            angles = mx.concatenate((angles, angles), axis=-1).astype(dtype)
            self.frequency_cache[cache_key] = (mx.cos(angles), mx.sin(angles))
        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(features):
        midpoint = features.shape[-1] // 2
        first = features[..., :midpoint]
        second = features[..., midpoint:]
        return mx.concatenate((-second, first), axis=-1)

    def _apply_1d_rope(self, tokens, positions, cosine, sine):
        cosine = cosine[positions][:, None, :, :]
        sine = sine[positions][:, None, :, :]
        return tokens * cosine + self._rotate_features(tokens) * sine

    def __call__(self, tokens, positions):
        if tokens.ndim != 4:
            raise ValueError(
                f"tokens must have shape [B,heads,T,dim], got {tokens.shape}"
            )
        if tokens.shape[-1] % 4:
            raise ValueError(
                f"Head dimension must be divisible by 4, got {tokens.shape[-1]}"
            )
        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError(
                f"positions must have shape [B,T,2], got {positions.shape}"
            )
        if positions.shape[0] != tokens.shape[0] or positions.shape[1] != tokens.shape[2]:
            raise ValueError(
                f"Position shape {positions.shape} does not match tokens {tokens.shape}"
            )

        per_axis_dim = tokens.shape[-1] // 2
        # Positions are integer grid coordinates. Special tokens use zero while
        # patch positions are shifted by one in the aggregator.
        max_position = int(mx.max(positions).item()) + 1
        cosine, sine = self._compute_frequency_components(
            per_axis_dim,
            max_position,
            tokens.dtype,
        )

        vertical, horizontal = mx.split(tokens, 2, axis=-1)
        vertical = self._apply_1d_rope(
            vertical,
            positions[..., 0],
            cosine,
            sine,
        )
        horizontal = self._apply_1d_rope(
            horizontal,
            positions[..., 1],
            cosine,
            sine,
        )
        return mx.concatenate((vertical, horizontal), axis=-1)
