"""VGGT alternating frame/global attention aggregator."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from vggt_mlx.config import VGGTConfig
from vggt_mlx.layers.block import Block
from vggt_mlx.layers.patch_embed import PatchEmbed
from vggt_mlx.layers.rope2d import PositionGetter, RotaryPositionEmbedding2D


def slice_expand_and_flatten(token_tensor, batch_size: int, num_frames: int):
    """Use token slot 0 for frame zero and slot 1 for every other frame."""
    if token_tensor.ndim != 4 or token_tensor.shape[1] != 2:
        raise ValueError(f"Expected token shape [1,2,X,C], got {token_tensor.shape}")
    if batch_size < 1 or num_frames < 1:
        raise ValueError("batch_size and num_frames must be positive")

    query = mx.broadcast_to(
        token_tensor[:, :1],
        (batch_size, 1, token_tensor.shape[2], token_tensor.shape[3]),
    )
    if num_frames == 1:
        combined = query
    else:
        others = mx.broadcast_to(
            token_tensor[:, 1:2],
            (
                batch_size,
                num_frames - 1,
                token_tensor.shape[2],
                token_tensor.shape[3],
            ),
        )
        combined = mx.concatenate((query, others), axis=1)
    return combined.reshape(
        batch_size * num_frames,
        token_tensor.shape[2],
        token_tensor.shape[3],
    )


class Aggregator(nn.Module):
    """DINOv2 tokenizer followed by alternating frame/global transformer blocks."""

    def __init__(self, config: Optional[VGGTConfig] = None) -> None:
        super().__init__()
        self.config = config or VGGTConfig()
        config = self.config

        if config.depth % config.aa_block_size:
            raise ValueError("depth must be divisible by aa_block_size")
        if tuple(config.aa_order) != ("frame", "global"):
            raise ValueError("VGGT-1B requires aa_order=('frame', 'global')")

        self.patch_embed = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_register_tokens=config.num_register_tokens,
            init_values=1.0,
        )
        self.rope = RotaryPositionEmbedding2D(frequency=config.rope_freq)
        self.position_getter = PositionGetter()
        block_kwargs = {
            "dim": config.embed_dim,
            "num_heads": config.num_heads,
            "mlp_ratio": config.mlp_ratio,
            "init_values": config.layerscale_init,
            "qk_norm": config.qk_norm,
            "rope": self.rope,
        }
        self.frame_blocks = [Block(**block_kwargs) for _ in range(config.depth)]
        self.global_blocks = [Block(**block_kwargs) for _ in range(config.depth)]

        self.depth = config.depth
        self.aa_order = tuple(config.aa_order)
        self.aa_block_size = config.aa_block_size
        self.aa_block_num = config.depth // config.aa_block_size
        self.patch_size = config.patch_size
        self.patch_start_idx = config.patch_start_idx
        self.cached_layer_indices = set(config.intermediate_layer_idx)
        self.cached_layer_indices.add(config.depth - 1)

        self.camera_token = mx.zeros((1, 2, 1, config.embed_dim))
        self.register_token = mx.zeros(
            (1, 2, config.num_register_tokens, config.embed_dim)
        )

    def _process_frame_attention(
        self,
        tokens,
        positions,
        batch_size: int,
        num_frames: int,
        tokens_per_frame: int,
        channels: int,
        block_index: int,
    ):
        tokens = tokens.reshape(
            batch_size * num_frames,
            tokens_per_frame,
            channels,
        )
        positions = positions.reshape(
            batch_size * num_frames,
            tokens_per_frame,
            2,
        )
        intermediates = []
        for _ in range(self.aa_block_size):
            tokens = self.frame_blocks[block_index](tokens, pos=positions)
            block_index += 1
            intermediates.append(
                tokens.reshape(
                    batch_size,
                    num_frames,
                    tokens_per_frame,
                    channels,
                )
            )
        return tokens, positions, block_index, intermediates

    def _process_global_attention(
        self,
        tokens,
        positions,
        batch_size: int,
        num_frames: int,
        tokens_per_frame: int,
        channels: int,
        block_index: int,
    ):
        tokens = tokens.reshape(
            batch_size,
            num_frames * tokens_per_frame,
            channels,
        )
        positions = positions.reshape(
            batch_size,
            num_frames * tokens_per_frame,
            2,
        )
        intermediates = []
        for _ in range(self.aa_block_size):
            tokens = self.global_blocks[block_index](tokens, pos=positions)
            block_index += 1
            intermediates.append(
                tokens.reshape(
                    batch_size,
                    num_frames,
                    tokens_per_frame,
                    channels,
                )
            )
        return tokens, positions, block_index, intermediates

    def __call__(self, images):
        if images.ndim != 5 or images.shape[-1] != 3:
            raise ValueError(f"Aggregator expects [B,S,H,W,3], got {images.shape}")
        batch_size, num_frames, height, width, _ = images.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Image height and width must be divisible by patch_size")

        mean = mx.array([0.485, 0.456, 0.406], dtype=images.dtype)
        std = mx.array([0.229, 0.224, 0.225], dtype=images.dtype)
        normalized = (images - mean) / std
        normalized = normalized.reshape(
            batch_size * num_frames,
            height,
            width,
            3,
        )
        patch_tokens = self.patch_embed(normalized)

        camera_token = slice_expand_and_flatten(
            self.camera_token,
            batch_size,
            num_frames,
        )
        register_token = slice_expand_and_flatten(
            self.register_token,
            batch_size,
            num_frames,
        )
        tokens = mx.concatenate(
            (camera_token, register_token, patch_tokens),
            axis=1,
        )
        _, tokens_per_frame, channels = tokens.shape
        if self.patch_start_idx != 1 + self.config.num_register_tokens:
            raise RuntimeError("patch_start_idx does not match special-token count")

        patch_positions = self.position_getter(
            batch_size * num_frames,
            height // self.patch_size,
            width // self.patch_size,
        )
        patch_positions = patch_positions + 1
        special_positions = mx.zeros(
            (batch_size * num_frames, self.patch_start_idx, 2),
            dtype=patch_positions.dtype,
        )
        positions = mx.concatenate(
            (special_positions, patch_positions),
            axis=1,
        )

        frame_index = 0
        global_index = 0
        outputs = []
        for _ in range(self.aa_block_num):
            tokens, positions, frame_index, frame_intermediates = (
                self._process_frame_attention(
                    tokens,
                    positions,
                    batch_size,
                    num_frames,
                    tokens_per_frame,
                    channels,
                    frame_index,
                )
            )
            tokens, positions, global_index, global_intermediates = (
                self._process_global_attention(
                    tokens,
                    positions,
                    batch_size,
                    num_frames,
                    tokens_per_frame,
                    channels,
                    global_index,
                )
            )
            for frame_tokens, global_tokens in zip(
                frame_intermediates,
                global_intermediates,
            ):
                layer_index = len(outputs)
                if layer_index in self.cached_layer_indices:
                    outputs.append(
                        mx.concatenate((frame_tokens, global_tokens), axis=-1)
                    )
                else:
                    outputs.append(None)

        if frame_index != self.depth or global_index != self.depth:
            raise RuntimeError("Alternating-attention block count did not reach depth")
        return outputs, self.patch_start_idx
