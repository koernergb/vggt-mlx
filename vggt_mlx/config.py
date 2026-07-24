"""Configuration for the canonical VGGT-1B MLX port."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class VGGTConfig:
    """Architecture settings verified against upstream ``facebook/VGGT-1B``."""

    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0

    num_register_tokens: int = 4
    patch_start_idx: int = 5

    rope_freq: int = 100
    aa_order: Tuple[str, ...] = ("frame", "global")
    aa_block_size: int = 1
    qk_norm: bool = True
    layerscale_init: float = 0.01
    intermediate_layer_idx: Tuple[int, ...] = (4, 11, 17, 23)

    dpt_features: int = 256
    dpt_out_channels: Tuple[int, ...] = (256, 512, 1024, 1024)

    pose_encoding_type: str = "absT_quaR_FoV"
    pose_dim: int = 9
