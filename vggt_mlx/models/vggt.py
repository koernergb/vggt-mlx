"""Top-level MLX assembly for VGGT inference."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from vggt_mlx.heads.camera_head import CameraHead
from vggt_mlx.heads.dpt_head import DPTHead
from vggt_mlx.models.aggregator import Aggregator
from vggt_mlx.utils.pose_enc import pose_encoding_to_extri_intri


class VGGT(nn.Module):
    """VGGT-1B without the intentionally out-of-scope tracking head."""

    def __init__(self) -> None:
        super().__init__()
        self.aggregator = Aggregator()
        self.camera_head = CameraHead()
        self.depth_head = DPTHead(output_dim=2, activation="exp")
        self.point_head = DPTHead(output_dim=4, activation="inv_log")

    def forward(self, images):
        if images.ndim == 4:
            images = images[None]
        if images.ndim != 5 or images.shape[-1] != 3:
            raise ValueError(f"Expected [B,S,H,W,3] or [S,H,W,3], got {images.shape}")

        aggregated, patch_start_idx = self.aggregator(images)
        pose_enc = self.camera_head(aggregated)[-1]
        depth, depth_conf = self.depth_head(
            aggregated, images, patch_start_idx
        )
        world_points, world_points_conf = self.point_head(
            aggregated, images, patch_start_idx
        )
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc, image_size_hw=tuple(int(size) for size in images.shape[2:4])
        )
        return {
            "pose_enc": pose_enc,
            "depth": depth,
            "depth_conf": depth_conf,
            "world_points": world_points,
            "world_points_conf": world_points_conf,
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
        }

    def __call__(self, images):
        return self.forward(images)
