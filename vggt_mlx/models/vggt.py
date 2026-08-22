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

    OUTPUT_ORDER = (
        "pose_enc",
        "depth",
        "depth_conf",
        "world_points",
        "world_points_conf",
        "extrinsic",
        "intrinsic",
    )
    OUTPUT_NAMES = frozenset(OUTPUT_ORDER)

    def forward(self, images, outputs=None):
        if images.ndim == 4:
            images = images[None]
        if images.ndim != 5 or images.shape[-1] != 3:
            raise ValueError(f"Expected [B,S,H,W,3] or [S,H,W,3], got {images.shape}")

        requested = self.OUTPUT_NAMES if outputs is None else frozenset(outputs)
        unknown = requested - self.OUTPUT_NAMES
        if unknown:
            raise ValueError(f"Unknown VGGT outputs: {sorted(unknown)}")
        if not requested:
            raise ValueError("At least one VGGT output must be requested")

        camera_outputs = {"pose_enc", "extrinsic", "intrinsic"}
        dense_outputs = {
            "depth",
            "depth_conf",
            "world_points",
            "world_points_conf",
        }
        cached_layers = (
            None
            if requested & dense_outputs
            else (self.aggregator.depth - 1,)
        )
        aggregated, patch_start_idx = self.aggregator(
            images, cached_layer_indices=cached_layers
        )
        prediction = {}
        if requested & camera_outputs:
            pose_enc = self.camera_head(aggregated)[-1]
            prediction["pose_enc"] = pose_enc
            if requested & {"extrinsic", "intrinsic"}:
                extrinsic, intrinsic = pose_encoding_to_extri_intri(
                    pose_enc,
                    image_size_hw=tuple(int(size) for size in images.shape[2:4]),
                )
                prediction["extrinsic"] = extrinsic
                prediction["intrinsic"] = intrinsic
        if requested & {"depth", "depth_conf"}:
            depth, depth_conf = self.depth_head(
                aggregated, images, patch_start_idx
            )
            prediction["depth"] = depth
            prediction["depth_conf"] = depth_conf
        if requested & {"world_points", "world_points_conf"}:
            world_points, world_points_conf = self.point_head(
                aggregated, images, patch_start_idx
            )
            prediction["world_points"] = world_points
            prediction["world_points_conf"] = world_points_conf
        return {
            name: prediction[name]
            for name in self.OUTPUT_ORDER
            if name in requested
        }

    def __call__(self, images, outputs=None):
        return self.forward(images, outputs=outputs)
