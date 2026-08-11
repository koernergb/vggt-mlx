"""Iterative VGGT camera-pose regression head."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from vggt_mlx.layers.block import Block
from vggt_mlx.layers.mlp import Mlp


class CameraHead(nn.Module):
    """Regress a 9-D translation/quaternion/FoV encoding."""

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",
    ) -> None:
        super().__init__()
        if pose_encoding_type != "absT_quaR_FoV":
            raise ValueError(f"Unsupported camera encoding: {pose_encoding_type}")

        self.target_dim = 9
        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth
        # Plain lists retain upstream numeric state-dict paths.
        self.trunk = [
            Block(
                dim=dim_in,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                init_values=init_values,
                qk_norm=False,
            )
            for _ in range(trunk_depth)
        ]
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)
        self.empty_pose_tokens = mx.zeros((1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)
        self.poseLN_modulation = [nn.SiLU(), nn.Linear(dim_in, 3 * dim_in)]
        self.adaln_norm = nn.LayerNorm(dim_in, eps=1e-6, affine=False)
        self.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_in // 2,
            out_features=self.target_dim,
        )

    def _activate_pose(self, pose):
        translation, quaternion, fov = pose[..., :3], pose[..., 3:7], pose[..., 7:]

        def activate(value, kind: str):
            if kind == "linear":
                return value
            if kind == "relu":
                return nn.relu(value)
            if kind == "exp":
                return mx.exp(value)
            if kind == "inv_log":
                return mx.sign(value) * mx.expm1(mx.abs(value))
            raise ValueError(f"Unsupported pose activation: {kind}")

        return mx.concatenate(
            (
                activate(translation, self.trans_act),
                activate(quaternion, self.quat_act),
                activate(fov, self.fl_act),
            ),
            axis=-1,
        )

    def __call__(self, aggregated_tokens_list, num_iterations: int = 4):
        tokens = aggregated_tokens_list[-1]
        if tokens is None:
            raise ValueError("Camera head requires the final aggregator output")
        pose_tokens = self.token_norm(tokens[:, :, 0])
        batch, frames, _ = pose_tokens.shape
        predicted = None
        predictions = []

        for _ in range(num_iterations):
            if predicted is None:
                pose_input = mx.broadcast_to(
                    self.empty_pose_tokens,
                    (batch, frames, self.target_dim),
                )
            else:
                pose_input = mx.stop_gradient(predicted)
            module_input = self.embed_pose(pose_input)
            modulation = module_input
            for layer in self.poseLN_modulation:
                modulation = layer(modulation)
            shift, scale, gate = mx.split(modulation, 3, axis=-1)
            trunk_tokens = gate * (
                self.adaln_norm(pose_tokens) * (1.0 + scale) + shift
            ) + pose_tokens
            for block in self.trunk:
                trunk_tokens = block(trunk_tokens)
            delta = self.pose_branch(self.trunk_norm(trunk_tokens))
            predicted = delta if predicted is None else predicted + delta
            predictions.append(self._activate_pose(predicted))

        return predictions
