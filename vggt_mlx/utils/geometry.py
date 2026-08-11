"""Camera geometry helpers for dense VGGT predictions."""

import mlx.core as mx


def unproject_depth_map_to_point_map(depth, extrinsic, intrinsic):
    """Unproject depth into world coordinates for batched OpenCV cameras."""
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 4:
        raise ValueError(f"Expected depth [B,S,H,W], got {depth.shape}")
    batch, frames, height, width = depth.shape
    if extrinsic.shape != (batch, frames, 3, 4):
        raise ValueError(f"Unexpected extrinsic shape: {extrinsic.shape}")
    if intrinsic.shape != (batch, frames, 3, 3):
        raise ValueError(f"Unexpected intrinsic shape: {intrinsic.shape}")

    u = mx.arange(width, dtype=depth.dtype)[None, None, None, :]
    v = mx.arange(height, dtype=depth.dtype)[None, None, :, None]
    x = (u - intrinsic[..., 0, 2, None, None]) * depth / intrinsic[
        ..., 0, 0, None, None
    ]
    y = (v - intrinsic[..., 1, 2, None, None]) * depth / intrinsic[
        ..., 1, 1, None, None
    ]
    camera_points = mx.stack((x, y, depth), axis=-1)

    rotation = extrinsic[..., :3, :3]
    translation = extrinsic[..., :3, 3]
    inverse_rotation = rotation.swapaxes(-1, -2)
    inverse_translation = -mx.matmul(
        inverse_rotation, translation[..., None]
    )[..., 0]
    return mx.einsum(
        "bsij,bshwj->bshwi", inverse_rotation, camera_points
    ) + inverse_translation[..., None, None, :]
