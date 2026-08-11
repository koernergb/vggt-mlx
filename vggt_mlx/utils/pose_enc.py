"""Conversions between VGGT's compact pose encoding and camera matrices."""

import mlx.core as mx


def _quaternion_xyzw_to_matrix(quaternion):
    x, y, z, w = [quaternion[..., index] for index in range(4)]
    two_s = 2.0 / mx.sum(quaternion * quaternion, axis=-1)
    entries = (
        1.0 - two_s * (y * y + z * z),
        two_s * (x * y - z * w),
        two_s * (x * z + y * w),
        two_s * (x * y + z * w),
        1.0 - two_s * (x * x + z * z),
        two_s * (y * z - x * w),
        two_s * (x * z - y * w),
        two_s * (y * z + x * w),
        1.0 - two_s * (x * x + y * y),
    )
    return mx.stack(entries, axis=-1).reshape(*quaternion.shape[:-1], 3, 3)


def pose_encoding_to_extri_intri(
    pose_encoding,
    image_size_hw=None,
    pose_encoding_type: str = "absT_quaR_FoV",
    build_intrinsics: bool = True,
    *,
    image_hw=None,
):
    """Decode XYZW pose vectors into OpenCV world-to-camera matrices."""
    if pose_encoding_type != "absT_quaR_FoV":
        raise NotImplementedError(pose_encoding_type)
    if image_size_hw is None:
        image_size_hw = image_hw
    if image_size_hw is None:
        raise ValueError("image_size_hw is required")

    translation = pose_encoding[..., :3]
    rotation = _quaternion_xyzw_to_matrix(pose_encoding[..., 3:7])
    extrinsic = mx.concatenate((rotation, translation[..., None]), axis=-1)
    if not build_intrinsics:
        return extrinsic, None

    height, width = image_size_hw
    fy = (height / 2.0) / mx.tan(pose_encoding[..., 7] / 2.0)
    fx = (width / 2.0) / mx.tan(pose_encoding[..., 8] / 2.0)
    zeros = mx.zeros_like(fx)
    ones = mx.ones_like(fx)
    intrinsic = mx.stack(
        (
            fx,
            zeros,
            mx.full_like(fx, width / 2.0),
            zeros,
            fy,
            mx.full_like(fy, height / 2.0),
            zeros,
            zeros,
            ones,
        ),
        axis=-1,
    ).reshape(*pose_encoding.shape[:-1], 3, 3)
    return extrinsic, intrinsic
