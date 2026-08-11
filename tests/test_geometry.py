import numpy as np
import pytest


mx = pytest.importorskip("mlx.core")

from vggt_mlx.utils.geometry import unproject_depth_map_to_point_map


def _camera(translation=(0.0, 0.0, 0.0)):
    extrinsic = np.array(
        [[1, 0, 0, translation[0]], [0, 1, 0, translation[1]], [0, 0, 1, translation[2]]],
        dtype=np.float32,
    )[None, None]
    intrinsic = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)[None, None]
    return mx.array(extrinsic), mx.array(intrinsic)


def test_identity_camera_unprojects_pixel_grid():
    depth = mx.ones((1, 1, 2, 2, 1))
    extrinsic, intrinsic = _camera()
    points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
    mx.eval(points)
    np.testing.assert_allclose(
        points,
        [[[[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 1]]]]],
        atol=1e-6,
    )


def test_world_to_camera_translation_is_inverted():
    extrinsic, intrinsic = _camera((1.0, 2.0, 3.0))
    points = unproject_depth_map_to_point_map(mx.ones((1, 1, 1, 1)), extrinsic, intrinsic)
    mx.eval(points)
    np.testing.assert_allclose(points, [[[[[-1, -2, -2]]]]], atol=1e-6)


def test_unprojection_rejects_bad_shapes():
    extrinsic, intrinsic = _camera()
    with pytest.raises(ValueError, match="depth"):
        unproject_depth_map_to_point_map(mx.ones((1, 2, 2)), extrinsic, intrinsic)
    with pytest.raises(ValueError, match="extrinsic"):
        unproject_depth_map_to_point_map(mx.ones((1, 1, 2, 2)), mx.zeros((1, 3, 4)), intrinsic)
