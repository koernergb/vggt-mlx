import numpy as np
import pytest


mx = pytest.importorskip("mlx.core")

from vggt_mlx.heads.camera_head import CameraHead


def test_camera_refinement_shapes_fov_and_determinism():
    model = CameraHead(dim_in=8, trunk_depth=2, num_heads=2, mlp_ratio=2)
    model.eval()
    aggregated = [None, mx.zeros((1, 2, 3, 8))]
    first = model(aggregated)
    second = model(aggregated)
    mx.eval(*first, *second)
    assert len(first) == 4
    assert all(output.shape == (1, 2, 9) for output in first)
    assert all(np.all(np.asarray(output)[..., 7:] >= 0) for output in first)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)


def test_camera_head_validates_encoding_and_final_tokens():
    with pytest.raises(ValueError, match="Unsupported camera encoding"):
        CameraHead(dim_in=8, num_heads=2, pose_encoding_type="unknown")
    model = CameraHead(dim_in=8, trunk_depth=1, num_heads=2)
    with pytest.raises(ValueError, match="final aggregator"):
        model([None])
