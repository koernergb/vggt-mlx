from pathlib import Path

import numpy as np
import pytest


FIXTURE_PATHS = (
    Path(__file__).parent / "fixtures" / "oracle_1view.npz",
    Path(__file__).parent / "fixtures" / "oracle_3view.npz",
)


@pytest.mark.parametrize("fixture_path", FIXTURE_PATHS)
def test_pose_encoding_matches_oracle_cameras(fixture_path):
    mx = pytest.importorskip("mlx.core")
    from vggt_mlx.utils.pose_enc import pose_encoding_to_extri_intri

    with np.load(fixture_path) as fixture:
        pose = fixture["pose_enc"].astype(np.float32)
        expected_extrinsic = fixture["extrinsic"]
        expected_intrinsic = fixture["intrinsic"]
        height, width = fixture["input"].shape[-2:]

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        mx.array(pose), image_size_hw=(height, width)
    )
    mx.eval(extrinsic, intrinsic)
    np.testing.assert_allclose(extrinsic, expected_extrinsic, atol=1e-4, rtol=0)
    np.testing.assert_allclose(intrinsic, expected_intrinsic, atol=1e-4, rtol=0)
