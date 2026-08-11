"""Full-model one-view parity gate."""

from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "vggt-1b-mlx.safetensors"
ORACLES = (
    ROOT / "tests" / "fixtures" / "oracle_1view.npz",
    ROOT / "tests" / "fixtures" / "oracle_3view.npz",
)


@pytest.mark.parametrize("oracle", ORACLES, ids=lambda path: path.stem)
def test_full_forward_matches_oracle_outputs(oracle):
    if not WEIGHTS.is_file():
        pytest.skip("Converted weights are required for end-to-end parity")
    mx = pytest.importorskip("mlx.core")
    from vggt_mlx.models.vggt import VGGT

    mx.set_default_device(mx.cpu)
    model = VGGT()
    model.load_weights(list(mx.load(str(WEIGHTS)).items()), strict=True)
    model.eval()
    with np.load(oracle) as fixture:
        images = fixture["input"].transpose(0, 1, 3, 4, 2).astype(np.float32)
        expected_depth = fixture["depth"].astype(np.float32)
        expected_pose = fixture["pose_enc"].astype(np.float32)
        expected_extrinsic = fixture["extrinsic"].astype(np.float32)
        expected_intrinsic = fixture["intrinsic"].astype(np.float32)
    output = model(mx.array(images))
    mx.eval(*output.values())
    assert set(output) == {
        "pose_enc", "depth", "depth_conf", "world_points",
        "world_points_conf", "extrinsic", "intrinsic",
    }
    assert np.max(np.abs(np.asarray(output["depth"]) - expected_depth)) < 1e-2
    assert np.max(np.abs(np.asarray(output["pose_enc"]) - expected_pose)) < 1e-3
    np.testing.assert_allclose(output["extrinsic"], expected_extrinsic, atol=1e-3, rtol=0)
    # The T4-to-MLX FoV drift is amplified slightly by focal = size/tan(FoV/2).
    np.testing.assert_allclose(output["intrinsic"], expected_intrinsic, atol=3e-3, rtol=0)
    batch, views, height, width, _ = images.shape
    assert output["world_points"].shape == (batch, views, height, width, 3)
    assert output["world_points_conf"].shape == (batch, views, height, width)
    assert all(np.isfinite(np.asarray(value)).all() for value in output.values())
