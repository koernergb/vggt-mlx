"""Full-model one-view parity gate."""

from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "vggt-1b-mlx.safetensors"
ORACLE = ROOT / "tests" / "fixtures" / "oracle_1view.npz"


def test_full_forward_matches_oracle_depth():
    if not WEIGHTS.is_file():
        pytest.skip("Converted weights are required for end-to-end parity")
    mx = pytest.importorskip("mlx.core")
    from vggt_mlx.models.vggt import VGGT

    mx.set_default_device(mx.cpu)
    model = VGGT()
    model.load_weights(list(mx.load(str(WEIGHTS)).items()), strict=True)
    model.eval()
    with np.load(ORACLE) as fixture:
        images = fixture["input"].transpose(0, 1, 3, 4, 2).astype(np.float32)
        expected_depth = fixture["depth"].astype(np.float32)
    output = model(mx.array(images))
    mx.eval(*output.values())
    assert set(output) == {
        "pose_enc", "depth", "depth_conf", "world_points",
        "world_points_conf", "extrinsic", "intrinsic",
    }
    assert np.max(np.abs(np.asarray(output["depth"]) - expected_depth)) < 1e-2
