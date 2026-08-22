"""Requested output groups must skip unused heads without changing results."""

from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "vggt-1b-mlx.safetensors"
ORACLE = ROOT / "tests" / "fixtures" / "oracle_1view.npz"


def test_depth_camera_selection_matches_full_forward():
    if not WEIGHTS.is_file():
        pytest.skip("converted weights are required for output-selection parity")
    mx = pytest.importorskip("mlx.core")
    from vggt_mlx.models.vggt import VGGT

    mx.set_default_device(mx.cpu)
    model = VGGT()
    model.load_weights(list(mx.load(str(WEIGHTS)).items()), strict=True)
    model.eval()
    with np.load(ORACLE) as fixture:
        images = mx.array(
            fixture["input"].transpose(0, 1, 3, 4, 2).astype(np.float32)
        )
    full = model(images)
    selected_names = {"pose_enc", "depth", "depth_conf", "extrinsic", "intrinsic"}
    selected = model(images, outputs=selected_names)
    mx.eval(full, selected)
    assert set(selected) == selected_names
    for name in selected_names:
        np.testing.assert_array_equal(selected[name], full[name])


def test_output_selection_rejects_empty_or_unknown_requests():
    mx = pytest.importorskip("mlx.core")
    from vggt_mlx.models.vggt import VGGT

    images = mx.zeros((1, 1, 14, 14, 3))
    model = VGGT()
    with pytest.raises(ValueError, match="At least one"):
        model(images, outputs=())
    with pytest.raises(ValueError, match="Unknown"):
        model(images, outputs=("tracks",))
