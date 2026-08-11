"""End-to-end parity gate for VGGT's camera and depth heads."""

from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = ROOT / "weights" / "vggt-1b-mlx.safetensors"
FIXTURE_PATHS = (
    ROOT / "tests" / "fixtures" / "oracle_1view.npz",
    ROOT / "tests" / "fixtures" / "oracle_3view.npz",
)


def _weights(checkpoint, prefix):
    return [
        (name.removeprefix(prefix), value)
        for name, value in checkpoint.items()
        if name.startswith(prefix)
    ]


@pytest.mark.parametrize("fixture_path", FIXTURE_PATHS, ids=lambda path: path.stem)
def test_camera_and_depth_heads_match_oracles(fixture_path):
    if not WEIGHTS_PATH.is_file():
        pytest.skip("Converted VGGT weights are required for head parity")
    mx = pytest.importorskip("mlx.core")
    from vggt_mlx.heads.camera_head import CameraHead
    from vggt_mlx.heads.dpt_head import DPTHead

    mx.set_default_device(mx.cpu)
    checkpoint = mx.load(str(WEIGHTS_PATH))
    camera = CameraHead()
    depth = DPTHead(output_dim=2, activation="exp")
    camera.load_weights(_weights(checkpoint, "camera_head."), strict=True)
    depth.load_weights(_weights(checkpoint, "depth_head."), strict=True)
    camera.eval()
    depth.eval()

    with np.load(fixture_path) as fixture:
        images = fixture["input"].astype(np.float32)
        aggregated = [None] * 24
        for layer in (4, 11, 17, 23):
            aggregated[layer] = mx.array(fixture[f"agg_l{layer}"])
        expected_pose = fixture["pose_enc"].astype(np.float32)
        expected_depth = fixture["depth"].astype(np.float32)
        expected_confidence = fixture["depth_conf"].astype(np.float32)

    mlx_images = mx.array(images.transpose(0, 1, 3, 4, 2))
    actual_pose = camera(aggregated)[-1]
    actual_depth, actual_confidence = depth(aggregated, mlx_images, 5)
    mx.eval(actual_pose, actual_depth, actual_confidence)
    actual_pose = np.asarray(actual_pose)
    actual_depth = np.asarray(actual_depth)
    actual_confidence = np.asarray(actual_confidence)

    assert np.max(np.abs(actual_pose - expected_pose)) < 1e-3
    assert np.max(np.abs(actual_depth - expected_depth)) < 1e-2
    assert np.isfinite(actual_confidence).all()
    correlation = np.corrcoef(
        actual_confidence.reshape(-1), expected_confidence.reshape(-1)
    )[0, 1]
    assert correlation > 0.99
