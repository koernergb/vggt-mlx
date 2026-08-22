"""Compiled inference must preserve the immutable fp32 reference."""

from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "vggt-1b-mlx.safetensors"
ORACLE = ROOT / "tests" / "fixtures" / "oracle_1view.npz"


def test_compiled_full_forward_matches_uncompiled_fp32():
    if not WEIGHTS.is_file():
        pytest.skip("converted weights are required for compiled parity")
    mx = pytest.importorskip("mlx.core")
    from vggt_mlx.benchmark.adapters import MLXAdapter
    from vggt_mlx.benchmark.parity import compare_arrays
    from vggt_mlx.models.vggt import VGGT

    mx.set_default_device(mx.cpu)
    model = VGGT()
    model.load_weights(list(mx.load(str(WEIGHTS)).items()), strict=True)
    model.eval()
    with np.load(ORACLE) as fixture:
        images = fixture["input"].transpose(0, 1, 3, 4, 2).astype(np.float32)

    expected = model(mx.array(images))
    mx.eval(expected)
    adapter = MLXAdapter(model, compile=True)
    actual = adapter.forward_tensors(adapter.prepare_input(images))
    adapter.evaluate(actual)
    max_abs_gates = {
        "pose_enc": 1e-3,
        "depth": 1e-2,
        "depth_conf": 1e-2,
        "world_points": 1e-2,
        "world_points_conf": 1e-2,
        "extrinsic": 1e-3,
        "intrinsic": 3e-3,
    }
    for name in expected:
        metrics = compare_arrays(np.asarray(expected[name]), np.asarray(actual[name]))
        assert metrics.max_abs < max_abs_gates[name], (name, metrics)
        assert metrics.rel_fro < 1e-4, (name, metrics)
        assert metrics.cosine > 0.999999, (name, metrics)
