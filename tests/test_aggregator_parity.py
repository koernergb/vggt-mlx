"""CPU/FP32 parity gate for alternating frame/global aggregation."""

from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = REPOSITORY_ROOT / "weights" / "vggt-1b-mlx.safetensors"
FIXTURE_PATHS = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "oracle_1view.npz",
    REPOSITORY_ROOT / "tests" / "fixtures" / "oracle_3view.npz",
)
AGGREGATOR_PREFIX = "aggregator."
LAYER_INDICES = (4, 11, 17, 23)
MAX_ABS_DIFF = 1e-3


def extract_aggregator_weights(weights):
    return [
        (name.removeprefix(AGGREGATOR_PREFIX), value)
        for name, value in weights.items()
        if name.startswith(AGGREGATOR_PREFIX)
    ]


def test_aggregator_matches_one_and_three_view_oracles():
    missing = [path for path in (*FIXTURE_PATHS, WEIGHTS_PATH) if not path.is_file()]
    if missing:
        pytest.skip(
            "Card 3.5 requires generated oracle fixtures and converted weights; "
            f"missing: {missing}"
        )

    mx = pytest.importorskip("mlx.core")

    from vggt_mlx.models.aggregator import Aggregator

    mx.set_default_device(mx.cpu)
    checkpoint = mx.load(str(WEIGHTS_PATH))
    aggregator_weights = extract_aggregator_weights(checkpoint)
    assert aggregator_weights, "Converted checkpoint contains no aggregator weights"

    model = Aggregator()
    model.load_weights(aggregator_weights, strict=True)
    model.eval()

    for fixture_path in FIXTURE_PATHS:
        with np.load(fixture_path) as fixture:
            required = {"input", *(f"agg_l{layer}" for layer in LAYER_INDICES)}
            assert required <= set(fixture.files)
            pytorch_input = fixture["input"].astype(np.float32, copy=False)
            expected = {
                layer: fixture[f"agg_l{layer}"].astype(np.float32, copy=False)
                for layer in LAYER_INDICES
            }

        assert pytorch_input.ndim == 5, (
            f"{fixture_path.name} input must be [B,S,C,H,W], "
            f"got {pytorch_input.shape}"
        )
        mlx_input = mx.array(pytorch_input.transpose(0, 1, 3, 4, 2))
        outputs, patch_start_idx = model(mlx_input)
        actual_outputs = [outputs[layer] for layer in LAYER_INDICES]
        assert all(output is not None for output in actual_outputs)
        mx.eval(*actual_outputs)

        assert patch_start_idx == 5
        for layer, actual in zip(LAYER_INDICES, actual_outputs):
            actual_array = np.asarray(actual)
            expected_array = expected[layer]
            assert actual_array.shape == expected_array.shape, (
                f"{fixture_path.name} layer {layer}: MLX shape "
                f"{actual_array.shape} != PyTorch shape {expected_array.shape}"
            )
            max_abs_diff = float(
                np.max(np.abs(actual_array - expected_array))
            )
            assert max_abs_diff < MAX_ABS_DIFF, (
                f"{fixture_path.name} layer {layer}: "
                f"max_abs_diff={max_abs_diff:.8g}, required < {MAX_ABS_DIFF}"
            )
