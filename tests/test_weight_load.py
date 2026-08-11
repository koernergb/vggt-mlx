"""Strict checkpoint-loading gate.

This test becomes active after Card 1.2's ignored local weight artifact exists.
It is intentionally retained while model modules land so any key or shape
drift is caught immediately.
"""

from pathlib import Path

import pytest


WEIGHTS_PATH = (
    Path(__file__).resolve().parents[1] / "weights" / "vggt-1b-mlx.safetensors"
)


def test_converted_weights_load_strictly():
    if not WEIGHTS_PATH.is_file():
        pytest.skip(
            "Run `python -m vggt_mlx.convert.torch_to_mlx` to generate local weights"
        )

    mx = pytest.importorskip("mlx.core")

    mx.set_default_device(mx.cpu)
    weights = list(mx.load(str(WEIGHTS_PATH)).items())

    assert weights, f"Converted checkpoint is empty: {WEIGHTS_PATH}"
    assert all(not name.startswith("track_head.") for name, _ in weights)

    from vggt_mlx.models.vggt import VGGT

    try:
        model = VGGT()
    except NotImplementedError:
        from vggt_mlx.models.aggregator import Aggregator

        model = Aggregator()
        aggregator_weights = [
            (name.removeprefix("aggregator."), value)
            for name, value in weights
            if name.startswith("aggregator.")
        ]
        assert aggregator_weights
        model.load_weights(aggregator_weights, strict=True)
    else:
        model.load_weights(weights, strict=True)
