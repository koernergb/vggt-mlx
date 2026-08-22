"""Compare an optimized MLX mode with the compiled fp32 reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from vggt_mlx.benchmark.adapters import MLXAdapter
from vggt_mlx.benchmark.parity import compare_arrays
from vggt_mlx.models.vggt import VGGT


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantize", type=int, choices=(4, 6, 8), required=True)
    parser.add_argument(
        "--weights", type=Path, default=ROOT / "weights" / "vggt-1b-mlx.safetensors"
    )
    parser.add_argument(
        "--oracle", type=Path, default=ROOT / "tests" / "fixtures" / "oracle_1view.npz"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with np.load(args.oracle) as fixture:
        images = fixture["input"].transpose(0, 1, 3, 4, 2).astype(np.float32)

    model = VGGT()
    model.load_weights(list(mx.load(str(args.weights)).items()), strict=True)
    model.eval()
    reference_adapter = MLXAdapter(model, compile=True)
    prepared = reference_adapter.prepare_input(images)
    reference = reference_adapter.to_numpy(
        reference_adapter.forward_tensors(prepared)
    )

    nn.quantize(
        model,
        group_size=64,
        bits=args.quantize,
        class_predicate=lambda _path, module: (
            isinstance(module, nn.Linear) and module.weight.shape[-1] % 64 == 0
        ),
    )
    optimized_adapter = MLXAdapter(model, compile=True)
    candidate = optimized_adapter.to_numpy(
        optimized_adapter.forward_tensors(prepared)
    )
    failed = False
    for name in reference:
        metrics = compare_arrays(reference[name], candidate[name])
        print(
            f"{name:22s} max_abs={metrics.max_abs:.8g} "
            f"mean_abs={metrics.mean_abs:.8g} rel_fro={metrics.rel_fro:.8g} "
            f"cosine={metrics.cosine:.10f}"
        )
        if not np.isfinite(candidate[name]).all():
            failed = True
    if failed:
        raise SystemExit("optimized mode produced non-finite outputs")


if __name__ == "__main__":
    main()
