"""Verify the equivalent PyTorch-MPS workload against a committed oracle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vggt_mlx.benchmark.adapters import PyTorchMPSAdapter
from vggt_mlx.benchmark.parity import compare_arrays


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "oracle_1view.npz"
DEFAULT_SOURCE = Path.home() / (
    ".cache/huggingface/hub/models--facebook--VGGT-1B/snapshots/"
    "860abec7937da0a4c03c41d3c269c366e82abdf9/model.safetensors"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Official checkpoint not found: {args.checkpoint}")
    if not args.fixture.is_file():
        raise SystemExit(f"Oracle fixture not found: {args.fixture}")
    with np.load(args.fixture) as fixture:
        shared = fixture["input"].transpose(0, 1, 3, 4, 2).astype(
            np.float32, copy=False
        )
        expected = {
            name: fixture[name].astype(np.float32, copy=False)
            for name in ("pose_enc", "depth", "depth_conf", "extrinsic", "intrinsic")
        }

    adapter = PyTorchMPSAdapter.from_local_safetensors(
        args.checkpoint, device=args.device
    )
    output = adapter.to_numpy(
        adapter.forward_tensors(adapter.prepare_input(shared))
    )
    metrics = {
        name: compare_arrays(expected[name], output[name])
        for name in expected
    }
    for name, value in metrics.items():
        print(
            f"{name:12s} max_abs={value.max_abs:.8g} "
            f"mean_abs={value.mean_abs:.8g} rel_fro={value.rel_fro:.8g} "
            f"cosine={value.cosine:.10f}"
        )

    if metrics["pose_enc"].max_abs >= 1e-3:
        raise SystemExit("MPS pose reference exceeds the 1e-3 baseline gate")
    if metrics["depth"].max_abs >= 1e-2:
        raise SystemExit("MPS depth reference exceeds the 1e-2 baseline gate")
    print("PyTorch-MPS reference workload verified")


if __name__ == "__main__":
    main()
