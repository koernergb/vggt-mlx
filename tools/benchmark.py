"""Run resumable, same-input VGGT benchmarks on Apple Silicon."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np

from vggt_mlx.benchmark.adapters import MLXAdapter, PyTorchMPSAdapter
from vggt_mlx.benchmark.runner import (
    build_benchmark_result,
    existing_run_ids,
    planned_run_id,
    run_trials,
    write_result,
)
from vggt_mlx.models.vggt import VGGT
from vggt_mlx.utils.load_fn import load_and_preprocess_images


ROOT = Path(__file__).resolve().parents[1]
REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"
DEFAULT_MLX_WEIGHTS = ROOT / "weights" / "vggt-1b-mlx.safetensors"
DEFAULT_TORCH_WEIGHTS = Path.home() / (
    ".cache/huggingface/hub/models--facebook--VGGT-1B/snapshots/"
    f"{REVISION}/model.safetensors"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument(
        "--framework", choices=("mlx", "mps", "both"), default="both"
    )
    parser.add_argument("--views", nargs="+", type=int, default=(1, 2, 3, 4))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--mlx-weights", type=Path, default=DEFAULT_MLX_WEIGHTS)
    parser.add_argument("--torch-weights", type=Path, default=DEFAULT_TORCH_WEIGHTS)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "benchmarks")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    if args.warmups < 0 or args.trials < 1:
        parser.error("warmups must be non-negative and trials must be positive")
    if any(view < 1 for view in args.views):
        parser.error("view counts must be positive")
    if max(args.views) > len(args.images):
        parser.error("provide at least as many images as the largest view count")
    return args


def load_adapter(name: str, args: argparse.Namespace):
    if name == "mlx":
        import mlx.core as mx

        if not args.mlx_weights.is_file():
            raise SystemExit(f"Converted MLX weights not found: {args.mlx_weights}")
        model = VGGT()
        model.load_weights(list(mx.load(str(args.mlx_weights)).items()), strict=True)
        return MLXAdapter(model, compile=not args.no_compile)
    if not args.torch_weights.is_file():
        raise SystemExit(f"Official PyTorch weights not found: {args.torch_weights}")
    return PyTorchMPSAdapter.from_local_safetensors(args.torch_weights)


def main() -> None:
    args = parse_args()
    frameworks = ("mlx", "mps") if args.framework == "both" else (args.framework,)
    completed = existing_run_ids(args.results)
    for view_count in args.views:
        # Preprocess once; both frameworks receive byte-identical float32 pixels.
        shared = np.asarray(load_and_preprocess_images(args.images[:view_count]))
        if shared.ndim == 4:
            shared = shared[None]
        shared = shared.astype(np.float32, copy=False)
        for framework in frameworks:
            framework_name = "mlx" if framework == "mlx" else "pytorch-mps"
            run_id = planned_run_id(
                repository=ROOT,
                framework=framework_name,
                precision="fp32",
                shared_input=shared,
                checkpoint_revision=REVISION,
            )
            print(
                f"{framework_name:11s} fp32 views={view_count} "
                f"shape={tuple(shared.shape)} run={run_id}"
            )
            if run_id in completed and not args.replace:
                print("  skip: valid result already exists")
                continue
            if args.dry_run:
                continue

            adapter = load_adapter(framework, args)
            prepared = adapter.prepare_input(shared)
            samples, _ = run_trials(
                adapter,
                prepared,
                warmups=args.warmups,
                trials=args.trials,
            )
            result = build_benchmark_result(
                repository=ROOT,
                adapter=adapter,
                shared_input=shared,
                checkpoint_revision=REVISION,
                warmups=args.warmups,
                samples_ms=samples,
                peak_memory_mb=adapter.last_peak_memory_mb,
            )
            if result["run_id"] != run_id:
                raise RuntimeError("planned and completed benchmark run IDs disagree")
            path = write_result(args.results, result)
            print(
                f"  median={result['summary']['median_ms']:.1f} ms "
                f"stable={result['validity']['thermally_stable']} -> {path}"
            )
            del prepared, adapter
            gc.collect()


if __name__ == "__main__":
    main()
