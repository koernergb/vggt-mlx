"""Profile VGGT-MLX inference stages on one shared scene."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import median

import mlx.core as mx

from vggt_mlx.models.vggt import VGGT
from vggt_mlx.utils.load_fn import load_and_preprocess_images
from vggt_mlx.utils.pose_enc import pose_encoding_to_extri_intri


ROOT = Path(__file__).resolve().parents[1]


def timed(function, *, trials: int):
    samples = []
    value = None
    for _ in range(trials):
        started = time.perf_counter()
        value = function()
        mx.eval(value)
        samples.append((time.perf_counter() - started) * 1000)
    return value, samples


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--weights", type=Path, default=ROOT / "weights" / "vggt-1b-mlx.safetensors"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    images = load_and_preprocess_images(args.images)
    if images.ndim == 4:
        images = images[None]
    model = VGGT()
    model.load_weights(list(mx.load(str(args.weights)).items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    forward = mx.compile(model, inputs=model.state) if args.compile else model

    # Untimed warmup establishes kernels and memory allocations.
    warmup = forward(images)
    mx.eval(warmup)

    aggregated, aggregator_ms = timed(
        lambda: model.aggregator(images), trials=args.trials
    )
    tokens, patch_start = aggregated
    _, camera_ms = timed(lambda: model.camera_head(tokens)[-1], trials=args.trials)
    _, depth_ms = timed(
        lambda: model.depth_head(tokens, images, patch_start), trials=args.trials
    )
    _, point_ms = timed(
        lambda: model.point_head(tokens, images, patch_start), trials=args.trials
    )
    pose = model.camera_head(tokens)[-1]
    _, geometry_ms = timed(
        lambda: pose_encoding_to_extri_intri(pose, images.shape[2:4]),
        trials=args.trials,
    )
    _, full_ms = timed(lambda: forward(images), trials=args.trials)

    stages = {
        "aggregator": aggregator_ms,
        "camera_head": camera_ms,
        "depth_head": depth_ms,
        "point_head": point_ms,
        "camera_geometry": geometry_ms,
        "full_forward": full_ms,
    }
    for name, samples in stages.items():
        print(
            f"{name:16s} median={median(samples):8.2f} ms "
            f"samples={','.join(f'{value:.2f}' for value in samples)}"
        )


if __name__ == "__main__":
    main()
