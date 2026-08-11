"""One-command VGGT-MLX depth and point-cloud demo."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from vggt_mlx.models.vggt import VGGT
from vggt_mlx.utils.geometry import unproject_depth_map_to_point_map
from vggt_mlx.utils.load_fn import load_and_preprocess_images


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "weights" / "vggt-1b-mlx.safetensors"


def ensure_weights(path: Path) -> None:
    if path.is_file():
        return
    print("Converted weights not found; downloading and converting facebook/VGGT-1B...")
    try:
        subprocess.run(
            [sys.executable, "-m", "vggt_mlx.convert.torch_to_mlx", "--output", str(path)],
            cwd=ROOT,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            "Automatic conversion requires the dev dependencies and upstream VGGT. "
            "See the README weight-conversion instructions."
        ) from error


def save_depth(depth: np.ndarray, path: Path) -> None:
    finite = depth[np.isfinite(depth)]
    low, high = np.percentile(finite, (2, 98))
    scaled = np.clip((depth - low) / max(high - low, 1e-8), 0.0, 1.0)
    # Compact blue-to-cyan-to-yellow visualization without matplotlib.
    red = np.clip(2.0 * scaled - 0.3, 0.0, 1.0)
    green = np.clip(2.0 * scaled, 0.0, 1.0)
    blue = np.clip(1.5 - 2.0 * scaled, 0.0, 1.0)
    rgb = (np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def save_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
    points = points.reshape(-1, 3)
    colors = np.clip(colors.reshape(-1, 3) * 255, 0, 255).astype(np.uint8)
    valid = np.isfinite(points).all(axis=1)
    points, colors = points[valid], colors[valid]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    if not 1 <= len(args.images) <= 4:
        parser.error("provide between one and four images")
    return args


def main() -> None:
    args = parse_args()
    ensure_weights(args.weights)
    args.output.mkdir(parents=True, exist_ok=True)
    images = load_and_preprocess_images(args.images)
    model = VGGT()
    model.load_weights(list(mx.load(str(args.weights)).items()), strict=True)
    model.eval()

    started = time.perf_counter()
    predictions = model(images)
    mx.eval(*predictions.values())
    elapsed = time.perf_counter() - started

    depth = np.asarray(predictions["depth"])[0, ..., 0]
    for index, frame_depth in enumerate(depth):
        save_depth(frame_depth, args.output / f"depth_{index:02d}.png")
    points = unproject_depth_map_to_point_map(
        predictions["depth"], predictions["extrinsic"], predictions["intrinsic"]
    )
    mx.eval(points)
    save_ply(np.asarray(points)[0], np.asarray(images), args.output / "points.ply")
    print(f"Wrote {len(depth)} depth map(s) and {args.output / 'points.ply'}")
    print(f"{elapsed * 1000 / len(depth):.1f} ms/frame (fp32)")


if __name__ == "__main__":
    main()
