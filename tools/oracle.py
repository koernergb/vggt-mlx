"""Generate FP32 PyTorch parity fixtures for the MLX port.

Run this in a Colab/upstream VGGT environment after placing three related
images in ``tests/fixtures/sample_images``:

    pip install vggt
    python tools/oracle.py tests/fixtures/sample_images/*.{jpg,png}

The first image produces the one-view fixture. All three images produce the
three-view fixture. Checkpoint weights are never written to this repository.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Optional


REQUIRED_KEYS = (
    "input",
    "patch_embed",
    "agg_l4",
    "agg_l11",
    "agg_l17",
    "agg_l23",
    "pose_enc",
    "depth",
    "depth_conf",
    "extrinsic",
    "intrinsic",
)
AGGREGATOR_LAYERS = (4, 11, 17, 23)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images",
        nargs=3,
        type=Path,
        help="Exactly three related input images; the first is the reference view",
    )
    parser.add_argument("--model-id", default="facebook/VGGT-1B")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/fixtures"),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="FP32 is enforced on every device; CPU is the cleanest parity reference",
    )
    parser.add_argument(
        "--preprocess-mode",
        choices=("crop", "pad"),
        default="crop",
    )
    return parser.parse_args(argv)


def resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return torch.device(requested)


def as_numpy(tensor: Any) -> Any:
    return tensor.detach().float().cpu().numpy()


def extract_patch_tokens(output: Any) -> Any:
    if isinstance(output, dict):
        try:
            return output["x_norm_patchtokens"]
        except KeyError as error:
            raise RuntimeError(
                "Patch embed output did not contain `x_norm_patchtokens`"
            ) from error
    return output


def validate_fixture(fixture: dict[str, Any], expected_views: int) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in fixture]
    if missing:
        raise RuntimeError(f"Oracle fixture is missing keys: {missing}")

    for key, value in fixture.items():
        if value.dtype.name != "float32":
            raise RuntimeError(f"{key} has dtype {value.dtype}, expected float32")
        if not value.size:
            raise RuntimeError(f"{key} is empty")

    if fixture["input"].shape[:2] != (1, expected_views):
        raise RuntimeError(
            f"input has shape {fixture['input'].shape}; expected B=1, S={expected_views}"
        )
    if fixture["pose_enc"].shape != (1, expected_views, 9):
        raise RuntimeError(
            f"pose_enc has shape {fixture['pose_enc'].shape}; "
            f"expected (1, {expected_views}, 9)"
        )
    if fixture["extrinsic"].shape != (1, expected_views, 3, 4):
        raise RuntimeError(f"Unexpected extrinsic shape: {fixture['extrinsic'].shape}")
    if fixture["intrinsic"].shape != (1, expected_views, 3, 3):
        raise RuntimeError(f"Unexpected intrinsic shape: {fixture['intrinsic'].shape}")


def run_oracle(
    model: Any,
    image_paths: list[Path],
    device: Any,
    preprocess_mode: str,
    torch: Any,
    load_and_preprocess_images: Any,
    pose_encoding_to_extri_intri: Any,
) -> dict[str, Any]:
    captures: dict[str, Any] = {}

    def capture_patch(_module: Any, _inputs: Any, output: Any) -> None:
        captures["patch_embed"] = extract_patch_tokens(output)

    def capture_aggregator(_module: Any, _inputs: Any, output: Any) -> None:
        aggregated_tokens, patch_start_idx = output
        if patch_start_idx != 5:
            raise RuntimeError(f"Expected patch_start_idx=5, got {patch_start_idx}")
        for layer in AGGREGATOR_LAYERS:
            value = aggregated_tokens[layer]
            if value is None:
                raise RuntimeError(f"Aggregator layer {layer} was not cached")
            captures[f"agg_l{layer}"] = value

    patch_handle = model.aggregator.patch_embed.register_forward_hook(capture_patch)
    aggregator_handle = model.aggregator.register_forward_hook(capture_aggregator)

    try:
        images = load_and_preprocess_images(
            [str(path) for path in image_paths],
            mode=preprocess_mode,
        )
        # Upstream accepts [S,C,H,W], but storing an explicit batch dimension
        # avoids ambiguity in every downstream MLX parity test.
        images = images.unsqueeze(0).to(device=device, dtype=torch.float32)

        if device.type == "cuda":
            autocast_disabled = torch.autocast(device_type="cuda", enabled=False)
        else:
            autocast_disabled = nullcontext()

        with torch.inference_mode(), autocast_disabled:
            predictions = model(images)
            pose_enc = predictions["pose_enc"].float()
            image_hw = tuple(int(size) for size in images.shape[-2:])
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc,
                image_size_hw=image_hw,
            )
    finally:
        patch_handle.remove()
        aggregator_handle.remove()

    fixture = {
        "input": as_numpy(images),
        "patch_embed": as_numpy(captures["patch_embed"]),
        **{
            f"agg_l{layer}": as_numpy(captures[f"agg_l{layer}"])
            for layer in AGGREGATOR_LAYERS
        },
        "pose_enc": as_numpy(pose_enc),
        "depth": as_numpy(predictions["depth"].float()),
        "depth_conf": as_numpy(predictions["depth_conf"].float()),
        "extrinsic": as_numpy(extrinsic.float()),
        "intrinsic": as_numpy(intrinsic.float()),
    }
    validate_fixture(fixture, expected_views=len(image_paths))
    return fixture


def update_arch_notes(path: Path, fixtures: dict[str, dict[str, Any]]) -> None:
    marker = "## Oracle fixture shapes"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# VGGT Architecture Notes\n"
    before_marker = existing.split(marker, maxsplit=1)[0].rstrip()

    lines = [before_marker, "", marker, ""]
    for fixture_name, fixture in fixtures.items():
        lines.append(f"### `{fixture_name}`")
        lines.append("")
        lines.extend(f"- `{key}`: `{value.shape}`" for key, value in fixture.items())
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    missing_images = [path for path in args.images if not path.is_file()]
    if missing_images:
        raise SystemExit(f"Input images do not exist: {missing_images}")

    try:
        import numpy as np
        import torch
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except ImportError as error:
        raise SystemExit(
            "Card 0.3 requires numpy, torch, and upstream vggt. "
            "Install them in Colab before running this script."
        ) from error

    device = resolve_device(torch, args.device)
    print(f"Loading {args.model_id} on {device} in float32")
    model = VGGT.from_pretrained(args.model_id).eval().to(device=device, dtype=torch.float32)

    fixture_specs = {
        "oracle_1view.npz": [args.images[0]],
        "oracle_3view.npz": list(args.images),
    }
    fixtures = {
        name: run_oracle(
            model,
            paths,
            device,
            args.preprocess_mode,
            torch,
            load_and_preprocess_images,
            pose_encoding_to_extri_intri,
        )
        for name, paths in fixture_specs.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, fixture in fixtures.items():
        destination = args.output_dir / name
        np.savez_compressed(destination, **fixture)
        print(f"Wrote {destination}")

    update_arch_notes(args.output_dir / "ARCH_NOTES.md", fixtures)
    print("Updated ARCH_NOTES.md with oracle shapes")


if __name__ == "__main__":
    main()
