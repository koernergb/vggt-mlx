"""Convert the upstream VGGT-1B PyTorch checkpoint to MLX safetensors."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


MODEL_PREFIXES = ("aggregator.", "camera_head.", "depth_head.", "point_head.")
SKIPPED_PREFIXES = ("track_head.",)


@dataclass
class ConversionReport:
    source: str
    output: str
    total_keys: int = 0
    mapped_keys: int = 0
    permuted_convs: int = 0
    skipped_track_keys: list[str] = field(default_factory=list)
    unmapped_keys: list[str] = field(default_factory=list)
    shape_mismatches: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            "VGGT PyTorch -> MLX conversion report",
            f"source: {self.source}",
            f"output: {self.output}",
            f"total keys: {self.total_keys}",
            f"mapped keys: {self.mapped_keys}",
            f"permuted convolution weights: {self.permuted_convs}",
            f"skipped track keys: {len(self.skipped_track_keys)}",
            f"unmapped non-track keys: {len(self.unmapped_keys)}",
            f"shape mismatches: {len(self.shape_mismatches)}",
        ]
        for heading, values in (
            ("skipped track keys", self.skipped_track_keys),
            ("unmapped non-track keys", self.unmapped_keys),
            ("shape mismatches", self.shape_mismatches),
        ):
            if values:
                lines.extend(("", f"[{heading}]", *values))
        return "\n".join(lines) + "\n"

    def assert_complete(self) -> None:
        if self.unmapped_keys or self.shape_mismatches:
            raise RuntimeError(
                "Conversion was incomplete; inspect the conversion report for details"
            )


def parse_shape_inventory(path: Optional[Path]) -> dict[str, tuple[int, ...]]:
    """Parse the ``name<TAB>(shape)`` file produced by Card 0.2."""
    if path is None or not path.exists():
        return {}

    inventory: dict[str, tuple[int, ...]] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            name, shape_text = raw_line.split("\t", maxsplit=1)
            shape = ast.literal_eval(shape_text)
            inventory[name] = tuple(int(dimension) for dimension in shape)
        except (ValueError, SyntaxError, TypeError) as error:
            raise ValueError(
                f"Invalid state-dict inventory line {line_number}: {raw_line!r}"
            ) from error
    return inventory


def unwrap_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported checkpoint object: {type(checkpoint).__name__}")

    for wrapper_key in ("state_dict", "model"):
        wrapped = checkpoint.get(wrapper_key)
        if isinstance(wrapped, Mapping):
            return wrapped
        if hasattr(wrapped, "state_dict"):
            return wrapped.state_dict()
    return checkpoint


def load_checkpoint(torch: Any, checkpoint_path: Path) -> Mapping[str, Any]:
    if checkpoint_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise RuntimeError(
                "Reading a PyTorch safetensors checkpoint requires `safetensors`"
            ) from error
        return load_file(str(checkpoint_path), device="cpu")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    return unwrap_state_dict(checkpoint)


def instantiate_reference_model(torch: Any) -> Any:
    """Build VGGT on the meta device for type inspection without allocating 5 GB."""
    try:
        from vggt.models.vggt import VGGT
    except ImportError as error:
        raise RuntimeError(
            "The upstream `vggt` package is required for module-type inspection"
        ) from error

    # DINOv2 derives stochastic-depth scalars with ``torch.linspace(...).item()``.
    # Meta tensors do not support ``item()``, so keep only that tiny constructor
    # helper on CPU while every parameter remains allocation-free on ``meta``.
    original_linspace = torch.linspace

    def cpu_linspace(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("device", "cpu")
        return original_linspace(*args, **kwargs)

    torch.linspace = cpu_linspace
    try:
        with torch.device("meta"):
            return VGGT()
    finally:
        torch.linspace = original_linspace


def remap_key(torch_key: str) -> Optional[str]:
    if torch_key.startswith(SKIPPED_PREFIXES):
        return None
    if torch_key.startswith(MODEL_PREFIXES):
        # Card 1.2 intentionally preserves upstream container names. Later MLX
        # modules must match this inventory so strict loading remains possible.
        return torch_key
    raise KeyError(torch_key)


def convert_torch_to_mlx(
    state_dict: Mapping[str, Any],
    reference_model: Any,
    mx: Any,
    torch_nn: Any,
    expected_shapes: Optional[Mapping[str, tuple[int, ...]]] = None,
    source: str = "<in-memory>",
    output: str = "<in-memory>",
) -> tuple[dict[str, Any], ConversionReport]:
    expected_shapes = expected_shapes or {}
    modules = dict(reference_model.named_modules())
    converted: dict[str, Any] = {}
    report = ConversionReport(source=source, output=output, total_keys=len(state_dict))

    for torch_key, tensor in state_dict.items():
        try:
            mlx_key = remap_key(torch_key)
        except KeyError:
            report.unmapped_keys.append(torch_key)
            continue
        if mlx_key is None:
            report.skipped_track_keys.append(torch_key)
            continue

        actual_shape = tuple(int(dimension) for dimension in tensor.shape)
        expected_shape = expected_shapes.get(torch_key)
        if expected_shape is not None and actual_shape != expected_shape:
            report.shape_mismatches.append(
                f"{torch_key}: inventory={expected_shape}, checkpoint={actual_shape}"
            )
            continue

        module_name, _, parameter_name = torch_key.rpartition(".")
        module = modules.get(module_name)
        if module is None:
            report.unmapped_keys.append(f"{torch_key}: parent module not found")
            continue

        array = mx.array(tensor.detach().float().cpu().numpy())
        if isinstance(module, (torch_nn.Conv2d, torch_nn.ConvTranspose2d)):
            if parameter_name == "weight" and array.ndim != 4:
                report.shape_mismatches.append(
                    f"{torch_key}: convolution weight is {array.ndim}-D, expected 4-D"
                )
                continue
            if parameter_name == "weight" and isinstance(module, torch_nn.Conv2d):
                array = array.transpose(0, 2, 3, 1)
                report.permuted_convs += 1
            elif parameter_name == "weight":
                # torch ConvTranspose2d: [in, out/groups, kh, kw]
                # MLX ConvTranspose2d:   [out, kh, kw, in]
                if module.groups != 1:
                    report.unmapped_keys.append(
                        f"{torch_key}: grouped ConvTranspose2d is unsupported by MLX"
                    )
                    continue
                array = array.transpose(1, 2, 3, 0)
                report.permuted_convs += 1
        elif isinstance(module, torch_nn.Linear) and parameter_name == "weight":
            if array.ndim != 2:
                report.shape_mismatches.append(
                    f"{torch_key}: linear weight is {array.ndim}-D, expected 2-D"
                )
                continue
            # PyTorch and MLX both store Linear weights as [out, in].

        converted[mlx_key] = array
        report.mapped_keys += 1

    return converted, report


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Local model.pt or model.safetensors; downloads model.pt when omitted",
    )
    parser.add_argument("--model-id", default="facebook/VGGT-1B")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("weights/vggt-1b-mlx.safetensors"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("weights/conversion_report.txt"),
    )
    parser.add_argument(
        "--state-dict-keys",
        type=Path,
        default=Path("tests/fixtures/state_dict_keys.txt"),
        help="Optional Card 0.2 shape inventory",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    try:
        import mlx.core as mx
        import torch
        import torch.nn as torch_nn
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise SystemExit(
            "Conversion requires MLX, PyTorch, huggingface_hub, and upstream vggt"
        ) from error

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path(
            hf_hub_download(repo_id=args.model_id, filename="model.pt")
        )
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint_path}")

    state_dict = load_checkpoint(torch, checkpoint_path)
    reference_model = instantiate_reference_model(torch)
    expected_shapes = parse_shape_inventory(args.state_dict_keys)

    converted, report = convert_torch_to_mlx(
        state_dict,
        reference_model,
        mx,
        torch_nn,
        expected_shapes=expected_shapes,
        source=str(checkpoint_path),
        output=str(args.output),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report.render(), encoding="utf-8")
    report.assert_complete()

    mx.save_safetensors(
        str(args.output),
        converted,
        metadata={
            "source_model": args.model_id,
            "precision": "float32",
            "track_head": "omitted",
        },
    )
    print(report.render(), end="")


if __name__ == "__main__":
    main()
