"""Equivalent framework adapters for controlled VGGT comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np


PUBLIC_OUTPUTS = (
    "pose_enc",
    "depth",
    "depth_conf",
    "world_points",
    "world_points_conf",
    "extrinsic",
    "intrinsic",
)


class PyTorchMPSAdapter:
    """Official PyTorch VGGT with the same public workload as the MLX port.

    Inputs are shared, framework-neutral float32 arrays in ``[B,S,H,W,3]`` or
    ``[S,H,W,3]`` layout and range ``[0, 1]``. Tracking is not requested because
    the current MLX release intentionally omits that head.
    """

    framework = "pytorch-mps"
    precision = "fp32"
    output_names = PUBLIC_OUTPUTS

    def __init__(
        self,
        model: Any,
        *,
        device: str = "mps",
        pose_decoder: Callable[..., tuple[Any, Any]] | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("PyTorch is required for the MPS comparison extra") from error
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS is unavailable on this machine")
        if device not in {"mps", "cpu"}:
            raise ValueError("PyTorch reference device must be 'mps' or 'cpu'")
        self.torch = torch
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        if pose_decoder is None:
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri

            pose_decoder = pose_encoding_to_extri_intri
        self.pose_decoder = pose_decoder
        self.last_peak_memory_mb: float | None = None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str = "facebook/VGGT-1B",
        *,
        revision: str | None = None,
        device: str = "mps",
    ) -> "PyTorchMPSAdapter":
        """Load the official model without changing its checkpoint family."""

        try:
            from vggt.models.vggt import VGGT
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "The pinned upstream `vggt` package is required for MPS benchmarking"
            ) from error
        kwargs = {"revision": revision} if revision else {}
        model = VGGT.from_pretrained(checkpoint, **kwargs)
        return cls(model, device=device)

    @classmethod
    def from_local_safetensors(
        cls,
        checkpoint: Path,
        *,
        device: str = "mps",
    ) -> "PyTorchMPSAdapter":
        """Load the official architecture from a checksum-verified local file."""

        try:
            from safetensors.torch import load_file
            from vggt.models.vggt import VGGT
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("Upstream VGGT and safetensors are required") from error
        model = VGGT()
        model.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
        return cls(model, device=device)

    def synchronize(self) -> None:
        if self.device.type == "mps":
            self.torch.mps.synchronize()

    def reset_peak_memory(self) -> None:
        self.last_peak_memory_mb = None

    def capture_peak_memory(self) -> None:
        if self.device.type != "mps":
            return
        values = []
        for name in ("current_allocated_memory", "driver_allocated_memory"):
            function = getattr(self.torch.mps, name, None)
            if function is not None:
                values.append(float(function()) / 1024**2)
        if values:
            current = max(values)
            self.last_peak_memory_mb = max(self.last_peak_memory_mb or 0.0, current)

    def prepare_input(self, shared_images: np.ndarray) -> Any:
        images = np.asarray(shared_images)
        if images.ndim == 4:
            images = images[None]
        if images.ndim != 5 or images.shape[-1] != 3:
            raise ValueError(
                f"shared input must be [B,S,H,W,3] or [S,H,W,3], got {images.shape}"
            )
        if images.dtype != np.float32:
            raise ValueError(f"shared input must be float32, got {images.dtype}")
        if not np.isfinite(images).all():
            raise ValueError("shared input contains NaN or infinity")
        if images.min() < 0.0 or images.max() > 1.0:
            raise ValueError("shared input must be in range [0, 1]")
        nchw = np.ascontiguousarray(images.transpose(0, 1, 4, 2, 3))
        return self.torch.from_numpy(nchw).to(self.device)

    def forward_tensors(self, prepared_images: Any) -> dict[str, Any]:
        """Execute equivalent heads and retain device tensors for honest timing."""

        with self.torch.inference_mode():
            raw = self.model(prepared_images, query_points=None)
            required = {
                "pose_enc",
                "depth",
                "depth_conf",
                "world_points",
                "world_points_conf",
            }
            missing = required - set(raw)
            if missing:
                raise RuntimeError(f"official VGGT output missing keys: {sorted(missing)}")
            extrinsic, intrinsic = self.pose_decoder(
                raw["pose_enc"], prepared_images.shape[-2:]
            )
            output = {key: raw[key] for key in required}
            output["extrinsic"] = extrinsic
            output["intrinsic"] = intrinsic
        if set(output) != set(PUBLIC_OUTPUTS):
            raise RuntimeError("PyTorch and MLX output workloads are not equivalent")
        return output

    def evaluate(self, output: Mapping[str, Any]) -> None:
        """Complete queued device work without transferring results to the host."""

        for name in PUBLIC_OUTPUTS:
            if name not in output:
                raise ValueError(f"output missing {name!r}")
        self.synchronize()

    def to_numpy(self, output: Mapping[str, Any]) -> dict[str, np.ndarray]:
        self.evaluate(output)
        return {
            name: output[name].detach().to("cpu").numpy()
            for name in PUBLIC_OUTPUTS
        }


class MLXAdapter:
    """Thin adapter exposing the same shared-input contract for native MLX."""

    framework = "mlx"
    precision = "fp32"
    output_names = PUBLIC_OUTPUTS

    def __init__(self, model: Any) -> None:
        import mlx.core as mx

        self.mx = mx
        self.model = model
        self.model.eval()
        self.last_peak_memory_mb: float | None = None

    def reset_peak_memory(self) -> None:
        reset = getattr(self.mx.metal, "reset_peak_memory", None)
        if reset is not None:
            reset()
        self.last_peak_memory_mb = None

    def capture_peak_memory(self) -> None:
        get_peak = getattr(self.mx.metal, "get_peak_memory", None)
        if get_peak is not None:
            self.last_peak_memory_mb = float(get_peak()) / 1024**2

    def prepare_input(self, shared_images: np.ndarray) -> Any:
        images = np.asarray(shared_images)
        if images.ndim == 4:
            images = images[None]
        if images.ndim != 5 or images.shape[-1] != 3:
            raise ValueError(
                f"shared input must be [B,S,H,W,3] or [S,H,W,3], got {images.shape}"
            )
        if images.dtype != np.float32:
            raise ValueError(f"shared input must be float32, got {images.dtype}")
        if not np.isfinite(images).all() or images.min() < 0.0 or images.max() > 1.0:
            raise ValueError("shared input must be finite and in range [0, 1]")
        return self.mx.array(np.ascontiguousarray(images))

    def forward_tensors(self, prepared_images: Any) -> dict[str, Any]:
        output = self.model(prepared_images)
        if set(output) != set(PUBLIC_OUTPUTS):
            raise RuntimeError("MLX output workload does not match the public contract")
        return output

    def evaluate(self, output: Mapping[str, Any]) -> None:
        self.mx.eval(*(output[name] for name in PUBLIC_OUTPUTS))

    def to_numpy(self, output: Mapping[str, Any]) -> dict[str, np.ndarray]:
        self.evaluate(output)
        return {name: np.asarray(output[name]) for name in PUBLIC_OUTPUTS}
