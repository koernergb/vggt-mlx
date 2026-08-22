"""Reproducible benchmark and parity result primitives."""

from vggt_mlx.benchmark.adapters import MLXAdapter, PUBLIC_OUTPUTS, PyTorchMPSAdapter
from vggt_mlx.benchmark.parity import (
    PARITY_POLICY_VERSION,
    ParityMetrics,
    TapComparison,
    camera_rotation_geodesic_degrees,
    camera_translation_direction_degrees,
    compare_arrays,
    compare_taps,
)
from vggt_mlx.benchmark.schema import (
    BENCHMARK_SCHEMA_VERSION,
    PARITY_SCHEMA_VERSION,
    ResultValidationError,
    validate_benchmark_result,
    validate_parity_result,
)
from vggt_mlx.benchmark.runner import run_trials, summarize_samples, write_result

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "MLXAdapter",
    "PARITY_POLICY_VERSION",
    "PARITY_SCHEMA_VERSION",
    "ParityMetrics",
    "PUBLIC_OUTPUTS",
    "PyTorchMPSAdapter",
    "ResultValidationError",
    "TapComparison",
    "camera_rotation_geodesic_degrees",
    "camera_translation_direction_degrees",
    "compare_arrays",
    "compare_taps",
    "run_trials",
    "summarize_samples",
    "validate_benchmark_result",
    "validate_parity_result",
    "write_result",
]
