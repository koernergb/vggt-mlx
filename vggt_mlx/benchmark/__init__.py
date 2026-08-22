"""Reproducible benchmark and parity result primitives."""

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

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "PARITY_POLICY_VERSION",
    "PARITY_SCHEMA_VERSION",
    "ParityMetrics",
    "ResultValidationError",
    "TapComparison",
    "camera_rotation_geodesic_degrees",
    "camera_translation_direction_degrees",
    "compare_arrays",
    "compare_taps",
    "validate_benchmark_result",
    "validate_parity_result",
]
