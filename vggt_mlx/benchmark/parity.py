"""Framework-neutral numerical parity metrics and tap diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


PARITY_POLICY_VERSION = "vggt-mlx-parity/1.0"


@dataclass(frozen=True)
class ParityMetrics:
    """Scalar comparison metrics for one pair of equal-shaped tensors."""

    max_abs: float
    mean_abs: float
    rel_fro: float
    cosine: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class TapComparison:
    """Comparison result for one named activation tap."""

    name: str
    shape: tuple[int, ...]
    metrics: ParityMetrics
    passed: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "metrics": self.metrics.to_dict(),
            "passed": self.passed,
            "failures": list(self.failures),
        }


def _finite_float64(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def compare_arrays(reference: np.ndarray, candidate: np.ndarray) -> ParityMetrics:
    """Compare two tensors without hiding zero-norm edge cases."""

    reference = _finite_float64(reference, "reference")
    candidate = _finite_float64(candidate, "candidate")
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: reference {reference.shape} != candidate {candidate.shape}"
        )
    if reference.size == 0:
        raise ValueError("cannot compare empty tensors")

    difference = candidate - reference
    abs_difference = np.abs(difference)
    reference_norm = float(np.linalg.norm(reference.ravel()))
    candidate_norm = float(np.linalg.norm(candidate.ravel()))
    difference_norm = float(np.linalg.norm(difference.ravel()))

    if reference_norm == 0.0:
        rel_fro = 0.0 if difference_norm == 0.0 else float("inf")
    else:
        rel_fro = difference_norm / reference_norm

    if reference_norm == 0.0 or candidate_norm == 0.0:
        cosine = 1.0 if reference_norm == candidate_norm == 0.0 else 0.0
    else:
        cosine = float(
            np.dot(reference.ravel(), candidate.ravel())
            / (reference_norm * candidate_norm)
        )
        cosine = float(np.clip(cosine, -1.0, 1.0))

    return ParityMetrics(
        max_abs=float(abs_difference.max()),
        mean_abs=float(abs_difference.mean()),
        rel_fro=rel_fro,
        cosine=cosine,
    )


def compare_taps(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    order: Sequence[str],
    tolerances: Mapping[str, Mapping[str, float]],
) -> tuple[list[TapComparison], str | None]:
    """Compare ordered taps and identify the first failed computation boundary."""

    expected = set(order)
    missing_reference = expected - set(reference)
    missing_candidate = expected - set(candidate)
    unexpected_reference = set(reference) - expected
    unexpected_candidate = set(candidate) - expected
    if missing_reference or missing_candidate or unexpected_reference or unexpected_candidate:
        raise ValueError(
            "tap mismatch: "
            f"missing_reference={sorted(missing_reference)}, "
            f"missing_candidate={sorted(missing_candidate)}, "
            f"unexpected_reference={sorted(unexpected_reference)}, "
            f"unexpected_candidate={sorted(unexpected_candidate)}"
        )

    comparisons: list[TapComparison] = []
    first_failure: str | None = None
    for name in order:
        if name not in tolerances:
            raise ValueError(f"no tolerance policy for tap {name!r}")
        reference_array = np.asarray(reference[name])
        candidate_array = np.asarray(candidate[name])
        metrics = compare_arrays(reference_array, candidate_array)
        failures = []
        for metric_name, threshold in tolerances[name].items():
            if not hasattr(metrics, metric_name):
                raise ValueError(f"unknown parity metric {metric_name!r}")
            value = getattr(metrics, metric_name)
            if metric_name == "cosine":
                if value < threshold:
                    failures.append(f"cosine={value:.8g} < {threshold:.8g}")
            elif value > threshold:
                failures.append(f"{metric_name}={value:.8g} > {threshold:.8g}")
        passed = not failures
        if not passed and first_failure is None:
            first_failure = name
        comparisons.append(
            TapComparison(
                name=name,
                shape=reference_array.shape,
                metrics=metrics,
                passed=passed,
                failures=tuple(failures),
            )
        )
    return comparisons, first_failure


def camera_rotation_geodesic_degrees(
    reference: np.ndarray, candidate: np.ndarray
) -> np.ndarray:
    """Return SO(3) geodesic errors in degrees for arrays ending in ``[3, 3]``."""

    reference = _finite_float64(reference, "reference rotations")
    candidate = _finite_float64(candidate, "candidate rotations")
    if reference.shape != candidate.shape or reference.shape[-2:] != (3, 3):
        raise ValueError("rotation arrays must have equal shape ending in [3, 3]")
    relative = candidate @ np.swapaxes(reference, -1, -2)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def camera_translation_direction_degrees(
    reference: np.ndarray, candidate: np.ndarray
) -> np.ndarray:
    """Return angular errors in degrees for arrays ending in translation XYZ."""

    reference = _finite_float64(reference, "reference translations")
    candidate = _finite_float64(candidate, "candidate translations")
    if reference.shape != candidate.shape or reference.shape[-1] != 3:
        raise ValueError("translation arrays must have equal shape ending in [3]")
    reference_norm = np.linalg.norm(reference, axis=-1)
    candidate_norm = np.linalg.norm(candidate, axis=-1)
    if np.any(reference_norm == 0.0) or np.any(candidate_norm == 0.0):
        raise ValueError("translation direction is undefined for a zero vector")
    cosine = np.sum(reference * candidate, axis=-1) / (
        reference_norm * candidate_norm
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
