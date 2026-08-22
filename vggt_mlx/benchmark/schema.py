"""Strict validation for append-only benchmark and parity result documents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any


BENCHMARK_SCHEMA_VERSION = "vggt-mlx-benchmark/1.0"
PARITY_SCHEMA_VERSION = "vggt-mlx-parity-result/1.0"
KNOWN_PARITY_POLICIES = {"vggt-mlx-parity/1.0"}


class ResultValidationError(ValueError):
    """A public result is incomplete, inconsistent, or unsupported."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResultValidationError(f"{path} must be an object")
    return value


def _required(document: Mapping[str, Any], keys: set[str], path: str) -> None:
    missing = keys - set(document)
    if missing:
        raise ResultValidationError(f"{path} missing required fields: {sorted(missing)}")


def _text(document: Mapping[str, Any], key: str, path: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ResultValidationError(f"{path}.{key} must be a non-empty string")
    return value


def _finite_number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultValidationError(f"{path} must be a number")
    value = float(value)
    if not isfinite(value):
        raise ResultValidationError(f"{path} must be finite")
    if minimum is not None and value < minimum:
        raise ResultValidationError(f"{path} must be >= {minimum}")
    return value


def _validate_common(document: Mapping[str, Any], version: str) -> None:
    _required(
        document,
        {"schema_version", "run_id", "timestamp_utc", "git", "environment", "workload"},
        "result",
    )
    if document["schema_version"] != version:
        raise ResultValidationError(
            f"unsupported schema_version {document['schema_version']!r}; expected {version!r}"
        )
    _text(document, "run_id", "result")
    timestamp = _text(document, "timestamp_utc", "result")
    if not timestamp.endswith("Z"):
        raise ResultValidationError("result.timestamp_utc must be UTC and end in Z")

    git = _mapping(document["git"], "result.git")
    _required(git, {"revision", "dirty"}, "result.git")
    _text(git, "revision", "result.git")
    if not isinstance(git["dirty"], bool):
        raise ResultValidationError("result.git.dirty must be boolean")

    environment = _mapping(document["environment"], "result.environment")
    _required(
        environment,
        {"hardware", "memory_gb", "macos", "python", "mlx", "power"},
        "result.environment",
    )
    for key in ("hardware", "macos", "python", "mlx", "power"):
        _text(environment, key, "result.environment")
    _finite_number(environment["memory_gb"], "result.environment.memory_gb", minimum=0)

    workload = _mapping(document["workload"], "result.workload")
    _required(workload, {"model", "checkpoint_revision", "input_sha256", "views", "shape"}, "result.workload")
    for key in ("model", "checkpoint_revision", "input_sha256"):
        _text(workload, key, "result.workload")
    if not isinstance(workload["views"], int) or isinstance(workload["views"], bool) or workload["views"] < 1:
        raise ResultValidationError("result.workload.views must be a positive integer")
    shape = workload["shape"]
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, (str, bytes))
        or not shape
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in shape)
    ):
        raise ResultValidationError("result.workload.shape must contain positive integers")


def validate_benchmark_result(document: Mapping[str, Any]) -> None:
    """Validate one append-only hardware benchmark result."""

    document = _mapping(document, "result")
    _validate_common(document, BENCHMARK_SCHEMA_VERSION)
    _required(
        document,
        {"framework", "precision", "warmup_trials", "samples_ms", "summary", "validity"},
        "result",
    )
    _text(document, "framework", "result")
    _text(document, "precision", "result")
    if not isinstance(document["warmup_trials"], int) or document["warmup_trials"] < 0:
        raise ResultValidationError("result.warmup_trials must be a non-negative integer")
    samples = document["samples_ms"]
    if not isinstance(samples, list) or not samples:
        raise ResultValidationError("result.samples_ms must contain raw measured trials")
    numeric_samples = [
        _finite_number(value, f"result.samples_ms[{index}]", minimum=0)
        for index, value in enumerate(samples)
    ]

    summary = _mapping(document["summary"], "result.summary")
    _required(summary, {"median_ms", "iqr_ms", "minimum_ms"}, "result.summary")
    median = _finite_number(summary["median_ms"], "result.summary.median_ms", minimum=0)
    minimum = _finite_number(summary["minimum_ms"], "result.summary.minimum_ms", minimum=0)
    _finite_number(summary["iqr_ms"], "result.summary.iqr_ms", minimum=0)
    if abs(minimum - min(numeric_samples)) > 1e-6:
        raise ResultValidationError("result.summary.minimum_ms does not match raw samples")
    ordered = sorted(numeric_samples)
    middle = len(ordered) // 2
    expected_median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    if abs(median - expected_median) > 1e-6:
        raise ResultValidationError("result.summary.median_ms does not match raw samples")

    validity = _mapping(document["validity"], "result.validity")
    _required(validity, {"included", "thermally_stable", "reason"}, "result.validity")
    if not isinstance(validity["included"], bool) or not isinstance(validity["thermally_stable"], bool):
        raise ResultValidationError("result.validity flags must be boolean")
    reason = validity["reason"]
    if validity["included"]:
        if reason not in (None, ""):
            raise ResultValidationError("included result must not have an exclusion reason")
    elif not isinstance(reason, str) or not reason.strip():
        raise ResultValidationError("excluded result requires a reason")


def validate_parity_result(document: Mapping[str, Any]) -> None:
    """Validate named-tap and end-to-end parity evidence."""

    document = _mapping(document, "result")
    _validate_common(document, PARITY_SCHEMA_VERSION)
    _required(document, {"reference", "candidate", "policy_version", "taps", "status", "first_failure"}, "result")
    _text(document, "reference", "result")
    _text(document, "candidate", "result")
    if document["policy_version"] not in KNOWN_PARITY_POLICIES:
        raise ResultValidationError(f"unknown parity policy {document['policy_version']!r}")
    if document["status"] not in {"pass", "fail"}:
        raise ResultValidationError("result.status must be 'pass' or 'fail'")
    taps = document["taps"]
    if not isinstance(taps, list) or not taps:
        raise ResultValidationError("result.taps must be a non-empty ordered list")
    failed = []
    seen = set()
    for index, tap_value in enumerate(taps):
        tap = _mapping(tap_value, f"result.taps[{index}]")
        _required(tap, {"name", "shape", "metrics", "passed", "failures"}, f"result.taps[{index}]")
        name = _text(tap, "name", f"result.taps[{index}]")
        if name in seen:
            raise ResultValidationError(f"duplicate tap name {name!r}")
        seen.add(name)
        metrics = _mapping(tap["metrics"], f"result.taps[{index}].metrics")
        _required(metrics, {"max_abs", "mean_abs", "rel_fro", "cosine"}, f"result.taps[{index}].metrics")
        for metric_name in ("max_abs", "mean_abs", "rel_fro", "cosine"):
            _finite_number(metrics[metric_name], f"result.taps[{index}].metrics.{metric_name}")
        if not isinstance(tap["passed"], bool) or not isinstance(tap["failures"], list):
            raise ResultValidationError(f"result.taps[{index}] has invalid pass/failure fields")
        if tap["passed"] and tap["failures"]:
            raise ResultValidationError(f"passing tap {name!r} cannot contain failures")
        if not tap["passed"]:
            failed.append(name)
            if not tap["failures"]:
                raise ResultValidationError(f"failing tap {name!r} must explain its failure")
    expected_status = "fail" if failed else "pass"
    expected_first = failed[0] if failed else None
    if document["status"] != expected_status:
        raise ResultValidationError("result.status disagrees with tap results")
    if document["first_failure"] != expected_first:
        raise ResultValidationError("result.first_failure disagrees with ordered tap results")


def result_path_is_append_only(path: Path) -> bool:
    """Result names use a stable timestamp/run-id form and never a mutable latest file."""

    return path.suffix == ".json" and path.stem not in {"latest", "current", "result"}
