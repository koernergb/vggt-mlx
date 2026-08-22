"""Reproducible, append-only benchmark execution."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from vggt_mlx.benchmark.schema import (
    BENCHMARK_SCHEMA_VERSION,
    validate_benchmark_result,
)


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def summarize_samples(samples_ms: Sequence[float]) -> dict[str, float]:
    if not samples_ms:
        raise ValueError("at least one measured sample is required")
    samples = np.asarray(samples_ms, dtype=np.float64)
    if not np.isfinite(samples).all() or np.any(samples < 0):
        raise ValueError("timing samples must be finite and non-negative")
    q25, q75 = np.percentile(samples, [25, 75])
    return {
        "median_ms": float(median(samples_ms)),
        "iqr_ms": float(q75 - q25),
        "minimum_ms": float(samples.min()),
    }


def thermally_stable(samples_ms: Sequence[float]) -> bool:
    if not samples_ms:
        raise ValueError("at least one measured sample is required")
    minimum = min(samples_ms)
    if minimum == 0:
        return max(samples_ms) == 0
    return max(samples_ms) / minimum <= 1.2


def run_trials(
    adapter: Any,
    prepared_input: Any,
    *,
    warmups: int,
    trials: int,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[float], Any]:
    """Time forward execution only and force queued work inside every trial."""

    if warmups < 0 or trials < 1:
        raise ValueError("warmups must be non-negative and trials must be positive")
    output = None
    for _ in range(warmups):
        output = adapter.forward_tensors(prepared_input)
        adapter.evaluate(output)
    if hasattr(adapter, "reset_peak_memory"):
        adapter.reset_peak_memory()
    samples = []
    for _ in range(trials):
        adapter.synchronize() if hasattr(adapter, "synchronize") else None
        started = clock()
        output = adapter.forward_tensors(prepared_input)
        adapter.evaluate(output)
        if hasattr(adapter, "capture_peak_memory"):
            adapter.capture_peak_memory()
        samples.append((clock() - started) * 1000.0)
    return samples, output


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def capture_environment() -> dict[str, Any]:
    """Capture publishable environment data while excluding serials and UUIDs."""

    hardware = _command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
    memory_bytes = _command_output(["sysctl", "-n", "hw.memsize"])
    power_output = _command_output(["pmset", "-g", "batt"]) or ""
    power = "AC" if "AC Power" in power_output else "battery"
    result = {
        "hardware": hardware or platform.machine(),
        "memory_gb": round(int(memory_bytes) / 1024**3, 2) if memory_bytes else 0,
        "macos": platform.mac_ver()[0] or platform.platform(),
        "python": platform.python_version(),
        "mlx": package_version("mlx") or "not-installed",
        "pytorch": package_version("torch"),
        "power": power,
    }
    return result


def git_state(repository: Path) -> dict[str, Any]:
    revision = _command_output(["git", "-C", str(repository), "rev-parse", "HEAD"])
    status = _command_output(["git", "-C", str(repository), "status", "--porcelain"])
    if not revision:
        raise RuntimeError("benchmark must run inside a git repository")
    return {"revision": revision, "dirty": bool(status)}


def stable_run_id(cell: Mapping[str, Any]) -> str:
    encoded = json.dumps(cell, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def planned_run_id(
    *,
    repository: Path,
    framework: str,
    precision: str,
    shared_input: np.ndarray,
    checkpoint_revision: str,
) -> str:
    """Compute a benchmark cell ID before loading its heavyweight model."""

    shape = list(shared_input.shape)
    views = shape[1] if len(shape) == 5 else shape[0]
    environment = capture_environment()
    git = git_state(repository)
    return stable_run_id(
        {
            "git_revision": git["revision"],
            "hardware": environment["hardware"],
            "framework": framework,
            "precision": precision,
            "checkpoint_revision": checkpoint_revision,
            "input_sha256": sha256_array(shared_input),
            "views": views,
            "shape": shape,
        }
    )


def build_benchmark_result(
    *,
    repository: Path,
    adapter: Any,
    shared_input: np.ndarray,
    checkpoint_revision: str,
    warmups: int,
    samples_ms: Sequence[float],
    peak_memory_mb: float | None = None,
) -> dict[str, Any]:
    shape = list(shared_input.shape)
    views = shape[1] if len(shape) == 5 else shape[0]
    environment = capture_environment()
    git = git_state(repository)
    cell = {
        "git_revision": git["revision"],
        "hardware": environment["hardware"],
        "framework": adapter.framework,
        "precision": adapter.precision,
        "checkpoint_revision": checkpoint_revision,
        "input_sha256": sha256_array(shared_input),
        "views": views,
        "shape": shape,
    }
    stable = thermally_stable(samples_ms)
    result = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "run_id": stable_run_id(cell),
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git": git,
        "environment": environment,
        "workload": {
            "model": "VGGT-1B",
            "checkpoint_revision": checkpoint_revision,
            "input_sha256": cell["input_sha256"],
            "views": views,
            "shape": shape,
            "outputs": list(getattr(adapter, "output_names", ())) or None,
        },
        "framework": adapter.framework,
        "precision": adapter.precision,
        "warmup_trials": warmups,
        "samples_ms": [float(value) for value in samples_ms],
        "summary": summarize_samples(samples_ms),
        "peak_memory_mb": peak_memory_mb,
        "validity": {
            "included": stable and environment["power"] == "AC" and not git["dirty"],
            "thermally_stable": stable,
            "reason": None,
        },
    }
    reasons = []
    if not stable:
        reasons.append("max/min timing ratio exceeds 1.2")
    if environment["power"] != "AC":
        reasons.append("machine was not on AC power")
    if git["dirty"]:
        reasons.append("repository had uncommitted changes")
    if reasons:
        result["validity"]["reason"] = "; ".join(reasons)
    validate_benchmark_result(result)
    return result


def result_path(results_dir: Path, result: Mapping[str, Any]) -> Path:
    timestamp = str(result["timestamp_utc"]).replace(":", "").replace("-", "")
    return results_dir / f"{timestamp}_{result['run_id']}.json"


def write_result(results_dir: Path, result: Mapping[str, Any]) -> Path:
    """Atomically append a new result and refuse any overwrite."""

    validate_benchmark_result(result)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = result_path(results_dir, result)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite benchmark result {path}")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def existing_run_ids(results_dir: Path) -> set[str]:
    run_ids = set()
    if not results_dir.exists():
        return run_ids
    for path in results_dir.glob("*.json"):
        document = json.loads(path.read_text())
        validate_benchmark_result(document)
        run_ids.add(document["run_id"])
    return run_ids
