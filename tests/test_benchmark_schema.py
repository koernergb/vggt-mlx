"""Validation gates for append-only public result documents."""

from copy import deepcopy

import pytest

from vggt_mlx.benchmark.schema import (
    ResultValidationError,
    validate_benchmark_result,
    validate_parity_result,
)


def common(version):
    return {
        "schema_version": version,
        "run_id": "m4-fp32-2view-001",
        "timestamp_utc": "2026-08-21T12:00:00Z",
        "git": {"revision": "abc1234", "dirty": False},
        "environment": {
            "hardware": "Apple M4",
            "memory_gb": 16,
            "macos": "26.1",
            "python": "3.12.13",
            "mlx": "0.32.0",
            "power": "AC",
        },
        "workload": {
            "model": "VGGT-1B",
            "checkpoint_revision": "860abec",
            "input_sha256": "deadbeef",
            "views": 2,
            "shape": [1, 2, 126, 518, 3],
        },
    }


def benchmark():
    return {
        **common("vggt-mlx-benchmark/1.0"),
        "framework": "mlx",
        "precision": "fp32",
        "warmup_trials": 5,
        "samples_ms": [10.0, 12.0, 11.0],
        "summary": {"median_ms": 11.0, "iqr_ms": 1.0, "minimum_ms": 10.0},
        "validity": {"included": True, "thermally_stable": True, "reason": None},
    }


def parity():
    return {
        **common("vggt-mlx-parity-result/1.0"),
        "reference": "pytorch-cpu-fp32",
        "candidate": "mlx-cpu-fp32",
        "policy_version": "vggt-mlx-parity/1.0",
        "taps": [
            {
                "name": "patch",
                "shape": [1, 4],
                "metrics": {"max_abs": 1e-5, "mean_abs": 1e-6, "rel_fro": 1e-6, "cosine": 1.0},
                "passed": True,
                "failures": [],
            }
        ],
        "status": "pass",
        "first_failure": None,
    }


def test_valid_documents_pass():
    validate_benchmark_result(benchmark())
    validate_parity_result(parity())


@pytest.mark.parametrize("field", ["samples_ms", "environment", "workload"])
def test_missing_benchmark_evidence_fails(field):
    document = benchmark()
    del document[field]
    with pytest.raises(ResultValidationError, match="missing required"):
        validate_benchmark_result(document)


def test_raw_samples_must_agree_with_summary():
    document = benchmark()
    document["summary"]["median_ms"] = 99
    with pytest.raises(ResultValidationError, match="median_ms"):
        validate_benchmark_result(document)


def test_exclusion_requires_reason():
    document = benchmark()
    document["validity"] = {"included": False, "thermally_stable": False, "reason": ""}
    with pytest.raises(ResultValidationError, match="requires a reason"):
        validate_benchmark_result(document)


def test_unknown_policy_and_inconsistent_failure_are_rejected():
    document = parity()
    document["policy_version"] = "loosened-unreviewed-policy"
    with pytest.raises(ResultValidationError, match="unknown parity policy"):
        validate_parity_result(document)

    document = deepcopy(parity())
    document["taps"][0]["passed"] = False
    with pytest.raises(ResultValidationError, match="must explain"):
        validate_parity_result(document)
