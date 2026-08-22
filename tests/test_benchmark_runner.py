"""Timing, resumability, and raw-evidence tests."""

import json

import numpy as np
import pytest

from vggt_mlx.benchmark.runner import (
    existing_run_ids,
    run_trials,
    summarize_samples,
    thermally_stable,
    write_result,
)


class DummyAdapter:
    framework = "dummy"
    precision = "fp32"

    def __init__(self):
        self.forwards = 0
        self.evaluations = 0

    def forward_tensors(self, prepared):
        self.forwards += 1
        return prepared

    def evaluate(self, output):
        self.evaluations += 1

    def synchronize(self):
        pass


def test_warmups_are_separate_and_every_trial_is_evaluated():
    ticks = iter([0.0, 0.01, 1.0, 1.02, 2.0, 2.03])
    adapter = DummyAdapter()
    samples, output = run_trials(
        adapter, "output", warmups=2, trials=3, clock=lambda: next(ticks)
    )
    assert samples == pytest.approx([10.0, 20.0, 30.0])
    assert adapter.forwards == 5
    assert adapter.evaluations == 5
    assert output == "output"


def test_summary_and_thermal_flag_are_derived_from_raw_samples():
    assert summarize_samples([10, 20, 30])["median_ms"] == 20
    assert summarize_samples([10, 20, 30])["minimum_ms"] == 10
    assert thermally_stable([10, 11, 12])
    assert not thermally_stable([10, 11, 12.01])


def valid_result():
    return {
        "schema_version": "vggt-mlx-benchmark/1.0",
        "run_id": "abcdef1234567890",
        "timestamp_utc": "2026-08-21T12:00:00Z",
        "git": {"revision": "abc", "dirty": False},
        "environment": {
            "hardware": "Apple M4", "memory_gb": 16, "macos": "26.1",
            "python": "3.12", "mlx": "0.32", "power": "AC"
        },
        "workload": {
            "model": "VGGT-1B", "checkpoint_revision": "rev",
            "input_sha256": "hash", "views": 1, "shape": [1, 1, 126, 518, 3]
        },
        "framework": "mlx", "precision": "fp32", "warmup_trials": 1,
        "samples_ms": [10, 11, 12],
        "summary": {"median_ms": 11, "iqr_ms": 1, "minimum_ms": 10},
        "validity": {"included": True, "thermally_stable": True, "reason": None}
    }


def test_results_are_append_only_and_discoverable(tmp_path):
    result = valid_result()
    path = write_result(tmp_path, result)
    assert json.loads(path.read_text())["run_id"] == result["run_id"]
    assert existing_run_ids(tmp_path) == {result["run_id"]}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_result(tmp_path, result)
