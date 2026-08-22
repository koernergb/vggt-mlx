"""Analytical tests for public numerical-parity evidence."""

import numpy as np
import pytest

from vggt_mlx.benchmark.parity import (
    camera_rotation_geodesic_degrees,
    camera_translation_direction_degrees,
    compare_arrays,
    compare_taps,
)


def test_array_metrics_match_analytical_case():
    reference = np.array([3.0, 4.0])
    candidate = np.array([0.0, 4.0])
    metrics = compare_arrays(reference, candidate)
    assert metrics.max_abs == 3.0
    assert metrics.mean_abs == 1.5
    assert metrics.rel_fro == pytest.approx(0.6)
    assert metrics.cosine == pytest.approx(0.8)


def test_array_metrics_handle_zero_norms_and_reject_bad_inputs():
    assert compare_arrays(np.zeros(2), np.zeros(2)).cosine == 1.0
    assert compare_arrays(np.zeros(2), np.ones(2)).rel_fro == float("inf")
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_arrays(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match="NaN"):
        compare_arrays(np.array([np.nan]), np.zeros(1))


def test_tap_comparison_reports_first_failure_in_model_order():
    reference = {"patch": np.ones(2), "block": np.ones(2), "depth": np.ones(2)}
    candidate = {"patch": np.ones(2), "block": np.array([1.0, 1.1]), "depth": np.zeros(2)}
    tolerances = {
        name: {"max_abs": 0.01, "cosine": 0.99}
        for name in ("patch", "block", "depth")
    }
    results, first_failure = compare_taps(
        reference, candidate, ("patch", "block", "depth"), tolerances
    )
    assert [result.name for result in results] == ["patch", "block", "depth"]
    assert [result.passed for result in results] == [True, False, False]
    assert first_failure == "block"


def test_tap_comparison_rejects_missing_or_shape_mismatched_tensors():
    with pytest.raises(ValueError, match="tap mismatch"):
        compare_taps({"a": np.ones(1)}, {}, ("a",), {"a": {"max_abs": 1}})
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_taps(
            {"a": np.ones(1)},
            {"a": np.ones(2)},
            ("a",),
            {"a": {"max_abs": 1}},
        )


def test_camera_angular_metrics_match_known_rotations_and_directions():
    angle = np.radians(30.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    assert camera_rotation_geodesic_degrees(np.eye(3), rotation) == pytest.approx(30.0)
    direction_error = camera_translation_direction_degrees(
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    )
    assert direction_error == pytest.approx(90.0)
    with pytest.raises(ValueError, match="zero vector"):
        camera_translation_direction_degrees(np.zeros(3), np.ones(3))
