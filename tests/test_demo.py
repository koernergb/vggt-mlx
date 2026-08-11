import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import pytest

from demo import ensure_weights, parse_args, save_depth, save_ply


def test_demo_writes_valid_depth_png(tmp_path: Path):
    path = tmp_path / "depth.png"
    save_depth(np.arange(16, dtype=np.float32).reshape(4, 4), path)
    with Image.open(path) as image:
        assert image.mode == "RGB"
        assert image.size == (4, 4)


def test_demo_ply_filters_invalid_points_and_records_colors(tmp_path: Path):
    path = tmp_path / "points.ply"
    points = np.array([[[[1, 2, 3], [np.nan, 0, 0]]]], dtype=np.float32)
    colors = np.array([[[[1, 0.5, 0], [0, 0, 0]]]], dtype=np.float32)
    save_ply(points, colors, path)
    text = path.read_text(encoding="utf-8")
    assert "element vertex 1" in text
    assert "1 2 3 255 127 0" in text


def test_demo_rejects_more_than_four_views(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["demo.py", "a", "b", "c", "d", "e"])
    with pytest.raises(SystemExit):
        parse_args()


def test_missing_weight_conversion_failure_is_actionable(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(SystemExit, match="dev dependencies"):
        ensure_weights(tmp_path / "missing.safetensors")
