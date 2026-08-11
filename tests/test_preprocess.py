from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vggt_mlx.utils.load_fn import load_and_preprocess_images


ROOT = Path(__file__).parent / "fixtures"


def test_preprocessing_reproduces_oracle_input():
    paths = [ROOT / "sample_images" / f"view_{index}.jpg" for index in range(3)]
    actual = np.asarray(load_and_preprocess_images(paths))
    with np.load(ROOT / "oracle_3view.npz") as fixture:
        expected = fixture["input"][0].transpose(0, 2, 3, 1)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)


def test_pad_mode_handles_portrait_rgba_and_white_transparency(tmp_path):
    rgba = np.zeros((20, 10, 4), dtype=np.uint8)
    path = tmp_path / "portrait.png"
    Image.fromarray(rgba, "RGBA").save(path)
    actual = np.asarray(load_and_preprocess_images([path], mode="pad"))
    assert actual.shape == (1, 518, 518, 3)
    assert np.all(actual[:, :, :120] == 1.0)


def test_mixed_shapes_are_center_padded_with_white(tmp_path):
    wide = tmp_path / "wide.png"
    square = tmp_path / "square.png"
    Image.new("RGB", (40, 10), "black").save(wide)
    Image.new("RGB", (20, 20), "black").save(square)
    actual = np.asarray(load_and_preprocess_images([wide, square]))
    assert actual.shape == (2, 518, 518, 3)
    assert np.all(actual[0, 0] == 1.0)
    assert np.all(actual[1] == 0.0)


def test_preprocess_validates_inputs():
    with pytest.raises(ValueError, match="At least one"):
        load_and_preprocess_images([])
    with pytest.raises(ValueError, match="mode"):
        load_and_preprocess_images([ROOT / "sample_images" / "view_0.jpg"], mode="bad")
