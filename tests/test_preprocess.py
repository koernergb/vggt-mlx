from pathlib import Path

import numpy as np

from vggt_mlx.utils.load_fn import load_and_preprocess_images


ROOT = Path(__file__).parent / "fixtures"


def test_preprocessing_reproduces_oracle_input():
    paths = [ROOT / "sample_images" / f"view_{index}.jpg" for index in range(3)]
    actual = np.asarray(load_and_preprocess_images(paths))
    with np.load(ROOT / "oracle_3view.npz") as fixture:
        expected = fixture["input"][0].transpose(0, 2, 3, 1)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
