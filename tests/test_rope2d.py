import math

import numpy as np
import pytest


mx = pytest.importorskip("mlx.core")

from vggt_mlx.layers.rope2d import PositionGetter, RotaryPositionEmbedding2D


def test_position_getter_uses_row_major_yx_coordinates():
    positions = PositionGetter()(batch_size=1, height=2, width=2)
    mx.eval(positions)
    np.testing.assert_array_equal(
        np.asarray(positions),
        np.array([[[0, 0], [0, 1], [1, 0], [1, 1]]], dtype=np.int32),
    )


def test_rope2d_matches_analytic_four_dimensional_rotation():
    mx.set_default_device(mx.cpu)
    rope = RotaryPositionEmbedding2D(frequency=100.0)
    tokens = mx.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=mx.float32)
    positions = mx.array([[[1, 1]]], dtype=mx.int32)

    actual = rope(tokens, positions)
    mx.eval(actual)

    cosine = math.cos(1.0)
    sine = math.sin(1.0)
    expected = np.array(
        [
            [
                [
                    1.0 * cosine - 2.0 * sine,
                    2.0 * cosine + 1.0 * sine,
                    3.0 * cosine - 4.0 * sine,
                    4.0 * cosine + 3.0 * sine,
                ]
            ]
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-6, rtol=0.0)


def test_zero_position_is_identity():
    rope = RotaryPositionEmbedding2D()
    tokens = mx.arange(8, dtype=mx.float32).reshape(1, 1, 2, 4)
    positions = mx.zeros((1, 2, 2), dtype=mx.int32)
    actual = rope(tokens, positions)
    mx.eval(actual)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(tokens))
