import numpy as np
import pytest


mx = pytest.importorskip("mlx.core")

from vggt_mlx.heads.dpt_head import DPTHead, ResidualConvUnit


def test_residual_unit_matches_upstream_inplace_relu_skip():
    unit = ResidualConvUnit(1)
    unit.conv1.weight = mx.zeros_like(unit.conv1.weight)
    unit.conv1.bias = mx.zeros_like(unit.conv1.bias)
    unit.conv2.weight = mx.zeros_like(unit.conv2.weight)
    unit.conv2.bias = mx.zeros_like(unit.conv2.bias)
    actual = unit(mx.array([[[[-2.0], [3.0]]]]))
    mx.eval(actual)
    np.testing.assert_array_equal(actual, [[[[0.0], [3.0]]]])


def test_dpt_rejects_missing_features_and_unknown_activation():
    model = DPTHead(dim_in=8, output_dim=2, features=4, out_channels=(4, 4, 4, 4))
    images = mx.zeros((1, 1, 28, 28, 3))
    with pytest.raises(ValueError, match="Missing aggregator"):
        model([None] * 24, images, 5)
    model.activation = "mystery"
    with pytest.raises(ValueError, match="Unsupported DPT activation"):
        model._activate(mx.zeros((1, 1, 1, 2)))
