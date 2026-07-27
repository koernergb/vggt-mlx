import numpy as np
import pytest


mx = pytest.importorskip("mlx.core")

from vggt_mlx.layers.block import Block


def test_zeroed_block_is_residual_identity():
    mx.set_default_device(mx.cpu)
    block = Block(
        dim=4,
        num_heads=1,
        mlp_ratio=2.0,
        qk_norm=False,
        init_values=0.01,
    )

    block.attn.qkv.weight = mx.zeros((12, 4))
    block.attn.qkv.bias = mx.zeros((12,))
    block.attn.proj.weight = mx.zeros((4, 4))
    block.attn.proj.bias = mx.zeros((4,))
    block.mlp.fc1.weight = mx.zeros((8, 4))
    block.mlp.fc1.bias = mx.zeros((8,))
    block.mlp.fc2.weight = mx.zeros((4, 8))
    block.mlp.fc2.bias = mx.zeros((4,))
    block.eval()

    inputs = mx.arange(24, dtype=mx.float32).reshape(2, 3, 4)
    outputs = block(inputs)
    mx.eval(outputs)
    np.testing.assert_array_equal(np.asarray(outputs), np.asarray(inputs))


def test_layerscale_matches_verified_aggregator_initialization():
    block = Block(dim=8, num_heads=2, qk_norm=True, init_values=0.01)
    mx.eval(block.ls1.gamma, block.ls2.gamma)
    np.testing.assert_allclose(np.asarray(block.ls1.gamma), 0.01, atol=1e-7)
    np.testing.assert_allclose(np.asarray(block.ls2.gamma), 0.01, atol=1e-7)
    assert block.mlp.fc1.weight.shape == (32, 8)
    assert block.mlp.fc2.weight.shape == (8, 32)
