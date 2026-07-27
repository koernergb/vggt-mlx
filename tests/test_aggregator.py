from dataclasses import replace

import numpy as np
import pytest


mx = pytest.importorskip("mlx.core")

from vggt_mlx.config import VGGTConfig
from vggt_mlx.models.aggregator import Aggregator, slice_expand_and_flatten


def test_special_tokens_use_reference_and_shared_other_slots():
    token = mx.array([[[[1.0]], [[2.0]]]])
    expanded = slice_expand_and_flatten(token, batch_size=2, num_frames=3)
    mx.eval(expanded)
    np.testing.assert_array_equal(
        np.asarray(expanded).reshape(2, 3),
        np.array([[1.0, 2.0, 2.0], [1.0, 2.0, 2.0]]),
    )


def test_small_aggregator_returns_cached_frame_global_concatenations():
    mx.set_default_device(mx.cpu)
    config = replace(
        VGGTConfig(),
        img_size=28,
        embed_dim=8,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        intermediate_layer_idx=(0, 1),
        dpt_out_channels=(8, 8, 8, 8),
    )
    model = Aggregator(config)
    model.eval()
    images = mx.zeros((1, 2, 28, 28, 3), dtype=mx.float32)
    outputs, patch_start_idx = model(images)
    mx.eval(*outputs)

    assert patch_start_idx == 5
    assert len(outputs) == 2
    # Four patch tokens plus five special tokens, frame/global concatenated.
    assert outputs[0].shape == (1, 2, 9, 16)
    assert outputs[1].shape == (1, 2, 9, 16)
