import numpy as np
import pytest


mx = pytest.importorskip("mlx.core")

from vggt_mlx.layers.attention import Attention


def configure_identity_attention():
    attention = Attention(
        dim=4,
        num_heads=1,
        qkv_bias=True,
        proj_bias=True,
        qk_norm=False,
    )
    identity = mx.eye(4, dtype=mx.float32)
    attention.qkv.weight = mx.concatenate((identity, identity, identity), axis=0)
    attention.qkv.bias = mx.zeros((12,), dtype=mx.float32)
    attention.proj.weight = identity
    attention.proj.bias = mx.zeros((4,), dtype=mx.float32)
    attention.eval()
    return attention


def manual_attention(streams):
    scale = streams.shape[-1] ** -0.5
    scores = (streams @ streams.transpose(0, 2, 1)) * scale
    return mx.softmax(scores, axis=-1) @ streams


def test_frame_and_global_scopes_match_manual_references():
    mx.set_default_device(mx.cpu)
    attention = configure_identity_attention()
    # Two frames with two tokens each. Frame attention treats frames as batch
    # items; global attention flattens them into one four-token stream.
    frame_streams = mx.array(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        ],
        dtype=mx.float32,
    )

    frame_actual = attention(frame_streams)
    frame_expected = manual_attention(frame_streams)

    global_stream = frame_streams.reshape(1, 4, 4)
    global_actual = attention(global_stream)
    global_expected = manual_attention(global_stream)
    mx.eval(frame_actual, frame_expected, global_actual, global_expected)

    np.testing.assert_allclose(
        np.asarray(frame_actual),
        np.asarray(frame_expected),
        atol=1e-5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(global_actual),
        np.asarray(global_expected),
        atol=1e-5,
        rtol=0.0,
    )
    assert not np.allclose(
        np.asarray(frame_actual).reshape(1, 4, 4),
        np.asarray(global_actual),
    )


def test_qk_norm_matches_verified_head_dimension():
    attention = Attention(dim=1024, num_heads=16, qk_norm=True)
    assert attention.head_dim == 64
    assert tuple(attention.q_norm.weight.shape) == (64,)
    assert tuple(attention.k_norm.weight.shape) == (64,)
