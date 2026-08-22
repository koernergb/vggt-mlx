"""Equivalent shared-input contracts for MLX and the PyTorch reference."""

import numpy as np
import pytest

from vggt_mlx.benchmark.adapters import PUBLIC_OUTPUTS, PyTorchMPSAdapter


torch = pytest.importorskip("torch")


class FakeOfficialVGGT:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, images, query_points=None):
        assert query_points is None
        batch, views, _, height, width = images.shape
        return {
            "pose_enc": torch.zeros((batch, views, 9), device=images.device),
            "pose_enc_list": [],
            "depth": torch.ones((batch, views, height, width, 1), device=images.device),
            "depth_conf": torch.ones((batch, views, height, width), device=images.device),
            "world_points": torch.ones((batch, views, height, width, 3), device=images.device),
            "world_points_conf": torch.ones((batch, views, height, width), device=images.device),
            "images": images,
        }


def fake_pose_decoder(pose, image_size):
    batch, views, _ = pose.shape
    device = pose.device
    return (
        torch.zeros((batch, views, 3, 4), device=device),
        torch.eye(3, device=device).expand(batch, views, 3, 3).clone(),
    )


def adapter():
    return PyTorchMPSAdapter(
        FakeOfficialVGGT(), device="cpu", pose_decoder=fake_pose_decoder
    )


def test_pytorch_adapter_preserves_shared_pixels_and_layout_contract():
    shared = np.arange(2 * 14 * 28 * 3, dtype=np.float32).reshape(2, 14, 28, 3)
    shared /= shared.max()
    prepared = adapter().prepare_input(shared)
    assert prepared.shape == (1, 2, 3, 14, 28)
    np.testing.assert_array_equal(
        prepared.numpy().transpose(0, 1, 3, 4, 2), shared[None]
    )


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 14, 28, 3), dtype=np.float64),
        np.full((2, 14, 28, 3), np.nan, dtype=np.float32),
        np.full((2, 14, 28, 3), 2.0, dtype=np.float32),
        np.zeros((2, 14, 28, 4), dtype=np.float32),
    ],
)
def test_pytorch_adapter_rejects_non_equivalent_inputs(bad):
    with pytest.raises(ValueError):
        adapter().prepare_input(bad)


def test_pytorch_adapter_returns_exact_public_output_workload():
    shared = np.zeros((2, 14, 28, 3), dtype=np.float32)
    reference = adapter()
    output = reference.forward_tensors(reference.prepare_input(shared))
    assert set(output) == set(PUBLIC_OUTPUTS)
    arrays = reference.to_numpy(output)
    assert arrays["depth"].shape == (1, 2, 14, 28, 1)
    assert arrays["extrinsic"].shape == (1, 2, 3, 4)
    assert arrays["intrinsic"].shape == (1, 2, 3, 3)
