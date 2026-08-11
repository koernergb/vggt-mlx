from pathlib import Path

import pytest

from vggt_mlx.convert.torch_to_mlx import (
    ConversionReport,
    convert_torch_to_mlx,
    open_checkpoint,
    parse_shape_inventory,
    remap_key,
)


def test_key_mapping_preserves_model_names_and_drops_track_head():
    assert remap_key("aggregator.frame_blocks.0.attn.qkv.weight") == (
        "aggregator.frame_blocks.0.attn.qkv.weight"
    )
    assert remap_key("track_head.blocks.0.weight") is None
    with pytest.raises(KeyError):
        remap_key("unexpected.weight")


def test_shape_inventory_parser(tmp_path: Path):
    inventory = tmp_path / "state_dict_keys.txt"
    inventory.write_text("linear.weight\t(4, 8)\nscalar\t()\n", encoding="utf-8")
    assert parse_shape_inventory(inventory) == {
        "linear.weight": (4, 8),
        "scalar": (),
    }


def test_report_gate_rejects_unmapped_keys():
    report = ConversionReport(source="source", output="output")
    report.unmapped_keys.append("unexpected.weight")
    with pytest.raises(RuntimeError):
        report.assert_complete()


def test_converter_permutations_and_report_are_exact():
    torch = pytest.importorskip("torch")
    mx = pytest.importorskip("mlx.core")

    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.aggregator = torch.nn.Module()
            self.aggregator.conv = torch.nn.Conv2d(2, 3, 2, bias=False)
            self.aggregator.deconv = torch.nn.ConvTranspose2d(2, 3, 2, bias=False)
            self.aggregator.linear = torch.nn.Linear(2, 3, bias=False)
            self.track_head = torch.nn.Linear(2, 1)

    model = Root()
    state = model.state_dict()
    converted, report = convert_torch_to_mlx(
        state, model, mx, torch.nn, expected_shapes={k: tuple(v.shape) for k, v in state.items()}
    )
    np = pytest.importorskip("numpy")
    np.testing.assert_array_equal(
        converted["aggregator.conv.weight"],
        state["aggregator.conv.weight"].numpy().transpose(0, 2, 3, 1),
    )
    np.testing.assert_array_equal(
        converted["aggregator.deconv.weight"],
        state["aggregator.deconv.weight"].numpy().transpose(1, 2, 3, 0),
    )
    np.testing.assert_array_equal(
        converted["aggregator.linear.weight"], state["aggregator.linear.weight"].numpy()
    )
    assert report.mapped_keys == 3
    assert len(report.skipped_track_keys) == 2
    assert report.permuted_convs == 2
    report.assert_complete()


def test_safetensors_checkpoint_opens_as_mapping(tmp_path: Path):
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    path = tmp_path / "model.safetensors"
    save_file({"aggregator.value": torch.arange(3)}, path)
    with open_checkpoint(torch, path) as checkpoint:
        assert list(checkpoint) == ["aggregator.value"]
        assert checkpoint["aggregator.value"].tolist() == [0, 1, 2]
