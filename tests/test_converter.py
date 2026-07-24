from pathlib import Path

import pytest

from vggt_mlx.convert.torch_to_mlx import (
    ConversionReport,
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
