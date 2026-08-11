"""CPU/FP32 parity gate for the DINOv2 patch-embedding backbone."""

from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = REPOSITORY_ROOT / "weights" / "vggt-1b-mlx.safetensors"
FIXTURE_PATHS = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "oracle_1view.npz",
    REPOSITORY_ROOT / "tests" / "fixtures" / "oracle_3view.npz",
)
PATCH_PREFIX = "aggregator.patch_embed."
# The oracle is generated on a CUDA T4. PyTorch CPU alone differs from that
# fixture by up to 7.5e-5, so leave a similarly small allowance for MLX Metal.
MAX_ABS_DIFF = 2e-4


def extract_patch_embed_weights(weights):
    """Strip the top-level prefix while retaining canonical DINO key names."""
    return [
        (name.removeprefix(PATCH_PREFIX), value)
        for name, value in weights.items()
        if name.startswith(PATCH_PREFIX)
    ]


def test_patch_embed_matches_pytorch_oracles():
    missing = [path for path in (*FIXTURE_PATHS, WEIGHTS_PATH) if not path.is_file()]
    if missing:
        pytest.skip(
            "Card 2.2 requires generated oracle fixtures and converted weights; "
            f"missing: {missing}"
        )

    mx = pytest.importorskip("mlx.core")

    from vggt_mlx.layers.patch_embed import PatchEmbed

    mx.set_default_device(mx.cpu)
    checkpoint = mx.load(str(WEIGHTS_PATH))
    patch_weights = extract_patch_embed_weights(checkpoint)
    assert patch_weights, "Converted checkpoint contains no patch-embed weights"

    model = PatchEmbed()
    model.load_weights(patch_weights, strict=True)
    model.eval()

    for fixture_path in FIXTURE_PATHS:
        with np.load(fixture_path) as fixture:
            assert {"input", "patch_embed"} <= set(fixture.files)
            pytorch_input = fixture["input"].astype(np.float32, copy=False)
            expected = fixture["patch_embed"].astype(np.float32, copy=False)

        assert pytorch_input.ndim == 5, (
            f"{fixture_path.name} input must be [B,S,C,H,W], "
            f"got {pytorch_input.shape}"
        )
        batch, views, channels, height, width = pytorch_input.shape
        assert channels == 3
        mlx_input = pytorch_input.transpose(0, 1, 3, 4, 2).reshape(
            batch * views,
            height,
            width,
            channels,
        )
        mlx_input = (
            mlx_input - np.array([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)

        actual = model(mx.array(mlx_input))
        mx.eval(actual)
        actual = np.asarray(actual)

        assert actual.shape == expected.shape, (
            f"{fixture_path.name}: MLX shape {actual.shape} != "
            f"PyTorch shape {expected.shape}"
        )
        max_abs_diff = float(np.max(np.abs(actual - expected)))
        assert max_abs_diff < MAX_ABS_DIFF, (
            f"{fixture_path.name}: max_abs_diff={max_abs_diff:.8g}, "
            f"required < {MAX_ABS_DIFF}"
        )
