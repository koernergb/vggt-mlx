"""Profile VGGT-MLX inference stages on one shared scene."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import median

import mlx.core as mx
import mlx.nn as nn

from vggt_mlx.models.aggregator import slice_expand_and_flatten
from vggt_mlx.models.vggt import VGGT
from vggt_mlx.utils.load_fn import load_and_preprocess_images
from vggt_mlx.utils.pose_enc import pose_encoding_to_extri_intri


ROOT = Path(__file__).resolve().parents[1]


def timed(function, *, trials: int):
    samples = []
    value = None
    for _ in range(trials):
        started = time.perf_counter()
        value = function()
        mx.eval(value)
        samples.append((time.perf_counter() - started) * 1000)
    return value, samples


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="fp32"
    )
    parser.add_argument("--quantize", type=int, choices=(4, 6, 8))
    parser.add_argument(
        "--weights", type=Path, default=ROOT / "weights" / "vggt-1b-mlx.safetensors"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    images = load_and_preprocess_images(args.images)
    if images.ndim == 4:
        images = images[None]
    model = VGGT()
    model.load_weights(list(mx.load(str(args.weights)).items()), strict=True)
    dtype = {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}[
        args.precision
    ]
    model.set_dtype(dtype)
    if args.quantize:
        nn.quantize(
            model,
            group_size=64,
            bits=args.quantize,
            class_predicate=lambda _path, module: (
                isinstance(module, nn.Linear) and module.weight.shape[-1] % 64 == 0
            ),
        )
    images = images.astype(dtype)
    model.eval()
    mx.eval(model.parameters())
    forward = mx.compile(model, inputs=model.state) if args.compile else model
    demo_function = lambda value: model(
        value,
        outputs=("pose_enc", "depth", "depth_conf", "extrinsic", "intrinsic"),
    )
    demo_forward = (
        mx.compile(demo_function, inputs=model.state)
        if args.compile
        else demo_function
    )
    pose_function = lambda value: model(value, outputs=("pose_enc",))
    pose_forward = (
        mx.compile(pose_function, inputs=model.state)
        if args.compile
        else pose_function
    )
    def legacy_pose_function(value):
        tokens, _ = model.aggregator(value)
        return model.camera_head(tokens)[-1]

    legacy_pose_forward = (
        mx.compile(legacy_pose_function, inputs=model.state)
        if args.compile
        else legacy_pose_function
    )

    # Untimed warmup establishes kernels and memory allocations.
    warmup = forward(images)
    mx.eval(warmup)
    mx.eval(demo_forward(images))
    mx.eval(pose_forward(images))
    mx.eval(legacy_pose_forward(images))

    aggregator = model.aggregator
    batch, frames, height, width, _ = images.shape
    mean = mx.array([0.485, 0.456, 0.406], dtype=images.dtype)
    std = mx.array([0.229, 0.224, 0.225], dtype=images.dtype)
    normalized = ((images - mean) / std).reshape(
        batch * frames, height, width, 3
    )
    patch_tokens, patch_embed_ms = timed(
        lambda: aggregator.patch_embed(normalized), trials=args.trials
    )
    camera_token = slice_expand_and_flatten(
        aggregator.camera_token, batch, frames
    )
    register_token = slice_expand_and_flatten(
        aggregator.register_token, batch, frames
    )
    block_tokens = mx.concatenate(
        (camera_token, register_token, patch_tokens), axis=1
    )
    patch_positions = aggregator.position_getter(
        batch * frames,
        height // aggregator.patch_size,
        width // aggregator.patch_size,
    ) + 1
    special_positions = mx.zeros(
        (batch * frames, aggregator.patch_start_idx, 2),
        dtype=patch_positions.dtype,
    )
    block_positions = mx.concatenate(
        (special_positions, patch_positions), axis=1
    )
    block = aggregator.frame_blocks[0]
    norm1 = (
        mx.compile(block.norm1, inputs=block.norm1.state)
        if args.compile
        else block.norm1
    )
    attention_function = lambda value: block.attn(value, pos=block_positions)
    attention = (
        mx.compile(attention_function, inputs=block.attn.state)
        if args.compile
        else attention_function
    )
    norm2 = (
        mx.compile(block.norm2, inputs=block.norm2.state)
        if args.compile
        else block.norm2
    )
    mlp = (
        mx.compile(block.mlp, inputs=block.mlp.state)
        if args.compile
        else block.mlp
    )
    normed1 = norm1(block_tokens)
    attended = attention(normed1)
    residual = block_tokens + block.ls1(attended)
    normed2 = norm2(residual)
    mx.eval(normed1, attended, residual, normed2, mlp(normed2))
    _, norm1_ms = timed(lambda: norm1(block_tokens), trials=args.trials)
    _, attention_ms = timed(lambda: attention(normed1), trials=args.trials)
    _, norm2_ms = timed(lambda: norm2(residual), trials=args.trials)
    _, mlp_ms = timed(lambda: mlp(normed2), trials=args.trials)

    aggregated, aggregator_ms = timed(
        lambda: model.aggregator(images), trials=args.trials
    )
    tokens, patch_start = aggregated
    _, camera_ms = timed(lambda: model.camera_head(tokens)[-1], trials=args.trials)
    _, depth_ms = timed(
        lambda: model.depth_head(tokens, images, patch_start), trials=args.trials
    )
    _, point_ms = timed(
        lambda: model.point_head(tokens, images, patch_start), trials=args.trials
    )
    pose = model.camera_head(tokens)[-1]
    _, geometry_ms = timed(
        lambda: pose_encoding_to_extri_intri(pose, images.shape[2:4]),
        trials=args.trials,
    )
    _, full_ms = timed(lambda: forward(images), trials=args.trials)
    _, demo_ms = timed(lambda: demo_forward(images), trials=args.trials)
    _, pose_ms = timed(lambda: pose_forward(images), trials=args.trials)
    _, legacy_pose_ms = timed(
        lambda: legacy_pose_forward(images), trials=args.trials
    )

    stages = {
        "aggregator": aggregator_ms,
        "dino_backbone": patch_embed_ms,
        "block_norm1": norm1_ms,
        "block_attention": attention_ms,
        "block_norm2": norm2_ms,
        "block_mlp": mlp_ms,
        "camera_head": camera_ms,
        "depth_head": depth_ms,
        "point_head": point_ms,
        "camera_geometry": geometry_ms,
        "full_forward": full_ms,
        "demo_forward": demo_ms,
        "pose_forward": pose_ms,
        "legacy_pose": legacy_pose_ms,
    }
    for name, samples in stages.items():
        print(
            f"{name:16s} median={median(samples):8.2f} ms "
            f"samples={','.join(f'{value:.2f}' for value in samples)}"
        )


if __name__ == "__main__":
    main()
