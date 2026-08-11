"""MLX implementation of VGGT's DPT dense-prediction decoder."""

from __future__ import annotations

from typing import Optional, Sequence

import mlx.core as mx
import mlx.nn as nn


def _resize(x, height: int, width: int):
    if x.shape[1:3] == (height, width):
        return x
    return nn.Upsample(
        scale_factor=(height / x.shape[1], width / x.shape[2]),
        mode="linear",
        align_corners=True,
    )(x)


def _position_embedding(height: int, width: int, channels: int, aspect: float):
    diagonal = (aspect * aspect + 1.0) ** 0.5
    span_x, span_y = aspect / diagonal, 1.0 / diagonal
    x = mx.linspace(
        -span_x * (width - 1) / width,
        span_x * (width - 1) / width,
        width,
    )
    y = mx.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        height,
    )
    yy, xx = mx.meshgrid(y, x, indexing="ij")

    half = channels // 2
    omega = 1.0 / (100.0 ** (mx.arange(half // 2) / (half / 2.0)))

    def encode(position):
        phase = position[..., None] * omega
        return mx.concatenate((mx.sin(phase), mx.cos(phase)), axis=-1)

    return mx.concatenate((encode(xx), encode(yy)), axis=-1)


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)

    def __call__(self, x):
        # Upstream uses an in-place ReLU, so its skip tensor has already been
        # rectified by the time it is added back. Preserve that exact behavior.
        skip = nn.relu(x)
        residual = self.conv1(skip)
        residual = self.conv2(nn.relu(residual))
        return residual + skip


class FeatureFusionBlock(nn.Module):
    def __init__(self, features: int, has_residual: bool = True) -> None:
        super().__init__()
        self.has_residual = has_residual
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, 1)

    def __call__(self, x, residual=None, size=None):
        if self.has_residual:
            if residual is None:
                raise ValueError("Fusion block requires a residual feature map")
            x = x + self.resConfUnit1(residual)
        x = self.resConfUnit2(x)
        if size is None:
            size = (x.shape[1] * 2, x.shape[2] * 2)
        return self.out_conv(_resize(x, int(size[0]), int(size[1])))


class _Scratch(nn.Module):
    def __init__(self, out_channels: Sequence[int], features: int, output_dim: int):
        super().__init__()
        self.layer1_rn = nn.Conv2d(out_channels[0], features, 3, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(out_channels[1], features, 3, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(out_channels[2], features, 3, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(out_channels[3], features, 3, padding=1, bias=False)
        self.refinenet1 = FeatureFusionBlock(features)
        self.refinenet2 = FeatureFusionBlock(features)
        self.refinenet3 = FeatureFusionBlock(features)
        self.refinenet4 = FeatureFusionBlock(features, has_residual=False)
        self.output_conv1 = nn.Conv2d(features, features // 2, 3, padding=1)
        # A plain list preserves upstream numeric keys (`output_conv2.0/2`).
        self.output_conv2 = [
            nn.Conv2d(features // 2, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, output_dim, 1),
        ]

    def finish(self, x):
        for layer in self.output_conv2:
            x = layer(x)
        return x


class DPTHead(nn.Module):
    """Shared DPT decoder used by VGGT's depth and point heads."""

    def __init__(
        self,
        dim_in: int = 2048,
        patch_size: int = 14,
        output_dim: int = 4,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        intermediate_layer_idx: Sequence[int] = (4, 11, 17, 23),
        pos_embed: bool = True,
        down_ratio: int = 1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = tuple(intermediate_layer_idx)
        self.norm = nn.LayerNorm(dim_in)
        self.projects = [nn.Conv2d(dim_in, channels, 1) for channels in out_channels]
        self.resize_layers = [
            nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, stride=4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, stride=2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], 3, stride=2, padding=1),
        ]
        self.scratch = _Scratch(out_channels, features, output_dim)

    def _add_position(self, x, image_height: int, image_width: int):
        position = _position_embedding(
            x.shape[1], x.shape[2], x.shape[-1], image_width / image_height
        )
        return x + position[None] * 0.1

    def _activate(self, output):
        prediction, confidence = output[..., :-1], output[..., -1]
        if self.activation == "exp":
            prediction = mx.exp(prediction)
        elif self.activation == "inv_log":
            prediction = mx.sign(prediction) * mx.expm1(mx.abs(prediction))
        elif self.activation == "linear":
            pass
        else:
            raise ValueError(f"Unsupported DPT activation: {self.activation}")

        if self.conf_activation == "expp1":
            confidence = 1.0 + mx.exp(confidence)
        elif self.conf_activation == "expp0":
            confidence = mx.exp(confidence)
        elif self.conf_activation == "sigmoid":
            confidence = mx.sigmoid(confidence)
        else:
            raise ValueError(
                f"Unsupported confidence activation: {self.conf_activation}"
            )
        return prediction, confidence

    def __call__(
        self,
        aggregated_tokens_list,
        images,
        patch_start_idx: int,
        frames_chunk_size: Optional[int] = None,
    ):
        del frames_chunk_size  # MLX unified memory does not need PyTorch chunking.
        if images.ndim != 5:
            raise ValueError(f"Expected five-dimensional images, got {images.shape}")
        if images.shape[-1] == 3:
            batch, frames, height, width, _ = images.shape
        elif images.shape[2] == 3:
            batch, frames, _, height, width = images.shape
        else:
            raise ValueError("Images must be NCHW or NHWC with three channels")

        patch_height, patch_width = height // self.patch_size, width // self.patch_size
        pyramid = []
        for decoder_index, layer_index in enumerate(self.intermediate_layer_idx):
            tokens = aggregated_tokens_list[layer_index]
            if tokens is None:
                raise ValueError(f"Missing aggregator output at layer {layer_index}")
            tokens = tokens[:, :, patch_start_idx:].reshape(
                batch * frames, -1, tokens.shape[-1]
            )
            feature = self.norm(tokens).reshape(
                batch * frames, patch_height, patch_width, tokens.shape[-1]
            )
            feature = self.projects[decoder_index](feature)
            if self.pos_embed:
                feature = self._add_position(feature, height, width)
            pyramid.append(self.resize_layers[decoder_index](feature))

        layer1, layer2, layer3, layer4 = pyramid
        layer1 = self.scratch.layer1_rn(layer1)
        layer2 = self.scratch.layer2_rn(layer2)
        layer3 = self.scratch.layer3_rn(layer3)
        layer4 = self.scratch.layer4_rn(layer4)
        output = self.scratch.refinenet4(layer4, size=layer3.shape[1:3])
        output = self.scratch.refinenet3(output, layer3, size=layer2.shape[1:3])
        output = self.scratch.refinenet2(output, layer2, size=layer1.shape[1:3])
        output = self.scratch.refinenet1(output, layer1)
        output = self.scratch.output_conv1(output)
        output = _resize(
            output,
            int(patch_height * self.patch_size / self.down_ratio),
            int(patch_width * self.patch_size / self.down_ratio),
        )
        if self.pos_embed:
            output = self._add_position(output, height, width)
        output = self.scratch.finish(output)
        prediction, confidence = self._activate(output)
        return (
            prediction.reshape(batch, frames, *prediction.shape[1:]),
            confidence.reshape(batch, frames, *confidence.shape[1:]),
        )
