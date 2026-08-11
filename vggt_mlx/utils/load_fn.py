"""Image loading and preprocessing compatible with upstream VGGT."""

from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image


def load_and_preprocess_images(image_path_list, mode: str = "crop"):
    if not image_path_list:
        raise ValueError("At least one image is required")
    if mode not in {"crop", "pad"}:
        raise ValueError("mode must be 'crop' or 'pad'")

    target = 518
    images = []
    for path in image_path_list:
        image = Image.open(Path(path))
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        image = image.convert("RGB")
        width, height = image.size
        if mode == "crop" or width >= height:
            new_width = target
            new_height = round(height * target / width / 14) * 14
        else:
            new_height = target
            new_width = round(width * target / height / 14) * 14
        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        if mode == "crop" and new_height > target:
            start = (new_height - target) // 2
            array = array[start : start + target]
        if mode == "pad":
            canvas = np.ones((target, target, 3), dtype=np.float32)
            top = (target - array.shape[0]) // 2
            left = (target - array.shape[1]) // 2
            canvas[top : top + array.shape[0], left : left + array.shape[1]] = array
            array = canvas
        images.append(array)

    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)
    padded = []
    for image in images:
        canvas = np.ones((max_height, max_width, 3), dtype=np.float32)
        top = (max_height - image.shape[0]) // 2
        left = (max_width - image.shape[1]) // 2
        canvas[top : top + image.shape[0], left : left + image.shape[1]] = image
        padded.append(canvas)
    return mx.array(np.stack(padded))
