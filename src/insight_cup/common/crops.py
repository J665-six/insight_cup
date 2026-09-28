"""Shared crop policies for downstream recognition modules."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


KNIFE_CROP_CONTEXT_SCALE = 1.15


def square_context_crop(
    image: np.ndarray,
    bbox_xyxy: Sequence[float],
    context_scale: float = KNIFE_CROP_CONTEXT_SCALE,
    fill_value: int = 127,
) -> np.ndarray:
    """Crop a square around an xyxy box while preserving out-of-frame padding."""

    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a non-empty three-channel image")
    if len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must contain four values")
    if context_scale < 1.0:
        raise ValueError("context_scale must be at least 1.0")
    if not 0 <= fill_value <= 255:
        raise ValueError("fill_value must be between 0 and 255")

    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError("bbox_xyxy contains a non-finite value")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_xyxy must have positive width and height")

    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    side = max(2, int(math.ceil(max(x2 - x1, y2 - y1) * context_scale)))
    left = int(math.floor(center_x - side * 0.5))
    top = int(math.floor(center_y - side * 0.5))
    right = left + side
    bottom = top + side

    height, width = image.shape[:2]
    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(width, right)
    source_bottom = min(height, bottom)

    crop = np.full((side, side, 3), fill_value, dtype=image.dtype)
    if source_right <= source_left or source_bottom <= source_top:
        return crop

    destination_left = source_left - left
    destination_top = source_top - top
    destination_right = destination_left + (source_right - source_left)
    destination_bottom = destination_top + (source_bottom - source_top)
    crop[destination_top:destination_bottom, destination_left:destination_right] = (
        image[source_top:source_bottom, source_left:source_right]
    )
    return crop
