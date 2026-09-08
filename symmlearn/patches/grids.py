"""Coordinate grids for patch-based model inputs."""

from __future__ import annotations

import numpy as np


def valid_center_bounds(
    image_shape: tuple[int, int],
    patch_size: int,
) -> tuple[int, int, int, int]:
    """Return inclusive XY bounds whose patches remain inside an image."""
    height, width = image_shape
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    before = patch_size // 2
    after = patch_size - before
    if height < patch_size or width < patch_size:
        raise ValueError("The image is smaller than the classifier patch.")
    return before, width - after, before, height - after


def dense_coordinate_grid(
    image_shape: tuple[int, int],
    patch_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create an XY coordinate grid for dense patch inference."""
    if stride <= 0:
        raise ValueError("stride must be positive.")
    min_x, max_x, min_y, max_y = valid_center_bounds(image_shape, patch_size)
    x_values = np.arange(min_x, max_x + 1, stride, dtype=np.int32)
    y_values = np.arange(min_y, max_y + 1, stride, dtype=np.int32)
    xx, yy = np.meshgrid(x_values, y_values, indexing="xy")
    coordinates = np.column_stack((xx.ravel(), yy.ravel())).astype(np.int32)
    return coordinates, x_values, y_values
