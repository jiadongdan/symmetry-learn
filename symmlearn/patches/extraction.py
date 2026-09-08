"""Extract channel-first square patches at XY coordinates."""

from __future__ import annotations

from typing import Iterator

import numpy as np


def extract_patches(
    features: np.ndarray,
    coordinates_xy: np.ndarray,
    patch_size: int,
    *,
    expected_channels: int | None = None,
) -> np.ndarray:
    """Extract a finite batch of channel-first square patches."""
    values = np.asarray(features, dtype=np.float32)
    coordinates = np.asarray(coordinates_xy, dtype=np.int32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Features must be a finite channel-first array.")
    if coordinates.ndim != 2 or coordinates.shape[1:] != (2,):
        raise ValueError("Patch coordinates must have shape (samples, 2).")
    if len(coordinates) == 0:
        raise ValueError("At least one patch coordinate is required.")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    half = patch_size // 2
    patches = [
        values[:, y - half : y - half + patch_size, x - half : x - half + patch_size]
        for x, y in coordinates
    ]
    result = np.stack(patches).astype(np.float32, copy=False)
    channels = values.shape[0] if expected_channels is None else expected_channels
    if result.shape[1:] != (channels, patch_size, patch_size):
        raise RuntimeError("Patch extraction produced an incomplete patch.")
    return result


def iter_patch_batches(
    features: np.ndarray,
    coordinates_xy: np.ndarray,
    *,
    patch_size: int,
    batch_size: int,
    expected_channels: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield bounded patch batches without materializing the full dense set."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(coordinates_xy), batch_size):
        yield extract_patches(
            features,
            coordinates_xy[start : start + batch_size],
            patch_size,
            expected_channels=expected_channels,
        )
