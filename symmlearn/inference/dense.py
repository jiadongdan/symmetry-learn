"""Batched probability and dense-grid inference."""

from __future__ import annotations

from collections.abc import Callable
from math import ceil
from typing import Any

import numpy as np

from symmlearn.patches import dense_coordinate_grid, iter_patch_batches

from .results import DensePrediction


def predict_probabilities(
    model: Any,
    patches: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return logits and probabilities for a finite patch batch."""
    import torch

    values = np.asarray(patches, dtype=np.float32)
    if values.ndim != 4 or not np.isfinite(values).all():
        raise ValueError("Prediction patches must be a finite four-dimensional array.")
    model.to(device).eval()
    logits_batches = []
    probability_batches = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            inputs = torch.from_numpy(values[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            )
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            logits_batches.append(logits.cpu().numpy())
            probability_batches.append(probabilities.cpu().numpy())
    return (
        np.concatenate(logits_batches).astype(np.float32, copy=False),
        np.concatenate(probability_batches).astype(np.float32, copy=False),
    )


def predict_dense(
    model: Any,
    features: np.ndarray,
    *,
    patch_size: int,
    stride: int,
    batch_size: int,
    device: str,
    expected_channels: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> DensePrediction:
    """Classify a dense grid without materializing every patch simultaneously."""
    coordinates, x_values, y_values = dense_coordinate_grid(
        (int(features.shape[1]), int(features.shape[2])),
        patch_size,
        stride,
    )
    logits_batches = []
    probability_batches = []
    total_batches = ceil(len(coordinates) / batch_size)
    if progress_callback is not None:
        progress_callback(0, total_batches)
    for batch_index, patches in enumerate(iter_patch_batches(
        features,
        coordinates,
        patch_size=patch_size,
        batch_size=batch_size,
        expected_channels=expected_channels,
    )):
        logits, probabilities = predict_probabilities(
            model,
            patches,
            batch_size=batch_size,
            device=device,
        )
        logits_batches.append(logits)
        probability_batches.append(probabilities)
        if progress_callback is not None:
            progress_callback(batch_index + 1, total_batches)
    logits = np.concatenate(logits_batches)
    probabilities = np.concatenate(probability_batches)
    predictions = probabilities.argmax(axis=1).astype(np.int16)
    confidence = probabilities.max(axis=1).astype(np.float32)
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1
    ).astype(np.float32)
    grid_shape = (len(y_values), len(x_values))
    return DensePrediction(
        coordinates_xy=coordinates,
        x_coordinates=x_values,
        y_coordinates=y_values,
        logits=logits,
        probabilities=probabilities,
        predictions=predictions,
        confidence=confidence,
        entropy=entropy,
        prediction_grid=predictions.reshape(grid_shape),
        confidence_grid=confidence.reshape(grid_shape),
        entropy_grid=entropy.reshape(grid_shape),
    )
