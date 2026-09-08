"""Validation for user-selected few-shot support points."""

from __future__ import annotations

from typing import Any

import numpy as np

from .grids import valid_center_bounds


def validate_support(
    image_shape: tuple[int, int],
    coordinates_xy: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    options: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Validate support coordinates, labels, names, counts, and boundaries."""
    coordinates = np.asarray(coordinates_xy, dtype=np.int32)
    resolved_labels = np.asarray(labels, dtype=np.int64)
    names = [str(name).strip() for name in class_names]
    if coordinates.ndim != 2 or coordinates.shape[1:] != (2,):
        raise ValueError("Support coordinates must have shape (samples, 2).")
    if resolved_labels.shape != (len(coordinates),):
        raise ValueError("Support labels must align with support coordinates.")
    if len(names) < 2 or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("At least two unique nonempty local class names are required.")
    expected_labels = set(range(len(names)))
    if set(resolved_labels.tolist()) != expected_labels:
        raise ValueError("Support labels must cover every contiguous local class index.")
    counts = np.bincount(resolved_labels, minlength=len(names))
    if np.any(counts < options["minimum_shots_per_class"]):
        raise ValueError("Every class must satisfy the minimum support-point count.")
    if np.any(counts > options["maximum_shots_per_class"]):
        raise ValueError("A class exceeds the maximum support-point count.")
    if len({tuple(point) for point in coordinates.tolist()}) != len(coordinates):
        raise ValueError("Support coordinates must be unique across all classes.")
    min_x, max_x, min_y, max_y = valid_center_bounds(
        image_shape, options["classifier_patch_size"]
    )
    for x, y in coordinates:
        if not min_x <= x <= max_x or not min_y <= y <= max_y:
            raise ValueError(f"Support point ({x}, {y}) cannot provide a full patch.")
    return coordinates, resolved_labels, names
