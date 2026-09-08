"""Input and output validation for the eight-channel pipeline."""

from __future__ import annotations

import numpy as np


def validate_unit_image(
    image: np.ndarray,
    *,
    minimum_size: int | None = None,
    subject: str = "Input",
) -> np.ndarray:
    """Return a finite float32 image in the shared unit range."""
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"{subject} must be a finite two-dimensional array.")
    if minimum_size is not None and min(values.shape) < minimum_size:
        raise ValueError(
            f"Both {subject.lower()} dimensions must be at least "
            f"{minimum_size} pixels."
        )
    if float(values.min()) < -1e-6 or float(values.max()) > 1.000001:
        raise ValueError(f"{subject} must lie in the shared [0, 1] space.")
    return values


def validate_feature_array(features: np.ndarray, image_shape: tuple[int, int]) -> None:
    """Validate the fixed output shape and finite-value contract."""
    if features.shape != (8, *image_shape) or not np.isfinite(features).all():
        raise RuntimeError("The symmetry representation failed shape or finite validation.")
