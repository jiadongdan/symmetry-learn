"""Common validation for few-shot fine-tuning inputs."""

from __future__ import annotations

import numpy as np


def validate_training_arrays(
    support_patches: np.ndarray,
    support_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated float32 patches and integer labels."""
    patches = np.asarray(support_patches, dtype=np.float32)
    labels = np.asarray(support_labels, dtype=np.int64)
    if patches.ndim != 4 or patches.shape[0] != labels.shape[0]:
        raise ValueError("Support patches and labels have incompatible shapes.")
    if not np.isfinite(patches).all():
        raise ValueError("Support patches contain nonfinite values.")
    return patches, labels
