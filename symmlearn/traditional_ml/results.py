"""Immutable numerical result of one traditional ML validation run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TraditionalMLResult:
    """Dense traditional-ML outputs with a JSON-safe Provider record.

    No serialized estimator object is ever part of the result: the first version
    intentionally keeps training and prediction inside one job.
    """

    arrays: dict[str, np.ndarray]
    record: dict[str, Any]

    @property
    def prediction_grid(self) -> np.ndarray:
        """Return the dense predicted-class grid."""
        return self.arrays["prediction_grid"]

    @property
    def confidence_grid(self) -> np.ndarray:
        """Return the dense maximum-probability grid."""
        return self.arrays["confidence_grid"]

    @property
    def entropy_grid(self) -> np.ndarray:
        """Return the dense predictive-entropy grid."""
        return self.arrays["entropy_grid"]

    @property
    def probabilities(self) -> np.ndarray:
        """Return per-sample class probabilities on the dense grid."""
        return self.arrays["probabilities"]


__all__ = ["TraditionalMLResult"]
