"""Structured dense-inference outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DensePrediction:
    """Dense local-class predictions and uncertainty diagnostics."""

    coordinates_xy: np.ndarray
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    confidence: np.ndarray
    entropy: np.ndarray
    prediction_grid: np.ndarray
    confidence_grid: np.ndarray
    entropy_grid: np.ndarray

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Return the stable array mapping used by Provider transport."""
        return {
            "coordinates_xy": self.coordinates_xy,
            "x_coordinates": self.x_coordinates,
            "y_coordinates": self.y_coordinates,
            "logits": self.logits,
            "probabilities": self.probabilities,
            "predictions": self.predictions,
            "confidence": self.confidence,
            "entropy": self.entropy,
            "prediction_grid": self.prediction_grid,
            "confidence_grid": self.confidence_grid,
            "entropy_grid": self.entropy_grid,
        }
