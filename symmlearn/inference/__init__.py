"""Patch-batch and dense inference for fine-tuned models."""

from .dense import predict_dense, predict_probabilities
from .results import DensePrediction

__all__ = ["DensePrediction", "predict_dense", "predict_probabilities"]
