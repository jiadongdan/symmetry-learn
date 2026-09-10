"""Versioned symmetry provider API for external orchestrators."""

from .contracts import PROVIDER_CONTRACT_VERSION, WORKER_SCHEMA_VERSION
from .registry import model_capabilities

__all__ = [
    "PROVIDER_CONTRACT_VERSION",
    "ProviderPredictionResult",
    "ProviderResult",
    "WORKER_SCHEMA_VERSION",
    "compute_features",
    "few_shot_analyze",
    "model_capabilities",
    "predict_with_fine_tuned_model",
]


def __getattr__(name: str):
    if name in {
        "ProviderPredictionResult",
        "ProviderResult",
        "compute_features",
        "few_shot_analyze",
        "predict_with_fine_tuned_model",
    }:
        from .api import (
            ProviderPredictionResult,
            ProviderResult,
            compute_features,
            few_shot_analyze,
            predict_with_fine_tuned_model,
        )

        return {
            "ProviderPredictionResult": ProviderPredictionResult,
            "ProviderResult": ProviderResult,
            "compute_features": compute_features,
            "few_shot_analyze": few_shot_analyze,
            "predict_with_fine_tuned_model": predict_with_fine_tuned_model,
        }[name]
    raise AttributeError(name)
