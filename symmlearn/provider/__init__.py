"""Versioned symmetry provider API for external orchestrators."""

from .contracts import PROVIDER_CONTRACT_VERSION, WORKER_SCHEMA_VERSION
from .registry import model_capabilities, traditional_ml_capability

__all__ = [
    "PROVIDER_CONTRACT_VERSION",
    "ProviderPredictionResult",
    "ProviderResult",
    "ProviderTraditionalMLResult",
    "WORKER_SCHEMA_VERSION",
    "compute_features",
    "few_shot_analyze",
    "model_capabilities",
    "predict_with_fine_tuned_model",
    "traditional_ml_analyze",
    "traditional_ml_capability",
]


def __getattr__(name: str):
    if name in {
        "ProviderPredictionResult",
        "ProviderResult",
        "ProviderTraditionalMLResult",
        "compute_features",
        "few_shot_analyze",
        "predict_with_fine_tuned_model",
        "traditional_ml_analyze",
    }:
        from .api import (
            ProviderPredictionResult,
            ProviderResult,
            ProviderTraditionalMLResult,
            compute_features,
            few_shot_analyze,
            predict_with_fine_tuned_model,
            traditional_ml_analyze,
        )

        return {
            "ProviderPredictionResult": ProviderPredictionResult,
            "ProviderResult": ProviderResult,
            "ProviderTraditionalMLResult": ProviderTraditionalMLResult,
            "compute_features": compute_features,
            "few_shot_analyze": few_shot_analyze,
            "predict_with_fine_tuned_model": predict_with_fine_tuned_model,
            "traditional_ml_analyze": traditional_ml_analyze,
        }[name]
    raise AttributeError(name)
