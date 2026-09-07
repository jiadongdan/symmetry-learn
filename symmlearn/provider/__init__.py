"""Versioned symmetry provider API for external orchestrators."""

from .contracts import PROVIDER_CONTRACT_VERSION, WORKER_SCHEMA_VERSION
from .registry import model_capabilities

__all__ = [
    "PROVIDER_CONTRACT_VERSION",
    "ProviderResult",
    "WORKER_SCHEMA_VERSION",
    "compute_features",
    "few_shot_analyze",
    "model_capabilities",
]


def __getattr__(name: str):
    if name in {"ProviderResult", "compute_features", "few_shot_analyze"}:
        from .api import ProviderResult, compute_features, few_shot_analyze

        return {
            "ProviderResult": ProviderResult,
            "compute_features": compute_features,
            "few_shot_analyze": few_shot_analyze,
        }[name]
    raise AttributeError(name)
