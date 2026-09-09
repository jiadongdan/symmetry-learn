"""Provider capability document backed by the public model registry."""

from __future__ import annotations

from typing import Any

from symmlearn import __version__
from symmlearn.models.registry import build_registered_model, model_capabilities

from .contracts import PROVIDER_CONTRACT_VERSION


def provider_capabilities() -> dict[str, Any]:
    """Return the complete versioned Provider capability document."""
    return {
        "contract_version": PROVIDER_CONTRACT_VERSION,
        "provider": "symmetry-learn",
        "provider_version": __version__,
        "operations": [
            "compute_features",
            "few_shot_analyze",
            "few_shot_analyze_precomputed_features",
            "probe_model",
        ],
        "models": model_capabilities(),
    }


__all__ = [
    "build_registered_model",
    "model_capabilities",
    "provider_capabilities",
]
