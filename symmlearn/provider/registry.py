"""Stable capabilities and model construction for provider clients."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

from symmlearn import __version__

from .contracts import (
    CHANNEL_NAMES,
    DEFAULT_OPTIONS,
    MODEL_IDENTIFIER,
    PROVIDER_CONTRACT_VERSION,
)


def _module_available(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def model_capabilities() -> list[dict[str, Any]]:
    """Return model metadata without importing the optional PyTorch runtime."""
    return [
        {
            "identifier": MODEL_IDENTIFIER,
            "kind": "few_shot_symmetry",
            "family": "deep_learning",
            "available": _module_available("torch"),
            "device_kind": "cuda_or_cpu",
            "checkpoint_required": True,
            "input_channels": 8,
            "pretrained_classes": 17,
            "classifier_patch_size": 64,
            "feature_channels": list(CHANNEL_NAMES),
            "fine_tuning_strategy": (
                "frozen pretrained network with residual adapters and a new local head"
            ),
            "defaults": dict(DEFAULT_OPTIONS),
        }
    ]


def provider_capabilities() -> dict[str, Any]:
    """Return the complete provider capability document."""
    return {
        "contract_version": PROVIDER_CONTRACT_VERSION,
        "provider": "symmetry-learn",
        "provider_version": __version__,
        "models": model_capabilities(),
    }


def build_registered_model(identifier: str, *, num_classes: int = 17):
    """Construct a supported checkpoint-compatible model without loading weights."""
    if identifier != MODEL_IDENTIFIER:
        raise ValueError(
            f"Unsupported provider model {identifier!r}; available model: {MODEL_IDENTIFIER}"
        )
    from .model import build_checkpoint_compatible_model

    return build_checkpoint_compatible_model(num_classes=num_classes)
