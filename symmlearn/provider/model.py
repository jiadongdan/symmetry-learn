"""Compatibility imports for the former Provider model implementation.

New code should import these operations from models, finetuning, or inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from symmlearn.finetuning.adapters import (
    AdaptedBlock,
    AdapterConv,
    AdapterLinear,
    add_task_adapters,
    set_adapter_training_mode,
    trainable_state_dict,
)
from symmlearn.finetuning.engine import fit_adapters, set_deterministic_seed
from symmlearn.inference.dense import predict_probabilities
from symmlearn.models.base import resolve_device
from symmlearn.models.cnn_8ch_pg17.architecture import build_model
from symmlearn.models.weights import file_sha256, load_pretrained_checkpoint


def build_checkpoint_compatible_model(num_classes: int = 17):
    """Construct the registered checkpoint-compatible architecture."""
    return build_model(num_classes=num_classes)


def _adapter_types():
    """Return the historical adapter class tuple."""
    return AdapterConv, AdapterLinear, AdaptedBlock


def _set_adapter_training_mode(model) -> None:
    """Preserve the former private helper for downstream compatibility."""
    set_adapter_training_mode(model)


def train_adapters(
    model,
    support_patches: np.ndarray,
    support_labels: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Preserve the former dictionary-returning training function."""
    return fit_adapters(
        model,
        support_patches,
        support_labels,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
    ).to_record()


__all__ = [
    "add_task_adapters",
    "build_checkpoint_compatible_model",
    "file_sha256",
    "load_pretrained_checkpoint",
    "predict_probabilities",
    "resolve_device",
    "set_deterministic_seed",
    "train_adapters",
    "trainable_state_dict",
]
