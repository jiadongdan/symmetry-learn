"""Save and load compact task-specific adapter artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


ADAPTER_SCHEMA_VERSION = "symmetry-adapter-head-v1"
FINE_TUNED_MODEL_STATE_SCHEMA_VERSION = "symmetry-fine-tuned-model-state-v1"


class FineTunedModelStateError(ValueError):
    """Raised when a fine-tuned model state artifact is structurally invalid."""


def save_adapter_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    """Persist adapters and the local head without duplicating base weights."""
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_adapter_artifact(path: str | Path) -> dict[str, Any]:
    """Load a validated adapter artifact mapping."""
    import torch

    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("The adapter artifact must decode to a mapping.")
    if payload.get("schema_version") != ADAPTER_SCHEMA_VERSION:
        raise ValueError("Unsupported adapter artifact schema version.")
    return payload


def save_fine_tuned_model_state(path: str | Path, payload: dict[str, Any]) -> None:
    """Persist a complete fine-tuned model state for portable inference."""
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def validate_fine_tuned_model_state(payload: Any) -> dict[str, Any]:
    """Validate one decoded fine-tuned model state before it is ever restored."""
    import torch

    if isinstance(payload, torch.nn.Module):
        raise FineTunedModelStateError(
            "A serialized nn.Module is not accepted. Save the versioned mapping "
            "produced by the fine-tuning workflow instead."
        )
    if not isinstance(payload, dict):
        raise FineTunedModelStateError(
            "The fine-tuned model state must decode to a mapping."
        )
    if payload.get("schema_version") != FINE_TUNED_MODEL_STATE_SCHEMA_VERSION:
        raise FineTunedModelStateError(
            "Unsupported fine-tuned model state schema version."
        )
    identifier = payload.get("model_identifier")
    if not isinstance(identifier, str) or not identifier.strip():
        raise FineTunedModelStateError(
            "The fine-tuned model state has no registered model identifier."
        )
    bottleneck = payload.get("adapter_bottleneck")
    if not isinstance(bottleneck, int) or isinstance(bottleneck, bool) or bottleneck <= 0:
        raise FineTunedModelStateError(
            "The fine-tuned model state has an invalid adapter bottleneck."
        )
    task_classes = payload.get("task_classes")
    if not isinstance(task_classes, int) or isinstance(task_classes, bool) or task_classes < 2:
        raise FineTunedModelStateError(
            "The fine-tuned model state requires at least two task classes."
        )
    class_names = payload.get("class_names")
    if not isinstance(class_names, list) or any(
        not isinstance(name, str) or not name.strip() for name in class_names
    ):
        raise FineTunedModelStateError(
            "The fine-tuned model state has invalid class names."
        )
    if len(class_names) != task_classes:
        raise FineTunedModelStateError(
            "The fine-tuned model state class names do not match task_classes."
        )
    if len({name.strip() for name in class_names}) != len(class_names):
        raise FineTunedModelStateError(
            "The fine-tuned model state class names must be unique."
        )
    checksum = payload.get("base_checkpoint_sha256")
    if checksum is not None and (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum.lower())
    ):
        raise FineTunedModelStateError(
            "The fine-tuned model state has a malformed base checkpoint checksum."
        )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise FineTunedModelStateError(
            "The fine-tuned model state has no state_dict mapping."
        )
    for name, value in state_dict.items():
        if not isinstance(name, str) or not name.strip():
            raise FineTunedModelStateError(
                "The fine-tuned model state contains an invalid parameter name."
            )
        if not isinstance(value, torch.Tensor):
            raise FineTunedModelStateError(
                f"The fine-tuned model state value {name!r} is not a tensor."
            )
        if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
            raise FineTunedModelStateError(
                f"The fine-tuned model state value {name!r} contains nonfinite values."
            )
    return payload


def load_fine_tuned_model_state(path: str | Path) -> dict[str, Any]:
    """Load and validate a complete fine-tuned model state artifact."""
    import torch

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"The fine-tuned model state does not exist: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    return validate_fine_tuned_model_state(payload)
