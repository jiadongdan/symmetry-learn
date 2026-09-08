"""Registry for model metadata, construction, and model-specific behavior."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Callable

from symmlearn.exceptions import ModelNotFoundError

from .cnn_8ch_pg17.specification import (
    MODEL_SPECIFICATION as CNN_8CH_PG17_SPECIFICATION,
)
from .specifications import ModelSpecification
from .weights import inspect_weight


@dataclass(frozen=True)
class _ModelRegistration:
    specification: ModelSpecification
    builder: str
    option_validator: str
    fine_tuner: str


_REGISTRATIONS = {
    CNN_8CH_PG17_SPECIFICATION.identifier: _ModelRegistration(
        specification=CNN_8CH_PG17_SPECIFICATION,
        builder="symmlearn.models.cnn_8ch_pg17.architecture:build_model",
        option_validator=(
            "symmlearn.models.cnn_8ch_pg17.specification:validate_options"
        ),
        fine_tuner="symmlearn.models.cnn_8ch_pg17.fine_tuning:fine_tune",
    ),
}


def _module_available(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def registered_model_identifiers() -> tuple[str, ...]:
    """Return model identifiers in stable display order."""
    return tuple(_REGISTRATIONS)


def _get_registration(identifier: str) -> _ModelRegistration:
    try:
        return _REGISTRATIONS[identifier]
    except KeyError as error:
        available = ", ".join(registered_model_identifiers())
        raise ModelNotFoundError(
            f"Unsupported model {identifier!r}; available models: {available}"
        ) from error


def _load_callable(reference: str) -> Callable[..., Any]:
    module_name, attribute = reference.split(":", maxsplit=1)
    value = getattr(import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"Registered target {reference!r} is not callable.")
    return value


def get_model_specification(identifier: str) -> ModelSpecification:
    """Return metadata for a registered model."""
    return _get_registration(identifier).specification


def build_registered_model(identifier: str, *, num_classes: int | None = None):
    """Construct a registered model without loading pretrained weights."""
    registration = _get_registration(identifier)
    builder = _load_callable(registration.builder)
    return builder(
        num_classes=(
            registration.specification.pretrained_classes
            if num_classes is None
            else int(num_classes)
        )
    )


def validate_model_options(
    identifier: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate options using the selected model's complete runtime contract."""
    registration = _get_registration(identifier)
    validator = _load_callable(registration.option_validator)
    return validator(overrides)


def fine_tune_registered_model(
    identifier: str,
    model: Any,
    support_patches: Any,
    support_labels: Any,
    *,
    task_classes: int,
    options: dict[str, Any],
    device: str,
):
    """Apply the fine-tuning recipe registered for a model."""
    registration = _get_registration(identifier)
    fine_tuner = _load_callable(registration.fine_tuner)
    return fine_tuner(
        model,
        support_patches,
        support_labels,
        task_classes=task_classes,
        options=options,
        device=device,
    )


def model_capabilities() -> list[dict[str, Any]]:
    """Return serializable metadata without importing PyTorch."""
    capabilities = []
    torch_available = _module_available("torch")
    for registration in _REGISTRATIONS.values():
        specification = registration.specification
        weights = [inspect_weight(weight) for weight in specification.weights]
        default_weight = next(
            weight
            for weight in weights
            if weight["identifier"] == specification.default_weight.identifier
        )
        capabilities.append(
            {
                "identifier": specification.identifier,
                "display_name": specification.display_name,
                "kind": specification.kind,
                "family": specification.family,
                "available": torch_available,
                "device_kind": "cuda_or_cpu",
                "checkpoint_required": default_weight["status"] != "installed",
                "input_channels": specification.input_channels,
                "pretrained_classes": specification.pretrained_classes,
                "classifier_patch_size": specification.classifier_patch_size,
                "feature_pipeline": specification.feature_pipeline,
                "feature_channels": list(specification.feature_channels),
                "fine_tuning_strategy": specification.fine_tuning.strategy,
                "minimum_shots_per_class": (
                    specification.fine_tuning.minimum_shots_per_class
                ),
                "recommended_shots_per_class": (
                    specification.fine_tuning.recommended_shots_per_class
                ),
                "maximum_shots_per_class": (
                    specification.fine_tuning.maximum_shots_per_class
                ),
                "default_weight": default_weight,
                "weights": weights,
                "defaults": dict(specification.default_options),
            }
        )
    return capabilities
