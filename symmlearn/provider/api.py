"""Thin direct API over registered symmetry-learn workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import platform
from typing import Any

import numpy as np

from symmlearn import __version__
from symmlearn.features import compute_registered_features
from symmlearn.features.eight_channel.validation import validate_unit_image
from symmlearn.models.base import resolve_device
from symmlearn.models.registry import (
    build_registered_model,
    get_model_specification,
    validate_model_options,
)
from symmlearn.models.weights import (
    load_pretrained_checkpoint,
    resolve_model_weight,
)
from symmlearn.traditional_ml import (
    IMAGE_PLUS_SYMMETRY_MAPS_MODE,
    TraditionalMLOptions,
    analyze_traditional_ml,
)
from symmlearn.workflows import predict_with_saved_model, run_few_shot

from .contracts import (
    ADAPTER_SCHEMA_VERSION,
    MODEL_IDENTIFIER,
    PROVIDER_CONTRACT_VERSION,
    WORKER_SCHEMA_VERSION,
)


@dataclass(frozen=True)
class ProviderResult:
    """In-memory output from one symmetry-learn few-shot analysis."""

    arrays: dict[str, np.ndarray]
    adapter_checkpoint: dict[str, Any]
    fine_tuned_model_state: dict[str, Any]
    record: dict[str, Any]

    @property
    def prediction_grid(self) -> np.ndarray:
        """Return the dense predicted-class grid."""
        return self.arrays["prediction_grid"]

    @property
    def confidence_grid(self) -> np.ndarray:
        """Return the dense maximum-probability grid."""
        return self.arrays["confidence_grid"]

    @property
    def entropy_grid(self) -> np.ndarray:
        """Return the dense predictive-entropy grid."""
        return self.arrays["entropy_grid"]


@dataclass(frozen=True)
class ProviderPredictionResult:
    """In-memory dense prediction from one saved fine-tuned model."""

    arrays: dict[str, np.ndarray]
    record: dict[str, Any]


@dataclass(frozen=True)
class ProviderTraditionalMLResult:
    """In-memory traditional ML prediction from one conventional classifier.

    No estimator object is part of the result: the first version intentionally
    fits and predicts inside a single job so that no pickle or joblib artifact
    is ever written or read.
    """

    arrays: dict[str, np.ndarray]
    record: dict[str, Any]

    @property
    def prediction_grid(self) -> np.ndarray:
        """Return the dense predicted-class grid."""
        return self.arrays["prediction_grid"]

    @property
    def confidence_grid(self) -> np.ndarray:
        """Return the dense maximum-probability grid."""
        return self.arrays["confidence_grid"]

    @property
    def entropy_grid(self) -> np.ndarray:
        """Return the dense predictive-entropy grid."""
        return self.arrays["entropy_grid"]


def _provider_result(result: Any) -> ProviderResult:
    adapter_checkpoint = {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "provider_contract_version": PROVIDER_CONTRACT_VERSION,
        **result.adapter_checkpoint,
    }
    model_record = {
        **result.record["model"],
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
    }
    record = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "provider_contract_version": PROVIDER_CONTRACT_VERSION,
        "provider": "symmetry-learn",
        "provider_version": __version__,
        **result.record,
        "model": model_record,
    }
    return ProviderResult(
        arrays=result.arrays,
        adapter_checkpoint=adapter_checkpoint,
        fine_tuned_model_state=result.fine_tuned_model_state,
        record=record,
    )


def compute_features(
    image: np.ndarray,
    *,
    options: dict[str, Any] | None = None,
    device: str = "auto",
    model: str = MODEL_IDENTIFIER,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the selected model's feature representation."""
    specification = get_model_specification(model)
    values = validate_unit_image(image)
    resolved_options = validate_model_options(model, options)
    resolved_options["device"] = device
    resolved_device = resolve_device(device)
    features, record = compute_registered_features(
        specification.feature_pipeline,
        values,
        resolved_options,
        device=resolved_device,
    )
    return features, {"identifier": specification.feature_pipeline, **record}


def few_shot_analyze(
    image: np.ndarray,
    *,
    coordinates_xy: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    checkpoint_path: str | Path | None = None,
    checkpoint_sha256: str | None = None,
    weight: str | None = None,
    options: dict[str, Any] | None = None,
    model: str = MODEL_IDENTIFIER,
    precomputed_features: np.ndarray | None = None,
    precomputed_feature_record: dict[str, Any] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> ProviderResult:
    """Fine-tune adapters and return dense local-class predictions."""
    result = run_few_shot(
        np.asarray(image, dtype=np.float32),
        coordinates_xy=np.asarray(coordinates_xy, dtype=np.int32),
        labels=np.asarray(labels, dtype=np.int64),
        class_names=list(class_names),
        model_identifier=model,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        weight_identifier=weight,
        options=options,
        precomputed_features=precomputed_features,
        precomputed_feature_record=precomputed_feature_record,
        progress_callback=progress_callback,
    )
    return _provider_result(result)


def predict_with_fine_tuned_model(
    image: np.ndarray,
    *,
    model_state_path: str | Path,
    feature_options: dict[str, Any],
    prediction_options: dict[str, Any] | None = None,
    expected_model: dict[str, Any] | None = None,
    prediction_defaults: dict[str, Any] | None = None,
    precomputed_features: np.ndarray | None = None,
    precomputed_feature_record: dict[str, Any] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> ProviderPredictionResult:
    """Restore one saved fine-tuned model and predict one normalized image."""
    result = predict_with_saved_model(
        np.asarray(image, dtype=np.float32),
        model_state_path=model_state_path,
        feature_options=dict(feature_options),
        prediction_options=prediction_options,
        expected_model=expected_model,
        prediction_defaults=prediction_defaults,
        precomputed_features=precomputed_features,
        precomputed_feature_record=precomputed_feature_record,
        progress_callback=progress_callback,
    )
    return ProviderPredictionResult(
        arrays=result.arrays,
        record={
            "schema_version": WORKER_SCHEMA_VERSION,
            "provider_contract_version": PROVIDER_CONTRACT_VERSION,
            "provider": "symmetry-learn",
            "provider_version": __version__,
            **result.record,
        },
    )


def probe_model(
    *,
    model: str = MODEL_IDENTIFIER,
    checkpoint_path: str | Path | None = None,
    checkpoint_sha256: str | None = None,
    weight: str | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve and strictly load a model without running inference."""
    import torch

    resolved_options = validate_model_options(model, options)
    device = resolve_device(resolved_options["device"])
    instance = build_registered_model(model)
    with resolve_model_weight(
        model,
        weight_identifier=weight,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
    ) as resolved_weight:
        checkpoint = load_pretrained_checkpoint(
            instance,
            resolved_weight.path,
            resolved_weight.sha256,
        )
        checkpoint.update(
            {
                "weight_identifier": resolved_weight.identifier,
                "source": resolved_weight.source,
                "bundled": resolved_weight.bundled,
            }
        )
    return {
        "schema_version": WORKER_SCHEMA_VERSION,
        "provider_contract_version": PROVIDER_CONTRACT_VERSION,
        "provider": "symmetry-learn",
        "identifier": model,
        "available": True,
        "details": {
            "model_class": instance.__class__.__module__
            + "."
            + instance.__class__.__name__,
            "torch_version": str(torch.__version__),
            "requested_device": resolved_options["device"],
            "resolved_device": device,
            "cuda_device": (
                torch.cuda.get_device_name(torch.device(device))
                if device.startswith("cuda")
                else None
            ),
            "platform": platform.platform(),
            "checkpoint": checkpoint,
        },
    }


def traditional_ml_analyze(
    features: np.ndarray,
    *,
    coordinates_xy: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    classifier: str,
    parameters: dict[str, Any] | None = None,
    feature_mode: str = IMAGE_PLUS_SYMMETRY_MAPS_MODE,
    options: TraditionalMLOptions | dict[str, Any] | None = None,
    feature_record: dict[str, Any] | None = None,
    input_shape: tuple[int, int] | None = None,
    input_sha256: str | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> ProviderTraditionalMLResult:
    """Train one conventional classifier and densely predict one image.

    This entry point accepts in-memory arrays only. It never requires a model
    identifier, weight, checkpoint, model-state path, or neural-inference
    device. Map computation device remains relevant to the earlier
    ``compute_features`` operation, not to scikit-learn training.
    """
    resolved_options = options
    if resolved_options is None:
        resolved_options = TraditionalMLOptions()
    elif isinstance(resolved_options, dict):
        resolved_options = TraditionalMLOptions.from_mapping(resolved_options)
    result = analyze_traditional_ml(
        np.asarray(features, dtype=np.float32),
        coordinates_xy=np.asarray(coordinates_xy, dtype=np.int32),
        labels=np.asarray(labels, dtype=np.int64),
        class_names=list(class_names),
        classifier=classifier,
        parameters=None if parameters is None else dict(parameters),
        feature_mode=feature_mode,
        options=resolved_options,
        feature_record=feature_record,
        input_shape=input_shape,
        input_sha256=input_sha256,
        progress_callback=progress_callback,
    )
    record = {
        "worker_schema_version": WORKER_SCHEMA_VERSION,
        "provider_contract_version": PROVIDER_CONTRACT_VERSION,
        "provider": "symmetry-learn",
        "provider_version": __version__,
        **result.record,
    }
    return ProviderTraditionalMLResult(arrays=result.arrays, record=record)
