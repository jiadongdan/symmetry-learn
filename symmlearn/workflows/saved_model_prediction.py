"""Prediction with a previously fine-tuned symmetry model.

This workflow never fine-tunes. It restores a saved model, recomputes the
feature contract recorded in the model package, and runs dense inference.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any

import numpy as np

from symmlearn.features import compute_registered_features
from symmlearn.features.eight_channel.validation import (
    validate_feature_array,
    validate_unit_image,
)
from symmlearn.finetuning.artifacts import load_fine_tuned_model_state
from symmlearn.inference import predict_dense
from symmlearn.models.base import resolve_device
from symmlearn.models.registry import (
    get_model_specification,
    restore_fine_tuned_registered_model,
    validate_model_options,
)


FEATURE_OPTION_NAMES = (
    "n_max",
    "symmetry_patch_size",
    "rotation_folds",
    "reflection_p",
    "normalize_rotation_maps",
)

PREDICTION_OPTION_NAMES = ("device", "stride", "batch_size")


@dataclass(frozen=True)
class SavedModelPredictionResult:
    """In-memory dense prediction produced by one restored fine-tuned model."""

    arrays: dict[str, np.ndarray]
    record: dict[str, Any]


def _resolved_feature_options(
    identifier: str, feature_options: dict[str, Any]
) -> dict[str, Any]:
    """Merge saved feature options with the registered model defaults."""
    specification = get_model_specification(identifier)
    unknown = sorted(set(feature_options).difference(FEATURE_OPTION_NAMES))
    if unknown:
        raise ValueError(f"Unsupported saved feature options: {unknown}")
    merged = dict(specification.default_options)
    merged.update(feature_options)
    return validate_model_options(identifier, merged)


def _resolved_prediction_options(
    identifier: str,
    prediction_options: dict[str, Any] | None,
    defaults: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge runtime-only overrides onto the saved prediction defaults."""
    specification = get_model_specification(identifier)
    merged = {
        "device": specification.default_options["device"],
        "stride": specification.default_options["stride"],
        "batch_size": specification.default_options["batch_size"],
    }
    merged.update(
        {
            name: value
            for name, value in dict(defaults or {}).items()
            if name in PREDICTION_OPTION_NAMES
        }
    )
    merged.update(
        {
            name: value
            for name, value in dict(prediction_options or {}).items()
            if value is not None
        }
    )
    unknown = sorted(set(merged).difference(PREDICTION_OPTION_NAMES))
    if unknown:
        raise ValueError(f"Unsupported prediction options: {unknown}")
    return validate_model_options(identifier, merged)


def _check_expected_model(
    model_state: dict[str, Any], expected: dict[str, Any] | None
) -> None:
    """Compare the restored artifact with the manifest values from Harness."""
    if not expected:
        return
    identifier = expected.get("identifier")
    if identifier is not None and model_state["model_identifier"] != identifier:
        raise ValueError(
            "The saved model state does not match the package model identifier."
        )
    task_classes = expected.get("task_classes")
    if task_classes is not None and model_state["task_classes"] != int(task_classes):
        raise ValueError(
            "The saved model state does not match the package class count."
        )
    bottleneck = expected.get("adapter_bottleneck")
    if bottleneck is not None and model_state["adapter_bottleneck"] != int(bottleneck):
        raise ValueError(
            "The saved model state does not match the package adapter bottleneck."
        )
    class_names = expected.get("class_names")
    if class_names is not None and list(model_state["class_names"]) != list(
        class_names
    ):
        raise ValueError(
            "The saved model state does not match the package class names."
        )
    base_checksum = expected.get("base_checkpoint_sha256")
    if (
        base_checksum is not None
        and model_state.get("base_checkpoint_sha256") != base_checksum
    ):
        raise ValueError(
            "The saved model state does not match the package base checkpoint."
        )


def _file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_precomputed_features(
    features: np.ndarray,
    feature_record: dict[str, Any] | None,
    *,
    image_shape: tuple[int, int],
    feature_pipeline: str,
    feature_channels: tuple[str, ...],
    options: dict[str, Any],
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate cached features before they replace feature computation."""
    values = np.asarray(features, dtype=np.float32)
    validate_feature_array(values, image_shape)
    record = dict(feature_record or {})
    if record.get("identifier") != feature_pipeline:
        raise ValueError("Precomputed features use a different feature pipeline.")
    if list(record.get("channel_names", [])) != list(feature_channels):
        raise ValueError("Precomputed feature channels do not match the selected model.")
    if list(record.get("shape", [])) != list(values.shape):
        raise ValueError("Precomputed feature metadata has an incompatible shape.")
    for name in FEATURE_OPTION_NAMES:
        recorded = record.get(name)
        expected = options[name]
        if name == "rotation_folds":
            recorded = list(recorded or [])
            expected = list(expected)
        if recorded != expected:
            raise ValueError(
                f"Precomputed features do not match the requested {name} option."
            )
    if record.get("device") != device:
        raise ValueError("Precomputed features were calculated on a different device.")
    record["cache_reused"] = True
    return values, record


class SavedModelPredictor:
    """One restored fine-tuned model reused for any number of images."""

    def __init__(
        self,
        model: Any,
        *,
        identifier: str,
        specification: Any,
        feature_options: dict[str, Any],
        prediction_options: dict[str, Any],
        restoration_record: dict[str, Any],
        device: str,
    ) -> None:
        self.model = model
        self.identifier = identifier
        self.specification = specification
        self.feature_options = feature_options
        self.prediction_options = prediction_options
        self.restoration_record = restoration_record
        self.device = device

    @classmethod
    def from_state_path(
        cls,
        model_state_path: str | Path,
        *,
        feature_options: dict[str, Any],
        prediction_options: dict[str, Any] | None = None,
        expected_model: dict[str, Any] | None = None,
        prediction_defaults: dict[str, Any] | None = None,
    ) -> SavedModelPredictor:
        """Validate one saved state and restore its model exactly once."""
        expected_checksum = dict(expected_model or {}).get("model_state_sha256")
        if expected_checksum is not None and _file_sha256(model_state_path) != str(
            expected_checksum
        ).lower():
            raise ValueError(
                "The materialized model state checksum does not match the package."
            )
        model_state = load_fine_tuned_model_state(model_state_path)
        identifier = str(model_state["model_identifier"])
        _check_expected_model(model_state, expected_model)
        specification = get_model_specification(identifier)
        resolved_features = _resolved_feature_options(identifier, feature_options)
        resolved_prediction = _resolved_prediction_options(
            identifier, prediction_options, prediction_defaults
        )
        device = resolve_device(resolved_prediction["device"])
        model, restoration_record = restore_fine_tuned_registered_model(
            identifier, model_state, device=device
        )
        return cls(
            model,
            identifier=identifier,
            specification=specification,
            feature_options=resolved_features,
            prediction_options=resolved_prediction,
            restoration_record=restoration_record,
            device=device,
        )

    def predict(
        self,
        image: np.ndarray,
        *,
        precomputed_features: np.ndarray | None = None,
        precomputed_feature_record: dict[str, Any] | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> SavedModelPredictionResult:
        """Compute features and dense prediction for one normalized image."""
        import torch

        started = perf_counter()
        specification = self.specification
        values = validate_unit_image(
            image,
            minimum_size=specification.classifier_patch_size,
            subject="Prediction input",
        )
        if precomputed_features is None:
            if precomputed_feature_record is not None:
                raise ValueError(
                    "A precomputed feature record requires precomputed feature values."
                )
            if progress_callback is not None:
                progress_callback("features", 0, 1)
            features, feature_record = compute_registered_features(
                specification.feature_pipeline,
                values,
                self.feature_options,
                device=self.device,
            )
            feature_record = {
                "identifier": specification.feature_pipeline,
                **feature_record,
                "cache_reused": False,
            }
            if progress_callback is not None:
                progress_callback("features", 1, 1)
        else:
            features, feature_record = _validated_precomputed_features(
                precomputed_features,
                precomputed_feature_record,
                image_shape=values.shape,
                feature_pipeline=specification.feature_pipeline,
                feature_channels=specification.feature_channels,
                options=self.feature_options,
                device=self.device,
            )

        prediction = predict_dense(
            self.model,
            features,
            patch_size=specification.classifier_patch_size,
            stride=self.prediction_options["stride"],
            batch_size=self.prediction_options["batch_size"],
            device=self.device,
            expected_channels=specification.input_channels,
            progress_callback=(
                None
                if progress_callback is None
                else lambda current, total: progress_callback(
                    "prediction", current, total
                )
            ),
        )
        arrays = {
            "features": features,
            "channel_names": np.asarray(specification.feature_channels),
            **prediction.to_arrays(),
        }
        record = {
            "identifier": self.identifier,
            "options": {
                "features": {
                    name: self.feature_options[name]
                    for name in FEATURE_OPTION_NAMES
                },
                "prediction": {
                    name: self.prediction_options[name]
                    for name in PREDICTION_OPTION_NAMES
                },
            },
            "model": {
                "identifier": self.identifier,
                "input_channels": specification.input_channels,
                "classifier_patch_size": specification.classifier_patch_size,
                "feature_pipeline": specification.feature_pipeline,
                **self.restoration_record,
            },
            "features": feature_record,
            "prediction": {
                "sample_count": int(len(prediction.coordinates_xy)),
                "grid_shape": list(prediction.prediction_grid.shape),
                "confidence_range": [
                    float(prediction.confidence.min()),
                    float(prediction.confidence.max()),
                ],
                "entropy_range": [
                    float(prediction.entropy.min()),
                    float(prediction.entropy.max()),
                ],
            },
            "runtime": {
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy_version": str(np.__version__),
                "torch_version": str(torch.__version__),
                "requested_device": self.prediction_options["device"],
                "resolved_device": self.device,
                "total_seconds": perf_counter() - started,
            },
            "warnings": [
                "Confidence and entropy do not establish physical correctness."
            ],
        }
        return SavedModelPredictionResult(arrays=arrays, record=record)


def predict_with_saved_model(
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
) -> SavedModelPredictionResult:
    """Restore one fine-tuned model and run dense prediction on one image."""
    predictor = SavedModelPredictor.from_state_path(
        model_state_path,
        feature_options=feature_options,
        prediction_options=prediction_options,
        expected_model=expected_model,
        prediction_defaults=prediction_defaults,
    )
    return predictor.predict(
        image,
        precomputed_features=precomputed_features,
        precomputed_feature_record=precomputed_feature_record,
        progress_callback=progress_callback,
    )
