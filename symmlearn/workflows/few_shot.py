"""Authoritative few-shot symmetry workflow for Python and Provider clients."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
from symmlearn.finetuning.adapters import trainable_state_dict
from symmlearn.finetuning.engine import set_deterministic_seed
from symmlearn.inference import predict_dense
from symmlearn.models.base import resolve_device
from symmlearn.models.registry import (
    build_registered_model,
    fine_tune_registered_model,
    get_model_specification,
    validate_model_options,
)
from symmlearn.models.weights import (
    load_pretrained_checkpoint,
    resolve_model_weight,
)
from symmlearn.patches import extract_patches, validate_support


@dataclass(frozen=True)
class FewShotResult:
    """In-memory result independent of Provider transport."""

    arrays: dict[str, np.ndarray]
    adapter_checkpoint: dict[str, Any]
    record: dict[str, Any]


_FEATURE_OPTION_NAMES = (
    "n_max",
    "symmetry_patch_size",
    "rotation_folds",
    "reflection_p",
    "normalize_rotation_maps",
)


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
    """Validate cached features against the complete feature-computation contract."""
    values = np.asarray(features, dtype=np.float32)
    validate_feature_array(values, image_shape)
    record = dict(feature_record or {})
    if record.get("identifier") != feature_pipeline:
        raise ValueError("Precomputed features use a different feature pipeline.")
    if list(record.get("channel_names", [])) != list(feature_channels):
        raise ValueError("Precomputed feature channels do not match the selected model.")
    if list(record.get("shape", [])) != list(values.shape):
        raise ValueError("Precomputed feature metadata has an incompatible shape.")
    for name in _FEATURE_OPTION_NAMES:
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


def run_few_shot(
    image: np.ndarray,
    *,
    coordinates_xy: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    model_identifier: str,
    checkpoint_path: str | Path | None = None,
    checkpoint_sha256: str | None = None,
    weight_identifier: str | None = None,
    options: dict[str, Any] | None = None,
    precomputed_features: np.ndarray | None = None,
    precomputed_feature_record: dict[str, Any] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> FewShotResult:
    """Fine-tune the selected model and produce dense local-class predictions."""
    import torch

    specification = get_model_specification(model_identifier)
    resolved_options = validate_model_options(model_identifier, options)
    values = validate_unit_image(
        image,
        minimum_size=specification.classifier_patch_size,
        subject="Provider input",
    )
    coordinates, resolved_labels, names = validate_support(
        values.shape,
        np.asarray(coordinates_xy),
        np.asarray(labels),
        list(class_names),
        resolved_options,
    )
    device = resolve_device(resolved_options["device"])
    started = perf_counter()
    if precomputed_features is None:
        if precomputed_feature_record is not None:
            raise ValueError(
                "A precomputed feature record requires precomputed feature values."
            )
        features, feature_record = compute_registered_features(
            specification.feature_pipeline,
            values,
            resolved_options,
            device=device,
        )
        feature_record = {
            "identifier": specification.feature_pipeline,
            **feature_record,
            "cache_reused": False,
        }
    else:
        features, feature_record = _validated_precomputed_features(
            precomputed_features,
            precomputed_feature_record,
            image_shape=values.shape,
            feature_pipeline=specification.feature_pipeline,
            feature_channels=specification.feature_channels,
            options=resolved_options,
            device=device,
        )
    support_patches = extract_patches(
        features,
        coordinates,
        specification.classifier_patch_size,
        expected_channels=specification.input_channels,
    )
    set_deterministic_seed(resolved_options["seed"])
    model = build_registered_model(model_identifier)
    with resolve_model_weight(
        model_identifier,
        weight_identifier=weight_identifier,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
    ) as resolved_weight:
        checkpoint_record = load_pretrained_checkpoint(
            model,
            resolved_weight.path,
            resolved_weight.sha256,
        )
        checkpoint_record.update(
            {
                "weight_identifier": resolved_weight.identifier,
                "source": resolved_weight.source,
                "bundled": resolved_weight.bundled,
            }
        )
    model, training_result = fine_tune_registered_model(
        model_identifier,
        model,
        support_patches,
        resolved_labels,
        task_classes=len(names),
        options=resolved_options,
        device=device,
        progress_callback=(
            None
            if progress_callback is None
            else lambda current, total: progress_callback(
                "fine_tuning", current, total
            )
        ),
    )
    training = training_result.to_record()
    prediction_result = predict_dense(
        model,
        features,
        patch_size=specification.classifier_patch_size,
        stride=resolved_options["stride"],
        batch_size=resolved_options["batch_size"],
        device=device,
        expected_channels=specification.input_channels,
        progress_callback=(
            None
            if progress_callback is None
            else lambda current, total: progress_callback("prediction", current, total)
        ),
    )
    prediction = prediction_result.to_arrays()
    arrays = {
        "features": features,
        "channel_names": np.asarray(specification.feature_channels),
        "support_patches": support_patches,
        "support_labels": resolved_labels,
        "support_coordinates_xy": coordinates,
        **prediction,
    }
    adapter_checkpoint = {
        "model_identifier": model_identifier,
        "base_checkpoint_sha256": checkpoint_record["sha256"],
        "base_weight_identifier": checkpoint_record["weight_identifier"],
        "class_names": names,
        "adapter_bottleneck": resolved_options["adapter_bottleneck"],
        "training": {
            "epochs": training["epochs"],
            "best_support_loss": training["best_support_loss"],
            "final_support_accuracy": training["final_support_accuracy"],
        },
        "trainable_state_dict": trainable_state_dict(model),
    }
    runtime = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "requested_device": resolved_options["device"],
        "resolved_device": device,
        "cuda_device": (
            torch.cuda.get_device_name(torch.device(device))
            if device.startswith("cuda")
            else None
        ),
        "total_seconds": perf_counter() - started,
    }
    record = {
        "identifier": model_identifier,
        "options": resolved_options,
        "support": {
            "class_names": names,
            "counts": np.bincount(
                resolved_labels, minlength=len(names)
            ).tolist(),
            "sample_count": int(len(resolved_labels)),
        },
        "model": {
            "identifier": model_identifier,
            "input_channels": specification.input_channels,
            "pretrained_classes": specification.pretrained_classes,
            "classifier_patch_size": specification.classifier_patch_size,
            "feature_pipeline": specification.feature_pipeline,
            "checkpoint": checkpoint_record,
        },
        "features": feature_record,
        "training": training,
        "prediction": {
            "sample_count": int(len(prediction["coordinates_xy"])),
            "grid_shape": list(prediction["prediction_grid"].shape),
            "confidence_range": [
                float(prediction["confidence"].min()),
                float(prediction["confidence"].max()),
            ],
            "entropy_range": [
                float(prediction["entropy"].min()),
                float(prediction["entropy"].max()),
            ],
        },
        "runtime": runtime,
        "warnings": [
            training["warning"],
            "Confidence and entropy do not establish physical correctness.",
        ],
    }
    return FewShotResult(
        arrays=arrays,
        adapter_checkpoint=adapter_checkpoint,
        record=record,
    )
