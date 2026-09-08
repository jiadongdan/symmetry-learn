"""Authoritative few-shot symmetry workflow for Python and Provider clients."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any

import numpy as np

from symmlearn.features import compute_registered_features
from symmlearn.features.eight_channel.validation import validate_unit_image
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
    features, feature_record = compute_registered_features(
        specification.feature_pipeline,
        values,
        resolved_options,
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
