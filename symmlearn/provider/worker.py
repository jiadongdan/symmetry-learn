"""Execute one versioned few-shot symmetry job for an external orchestrator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any, Iterator

import numpy as np

from .contracts import (
    ADAPTER_SCHEMA_VERSION,
    CHANNEL_NAMES,
    DEFAULT_OPTIONS,
    MODEL_IDENTIFIER,
    PROVIDER_CONTRACT_VERSION,
    WORKER_SCHEMA_VERSION,
)
from .registry import build_registered_model, provider_capabilities


def _load_job(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != WORKER_SCHEMA_VERSION:
        raise ValueError("Unsupported worker job schema version.")
    return payload


def _load_input(path: Path) -> np.ndarray:
    image = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Worker input must be a finite float32 2D array.")
    if min(image.shape) < 64:
        raise ValueError("Both worker input dimensions must be at least 64 pixels.")
    if float(image.min()) < -1e-6 or float(image.max()) > 1.000001:
        raise ValueError("Worker input must lie in the shared [0, 1] space.")
    return image


def _validated_options(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    unknown = sorted(set(overrides or {}).difference(DEFAULT_OPTIONS))
    if unknown:
        raise ValueError(f"Unsupported provider options: {unknown}")
    options = dict(DEFAULT_OPTIONS)
    options.update(dict(overrides or {}))
    integer_names = (
        "n_max",
        "symmetry_patch_size",
        "classifier_patch_size",
        "minimum_shots_per_class",
        "maximum_shots_per_class",
        "adapter_bottleneck",
        "epochs",
        "seed",
        "stride",
        "batch_size",
    )
    for name in integer_names:
        options[name] = int(options[name])
    for name in ("reflection_p", "learning_rate", "weight_decay"):
        options[name] = float(options[name])
    options["rotation_folds"] = [int(value) for value in options["rotation_folds"]]
    if options["n_max"] <= 0:
        raise ValueError("n_max must be positive.")
    if options["symmetry_patch_size"] <= 0 or options["symmetry_patch_size"] % 2 == 0:
        raise ValueError("symmetry_patch_size must be a positive odd number.")
    if options["rotation_folds"] != [2, 3, 4, 6]:
        raise ValueError("The v1 channel contract requires rotation_folds [2, 3, 4, 6].")
    if options["reflection_p"] <= 0:
        raise ValueError("reflection_p must be positive.")
    if not isinstance(options["normalize_rotation_maps"], bool):
        raise ValueError("normalize_rotation_maps must be boolean.")
    if options["classifier_patch_size"] != 64:
        raise ValueError("The v1 model contract requires a 64-pixel classifier patch.")
    if options["minimum_shots_per_class"] <= 0:
        raise ValueError("minimum_shots_per_class must be positive.")
    if options["maximum_shots_per_class"] < options["minimum_shots_per_class"]:
        raise ValueError("maximum_shots_per_class must not be smaller than the minimum.")
    for name in ("adapter_bottleneck", "epochs", "stride", "batch_size"):
        if options[name] <= 0:
            raise ValueError(f"{name} must be positive.")
    if options["learning_rate"] <= 0 or options["weight_decay"] < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative.")
    device = str(options["device"])
    if device not in {"auto", "cpu"} and not device.startswith("cuda"):
        raise ValueError("device must be auto, cpu, or a CUDA device.")
    options["device"] = device
    return options


def _compute_features(
    image: np.ndarray,
    options: dict[str, Any],
    *,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    from symmlearn.maps import get_ref_map, get_rot_maps
    import torch

    started = perf_counter()
    rotation_maps = get_rot_maps(
        image,
        patch_size=options["symmetry_patch_size"],
        n_max=options["n_max"],
        n_folds=options["rotation_folds"],
        device=torch.device(device),
        normalize_output=options["normalize_rotation_maps"],
    )
    reflection_map, theta_map = get_ref_map(
        image,
        patch_size=options["symmetry_patch_size"],
        n_max=options["n_max"],
        device=torch.device(device),
        return_angle=True,
        p=options["reflection_p"],
    )
    rotation_maps = np.asarray(rotation_maps, dtype=np.float32)
    reflection_map = np.asarray(reflection_map, dtype=np.float32)
    theta_map = np.asarray(theta_map, dtype=np.float32)
    if rotation_maps.shape != (4, *image.shape):
        raise RuntimeError("Rotational symmetry outputs do not match the v1 contract.")
    if reflection_map.shape != image.shape or theta_map.shape != image.shape:
        raise RuntimeError("Reflection symmetry outputs do not match the source shape.")
    features = np.stack(
        (
            image,
            reflection_map,
            np.sin(theta_map * 2.0),
            np.cos(theta_map * 2.0),
            rotation_maps[0],
            rotation_maps[1],
            rotation_maps[2],
            rotation_maps[3],
        )
    ).astype(np.float32, copy=False)
    if features.shape != (8, *image.shape) or not np.isfinite(features).all():
        raise RuntimeError("The symmetry representation failed shape or finite validation.")
    return features, {
        "channel_names": list(CHANNEL_NAMES),
        "shape": list(features.shape),
        "dtype": "float32",
        "channel_ranges": [
            [float(channel.min()), float(channel.max())] for channel in features
        ],
        "n_max": options["n_max"],
        "symmetry_patch_size": options["symmetry_patch_size"],
        "rotation_folds": options["rotation_folds"],
        "reflection_p": options["reflection_p"],
        "normalize_rotation_maps": options["normalize_rotation_maps"],
        "runtime_seconds": perf_counter() - started,
        "device": device,
    }


def _valid_center_bounds(
    image_shape: tuple[int, int], patch_size: int
) -> tuple[int, int, int, int]:
    height, width = image_shape
    half = patch_size // 2
    if height < patch_size or width < patch_size:
        raise ValueError("The image is smaller than the classifier patch.")
    return half, width - half, half, height - half


def _validate_support(
    image_shape: tuple[int, int],
    coordinates_xy: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    options: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    coordinates = np.asarray(coordinates_xy, dtype=np.int32)
    resolved_labels = np.asarray(labels, dtype=np.int64)
    names = [str(name).strip() for name in class_names]
    if coordinates.ndim != 2 or coordinates.shape[1:] != (2,):
        raise ValueError("Support coordinates must have shape (samples, 2).")
    if resolved_labels.shape != (len(coordinates),):
        raise ValueError("Support labels must align with support coordinates.")
    if len(names) < 2 or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("At least two unique nonempty local class names are required.")
    expected_labels = set(range(len(names)))
    if set(resolved_labels.tolist()) != expected_labels:
        raise ValueError("Support labels must cover every contiguous local class index.")
    counts = np.bincount(resolved_labels, minlength=len(names))
    if np.any(counts < options["minimum_shots_per_class"]):
        raise ValueError("Every class must satisfy the minimum support-point count.")
    if np.any(counts > options["maximum_shots_per_class"]):
        raise ValueError("A class exceeds the maximum support-point count.")
    if len({tuple(point) for point in coordinates.tolist()}) != len(coordinates):
        raise ValueError("Support coordinates must be unique across all classes.")
    min_x, max_x, min_y, max_y = _valid_center_bounds(
        image_shape, options["classifier_patch_size"]
    )
    for x, y in coordinates:
        if not min_x <= x <= max_x or not min_y <= y <= max_y:
            raise ValueError(f"Support point ({x}, {y}) cannot provide a full patch.")
    return coordinates, resolved_labels, names


def _extract_patches(
    features: np.ndarray,
    coordinates_xy: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    half = patch_size // 2
    patches = [
        features[:, y - half : y + half, x - half : x + half]
        for x, y in coordinates_xy
    ]
    result = np.stack(patches).astype(np.float32, copy=False)
    if result.shape[1:] != (8, patch_size, patch_size):
        raise RuntimeError("Patch extraction produced an incomplete patch.")
    return result


def _dense_coordinate_grid(
    image_shape: tuple[int, int], patch_size: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_x, max_x, min_y, max_y = _valid_center_bounds(image_shape, patch_size)
    x_values = np.arange(min_x, max_x + 1, stride, dtype=np.int32)
    y_values = np.arange(min_y, max_y + 1, stride, dtype=np.int32)
    xx, yy = np.meshgrid(x_values, y_values, indexing="xy")
    coordinates = np.column_stack((xx.ravel(), yy.ravel())).astype(np.int32)
    return coordinates, x_values, y_values


def _iter_patch_batches(
    features: np.ndarray,
    coordinates_xy: np.ndarray,
    *,
    patch_size: int,
    batch_size: int,
) -> Iterator[np.ndarray]:
    for start in range(0, len(coordinates_xy), batch_size):
        yield _extract_patches(
            features,
            coordinates_xy[start : start + batch_size],
            patch_size,
        )


def _predict_dense(
    model,
    features: np.ndarray,
    options: dict[str, Any],
    *,
    device: str,
) -> dict[str, np.ndarray]:
    from .model import predict_probabilities

    coordinates, x_values, y_values = _dense_coordinate_grid(
        (int(features.shape[1]), int(features.shape[2])),
        options["classifier_patch_size"],
        options["stride"],
    )
    logits_batches = []
    probability_batches = []
    for patches in _iter_patch_batches(
        features,
        coordinates,
        patch_size=options["classifier_patch_size"],
        batch_size=options["batch_size"],
    ):
        logits, probabilities = predict_probabilities(
            model,
            patches,
            batch_size=options["batch_size"],
            device=device,
        )
        logits_batches.append(logits)
        probability_batches.append(probabilities)
    logits = np.concatenate(logits_batches)
    probabilities = np.concatenate(probability_batches)
    predictions = probabilities.argmax(axis=1).astype(np.int16)
    confidence = probabilities.max(axis=1).astype(np.float32)
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1
    ).astype(np.float32)
    grid_shape = (len(y_values), len(x_values))
    return {
        "coordinates_xy": coordinates,
        "x_coordinates": x_values,
        "y_coordinates": y_values,
        "logits": logits,
        "probabilities": probabilities,
        "predictions": predictions,
        "confidence": confidence,
        "entropy": entropy,
        "prediction_grid": predictions.reshape(grid_shape),
        "confidence_grid": confidence.reshape(grid_shape),
        "entropy_grid": entropy.reshape(grid_shape),
    }


def _execute_analysis(
    image: np.ndarray,
    support: dict[str, Any],
    method: dict[str, Any],
    option_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    import torch

    from .model import (
        add_task_adapters,
        load_pretrained_checkpoint,
        resolve_device,
        set_deterministic_seed,
        train_adapters,
        trainable_state_dict,
    )

    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Provider input must be a finite two-dimensional array.")
    if min(values.shape) < 64:
        raise ValueError("Both provider input dimensions must be at least 64 pixels.")
    if float(values.min()) < -1e-6 or float(values.max()) > 1.000001:
        raise ValueError("Provider input must lie in the shared [0, 1] space.")
    options = _validated_options(option_overrides)
    if str(method.get("identifier")) != MODEL_IDENTIFIER:
        raise ValueError("The worker job requests an unsupported model identifier.")
    coordinates, labels, class_names = _validate_support(
        values.shape,
        np.asarray(support.get("coordinates_xy")),
        np.asarray(support.get("labels")),
        list(support.get("class_names", [])),
        options,
    )
    resolved_device = resolve_device(options["device"])
    started = perf_counter()
    features, feature_record = _compute_features(
        values, options, device=resolved_device
    )
    support_patches = _extract_patches(
        features, coordinates, options["classifier_patch_size"]
    )
    set_deterministic_seed(options["seed"])
    model = build_registered_model(MODEL_IDENTIFIER)
    checkpoint_record = load_pretrained_checkpoint(
        model,
        method["checkpoint_path"],
        method["checkpoint_sha256"],
    )
    model = add_task_adapters(
        model,
        bottleneck=options["adapter_bottleneck"],
        task_classes=len(class_names),
    )
    training = train_adapters(
        model,
        support_patches,
        labels,
        epochs=options["epochs"],
        learning_rate=options["learning_rate"],
        weight_decay=options["weight_decay"],
        seed=options["seed"],
        device=resolved_device,
    )
    prediction = _predict_dense(
        model, features, options, device=resolved_device
    )
    arrays = {
        "features": features,
        "channel_names": np.asarray(CHANNEL_NAMES),
        "support_patches": support_patches,
        "support_labels": labels,
        "support_coordinates_xy": coordinates,
        **prediction,
    }
    adapter_checkpoint = {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "provider_contract_version": PROVIDER_CONTRACT_VERSION,
        "model_identifier": MODEL_IDENTIFIER,
        "base_checkpoint_sha256": checkpoint_record["sha256"],
        "class_names": class_names,
        "adapter_bottleneck": options["adapter_bottleneck"],
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
        "requested_device": options["device"],
        "resolved_device": resolved_device,
        "cuda_device": (
            torch.cuda.get_device_name(torch.device(resolved_device))
            if resolved_device.startswith("cuda")
            else None
        ),
        "total_seconds": perf_counter() - started,
    }
    record = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "provider_contract_version": PROVIDER_CONTRACT_VERSION,
        "provider": "symmetry-learn",
        "provider_version": provider_capabilities()["provider_version"],
        "identifier": MODEL_IDENTIFIER,
        "options": options,
        "support": {
            "class_names": class_names,
            "counts": np.bincount(labels, minlength=len(class_names)).tolist(),
            "sample_count": int(len(labels)),
        },
        "model": {
            "identifier": MODEL_IDENTIFIER,
            "input_channels": 8,
            "pretrained_classes": 17,
            "classifier_patch_size": 64,
            "checkpoint": checkpoint_record,
            "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
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
    return {
        "arrays": arrays,
        "adapter_checkpoint": adapter_checkpoint,
        "record": record,
    }


def run_worker(job_path: Path) -> dict[str, Any]:
    """Execute one analysis job and persist arrays plus provider provenance."""
    import torch

    job = _load_job(job_path)
    result = _execute_analysis(
        _load_input(Path(job["input_path"])),
        dict(job["support"]),
        dict(job["method"]),
        dict(job.get("options", {})),
    )
    output_path = Path(job["output_path"])
    adapter_path = Path(job["adapter_path"])
    record_path = Path(job["record_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **result["arrays"])
    torch.save(result["adapter_checkpoint"], adapter_path)
    record = {
        **result["record"],
        "output_path": str(output_path.resolve()),
        "adapter_path": str(adapter_path.resolve()),
        "record_path": str(record_path.resolve()),
    }
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def probe_worker(job_path: Path) -> dict[str, Any]:
    """Validate the model, checkpoint, and requested device without inference."""
    import torch

    from .model import load_pretrained_checkpoint, resolve_device

    job = _load_job(job_path)
    method = dict(job["method"])
    identifier = str(method.get("identifier"))
    options = _validated_options(dict(job.get("options", {})))
    device = resolve_device(options["device"])
    model = build_registered_model(identifier)
    checkpoint = load_pretrained_checkpoint(
        model,
        method["checkpoint_path"],
        method["checkpoint_sha256"],
    )
    return {
        "schema_version": WORKER_SCHEMA_VERSION,
        "provider_contract_version": PROVIDER_CONTRACT_VERSION,
        "provider": "symmetry-learn",
        "identifier": identifier,
        "available": True,
        "details": {
            "model_class": model.__class__.__module__ + "." + model.__class__.__name__,
            "torch_version": str(torch.__version__),
            "requested_device": options["device"],
            "resolved_device": device,
            "cuda_device": (
                torch.cuda.get_device_name(torch.device(device))
                if device.startswith("cuda")
                else None
            ),
            "checkpoint": checkpoint,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job", type=Path)
    group.add_argument("--probe", type=Path)
    group.add_argument("--capabilities", action="store_true")
    arguments = parser.parse_args()
    if arguments.capabilities:
        result = provider_capabilities()
    elif arguments.job is not None:
        result = run_worker(arguments.job.resolve())
    else:
        result = probe_worker(arguments.probe.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
