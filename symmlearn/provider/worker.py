"""Execute one versioned few-shot symmetry job for an external orchestrator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from symmlearn import __version__
from symmlearn.features import compute_registered_features
from symmlearn.inference import predict_dense
from symmlearn.models.registry import (
    get_model_specification,
    validate_model_options,
)
from symmlearn.patches import (
    dense_coordinate_grid,
    extract_patches,
    iter_patch_batches,
    valid_center_bounds,
    validate_support,
)

from .api import compute_features as compute_provider_features
from .api import (
    few_shot_analyze,
    probe_model,
    predict_with_fine_tuned_model,
    traditional_ml_analyze,
)
from .contracts import (
    MODEL_IDENTIFIER,
    PROVIDER_CONTRACT_VERSION,
    WORKER_SCHEMA_VERSION,
)
from .registry import provider_capabilities
from .serialization import (
    persist_prediction_result,
    persist_provider_result,
    persist_traditional_result,
)
from symmlearn.features.eight_channel.contract import CHANNEL_NAMES
from symmlearn.traditional_ml.contracts import (
    IMAGE_PLUS_SYMMETRY_MAPS_MODE,
    TraditionalMLOptions,
)
from symmlearn.workflows.saved_model_prediction import SavedModelPredictor


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


def _load_precomputed_features(
    output_path: Path, record_path: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a feature artifact produced by this versioned Provider."""
    with np.load(output_path, allow_pickle=False) as archive:
        if "features" not in archive.files:
            raise ValueError("Precomputed feature artifact has no features array.")
        features = np.asarray(archive["features"], dtype=np.float32).copy()
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != WORKER_SCHEMA_VERSION:
        raise ValueError("Precomputed feature worker schema mismatch.")
    if payload.get("provider_contract_version") != PROVIDER_CONTRACT_VERSION:
        raise ValueError("Precomputed feature Provider contract mismatch.")
    feature_record = payload.get("features")
    if not isinstance(feature_record, dict):
        raise ValueError("Precomputed feature record is incomplete.")
    return features, dict(feature_record)


def _progress_reporter(progress_path: Path):
    """Return a Windows-safe writer for best-effort worker progress updates."""

    def write_progress(phase: str, current: int, total: int) -> None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(
            json.dumps(
                {"phase": phase, "current": int(current), "total": int(total)}
            ),
            encoding="utf-8",
        )

    return write_progress


def _validated_options(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Preserve the v1 helper while delegating model option validation."""
    return validate_model_options(MODEL_IDENTIFIER, overrides)


def _compute_features(
    image: np.ndarray,
    options: dict[str, Any],
    *,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Preserve the v1 helper while delegating feature computation."""
    specification = get_model_specification(MODEL_IDENTIFIER)
    return compute_registered_features(
        specification.feature_pipeline,
        image,
        options,
        device=device,
    )


def _valid_center_bounds(
    image_shape: tuple[int, int], patch_size: int
) -> tuple[int, int, int, int]:
    """Preserve the v1 helper while delegating boundary calculation."""
    return valid_center_bounds(image_shape, patch_size)


def _validate_support(
    image_shape: tuple[int, int],
    coordinates_xy: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    options: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Preserve the v1 helper while delegating support validation."""
    return validate_support(
        image_shape,
        coordinates_xy,
        labels,
        class_names,
        options,
    )


def _extract_patches(
    features: np.ndarray,
    coordinates_xy: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    """Preserve the v1 helper while delegating patch extraction."""
    return extract_patches(
        features,
        coordinates_xy,
        patch_size,
        expected_channels=8,
    )


def _dense_coordinate_grid(
    image_shape: tuple[int, int], patch_size: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preserve the v1 helper while delegating dense grid creation."""
    return dense_coordinate_grid(image_shape, patch_size, stride)


def _iter_patch_batches(
    features: np.ndarray,
    coordinates_xy: np.ndarray,
    *,
    patch_size: int,
    batch_size: int,
) -> Iterator[np.ndarray]:
    """Preserve the v1 helper while delegating bounded patch iteration."""
    yield from iter_patch_batches(
        features,
        coordinates_xy,
        patch_size=patch_size,
        batch_size=batch_size,
        expected_channels=8,
    )


def _predict_dense(
    model,
    features: np.ndarray,
    options: dict[str, Any],
    *,
    device: str,
) -> dict[str, np.ndarray]:
    """Preserve the v1 helper while delegating dense inference."""
    return predict_dense(
        model,
        features,
        patch_size=options["classifier_patch_size"],
        stride=options["stride"],
        batch_size=options["batch_size"],
        device=device,
        expected_channels=8,
    ).to_arrays()


def _execute_analysis(
    image: np.ndarray,
    support: dict[str, Any],
    method: dict[str, Any],
    option_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Preserve the v1 dictionary helper while delegating the workflow."""
    result = few_shot_analyze(
        image,
        coordinates_xy=np.asarray(support.get("coordinates_xy")),
        labels=np.asarray(support.get("labels")),
        class_names=list(support.get("class_names", [])),
        checkpoint_path=method.get("checkpoint_path"),
        checkpoint_sha256=method.get("checkpoint_sha256"),
        weight=method.get("weight_identifier"),
        options=option_overrides,
        model=str(method.get("identifier", MODEL_IDENTIFIER)),
    )
    return {
        "arrays": result.arrays,
        "adapter_checkpoint": result.adapter_checkpoint,
        "record": result.record,
    }


def run_worker(job_path: Path) -> dict[str, Any]:
    """Execute one analysis job and persist arrays plus provider provenance."""
    job = _load_job(job_path)
    support = dict(job["support"])
    method = dict(job["method"])
    precomputed_features = None
    precomputed_feature_record = None
    if job.get("features_path") is not None:
        if job.get("features_record_path") is None:
            raise ValueError(
                "A precomputed features path requires a feature record path."
            )
        precomputed_features, precomputed_feature_record = _load_precomputed_features(
            Path(job["features_path"]), Path(job["features_record_path"])
        )
    elif job.get("features_record_path") is not None:
        raise ValueError("A feature record path requires a precomputed features path.")
    progress_path = (
        None if job.get("progress_path") is None else Path(job["progress_path"])
    )

    result = few_shot_analyze(
        _load_input(Path(job["input_path"])),
        coordinates_xy=np.asarray(support.get("coordinates_xy")),
        labels=np.asarray(support.get("labels")),
        class_names=list(support.get("class_names", [])),
        checkpoint_path=method.get("checkpoint_path"),
        checkpoint_sha256=method.get("checkpoint_sha256"),
        weight=method.get("weight_identifier"),
        options=dict(job.get("options", {})),
        model=str(method.get("identifier", MODEL_IDENTIFIER)),
        precomputed_features=precomputed_features,
        precomputed_feature_record=precomputed_feature_record,
        progress_callback=(
            None if progress_path is None else _progress_reporter(progress_path)
        ),
    )
    return persist_provider_result(
        result,
        output_path=job["output_path"],
        adapter_path=job["adapter_path"],
        model_state_path=job.get("model_state_path"),
        record_path=job["record_path"],
    )


def run_features_worker(job_path: Path) -> dict[str, Any]:
    """Compute and persist the selected model's reusable feature representation."""
    job = _load_job(job_path)
    method = dict(job["method"])
    model = str(method.get("identifier", MODEL_IDENTIFIER))
    features, feature_record = compute_provider_features(
        _load_input(Path(job["input_path"])),
        options=dict(job.get("options", {})),
        device=str(dict(job.get("options", {})).get("device", "auto")),
        model=model,
    )
    output_path = Path(job["output_path"])
    record_path = Path(job["record_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=np.asarray(features, dtype=np.float32),
        channel_names=np.asarray(feature_record["channel_names"]),
    )
    record = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "provider_contract_version": PROVIDER_CONTRACT_VERSION,
        "provider": "symmetry-learn",
        "provider_version": __version__,
        "identifier": model,
        "features": feature_record,
        "output_path": str(output_path.resolve()),
        "record_path": str(record_path.resolve()),
    }
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def _load_prediction_items(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the batch items of one saved-model prediction job."""
    raw_items = job.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("A prediction job requires at least one item.")
    items = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise TypeError("Each prediction item must be an object.")
        for key in ("item_id", "input_path", "output_path", "record_path"):
            if not str(raw.get(key, "")).strip():
                raise ValueError(f"Prediction item {index} is missing {key!r}.")
        item_id = str(raw["item_id"])
        if item_id in seen:
            raise ValueError(f"Duplicate prediction item identifier: {item_id!r}")
        seen.add(item_id)
        items.append(
            {
                "item_id": item_id,
                "input_path": Path(str(raw["input_path"])),
                "output_path": Path(str(raw["output_path"])),
                "record_path": Path(str(raw["record_path"])),
            }
        )
    return items


def run_prediction_worker(job_path: Path) -> dict[str, Any]:
    """Predict one batch with one saved model that is restored exactly once."""
    job = _load_job(job_path)
    model_state_path = job.get("model_state_path")
    if not str(model_state_path or "").strip():
        raise ValueError("A prediction job requires a model_state_path.")
    items = _load_prediction_items(job)
    progress_path = (
        None if job.get("progress_path") is None else Path(job["progress_path"])
    )
    report_progress = (
        None if progress_path is None else _progress_reporter(progress_path)
    )
    total_items = len(items)

    # Model-level failures must stop the whole batch, so restoration happens
    # before any image is processed.
    predictor = SavedModelPredictor.from_state_path(
        model_state_path,
        feature_options=dict(job.get("feature_options", {})),
        prediction_options=dict(job.get("prediction_options", {})),
        expected_model=dict(job.get("expected_model", {}) or {}),
        prediction_defaults=dict(job.get("prediction_defaults", {}) or {}),
    )

    results: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if report_progress is not None:
            report_progress("item", index - 1, total_items)
        try:
            result = predictor.predict(
                _load_input(item["input_path"]),
                progress_callback=(
                    None
                    if report_progress is None
                    else lambda phase, current, total: report_progress(
                        f"item {index} of {total_items}: {phase}", current, total
                    )
                ),
            )
        except Exception as error:
            # An image-specific failure is recorded and the batch continues.
            results.append(
                {
                    "item_id": item["item_id"],
                    "status": "failed",
                    "phase": "prediction",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue
        persist_prediction_result(
            result,
            output_path=item["output_path"],
            record_path=item["record_path"],
        )
        results.append(
            {
                "item_id": item["item_id"],
                "status": "completed",
                "identifier": result.record["identifier"],
                "prediction": result.record["prediction"],
                "output_path": str(item["output_path"]),
                "record_path": str(item["record_path"]),
            }
        )
    if report_progress is not None:
        report_progress("item", total_items, total_items)
    return {
        "schema_version": WORKER_SCHEMA_VERSION,
        "provider_contract_version": PROVIDER_CONTRACT_VERSION,
        "provider": "symmetry-learn",
        "provider_version": __version__,
        "identifier": predictor.identifier,
        "model": predictor.restoration_record,
        "options": {
            "features": {
                name: predictor.feature_options[name]
                for name in (
                    "n_max",
                    "symmetry_patch_size",
                    "rotation_folds",
                    "reflection_p",
                    "normalize_rotation_maps",
                )
            },
            "prediction": {
                name: predictor.prediction_options[name]
                for name in ("device", "stride", "batch_size")
            },
        },
        "item_count": total_items,
        "completed_count": sum(1 for entry in results if entry["status"] == "completed"),
        "failed_count": sum(1 for entry in results if entry["status"] == "failed"),
        "items": results,
    }


def _required_job_string(job: dict[str, Any], key: str) -> str:
    value = str(job.get(key, "")).strip()
    if not value:
        raise ValueError(f"A traditional ML job requires {key!r}.")
    return value


def _validate_feature_provenance(
    image: np.ndarray,
    features: np.ndarray,
    feature_record: dict[str, Any],
) -> None:
    """Check that one feature artifact belongs to the supplied input image.

    The stored feature record carries the validated feature shape, the channel
    contract, and the map-computation options, so the Worker can refuse a job
    whose image and features do not belong together.
    """
    recorded_shape = feature_record.get("shape")
    if not isinstance(recorded_shape, (list, tuple)) or len(recorded_shape) != 3:
        raise ValueError("The feature record does not declare a validated shape.")
    if [int(value) for value in recorded_shape] != [
        int(value) for value in features.shape
    ]:
        raise ValueError(
            "The feature record shape disagrees with the stored feature array."
        )
    if (int(features.shape[1]), int(features.shape[2])) != (
        int(image.shape[0]),
        int(image.shape[1]),
    ):
        raise ValueError(
            "The feature artifact and the input image do not share a shape."
        )
    if list(feature_record.get("channel_names") or []) != list(CHANNEL_NAMES):
        raise ValueError(
            "The feature record does not carry the eight-channel contract."
        )


def run_traditional_ml_worker(job_path: Path) -> dict[str, Any]:
    """Train one conventional classifier and densely predict one image.

    This operation is independent of every pretrained checkpoint: it never
    resolves a model, weight, adapter, or model-state path.
    """
    job = _load_job(job_path)
    features_path = _required_job_string(job, "features_path")
    features_record_path = _required_job_string(job, "features_record_path")
    output_path = _required_job_string(job, "output_path")
    record_path = _required_job_string(job, "record_path")

    features, feature_record = _load_precomputed_features(
        Path(features_path), Path(features_record_path)
    )
    image = _load_input(Path(_required_job_string(job, "input_path")))
    _validate_feature_provenance(image, features, feature_record)

    support = job.get("support")
    if not isinstance(support, dict):
        raise ValueError("A traditional ML job requires a support object.")
    for key in ("coordinates_xy", "labels", "class_names"):
        if support.get(key) is None:
            raise ValueError(f"A traditional ML job requires support.{key}.")
    classifier = job.get("classifier")
    if not isinstance(classifier, dict):
        raise ValueError("A traditional ML job requires a classifier object.")
    identifier = str(classifier.get("identifier", "")).strip()
    if not identifier:
        raise ValueError("A traditional ML job requires classifier.identifier.")
    parameters = classifier.get("parameters")
    if parameters is not None and not isinstance(parameters, dict):
        raise TypeError("classifier.parameters must be an object when provided.")

    input_sha256 = job.get("input_sha256")
    if input_sha256 is not None:
        digest = str(input_sha256).strip().lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                "input_sha256 must be a 64-character hexadecimal digest."
            )
        input_sha256 = digest

    progress_path = (
        None if job.get("progress_path") is None else Path(job["progress_path"])
    )
    result = traditional_ml_analyze(
        features,
        coordinates_xy=np.asarray(support["coordinates_xy"]),
        labels=np.asarray(support["labels"]),
        class_names=list(support["class_names"]),
        classifier=identifier,
        parameters=parameters,
        feature_mode=str(job.get("feature_mode", IMAGE_PLUS_SYMMETRY_MAPS_MODE)),
        options=TraditionalMLOptions.from_mapping(dict(job.get("options") or {})),
        feature_record=feature_record,
        input_shape=(int(image.shape[0]), int(image.shape[1])),
        input_sha256=input_sha256,
        progress_callback=(
            None if progress_path is None else _progress_reporter(progress_path)
        ),
    )
    return persist_traditional_result(
        result, output_path=output_path, record_path=record_path
    )


def probe_worker(job_path: Path) -> dict[str, Any]:
    """Validate the model, checkpoint, and requested device without inference."""
    job = _load_job(job_path)
    method = dict(job["method"])
    return probe_model(
        model=str(method.get("identifier", MODEL_IDENTIFIER)),
        checkpoint_path=method.get("checkpoint_path"),
        checkpoint_sha256=method.get("checkpoint_sha256"),
        weight=method.get("weight_identifier"),
        options=dict(job.get("options", {})),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job", type=Path)
    group.add_argument("--features", type=Path)
    group.add_argument("--probe", type=Path)
    group.add_argument("--predict", type=Path)
    group.add_argument("--traditional-ml", type=Path)
    group.add_argument("--capabilities", action="store_true")
    arguments = parser.parse_args()
    if arguments.capabilities:
        result = provider_capabilities()
    elif arguments.features is not None:
        result = run_features_worker(arguments.features.resolve())
    elif arguments.predict is not None:
        result = run_prediction_worker(arguments.predict.resolve())
    elif arguments.traditional_ml is not None:
        result = run_traditional_ml_worker(arguments.traditional_ml.resolve())
    elif arguments.job is not None:
        result = run_worker(arguments.job.resolve())
    else:
        result = probe_worker(arguments.probe.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
