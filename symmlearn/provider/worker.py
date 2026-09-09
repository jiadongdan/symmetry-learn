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
from .api import few_shot_analyze, probe_model
from .contracts import (
    MODEL_IDENTIFIER,
    PROVIDER_CONTRACT_VERSION,
    WORKER_SCHEMA_VERSION,
)
from .registry import provider_capabilities
from .serialization import persist_provider_result


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
    )
    return persist_provider_result(
        result,
        output_path=job["output_path"],
        adapter_path=job["adapter_path"],
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
    group.add_argument("--capabilities", action="store_true")
    arguments = parser.parse_args()
    if arguments.capabilities:
        result = provider_capabilities()
    elif arguments.features is not None:
        result = run_features_worker(arguments.features.resolve())
    elif arguments.job is not None:
        result = run_worker(arguments.job.resolve())
    else:
        result = probe_worker(arguments.probe.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
