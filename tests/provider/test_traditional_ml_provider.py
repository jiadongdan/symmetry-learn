from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from symmlearn import __version__
from symmlearn.features.eight_channel.contract import CHANNEL_NAMES
from symmlearn.provider import traditional_ml_analyze
from symmlearn.provider.api import ProviderTraditionalMLResult
from symmlearn.provider.contracts import (
    PROVIDER_CONTRACT_VERSION,
    WORKER_SCHEMA_VERSION,
)
from symmlearn.provider.registry import provider_capabilities
from symmlearn.provider.worker import (
    _load_job,
    run_traditional_ml_worker,
)
from symmlearn.traditional_ml import (
    TRADITIONAL_ML_CAPABILITY_SCHEMA_VERSION,
    TRADITIONAL_ML_OPERATION,
    TRADITIONAL_ML_RECORD_SCHEMA_VERSION,
    TraditionalMLOptions,
    classifier_specification,
)


IMAGE_SIZE = 96
PATCH_SIZE = 32
CLASS_A = [(24, 24), (28, 28), (32, 32)]
CLASS_B = [(72, 72), (68, 68), (64, 64)]


def _image(size: int = IMAGE_SIZE) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    values = np.sin(x / 6.0) + np.cos(y / 7.0) + 0.01 * x
    return ((values - values.min()) / (values.max() - values.min())).astype(np.float32)


def _support(*groups) -> dict:
    coordinates, labels = [], []
    for label, group in enumerate(groups):
        for point in group:
            coordinates.append(list(point))
            labels.append(label)
    return {
        "coordinates_xy": coordinates,
        "labels": labels,
        "class_names": [f"Class {index}" for index in range(len(groups))],
    }


def _write_feature_artifact(
    directory: Path,
    image: np.ndarray,
    *,
    channel_names: list[str] | None = None,
    shape_override: list[int] | None = None,
) -> tuple[Path, Path, np.ndarray]:
    """Write a synthetic but contract-shaped feature artifact."""
    features = np.random.default_rng(7).random((8, *image.shape), dtype=np.float32)
    names = list(CHANNEL_NAMES if channel_names is None else channel_names)
    features_path = directory / "features.npz"
    record_path = directory / "features_record.json"
    np.savez_compressed(
        features_path,
        features=features,
        channel_names=np.asarray(names),
    )
    record_path.write_text(
        json.dumps(
            {
                "schema_version": WORKER_SCHEMA_VERSION,
                "provider_contract_version": PROVIDER_CONTRACT_VERSION,
                "provider": "symmetry-learn",
                "provider_version": __version__,
                "identifier": "cnn_8ch_pg17",
                "features": {
                    "channel_names": names,
                    "shape": shape_override or list(features.shape),
                    "dtype": "float32",
                    "n_max": 2,
                    "symmetry_patch_size": 5,
                    "rotation_folds": [2, 3, 4, 6],
                    "reflection_p": 2.0,
                    "normalize_rotation_maps": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return features_path, record_path, features


def _job(
    directory: Path,
    image: np.ndarray,
    *,
    features_path: Path,
    features_record_path: Path,
    classifier: str = "random_forest",
    parameters: dict | None = None,
    feature_mode: str = "image_plus_symmetry_maps",
    support: dict | None = None,
    options: dict | None = None,
    input_sha256: str | None = None,
    omit: set[str] | None = None,
) -> Path:
    input_path = directory / "input.npy"
    np.save(input_path, image)
    payload = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "input_path": str(input_path),
        "features_path": str(features_path),
        "features_record_path": str(features_record_path),
        "feature_mode": feature_mode,
        "support": support if support is not None else _support(CLASS_A, CLASS_B),
        "classifier": {
            "identifier": classifier,
            "parameters": (
                {"n_estimators": 8} if parameters is None else parameters
            ),
        },
        "options": options
        or {
            "classifier_patch_size": PATCH_SIZE,
            "seed": 42,
            "stride": 8,
            "batch_size": 32,
        },
        "output_path": str(directory / "traditional_prediction.npz"),
        "record_path": str(directory / "provider_record.json"),
        "progress_path": str(directory / "progress.json"),
    }
    if input_sha256 is not None:
        payload["input_sha256"] = input_sha256
    for key in omit or set():
        payload.pop(key, None)
    job_path = directory / "traditional_job.json"
    job_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return job_path


# --- capability document ----------------------------------------------------


def test_capabilities_advertise_the_traditional_operation_and_block() -> None:
    capabilities = provider_capabilities()
    assert TRADITIONAL_ML_OPERATION in capabilities["operations"]
    block = capabilities["traditional_ml"]
    assert block["schema_version"] == TRADITIONAL_ML_CAPABILITY_SCHEMA_VERSION
    assert block["feature_modes"] == ["raw_image", "image_plus_symmetry_maps"]
    assert [entry["identifier"] for entry in block["classifiers"]] == [
        "logistic_regression",
        "random_forest",
    ]
    assert all(entry["supports_predict_proba"] is True for entry in block["classifiers"])
    assert capabilities["provider_version"] == "0.1.1"


def test_capability_defaults_come_from_the_classifier_registry() -> None:
    block = provider_capabilities()["traditional_ml"]
    for entry in block["classifiers"]:
        assert entry["defaults"] == classifier_specification(
            entry["identifier"]
        ).defaults()


def test_capabilities_document_the_new_object_operation_versions() -> None:
    capabilities = provider_capabilities()
    assert capabilities["contract_version"] == PROVIDER_CONTRACT_VERSION
    assert TRADITIONAL_ML_RECORD_SCHEMA_VERSION == "symmetry-traditional-ml-run-v1"
    assert WORKER_SCHEMA_VERSION == "scientific-symmetry-worker-v1"


# --- public Python API ------------------------------------------------------


def test_public_api_returns_the_provider_record_envelope() -> None:
    image = _image()
    features = np.random.default_rng(3).random((8, *image.shape), dtype=np.float32)
    coordinates, labels = [], []
    for label, group in enumerate((CLASS_A, CLASS_B)):
        for point in group:
            coordinates.append(point)
            labels.append(label)
    result = traditional_ml_analyze(
        features,
        coordinates_xy=np.asarray(coordinates, dtype=np.int32),
        labels=np.asarray(labels, dtype=np.int64),
        class_names=["a", "b"],
        classifier="logistic_regression",
        feature_mode="raw_image",
        options=TraditionalMLOptions(
            classifier_patch_size=PATCH_SIZE, seed=42, stride=8, batch_size=16
        ),
        input_shape=(IMAGE_SIZE, IMAGE_SIZE),
    )
    assert isinstance(result, ProviderTraditionalMLResult)
    assert result.record["provider"] == "symmetry-learn"
    assert result.record["provider_version"] == __version__
    assert result.record["provider_contract_version"] == PROVIDER_CONTRACT_VERSION
    assert result.record["worker_schema_version"] == WORKER_SCHEMA_VERSION
    assert result.record["schema_version"] == TRADITIONAL_ML_RECORD_SCHEMA_VERSION
    assert result.record["operation"] == TRADITIONAL_ML_OPERATION
    assert result.record["classifier"]["identifier"] == "logistic_regression"
    assert result.prediction_grid.shape == result.confidence_grid.shape


def test_public_api_accepts_a_json_options_mapping() -> None:
    image = _image()
    features = np.random.default_rng(4).random((8, *image.shape), dtype=np.float32)
    coordinates, labels = [], []
    for label, group in enumerate((CLASS_A, CLASS_B)):
        for point in group:
            coordinates.append(point)
            labels.append(label)
    result = traditional_ml_analyze(
        features,
        coordinates_xy=np.asarray(coordinates, dtype=np.int32),
        labels=np.asarray(labels, dtype=np.int64),
        class_names=["a", "b"],
        classifier="random_forest",
        parameters={"n_estimators": 5},
        options={"classifier_patch_size": PATCH_SIZE, "stride": 8, "batch_size": 16},
    )
    assert result.record["options"]["classifier_patch_size"] == PATCH_SIZE
    assert result.record["options"]["seed"] == 42
    assert result.record["classifier"]["parameters"]["n_estimators"] == 5


def test_public_api_rejects_an_unsupported_classifier() -> None:
    image = _image()
    features = np.random.default_rng(5).random((8, *image.shape), dtype=np.float32)
    with pytest.raises(ValueError, match="Unsupported classifier"):
        traditional_ml_analyze(
            features,
            coordinates_xy=np.asarray(CLASS_A + CLASS_B, dtype=np.int32),
            labels=np.asarray([0] * len(CLASS_A) + [1] * len(CLASS_B), dtype=np.int64),
            class_names=["a", "b"],
            classifier="support_vector_machine",
        )


# --- worker -----------------------------------------------------------------


def test_worker_success_writes_arrays_record_and_progress(tmp_path: Path) -> None:
    image = _image()
    features_path, record_path, _ = _write_feature_artifact(tmp_path, image)
    job_path = _job(
        tmp_path,
        image,
        features_path=features_path,
        features_record_path=record_path,
    )
    summary = run_traditional_ml_worker(job_path)

    assert summary["operation"] == TRADITIONAL_ML_OPERATION
    assert summary["provider"] == "symmetry-learn"
    assert summary["provider_version"] == "0.1.1"
    assert summary["worker_schema_version"] == WORKER_SCHEMA_VERSION
    assert summary["schema_version"] == TRADITIONAL_ML_RECORD_SCHEMA_VERSION
    assert summary["classifier"]["identifier"] == "random_forest"
    assert summary["seed"] == 42
    assert summary["feature_mode"] == "image_plus_symmetry_maps"
    assert summary["class_names"] == ["Class 0", "Class 1"]
    assert summary["support_counts"] == {"Class 0": 3, "Class 1": 3}
    assert summary["input_shape"] == [IMAGE_SIZE, IMAGE_SIZE]
    assert summary["feature_shape"] == [8, IMAGE_SIZE, IMAGE_SIZE]
    assert summary["feature_provenance"]["channel_names"] == list(CHANNEL_NAMES)
    assert json.loads(json.dumps(summary, sort_keys=True)) == summary

    arrays_path = Path(summary["output_path"])
    with np.load(arrays_path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "coordinates_xy",
            "x_coordinates",
            "y_coordinates",
            "probabilities",
            "predictions",
            "confidence",
            "entropy",
            "prediction_grid",
            "confidence_grid",
            "entropy_grid",
        }
        assert "logits" not in archive.files
        grid = archive["prediction_grid"]
        assert grid.dtype == np.int16
    written = json.loads(Path(summary["record_path"]).read_text(encoding="utf-8"))
    assert written == summary

    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["phase"] == "prediction"
    assert progress["current"] == progress["total"] == 3


def test_worker_never_writes_an_estimator_artifact(tmp_path: Path) -> None:
    image = _image()
    features_path, record_path, _ = _write_feature_artifact(tmp_path, image)
    job_path = _job(
        tmp_path,
        image,
        features_path=features_path,
        features_record_path=record_path,
    )
    run_traditional_ml_worker(job_path)
    suffixes = {path.suffix for path in tmp_path.iterdir()}
    assert not suffixes & {".pkl", ".joblib", ".pt", ".pth", ".symmodel"}
    assert (tmp_path / "traditional_prediction.npz").is_file()
    assert (tmp_path / "provider_record.json").is_file()


def test_worker_job_schema_version_is_enforced(tmp_path: Path) -> None:
    image = _image()
    features_path, record_path, _ = _write_feature_artifact(tmp_path, image)
    job_path = _job(
        tmp_path,
        image,
        features_path=features_path,
        features_record_path=record_path,
    )
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "scientific-symmetry-worker-v0"
    job_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported worker job schema version"):
        _load_job(job_path)


@pytest.mark.parametrize(
    "omit,message",
    [
        ({"features_path"}, "features_path"),
        ({"features_record_path"}, "features_record_path"),
        ({"output_path"}, "output_path"),
        ({"record_path"}, "record_path"),
        ({"input_path"}, "input_path"),
        ({"support"}, "support object"),
        ({"classifier"}, "classifier object"),
    ],
)
def test_worker_requires_every_declared_path(
    tmp_path: Path, omit: set[str], message: str
) -> None:
    image = _image()
    features_path, record_path, _ = _write_feature_artifact(tmp_path, image)
    job_path = _job(
        tmp_path,
        image,
        features_path=features_path,
        features_record_path=record_path,
        omit=omit,
    )
    with pytest.raises(ValueError, match=message):
        run_traditional_ml_worker(job_path)


def test_worker_requires_a_classifier_identifier(tmp_path: Path) -> None:
    image = _image()
    features_path, record_path, _ = _write_feature_artifact(tmp_path, image)
    job_path = _job(
        tmp_path,
        image,
        features_path=features_path,
        features_record_path=record_path,
    )
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    payload["classifier"] = {"identifier": "", "parameters": {}}
    job_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="classifier.identifier"):
        run_traditional_ml_worker(job_path)


@pytest.mark.parametrize(
    "mismatch,message",
    [
        ("shape", "do not share a shape"),
        ("record_shape", "disagrees with the stored feature array"),
        ("channels", "eight-channel contract"),
    ],
)
def test_worker_rejects_mismatched_feature_provenance(
    tmp_path: Path, mismatch: str, message: str
) -> None:
    image = _image()
    if mismatch == "shape":
        features = np.random.default_rng(9).random(
            (8, IMAGE_SIZE, IMAGE_SIZE - 8), dtype=np.float32
        )
        features_path = tmp_path / "features.npz"
        record_path = tmp_path / "features_record.json"
        np.savez_compressed(
            features_path,
            features=features,
            channel_names=np.asarray(CHANNEL_NAMES),
        )
        record_path.write_text(
            json.dumps(
                {
                    "schema_version": WORKER_SCHEMA_VERSION,
                    "provider_contract_version": PROVIDER_CONTRACT_VERSION,
                    "provider": "symmetry-learn",
                    "identifier": "cnn_8ch_pg17",
                    "features": {
                        "channel_names": list(CHANNEL_NAMES),
                        "shape": list(features.shape),
                    },
                }
            ),
            encoding="utf-8",
        )
    elif mismatch == "record_shape":
        features_path, record_path, _ = _write_feature_artifact(
            tmp_path, image, shape_override=[8, IMAGE_SIZE - 4, IMAGE_SIZE - 4]
        )
    else:
        features_path, record_path, _ = _write_feature_artifact(
            tmp_path, image, channel_names=["image"] + ["other"] * 7
        )
    job_path = _job(
        tmp_path,
        image,
        features_path=features_path,
        features_record_path=record_path,
    )
    with pytest.raises(ValueError, match=message):
        run_traditional_ml_worker(job_path)


def test_worker_rejects_an_invalid_input_digest(tmp_path: Path) -> None:
    image = _image()
    features_path, record_path, _ = _write_feature_artifact(tmp_path, image)
    job_path = _job(
        tmp_path,
        image,
        features_path=features_path,
        features_record_path=record_path,
        input_sha256="not-a-digest",
    )
    with pytest.raises(ValueError, match="hexadecimal digest"):
        run_traditional_ml_worker(job_path)


def test_worker_records_a_valid_input_digest(tmp_path: Path) -> None:
    image = _image()
    features_path, record_path, _ = _write_feature_artifact(tmp_path, image)
    job_path = _job(
        tmp_path,
        image,
        features_path=features_path,
        features_record_path=record_path,
        input_sha256="B" * 64,
    )
    summary = run_traditional_ml_worker(job_path)
    assert summary["input_sha256"] == "b" * 64


@pytest.mark.parametrize(
    "feature_mode,channel_count",
    [("raw_image", 1), ("image_plus_symmetry_maps", 8)],
)
def test_worker_honors_both_feature_modes(
    tmp_path: Path, feature_mode: str, channel_count: int
) -> None:
    image = _image()
    features_path, record_path, _ = _write_feature_artifact(tmp_path, image)
    job_path = _job(
        tmp_path,
        image,
        features_path=features_path,
        features_record_path=record_path,
        feature_mode=feature_mode,
    )
    summary = run_traditional_ml_worker(job_path)
    assert summary["feature_mode"] == feature_mode
    assert summary["channel_count"] == channel_count
    assert len(summary["channel_names"]) == channel_count


def test_worker_rejects_an_unsupported_feature_mode(tmp_path: Path) -> None:
    image = _image()
    features_path, record_path, _ = _write_feature_artifact(tmp_path, image)
    job_path = _job(
        tmp_path,
        image,
        features_path=features_path,
        features_record_path=record_path,
        feature_mode="symmetry_only",
    )
    with pytest.raises(ValueError, match="Unsupported feature mode"):
        run_traditional_ml_worker(job_path)


def test_worker_rejects_unknown_job_options(tmp_path: Path) -> None:
    image = _image()
    features_path, record_path, _ = _write_feature_artifact(tmp_path, image)
    job_path = _job(
        tmp_path,
        image,
        features_path=features_path,
        features_record_path=record_path,
        options={"classifier_patch_size": PATCH_SIZE, "epochs": 5},
    )
    with pytest.raises(ValueError, match="Unsupported traditional ML options"):
        run_traditional_ml_worker(job_path)


def test_worker_reports_prediction_progress_for_every_batch(tmp_path: Path) -> None:
    image = _image()
    features_path, record_path, _ = _write_feature_artifact(tmp_path, image)
    job_path = _job(
        tmp_path,
        image,
        features_path=features_path,
        features_record_path=record_path,
        options={
            "classifier_patch_size": PATCH_SIZE,
            "seed": 42,
            "stride": 8,
            "batch_size": 8,
        },
    )
    run_traditional_ml_worker(job_path)
    final = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert final["phase"] == "prediction"
    assert final["total"] > 1
    assert final["current"] == final["total"]


def test_worker_result_is_reproducible_for_the_same_job(tmp_path: Path) -> None:
    image = _image()
    features_path, record_path, _ = _write_feature_artifact(tmp_path, image)
    job_path = _job(
        tmp_path,
        image,
        features_path=features_path,
        features_record_path=record_path,
    )
    first = run_traditional_ml_worker(job_path)
    first_arrays = np.load(first["output_path"], allow_pickle=False)
    second = run_traditional_ml_worker(job_path)
    second_arrays = np.load(second["output_path"], allow_pickle=False)
    assert np.array_equal(
        first_arrays["probabilities"], second_arrays["probabilities"]
    )
    assert np.array_equal(first_arrays["predictions"], second_arrays["predictions"])


# --- worker command line ----------------------------------------------------


def test_worker_cli_runs_a_traditional_job_end_to_end(tmp_path: Path) -> None:
    image = _image()
    features_path, record_path, _ = _write_feature_artifact(tmp_path, image)
    job_path = _job(
        tmp_path,
        image,
        features_path=features_path,
        features_record_path=record_path,
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "symmlearn.provider.worker",
            "--traditional-ml",
            str(job_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["operation"] == TRADITIONAL_ML_OPERATION
    assert payload["provider_version"] == "0.1.1"
    assert (tmp_path / "traditional_prediction.npz").is_file()
    assert (tmp_path / "provider_record.json").is_file()


def test_worker_cli_keeps_its_actions_mutually_exclusive(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "symmlearn.provider.worker",
            "--traditional-ml",
            str(tmp_path / "job.json"),
            "--features",
            str(tmp_path / "features.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "not allowed with argument" in completed.stderr


def test_worker_cli_capabilities_advertise_traditional_ml() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "symmlearn.provider.worker", "--capabilities"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert TRADITIONAL_ML_OPERATION in payload["operations"]
    assert (
        payload["traditional_ml"]["schema_version"]
        == TRADITIONAL_ML_CAPABILITY_SCHEMA_VERSION
    )
    assert payload["provider_version"] == "0.1.1"
