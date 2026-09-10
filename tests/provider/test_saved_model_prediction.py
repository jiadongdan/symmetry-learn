from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from symmlearn.finetuning.adapters import add_task_adapters
from symmlearn.finetuning.artifacts import (
    FINE_TUNED_MODEL_STATE_SCHEMA_VERSION,
    save_fine_tuned_model_state,
)
from symmlearn.models.registry import build_registered_model
from symmlearn.provider.worker import _load_job, run_prediction_worker
from symmlearn.workflows.saved_model_prediction import SavedModelPredictor


FEATURE_OPTIONS = {
    "n_max": 2,
    "symmetry_patch_size": 5,
    "rotation_folds": [2, 3, 4, 6],
    "reflection_p": 2.0,
    "normalize_rotation_maps": False,
}


def _unit_image(size: int = 64) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    image = np.sin(x / 4.0) + np.cos(y / 5.0) + 0.02 * x
    return ((image - image.min()) / (image.max() - image.min())).astype(np.float32)


def _saved_state(path: Path, *, task_classes: int = 2, bottleneck: int = 4) -> Path:
    torch = pytest.importorskip("torch")
    model = build_registered_model("cnn_8ch_pg17")
    adapted = add_task_adapters(
        model, bottleneck=bottleneck, task_classes=task_classes
    )
    save_fine_tuned_model_state(
        path,
        {
            "schema_version": FINE_TUNED_MODEL_STATE_SCHEMA_VERSION,
            "model_identifier": "cnn_8ch_pg17",
            "adapter_bottleneck": bottleneck,
            "task_classes": task_classes,
            "class_names": [f"Phase {index}" for index in range(task_classes)],
            "state_dict": {
                name: value.detach().cpu().clone()
                for name, value in adapted.state_dict().items()
            },
        },
    )
    return path


def test_saved_model_prediction_returns_the_standard_dense_arrays(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    state_path = _saved_state(tmp_path / "model_state.pt")

    predictor = SavedModelPredictor.from_state_path(
        state_path,
        feature_options=FEATURE_OPTIONS,
        prediction_options={"device": "cpu", "stride": 16, "batch_size": 8},
    )
    result = predictor.predict(_unit_image())

    assert set(result.arrays) >= {
        "features",
        "channel_names",
        "coordinates_xy",
        "predictions",
        "confidence",
        "entropy",
        "prediction_grid",
        "confidence_grid",
        "entropy_grid",
    }
    assert result.arrays["features"].shape == (8, 64, 64)
    assert result.arrays["prediction_grid"].shape == result.arrays["confidence_grid"].shape
    assert not set(result.arrays) & {"support_patches", "support_labels"}
    assert result.record["model"]["task_classes"] == 2
    assert result.record["runtime"]["resolved_device"] == "cpu"


def test_saved_feature_options_are_honored_and_reported(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    state_path = _saved_state(tmp_path / "model_state.pt")

    predictor = SavedModelPredictor.from_state_path(
        state_path,
        feature_options=FEATURE_OPTIONS,
        prediction_options={"device": "cpu"},
    )
    result = predictor.predict(_unit_image())

    reported = result.record["options"]["features"]
    assert reported["symmetry_patch_size"] == 5
    assert reported["n_max"] == 2
    assert reported["rotation_folds"] == [2, 3, 4, 6]


def test_prediction_controls_are_the_only_runtime_overrides(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    state_path = _saved_state(tmp_path / "model_state.pt")

    predictor = SavedModelPredictor.from_state_path(
        state_path,
        feature_options=FEATURE_OPTIONS,
        prediction_options={"device": "cpu", "stride": 32, "batch_size": 4},
    )

    assert predictor.prediction_options["stride"] == 32
    assert predictor.prediction_options["batch_size"] == 4
    with pytest.raises(ValueError, match="Unsupported prediction options"):
        SavedModelPredictor.from_state_path(
            state_path,
            feature_options=FEATURE_OPTIONS,
            prediction_options={"device": "cpu", "epochs": 5},
        )
    with pytest.raises(ValueError, match="Unsupported saved feature options"):
        SavedModelPredictor.from_state_path(
            state_path,
            feature_options={**FEATURE_OPTIONS, "epochs": 5},
            prediction_options={"device": "cpu"},
        )


def test_prediction_reports_progress_phases(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    state_path = _saved_state(tmp_path / "model_state.pt")
    seen: list[str] = []

    predictor = SavedModelPredictor.from_state_path(
        state_path,
        feature_options=FEATURE_OPTIONS,
        prediction_options={"device": "cpu", "stride": 32, "batch_size": 16},
    )
    predictor.predict(
        _unit_image(),
        progress_callback=lambda phase, current, total: seen.append(phase),
    )

    assert "features" in seen
    assert "prediction" in seen


def test_expected_model_mismatches_block_prediction(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    state_path = _saved_state(tmp_path / "model_state.pt", task_classes=3)

    with pytest.raises(ValueError, match="class count"):
        SavedModelPredictor.from_state_path(
            state_path,
            feature_options=FEATURE_OPTIONS,
            prediction_options={"device": "cpu"},
            expected_model={"task_classes": 2},
        )
    with pytest.raises(ValueError, match="model identifier"):
        SavedModelPredictor.from_state_path(
            state_path,
            feature_options=FEATURE_OPTIONS,
            prediction_options={"device": "cpu"},
            expected_model={"identifier": "other_model"},
        )
    with pytest.raises(ValueError, match="adapter bottleneck"):
        SavedModelPredictor.from_state_path(
            state_path,
            feature_options=FEATURE_OPTIONS,
            prediction_options={"device": "cpu"},
            expected_model={"adapter_bottleneck": 8},
        )
    with pytest.raises(ValueError, match="class names"):
        SavedModelPredictor.from_state_path(
            state_path,
            feature_options=FEATURE_OPTIONS,
            prediction_options={"device": "cpu"},
            expected_model={"class_names": ["Other A", "Other B", "Other C"]},
        )
    with pytest.raises(ValueError, match="base checkpoint"):
        SavedModelPredictor.from_state_path(
            state_path,
            feature_options=FEATURE_OPTIONS,
            prediction_options={"device": "cpu"},
            expected_model={"base_checkpoint_sha256": "0" * 64},
        )


def test_model_state_checksum_mismatch_blocks_restore(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    state_path = _saved_state(tmp_path / "model_state.pt")
    expected_checksum = sha256(state_path.read_bytes()).hexdigest()

    predictor = SavedModelPredictor.from_state_path(
        state_path,
        feature_options=FEATURE_OPTIONS,
        prediction_options={"device": "cpu"},
        expected_model={"model_state_sha256": expected_checksum},
    )
    assert predictor.identifier == "cnn_8ch_pg17"

    with pytest.raises(ValueError, match="checksum"):
        SavedModelPredictor.from_state_path(
            state_path,
            feature_options=FEATURE_OPTIONS,
            prediction_options={"device": "cpu"},
            expected_model={"model_state_sha256": "0" * 64},
        )


def _write_prediction_job(
    path: Path,
    *,
    state_path: Path,
    directory: Path,
    inputs: list[Path],
) -> Path:
    items = []
    for index, image in enumerate(inputs, start=1):
        item_id = f"image-{index:04d}"
        items.append(
            {
                "item_id": item_id,
                "input_path": str(image),
                "output_path": str(directory / item_id / "prediction.npz"),
                "record_path": str(directory / item_id / "provider_record.json"),
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": "scientific-symmetry-worker-v1",
                "model_state_path": str(state_path),
                "expected_model": {
                    "identifier": "cnn_8ch_pg17",
                    "task_classes": 2,
                    "adapter_bottleneck": 4,
                },
                "feature_options": FEATURE_OPTIONS,
                "prediction_options": {
                    "device": "cpu",
                    "stride": 32,
                    "batch_size": 16,
                },
                "items": items,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_prediction_worker_loads_the_model_once_for_a_batch(
    tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("torch")
    import symmlearn.provider.worker as worker_module

    state_path = _saved_state(tmp_path / "model_state.pt")
    directory = tmp_path / "batch"
    directory.mkdir()
    inputs = []
    for index in range(2):
        image_path = directory / f"input{index}.npy"
        np.save(image_path, _unit_image())
        inputs.append(image_path)
    job_path = _write_prediction_job(
        tmp_path / "job.json",
        state_path=state_path,
        directory=directory,
        inputs=inputs,
    )

    calls: list[Path] = []
    original = SavedModelPredictor.from_state_path.__func__

    def counting_from_state_path(cls, model_state_path, **kwargs):
        calls.append(Path(model_state_path))
        return original(cls, model_state_path, **kwargs)

    monkeypatch.setattr(
        worker_module.SavedModelPredictor,
        "from_state_path",
        classmethod(counting_from_state_path),
    )

    summary = run_prediction_worker(job_path)

    assert len(calls) == 1
    assert summary["item_count"] == 2
    assert summary["completed_count"] == 2
    assert summary["failed_count"] == 0
    for index in range(1, 3):
        item_directory = directory / f"image-{index:04d}"
        assert (item_directory / "prediction.npz").is_file()
        assert (item_directory / "provider_record.json").is_file()


def test_prediction_worker_records_one_image_failure_and_continues(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    state_path = _saved_state(tmp_path / "model_state.pt")
    directory = tmp_path / "batch"
    directory.mkdir()
    good = directory / "good.npy"
    np.save(good, _unit_image())
    too_small = directory / "small.npy"
    np.save(too_small, np.zeros((32, 32), dtype=np.float32))
    job_path = _write_prediction_job(
        tmp_path / "job.json",
        state_path=state_path,
        directory=directory,
        inputs=[too_small, good],
    )

    summary = run_prediction_worker(job_path)

    assert summary["completed_count"] == 1
    assert summary["failed_count"] == 1
    failed = next(entry for entry in summary["items"] if entry["status"] == "failed")
    assert failed["item_id"] == "image-0001"
    assert failed["error"]
    assert (directory / "image-0002" / "prediction.npz").is_file()


def test_prediction_worker_stops_the_batch_for_a_model_failure(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    directory = tmp_path / "batch"
    directory.mkdir()
    image_path = directory / "input.npy"
    np.save(image_path, _unit_image())
    corrupt_state = directory / "model_state.pt"
    corrupt_state.write_bytes(b"not a torch archive")
    job_path = _write_prediction_job(
        tmp_path / "job.json",
        state_path=corrupt_state,
        directory=directory,
        inputs=[image_path],
    )

    with pytest.raises(Exception):
        run_prediction_worker(job_path)

    assert not (directory / "image-0001" / "prediction.npz").exists()


def test_prediction_worker_rejects_incomplete_items(tmp_path: Path) -> None:
    state_path = tmp_path / "model_state.pt"
    job_path = tmp_path / "job.json"
    job_path.write_text(
        json.dumps(
            {
                "schema_version": "scientific-symmetry-worker-v1",
                "model_state_path": str(state_path),
                "items": [{"item_id": "image-0001"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing"):
        run_prediction_worker(job_path)

    _load_job(job_path)
