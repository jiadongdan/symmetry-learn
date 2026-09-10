from __future__ import annotations

from pathlib import Path

import pytest

from symmlearn.finetuning.artifacts import (
    FINE_TUNED_MODEL_STATE_SCHEMA_VERSION,
    FineTunedModelStateError,
    load_fine_tuned_model_state,
    save_fine_tuned_model_state,
    validate_fine_tuned_model_state,
)


def _state(**overrides) -> dict:
    torch = pytest.importorskip("torch")
    payload = {
        "schema_version": FINE_TUNED_MODEL_STATE_SCHEMA_VERSION,
        "model_identifier": "cnn_8ch_pg17",
        "adapter_bottleneck": 4,
        "task_classes": 2,
        "class_names": ["Phase A", "Phase B"],
        "base_checkpoint_sha256": "a" * 64,
        "state_dict": {
            "classifier.weight": torch.ones((2, 3)),
            "classifier.bias": torch.zeros(2),
        },
    }
    payload.update(overrides)
    return payload


def test_valid_state_passes_complete_structural_validation() -> None:
    pytest.importorskip("torch")

    validated = validate_fine_tuned_model_state(_state())

    assert validated["model_identifier"] == "cnn_8ch_pg17"
    assert validated["task_classes"] == 2


def test_state_validation_rejects_unsupported_schema() -> None:
    pytest.importorskip("torch")

    with pytest.raises(FineTunedModelStateError, match="schema"):
        validate_fine_tuned_model_state(_state(schema_version="other-v1"))


def test_state_validation_rejects_a_serialized_module() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(FineTunedModelStateError, match="nn.Module"):
        validate_fine_tuned_model_state(torch.nn.Linear(2, 2))


def test_state_validation_rejects_a_nonmapping_state_dict() -> None:
    pytest.importorskip("torch")

    with pytest.raises(FineTunedModelStateError, match="state_dict"):
        validate_fine_tuned_model_state(_state(state_dict=["not", "a", "mapping"]))


def test_state_validation_rejects_non_tensor_values() -> None:
    pytest.importorskip("torch")

    with pytest.raises(FineTunedModelStateError, match="not a tensor"):
        validate_fine_tuned_model_state(
            _state(state_dict={"classifier.weight": [[0.0, 1.0]]})
        )


def test_state_validation_rejects_nonfinite_tensors() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(FineTunedModelStateError, match="nonfinite"):
        validate_fine_tuned_model_state(
            _state(state_dict={"classifier.weight": torch.full((2, 3), float("nan"))})
        )


def test_state_validation_rejects_class_and_bottleneck_mismatches() -> None:
    pytest.importorskip("torch")

    with pytest.raises(FineTunedModelStateError, match="task_classes"):
        validate_fine_tuned_model_state(_state(class_names=["Only one"]))

    with pytest.raises(FineTunedModelStateError, match="adapter bottleneck"):
        validate_fine_tuned_model_state(_state(adapter_bottleneck=0))

    with pytest.raises(FineTunedModelStateError, match="at least two"):
        validate_fine_tuned_model_state(
            _state(task_classes=1, class_names=["Only one"])
        )


def test_loading_a_complete_state_round_trips_and_validates(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "model_state.pt"
    save_fine_tuned_model_state(path, _state())

    restored = load_fine_tuned_model_state(path)

    assert restored["model_identifier"] == "cnn_8ch_pg17"
    assert torch.equal(restored["state_dict"]["classifier.weight"], torch.ones((2, 3)))
    assert load_fine_tuned_model_state(path)["class_names"] == ["Phase A", "Phase B"]


def test_loading_rejects_invalid_and_absent_states(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    invalid = tmp_path / "invalid.pt"
    save_fine_tuned_model_state(invalid, _state(schema_version="other-v1"))

    with pytest.raises(FineTunedModelStateError, match="schema"):
        load_fine_tuned_model_state(invalid)

    with pytest.raises(FileNotFoundError):
        load_fine_tuned_model_state(tmp_path / "absent.pt")
