from __future__ import annotations

import pytest

from symmlearn.finetuning.artifacts import FINE_TUNED_MODEL_STATE_SCHEMA_VERSION
from symmlearn.finetuning.adapters import add_task_adapters
from symmlearn.models.registry import (
    build_registered_model,
    restore_fine_tuned_registered_model,
)


def _fine_tuned_state(bottleneck: int = 4, task_classes: int = 2) -> dict:
    torch = pytest.importorskip("torch")
    model = build_registered_model("cnn_8ch_pg17")
    adapted = add_task_adapters(
        model, bottleneck=bottleneck, task_classes=task_classes
    )
    return {
        "schema_version": FINE_TUNED_MODEL_STATE_SCHEMA_VERSION,
        "model_identifier": "cnn_8ch_pg17",
        "adapter_bottleneck": bottleneck,
        "task_classes": task_classes,
        "class_names": [f"Phase {index}" for index in range(task_classes)],
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in adapted.state_dict().items()
        },
    }


def test_restoration_rebuilds_adapters_and_the_task_head() -> None:
    torch = pytest.importorskip("torch")
    state = _fine_tuned_state(bottleneck=4, task_classes=3)

    model, record = restore_fine_tuned_registered_model(
        "cnn_8ch_pg17", state, device="cpu"
    )

    assert record["identifier"] == "cnn_8ch_pg17"
    assert record["task_classes"] == 3
    assert record["adapter_bottleneck"] == 4
    assert record["class_names"] == ["Phase 0", "Phase 1", "Phase 2"]
    assert list(model.classifier.weight.shape) == [3, 256]
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert torch.equal(
        model.state_dict()["classifier.weight"], state["state_dict"]["classifier.weight"]
    )


def test_restoration_rejects_missing_keys() -> None:
    pytest.importorskip("torch")
    state = _fine_tuned_state()
    state["state_dict"].pop("classifier.weight")

    with pytest.raises(RuntimeError):
        restore_fine_tuned_registered_model("cnn_8ch_pg17", state, device="cpu")


def test_restoration_rejects_unexpected_keys() -> None:
    torch = pytest.importorskip("torch")
    state = _fine_tuned_state()
    state["state_dict"]["classifier.unexpected"] = torch.zeros(2)

    with pytest.raises(RuntimeError):
        restore_fine_tuned_registered_model("cnn_8ch_pg17", state, device="cpu")


def test_restoration_rejects_shape_mismatches() -> None:
    torch = pytest.importorskip("torch")
    state = _fine_tuned_state()
    state["state_dict"]["classifier.weight"] = torch.zeros((2, 999))

    with pytest.raises(RuntimeError):
        restore_fine_tuned_registered_model("cnn_8ch_pg17", state, device="cpu")


def test_restoration_rejects_a_mismatched_model_identifier() -> None:
    pytest.importorskip("torch")
    state = _fine_tuned_state()
    state["model_identifier"] = "other_model"

    with pytest.raises(ValueError, match="not the selected model"):
        restore_fine_tuned_registered_model("cnn_8ch_pg17", state, device="cpu")


def test_restoration_rejects_an_unregistered_model() -> None:
    pytest.importorskip("torch")
    state = _fine_tuned_state()

    with pytest.raises(Exception, match="Unsupported model"):
        restore_fine_tuned_registered_model("not_a_model", state, device="cpu")


def test_restored_model_runs_cpu_inference() -> None:
    torch = pytest.importorskip("torch")
    state = _fine_tuned_state(task_classes=2)

    model, _ = restore_fine_tuned_registered_model(
        "cnn_8ch_pg17", state, device="cpu"
    )
    inputs = torch.zeros((2, 8, 64, 64), dtype=torch.float32)

    with torch.inference_mode():
        logits = model(inputs)

    assert logits.shape == (2, 2)
