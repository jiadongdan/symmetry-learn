"""Restore the fine-tuned form of the eight-channel PG17 model."""

from __future__ import annotations

from typing import Any

from symmlearn.finetuning.adapters import add_task_adapters


def restore_fine_tuned(
    model_state: dict[str, Any],
    *,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    """Rebuild this architecture with adapters and load its state strictly."""
    from .architecture import build_model

    task_classes = int(model_state["task_classes"])
    bottleneck = int(model_state["adapter_bottleneck"])
    model = build_model(num_classes=17)
    model = add_task_adapters(
        model, bottleneck=bottleneck, task_classes=task_classes
    )
    incompatible = model.load_state_dict(model_state["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "The saved model state does not match the registered architecture. "
            f"Missing keys: {incompatible.missing_keys}. "
            f"Unexpected keys: {incompatible.unexpected_keys}."
        )
    model.to(device)
    # Prediction only reads the restored model, so every parameter is frozen.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    record = {
        "class_names": list(model_state["class_names"]),
        "task_classes": task_classes,
        "adapter_bottleneck": bottleneck,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "classifier_shape": list(model.classifier.weight.shape),
    }
    return model, record
