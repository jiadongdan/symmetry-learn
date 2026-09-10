"""Fine-tuning recipe paired with the eight-channel PG17 architecture."""

from __future__ import annotations

from typing import Any

from symmlearn.finetuning.adapters import add_task_adapters
from symmlearn.finetuning.engine import fit_adapters


def fine_tune(
    model: Any,
    support_patches: Any,
    support_labels: Any,
    *,
    task_classes: int,
    options: dict[str, Any],
    device: str,
    progress_callback=None,
):
    """Insert this model's adapters and optimize its task-specific parameters."""
    adapted = add_task_adapters(
        model,
        bottleneck=options["adapter_bottleneck"],
        task_classes=task_classes,
    )
    result = fit_adapters(
        adapted,
        support_patches,
        support_labels,
        epochs=options["epochs"],
        learning_rate=options["learning_rate"],
        weight_decay=options["weight_decay"],
        seed=options["seed"],
        device=device,
        progress_callback=progress_callback,
    )
    return adapted, result
