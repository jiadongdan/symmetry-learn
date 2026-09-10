"""Generic optimization engine for adapter-based few-shot learning."""

from __future__ import annotations

from time import perf_counter
from collections.abc import Callable
from typing import Any

import numpy as np

from .adapters import set_adapter_training_mode
from .base import validate_training_arrays
from .results import FineTuningResult


def set_deterministic_seed(seed: int) -> None:
    """Set deterministic NumPy and PyTorch random state."""
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def fit_adapters(
    model: Any,
    support_patches: np.ndarray,
    support_labels: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> FineTuningResult:
    """Optimize trainable adapters and the local head on the support set."""
    import torch
    import torch.nn as nn

    patches, labels = validate_training_arrays(support_patches, support_labels)
    set_deterministic_seed(seed)
    torch_device = torch.device(device)
    model.to(torch_device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        trainable, lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    inputs = torch.from_numpy(patches).to(torch_device, dtype=torch.float32)
    targets = torch.from_numpy(labels).to(torch_device, dtype=torch.long)
    losses: list[float] = []
    accuracies: list[float] = []
    best_loss = float("inf")
    best_state = None
    started = perf_counter()
    if progress_callback is not None:
        progress_callback(0, epochs)
    for epoch_index in range(epochs):
        set_adapter_training_mode(model)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Few-shot training produced a nonfinite loss.")
        loss.backward()
        optimizer.step()
        set_adapter_training_mode(model)
        with torch.no_grad():
            updated_logits = model(inputs)
            updated_loss = criterion(updated_logits, targets)
        if not torch.isfinite(updated_loss):
            raise RuntimeError("Few-shot training produced a nonfinite updated loss.")
        loss_value = float(updated_loss.cpu())
        accuracy = float(
            (updated_logits.argmax(dim=1) == targets).float().mean().cpu()
        )
        losses.append(loss_value)
        accuracies.append(accuracy)
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
        if progress_callback is not None:
            progress_callback(epoch_index + 1, epochs)
    if best_state is None:
        raise RuntimeError("Few-shot training did not produce a model state.")
    named_parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in best_state.items():
            named_parameters[name].copy_(value.to(named_parameters[name].device))
    model.eval()
    return FineTuningResult(
        loss=losses,
        support_accuracy=accuracies,
        best_support_loss=best_loss,
        final_support_loss=losses[-1],
        final_support_accuracy=accuracies[-1],
        epochs=epochs,
        runtime_seconds=perf_counter() - started,
        trainable_parameters=sum(parameter.numel() for parameter in trainable),
        warning=(
            "Support accuracy and support loss are training diagnostics, not "
            "independent estimates of generalization accuracy."
        ),
    )
