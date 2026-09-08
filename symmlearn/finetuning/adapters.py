"""Generic residual adapters and task-specific head replacement."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


class AdapterConv(nn.Module):
    """Bottleneck residual adapter for a convolutional feature block."""

    def __init__(self, channels: int, bottleneck: int) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(bottleneck, channels, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.adapter[-1].weight)

    def forward(self, values):
        """Return the learned residual."""
        return self.adapter(values)


class AdapterLinear(nn.Module):
    """Bottleneck residual adapter for a linear feature block."""

    def __init__(self, features: int, bottleneck: int) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(features, bottleneck, bias=False),
            nn.ReLU(),
            nn.Linear(bottleneck, features, bias=False),
        )
        nn.init.zeros_(self.adapter[-1].weight)

    def forward(self, values):
        """Return the learned residual."""
        return self.adapter(values)


class AdaptedBlock(nn.Module):
    """Add a trainable adapter residual to a frozen original block."""

    def __init__(self, original: nn.Module, adapter: nn.Module) -> None:
        super().__init__()
        self.original = original
        self.adapter = adapter

    def forward(self, values):
        """Apply the original block and add its adapter residual."""
        original = self.original(values)
        return original + self.adapter(original)


def add_task_adapters(
    model: Any,
    *,
    bottleneck: int,
    task_classes: int,
):
    """Freeze a compatible base model, insert adapters, and replace its head."""
    if task_classes < 2:
        raise ValueError("Few-shot fine-tuning requires at least two local classes.")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    convolutional = nn.ModuleList()
    for block in model.conv_blocks:
        channels = int(block[0].out_channels)
        convolutional.append(
            AdaptedBlock(block, AdapterConv(channels, bottleneck))
        )
    model.conv_blocks = convolutional
    linear = nn.ModuleList()
    for block in model.fc_blocks:
        features = int(block[0].out_features)
        linear.append(AdaptedBlock(block, AdapterLinear(features, bottleneck)))
    model.fc_blocks = linear
    model.classifier = nn.Linear(int(model.classifier.in_features), task_classes)
    return model


def set_adapter_training_mode(model: Any) -> None:
    """Train only adapters and the task-specific classifier."""
    model.eval()
    for name, module in model.named_modules():
        if name.endswith(".adapter"):
            module.train()
    model.classifier.train()


def trainable_state_dict(model: Any) -> dict[str, Any]:
    """Return adapter and local-head tensors without copying the base model."""
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
