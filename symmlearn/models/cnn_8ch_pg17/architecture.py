"""Checkpoint-compatible architecture for the eight-channel PG17 model."""

from __future__ import annotations

import torch.nn as nn


BASE_CONV_LAYERS = (
    (32, 3, 1, 1, True),
    (64, 3, 1, 1, False),
    (128, 3, 1, 1, True),
)
BASE_FC_LAYERS = (512, 256)


class CheckpointCompatibleCNN(nn.Module):
    """CNN whose parameter keys match the published PG17 checkpoint."""

    def __init__(self, num_classes: int = 17) -> None:
        super().__init__()
        self.input_channels = 8
        self.input_h = 64
        self.input_w = 64
        self.num_classes = num_classes
        self.dropout_rate = 0.1
        self.use_gap = False
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_blocks = nn.ModuleList()
        self.pool_flags = []
        in_channels = self.input_channels
        for out_channels, kernel_size, stride, padding, use_pool in BASE_CONV_LAYERS:
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                )
            )
            self.pool_flags.append(use_pool)
            in_channels = out_channels
        self.gap = None
        self.flat_features = 128 * 16 * 16
        self.fc_blocks = nn.ModuleList()
        in_features = self.flat_features
        for hidden_size in BASE_FC_LAYERS:
            self.fc_blocks.append(
                nn.Sequential(
                    nn.Linear(in_features, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                )
            )
            in_features = hidden_size
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, values):
        """Return logits for a batch of eight-channel 64-pixel patches."""
        for block, use_pool in zip(self.conv_blocks, self.pool_flags):
            values = block(values)
            if use_pool:
                values = self.pool(values)
        values = values.view(values.size(0), -1)
        for block in self.fc_blocks:
            values = block(values)
        return self.classifier(values)


def build_model(num_classes: int = 17) -> CheckpointCompatibleCNN:
    """Construct the registered PG17 architecture."""
    return CheckpointCompatibleCNN(num_classes=num_classes)
