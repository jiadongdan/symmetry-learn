"""Checkpoint-compatible model, residual adapters, and few-shot optimization."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


BASE_CONV_LAYERS = (
    (32, 3, 1, 1, True),
    (64, 3, 1, 1, False),
    (128, 3, 1, 1, True),
)
BASE_FC_LAYERS = (512, 256)


def file_sha256(path: str | Path) -> str:
    """Calculate a file checksum without reading the complete file at once."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_device(requested: str) -> str:
    """Resolve auto while preserving errors for an explicitly requested CUDA device."""
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is not available.")
    if requested != "cpu" and not requested.startswith("cuda"):
        raise ValueError("device must be auto, cpu, or a CUDA device.")
    return requested


def set_deterministic_seed(seed: int) -> None:
    """Set deterministic NumPy and PyTorch random state for the supported path."""
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_checkpoint_compatible_model(num_classes: int = 17):
    """Construct the fixed CNN whose state keys match the registered checkpoint."""
    import torch.nn as nn

    class CheckpointCompatibleCNN(nn.Module):
        def __init__(self) -> None:
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
            for block, use_pool in zip(self.conv_blocks, self.pool_flags):
                values = block(values)
                if use_pool:
                    values = self.pool(values)
            values = values.view(values.size(0), -1)
            for block in self.fc_blocks:
                values = block(values)
            return self.classifier(values)

    return CheckpointCompatibleCNN()


def load_pretrained_checkpoint(
    model,
    checkpoint_path: str | Path,
    expected_sha256: str,
) -> dict[str, Any]:
    """Strictly load a user-managed checkpoint after checksum verification."""
    import torch

    source = Path(checkpoint_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {source}")
    actual_sha256 = file_sha256(source)
    if actual_sha256 != str(expected_sha256).lower():
        raise RuntimeError("Pretrained checkpoint SHA-256 mismatch.")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("The pretrained checkpoint must decode to a mapping.")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise TypeError("The pretrained checkpoint has no model_state_dict mapping.")
    cleaned = {str(key).removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(cleaned, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict checkpoint loading returned incompatible keys.")
    return {
        "path": str(source),
        "sha256": actual_sha256,
        "epoch": checkpoint.get("epoch"),
        "train_loss": checkpoint.get("train_loss"),
        "val_loss": checkpoint.get("val_loss"),
        "train_acc": checkpoint.get("train_acc"),
        "val_acc": checkpoint.get("val_acc"),
        "strict_load": True,
    }


def _adapter_types():
    import torch.nn as nn

    class AdapterConv(nn.Module):
        def __init__(self, channels: int, bottleneck: int) -> None:
            super().__init__()
            self.adapter = nn.Sequential(
                nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(bottleneck, channels, kernel_size=1, bias=False),
            )
            nn.init.zeros_(self.adapter[-1].weight)

        def forward(self, values):
            return self.adapter(values)

    class AdapterLinear(nn.Module):
        def __init__(self, features: int, bottleneck: int) -> None:
            super().__init__()
            self.adapter = nn.Sequential(
                nn.Linear(features, bottleneck, bias=False),
                nn.ReLU(),
                nn.Linear(bottleneck, features, bias=False),
            )
            nn.init.zeros_(self.adapter[-1].weight)

        def forward(self, values):
            return self.adapter(values)

    class AdaptedBlock(nn.Module):
        def __init__(self, original, adapter) -> None:
            super().__init__()
            self.original = original
            self.adapter = adapter

        def forward(self, values):
            original = self.original(values)
            return original + self.adapter(original)

    return AdapterConv, AdapterLinear, AdaptedBlock


def add_task_adapters(model, *, bottleneck: int, task_classes: int):
    """Freeze the base model, insert residual adapters, and replace its head."""
    import torch.nn as nn

    if task_classes < 2:
        raise ValueError("Few-shot fine-tuning requires at least two local classes.")
    adapter_conv, adapter_linear, adapted_block = _adapter_types()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    convolutional = nn.ModuleList()
    for block in model.conv_blocks:
        channels = int(block[0].out_channels)
        convolutional.append(
            adapted_block(block, adapter_conv(channels, bottleneck))
        )
    model.conv_blocks = convolutional
    linear = nn.ModuleList()
    for block in model.fc_blocks:
        features = int(block[0].out_features)
        linear.append(adapted_block(block, adapter_linear(features, bottleneck)))
    model.fc_blocks = linear
    model.classifier = nn.Linear(int(model.classifier.in_features), task_classes)
    return model


def _set_adapter_training_mode(model) -> None:
    model.eval()
    for name, module in model.named_modules():
        if name.endswith(".adapter"):
            module.train()
    model.classifier.train()


def train_adapters(
    model,
    support_patches: np.ndarray,
    support_labels: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Fit only residual adapters and the local head on support patches."""
    import torch
    import torch.nn as nn

    patches = np.asarray(support_patches, dtype=np.float32)
    labels = np.asarray(support_labels, dtype=np.int64)
    if patches.ndim != 4 or patches.shape[0] != labels.shape[0]:
        raise ValueError("Support patches and labels have incompatible shapes.")
    if not np.isfinite(patches).all():
        raise ValueError("Support patches contain nonfinite values.")
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
    losses = []
    accuracies = []
    best_loss = float("inf")
    best_state = None
    started = perf_counter()
    for _ in range(epochs):
        _set_adapter_training_mode(model)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Few-shot training produced a nonfinite loss.")
        loss.backward()
        optimizer.step()
        _set_adapter_training_mode(model)
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
    if best_state is None:
        raise RuntimeError("Few-shot training did not produce a model state.")
    named_parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in best_state.items():
            named_parameters[name].copy_(value.to(named_parameters[name].device))
    model.eval()
    return {
        "loss": losses,
        "support_accuracy": accuracies,
        "best_support_loss": best_loss,
        "final_support_loss": losses[-1],
        "final_support_accuracy": accuracies[-1],
        "epochs": epochs,
        "runtime_seconds": perf_counter() - started,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "warning": (
            "Support accuracy and support loss are training diagnostics, not "
            "independent estimates of generalization accuracy."
        ),
    }


def trainable_state_dict(model) -> dict[str, Any]:
    """Return the adapter and local-head tensors without copying the base model."""
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def predict_probabilities(
    model,
    patches: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return logits and probabilities for a finite patch batch."""
    import torch

    values = np.asarray(patches, dtype=np.float32)
    if values.ndim != 4 or not np.isfinite(values).all():
        raise ValueError("Prediction patches must be a finite four-dimensional array.")
    model.to(device).eval()
    logits_batches = []
    probability_batches = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            inputs = torch.from_numpy(values[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            )
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            logits_batches.append(logits.cpu().numpy())
            probability_batches.append(probabilities.cpu().numpy())
    return (
        np.concatenate(logits_batches).astype(np.float32, copy=False),
        np.concatenate(probability_batches).astype(np.float32, copy=False),
    )
