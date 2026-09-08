"""Static contract for the eight-channel PG17 model."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from symmlearn.features.eight_channel.contract import (
    CHANNEL_NAMES,
    FEATURE_IDENTIFIER,
)
from symmlearn.models.specifications import (
    FineTuningSpecification,
    ModelSpecification,
    WeightSpecification,
)


MODEL_IDENTIFIER = "cnn_8ch_pg17"
DEFAULT_WEIGHT_IDENTIFIER = "pg17-symmetry-v1"
DEFAULT_WEIGHT_SHA256 = (
    "5a151e7963b9065eba0baebdf4aa259fe0b9d5514af9a1a62589d7ce22e892ea"
)

DEFAULT_OPTIONS = {
    "n_max": 20,
    "symmetry_patch_size": 51,
    "rotation_folds": [2, 3, 4, 6],
    "reflection_p": 2.0,
    "normalize_rotation_maps": False,
    "classifier_patch_size": 64,
    "minimum_shots_per_class": 3,
    "maximum_shots_per_class": 50,
    "adapter_bottleneck": 16,
    "epochs": 150,
    "learning_rate": 0.0005,
    "weight_decay": 0.0,
    "seed": 42,
    "stride": 4,
    "batch_size": 512,
    "device": "auto",
}

MODEL_SPECIFICATION = ModelSpecification(
    identifier=MODEL_IDENTIFIER,
    display_name="Eight-channel CNN (PG17)",
    kind="few_shot_symmetry",
    family="deep_learning",
    input_channels=8,
    pretrained_classes=17,
    classifier_patch_size=64,
    feature_pipeline=FEATURE_IDENTIFIER,
    feature_channels=CHANNEL_NAMES,
    fine_tuning=FineTuningSpecification(
        strategy=(
            "frozen pretrained network with residual adapters and a new local head"
        ),
        minimum_shots_per_class=3,
        recommended_shots_per_class=5,
        maximum_shots_per_class=50,
    ),
    default_options=DEFAULT_OPTIONS,
    weights=(
        WeightSpecification(
            identifier=DEFAULT_WEIGHT_IDENTIFIER,
            version="1.0.0",
            sha256=DEFAULT_WEIGHT_SHA256,
            size_bytes=68045532,
            distribution="symmetry-learn-default-model",
            package="symmlearn_default_model",
            resource=f"weights/{MODEL_IDENTIFIER}/{DEFAULT_WEIGHT_IDENTIFIER}.pth",
            default=True,
            bundled=True,
        ),
    ),
)


def validate_options(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate the complete feature, fine-tuning, and inference contract."""
    unknown = sorted(set(overrides or {}).difference(DEFAULT_OPTIONS))
    if unknown:
        raise ValueError(f"Unsupported provider options: {unknown}")
    options = deepcopy(DEFAULT_OPTIONS)
    options.update(dict(overrides or {}))
    integer_names = (
        "n_max",
        "symmetry_patch_size",
        "classifier_patch_size",
        "minimum_shots_per_class",
        "maximum_shots_per_class",
        "adapter_bottleneck",
        "epochs",
        "seed",
        "stride",
        "batch_size",
    )
    for name in integer_names:
        options[name] = int(options[name])
    for name in ("reflection_p", "learning_rate", "weight_decay"):
        options[name] = float(options[name])
    options["rotation_folds"] = [int(value) for value in options["rotation_folds"]]
    if options["n_max"] <= 0:
        raise ValueError("n_max must be positive.")
    if options["symmetry_patch_size"] <= 0 or options["symmetry_patch_size"] % 2 == 0:
        raise ValueError("symmetry_patch_size must be a positive odd number.")
    if options["rotation_folds"] != [2, 3, 4, 6]:
        raise ValueError("The v1 channel contract requires rotation_folds [2, 3, 4, 6].")
    if options["reflection_p"] <= 0:
        raise ValueError("reflection_p must be positive.")
    if not isinstance(options["normalize_rotation_maps"], bool):
        raise ValueError("normalize_rotation_maps must be boolean.")
    if options["classifier_patch_size"] != 64:
        raise ValueError("The v1 model contract requires a 64-pixel classifier patch.")
    if options["minimum_shots_per_class"] <= 0:
        raise ValueError("minimum_shots_per_class must be positive.")
    if options["maximum_shots_per_class"] < options["minimum_shots_per_class"]:
        raise ValueError("maximum_shots_per_class must not be smaller than the minimum.")
    for name in ("adapter_bottleneck", "epochs", "stride", "batch_size"):
        if options[name] <= 0:
            raise ValueError(f"{name} must be positive.")
    if options["learning_rate"] <= 0 or options["weight_decay"] < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative.")
    device = str(options["device"])
    if device not in {"auto", "cpu"} and not device.startswith("cuda"):
        raise ValueError("device must be auto, cpu, or a CUDA device.")
    options["device"] = device
    return options
