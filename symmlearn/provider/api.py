"""Direct array API backed by the same code used by provider workers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import MODEL_IDENTIFIER
from .model import resolve_device
from .worker import _compute_features, _execute_analysis, _validated_options


@dataclass(frozen=True)
class ProviderResult:
    """In-memory output from one symmetry-learn few-shot analysis."""

    arrays: dict[str, np.ndarray]
    adapter_checkpoint: dict[str, Any]
    record: dict[str, Any]

    @property
    def prediction_grid(self) -> np.ndarray:
        return self.arrays["prediction_grid"]

    @property
    def confidence_grid(self) -> np.ndarray:
        return self.arrays["confidence_grid"]

    @property
    def entropy_grid(self) -> np.ndarray:
        return self.arrays["entropy_grid"]


def compute_features(
    image: np.ndarray,
    *,
    options: dict[str, Any] | None = None,
    device: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the fixed eight-channel representation from a unit-range image."""
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Input must be a finite two-dimensional array.")
    if float(values.min()) < -1e-6 or float(values.max()) > 1.000001:
        raise ValueError("Input must lie in the shared [0, 1] space.")
    resolved_options = _validated_options(options)
    resolved_options["device"] = device
    resolved_device = resolve_device(device)
    return _compute_features(values, resolved_options, device=resolved_device)


def few_shot_analyze(
    image: np.ndarray,
    *,
    coordinates_xy: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    options: dict[str, Any] | None = None,
    model: str = MODEL_IDENTIFIER,
) -> ProviderResult:
    """Fine-tune adapters and return dense local-class predictions."""
    result = _execute_analysis(
        np.asarray(image, dtype=np.float32),
        {
            "coordinates_xy": np.asarray(coordinates_xy, dtype=np.int32).tolist(),
            "labels": np.asarray(labels, dtype=np.int64).tolist(),
            "class_names": list(class_names),
        },
        {
            "identifier": model,
            "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
            "checkpoint_sha256": str(checkpoint_sha256).lower(),
        },
        options,
    )
    return ProviderResult(
        arrays=result["arrays"],
        adapter_checkpoint=result["adapter_checkpoint"],
        record=result["record"],
    )
