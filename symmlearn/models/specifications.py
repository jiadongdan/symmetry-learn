"""Immutable metadata structures for registered models and weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class WeightSpecification:
    """Describe one compatible pretrained weight asset."""

    identifier: str
    version: str
    sha256: str
    size_bytes: int
    distribution: str
    package: str
    resource: str
    default: bool = False
    bundled: bool = False


@dataclass(frozen=True)
class FineTuningSpecification:
    """Describe the supported few-shot strategy for one model."""

    strategy: str
    minimum_shots_per_class: int
    recommended_shots_per_class: int
    maximum_shots_per_class: int


@dataclass(frozen=True)
class ModelSpecification:
    """Describe a model without importing its numerical runtime."""

    identifier: str
    display_name: str
    kind: str
    family: str
    input_channels: int
    pretrained_classes: int
    classifier_patch_size: int
    feature_pipeline: str
    feature_channels: tuple[str, ...]
    fine_tuning: FineTuningSpecification
    default_options: Mapping[str, Any]
    weights: tuple[WeightSpecification, ...]

    @property
    def default_weight(self) -> WeightSpecification:
        """Return the single weight marked as the model default."""
        matches = [weight for weight in self.weights if weight.default]
        if len(matches) != 1:
            raise RuntimeError(
                f"Model {self.identifier!r} must define exactly one default weight."
            )
        return matches[0]
