"""Metadata structures shared by feature pipelines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpecification:
    """Describe the stable output contract of a feature pipeline."""

    identifier: str
    channel_names: tuple[str, ...]
