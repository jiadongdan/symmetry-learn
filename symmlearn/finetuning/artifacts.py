"""Save and load compact task-specific adapter artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


ADAPTER_SCHEMA_VERSION = "symmetry-adapter-head-v1"


def save_adapter_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    """Persist adapters and the local head without duplicating base weights."""
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_adapter_artifact(path: str | Path) -> dict[str, Any]:
    """Load a validated adapter artifact mapping."""
    import torch

    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("The adapter artifact must decode to a mapping.")
    if payload.get("schema_version") != ADAPTER_SCHEMA_VERSION:
        raise ValueError("Unsupported adapter artifact schema version.")
    return payload
