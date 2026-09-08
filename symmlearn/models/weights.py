"""Resolve and strictly validate pretrained model weights."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from importlib import resources
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Iterator

from symmlearn.exceptions import WeightIntegrityError, WeightNotInstalledError

from .specifications import WeightSpecification


@dataclass(frozen=True)
class ResolvedWeight:
    """A local checkpoint selected for one model invocation."""

    identifier: str
    path: Path
    sha256: str
    source: str
    bundled: bool


def file_sha256(path: str | Path) -> str:
    """Calculate a file checksum without reading the complete file at once."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_available(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def inspect_weight(specification: WeightSpecification) -> dict[str, Any]:
    """Report whether a packaged weight appears to be locally installed."""
    result: dict[str, Any] = {
        "identifier": specification.identifier,
        "version": specification.version,
        "sha256": specification.sha256,
        "size_bytes": specification.size_bytes,
        "distribution": specification.distribution,
        "default": specification.default,
        "bundled": specification.bundled,
        "status": "not_installed",
    }
    if not _package_available(specification.package):
        return result
    try:
        resource = resources.files(specification.package).joinpath(
            specification.resource
        )
        if not resource.is_file():
            return result
        with resources.as_file(resource) as materialized:
            size = materialized.stat().st_size
            result["installed_path"] = str(materialized.resolve())
            result["installed_size_bytes"] = size
            result["status"] = (
                "installed" if size == specification.size_bytes else "corrupted"
            )
    except (FileNotFoundError, ModuleNotFoundError):
        return result
    return result


def _select_weight(
    model_identifier: str,
    weight_identifier: str | None,
) -> WeightSpecification:
    from .registry import get_model_specification

    model = get_model_specification(model_identifier)
    if weight_identifier is None:
        return model.default_weight
    for candidate in model.weights:
        if candidate.identifier == weight_identifier:
            return candidate
    available = ", ".join(weight.identifier for weight in model.weights)
    raise ValueError(
        f"Unsupported weight {weight_identifier!r} for model "
        f"{model_identifier!r}; available weights: {available}"
    )


@contextmanager
def resolve_model_weight(
    model_identifier: str,
    *,
    weight_identifier: str | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_sha256: str | None = None,
) -> Iterator[ResolvedWeight]:
    """Resolve an explicit checkpoint or the installed default model asset."""
    if checkpoint_path is not None:
        source = Path(checkpoint_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {source}")
        if checkpoint_sha256 is None:
            raise ValueError("An explicit checkpoint requires checkpoint_sha256.")
        yield ResolvedWeight(
            identifier="custom",
            path=source,
            sha256=str(checkpoint_sha256).lower(),
            source="explicit",
            bundled=False,
        )
        return

    specification = _select_weight(model_identifier, weight_identifier)
    if not _package_available(specification.package):
        raise WeightNotInstalledError(
            f"Weight {specification.identifier!r} is not installed. Install "
            f"{specification.distribution!r} or provide an explicit checkpoint."
        )
    resource = resources.files(specification.package).joinpath(specification.resource)
    if not resource.is_file():
        raise WeightNotInstalledError(
            f"Installed distribution {specification.distribution!r} does not "
            f"contain weight {specification.identifier!r}."
        )
    with resources.as_file(resource) as materialized:
        if materialized.stat().st_size != specification.size_bytes:
            raise WeightIntegrityError(
                f"Weight {specification.identifier!r} has an unexpected file size."
            )
        yield ResolvedWeight(
            identifier=specification.identifier,
            path=materialized.resolve(),
            sha256=specification.sha256,
            source="bundled" if specification.bundled else "installed",
            bundled=specification.bundled,
        )


def load_pretrained_checkpoint(
    model: Any,
    checkpoint_path: str | Path,
    expected_sha256: str,
) -> dict[str, Any]:
    """Strictly load a checkpoint after checksum verification."""
    import torch

    source = Path(checkpoint_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {source}")
    actual_sha256 = file_sha256(source)
    if actual_sha256 != str(expected_sha256).lower():
        raise WeightIntegrityError("Pretrained checkpoint SHA-256 mismatch.")
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
