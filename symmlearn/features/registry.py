"""Registry for model input feature pipelines."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .eight_channel.contract import FEATURE_IDENTIFIER, FEATURE_SPECIFICATION


_FEATURES = {
    FEATURE_IDENTIFIER: (
        FEATURE_SPECIFICATION,
        "symmlearn.features.eight_channel.extractor:compute_features",
    ),
}


def _load_extractor(reference: str):
    module_name, attribute = reference.split(":", maxsplit=1)
    extractor = getattr(import_module(module_name), attribute)
    if not callable(extractor):
        raise TypeError(f"Registered target {reference!r} is not callable.")
    return extractor


def feature_capabilities() -> list[dict[str, Any]]:
    """Return stable feature metadata for external inspection."""
    return [
        {
            "identifier": specification.identifier,
            "channel_names": list(specification.channel_names),
        }
        for specification, _ in _FEATURES.values()
    ]


def compute_registered_features(
    identifier: str,
    image: Any,
    options: dict[str, Any],
    *,
    device: str,
):
    """Compute a registered feature pipeline."""
    try:
        _, reference = _FEATURES[identifier]
    except KeyError as error:
        available = ", ".join(_FEATURES)
        raise ValueError(
            f"Unsupported feature pipeline {identifier!r}; "
            f"available pipelines: {available}"
        ) from error
    return _load_extractor(reference)(image, options, device=device)
