"""Eight-channel symmetry representation."""

from .contract import CHANNEL_NAMES, FEATURE_IDENTIFIER, FEATURE_SPECIFICATION
from .extractor import compute_features

__all__ = [
    "CHANNEL_NAMES",
    "FEATURE_IDENTIFIER",
    "FEATURE_SPECIFICATION",
    "compute_features",
]
