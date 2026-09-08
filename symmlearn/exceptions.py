"""Shared exceptions for the public symmetry-learn APIs."""


class SymmetryLearnError(Exception):
    """Base class for symmetry-learn errors."""


class ModelNotFoundError(SymmetryLearnError, ValueError):
    """Raised when a requested model is not registered."""


class WeightNotInstalledError(SymmetryLearnError, FileNotFoundError):
    """Raised when a registered weight asset is not installed."""


class WeightIntegrityError(SymmetryLearnError, RuntimeError):
    """Raised when a weight asset fails integrity validation."""
