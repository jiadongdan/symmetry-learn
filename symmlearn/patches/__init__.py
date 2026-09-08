"""Patch extraction, dense grids, and support-set validation."""

from .extraction import extract_patches, iter_patch_batches
from .grids import dense_coordinate_grid, valid_center_bounds
from .validation import validate_support

__all__ = [
    "dense_coordinate_grid",
    "extract_patches",
    "iter_patch_batches",
    "valid_center_bounds",
    "validate_support",
]
