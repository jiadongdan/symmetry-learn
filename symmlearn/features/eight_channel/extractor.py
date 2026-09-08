"""Compute the eight-channel symmetry representation."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from .contract import CHANNEL_NAMES
from .validation import validate_feature_array, validate_unit_image


def compute_features(
    image: np.ndarray,
    options: dict[str, Any],
    *,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute image, reflection, orientation, and rotation channels."""
    from symmlearn.maps import get_ref_map, get_rot_maps
    import torch

    values = validate_unit_image(image)
    started = perf_counter()
    rotation_maps = get_rot_maps(
        values,
        patch_size=options["symmetry_patch_size"],
        n_max=options["n_max"],
        n_folds=options["rotation_folds"],
        device=torch.device(device),
        normalize_output=options["normalize_rotation_maps"],
    )
    reflection_map, theta_map = get_ref_map(
        values,
        patch_size=options["symmetry_patch_size"],
        n_max=options["n_max"],
        device=torch.device(device),
        return_angle=True,
        p=options["reflection_p"],
    )
    rotation_maps = np.asarray(rotation_maps, dtype=np.float32)
    reflection_map = np.asarray(reflection_map, dtype=np.float32)
    theta_map = np.asarray(theta_map, dtype=np.float32)
    if rotation_maps.shape != (4, *values.shape):
        raise RuntimeError("Rotational symmetry outputs do not match the v1 contract.")
    if reflection_map.shape != values.shape or theta_map.shape != values.shape:
        raise RuntimeError("Reflection symmetry outputs do not match the source shape.")
    features = np.stack(
        (
            values,
            reflection_map,
            np.sin(theta_map * 2.0),
            np.cos(theta_map * 2.0),
            rotation_maps[0],
            rotation_maps[1],
            rotation_maps[2],
            rotation_maps[3],
        )
    ).astype(np.float32, copy=False)
    validate_feature_array(features, values.shape)
    return features, {
        "channel_names": list(CHANNEL_NAMES),
        "shape": list(features.shape),
        "dtype": "float32",
        "channel_ranges": [
            [float(channel.min()), float(channel.max())] for channel in features
        ],
        "n_max": options["n_max"],
        "symmetry_patch_size": options["symmetry_patch_size"],
        "rotation_folds": options["rotation_folds"],
        "reflection_p": options["reflection_p"],
        "normalize_rotation_maps": options["normalize_rotation_maps"],
        "runtime_seconds": perf_counter() - started,
        "device": device,
    }
