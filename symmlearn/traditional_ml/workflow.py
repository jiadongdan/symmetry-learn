"""Traditional ML training and batched dense sklearn prediction.

The workflow extracts support patches from the validated eight-channel feature
representation, fits one conventional scikit-learn estimator, and then performs
dense full-image prediction with ``predict_proba``. It never loads, probes, or
requires a pretrained neural-network checkpoint, and it does not serialize the
fitted estimator.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import platform
from time import perf_counter
from typing import Any
import warnings

import numpy as np

from symmlearn import __version__
from symmlearn.features.eight_channel.contract import CHANNEL_NAMES
from symmlearn.patches import (
    dense_coordinate_grid,
    extract_patches,
    iter_patch_batches,
    validate_support,
)

from .contracts import (
    IMAGE_PLUS_SYMMETRY_MAPS_MODE,
    RAW_IMAGE_MODE,
    TRADITIONAL_ML_OPERATION,
    TRADITIONAL_ML_RECORD_SCHEMA_VERSION,
    TraditionalMLError,
    TraditionalMLOptions,
    classifier_specification,
    validate_feature_mode,
)
from .estimators import aligned_predict_proba, build_estimator
from .results import TraditionalMLResult


ProgressCallback = Callable[[str, int, int], None]
CHANNEL_COUNT = len(CHANNEL_NAMES)


@dataclass(frozen=True)
class DenseTraditionalPrediction:
    """Dense traditional-ML outputs on the valid image grid."""

    coordinates_xy: np.ndarray
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    confidence: np.ndarray
    entropy: np.ndarray


def _runtime_versions() -> dict[str, Any]:
    """Return the numerical provenance required by the run contract."""
    from importlib.metadata import PackageNotFoundError, version

    def _distribution(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "symmetry_learn": __version__,
        "numpy": np.__version__,
        "scipy": _distribution("scipy"),
        "scikit_learn": _distribution("scikit-learn"),
    }


def _array_sha256(values: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def validate_feature_tensor(features: np.ndarray) -> np.ndarray:
    """Validate the eight-channel feature tensor used by this workflow."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("Traditional ML features must be a channel-first array.")
    if values.shape[0] != CHANNEL_COUNT:
        raise ValueError(
            f"Traditional ML features must have exactly {CHANNEL_COUNT} channels."
        )
    if min(values.shape[1], values.shape[2]) <= 0:
        raise ValueError("Traditional ML features must have positive dimensions.")
    if not np.isfinite(values).all():
        raise ValueError("Traditional ML features must contain only finite values.")
    return values


def select_feature_channels(features: np.ndarray, feature_mode: str) -> np.ndarray:
    """Select the exact channels required by one feature mode.

    ``raw_image`` is contractually channel 0 of the same normalized
    representation. ``image_plus_symmetry_maps`` is all eight channels in their
    existing order. No other normalization, averaging, or downsampling exists.
    """
    mode = validate_feature_mode(feature_mode)
    values = validate_feature_tensor(features)
    if mode == RAW_IMAGE_MODE:
        return np.ascontiguousarray(values[0:1, :, :])
    return np.ascontiguousarray(values[:, :, :])


def channel_names_for_mode(feature_mode: str) -> list[str]:
    """Return the channel names used by one feature mode, in order."""
    mode = validate_feature_mode(feature_mode)
    if mode == RAW_IMAGE_MODE:
        return [CHANNEL_NAMES[0]]
    return list(CHANNEL_NAMES)


def flatten_patches(patches: np.ndarray) -> np.ndarray:
    """Flatten channel-first patches into a stable NumPy C-order matrix."""
    values = np.asarray(patches, dtype=np.float32)
    if values.ndim != 4:
        raise ValueError("Patch batches must be a four-dimensional array.")
    if values.shape[0] == 0:
        raise ValueError("Patch batches cannot be empty.")
    if not np.isfinite(values).all():
        raise ValueError("Patch batches must contain only finite values.")
    return values.reshape(values.shape[0], -1)


def validate_features_and_support(
    features: np.ndarray,
    *,
    coordinates_xy: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    options: TraditionalMLOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return validated features, coordinates, contiguous labels, and names."""
    values = validate_feature_tensor(features)
    image_shape = (int(values.shape[1]), int(values.shape[2]))
    coordinates, resolved_labels, names = validate_support(
        image_shape,
        coordinates_xy,
        labels,
        list(class_names),
        options.support_options(),
    )
    return values, coordinates, resolved_labels, names


def _fit_estimator(
    identifier: str,
    parameters: Mapping[str, Any] | None,
    *,
    seed: int,
    matrix: np.ndarray,
    labels: np.ndarray,
):
    """Fit one estimator, converting fit failure into an actionable error."""
    from sklearn.exceptions import ConvergenceWarning

    estimator = build_estimator(identifier, parameters, seed=seed)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            estimator.fit(matrix, labels)
        except Exception as error:  # noqa: BLE001 - re-raised as a typed error
            raise TraditionalMLError(
                f"{identifier} failed to fit: {type(error).__name__}: {error}"
            ) from error
    convergence = [
        str(item.message)
        for item in captured
        if issubclass(item.category, ConvergenceWarning)
    ]
    if convergence:
        raise TraditionalMLError(
            f"{identifier} did not converge. Raise max_iter or simplify the "
            f"requested parameters. Reported: {'; '.join(convergence)}"
        )
    return estimator


def _dense_traditional_prediction(
    estimator: Any,
    selected: np.ndarray,
    *,
    patch_size: int,
    stride: int,
    batch_size: int,
    class_count: int,
    progress_callback: ProgressCallback | None,
) -> DenseTraditionalPrediction:
    """Predict probabilities and derived vectors on the dense image grid."""
    image_shape = (int(selected.shape[1]), int(selected.shape[2]))
    coordinates, x_values, y_values = dense_coordinate_grid(
        image_shape, patch_size, stride
    )
    total_batches = int(np.ceil(len(coordinates) / batch_size))
    if progress_callback is not None:
        progress_callback("prediction", 0, total_batches)
    batches: list[np.ndarray] = []
    for index, patches in enumerate(
        iter_patch_batches(
            selected,
            coordinates,
            patch_size=patch_size,
            batch_size=batch_size,
            expected_channels=int(selected.shape[0]),
        )
    ):
        batches.append(
            aligned_predict_proba(
                estimator,
                flatten_patches(patches),
                class_count=class_count,
            )
        )
        if progress_callback is not None:
            progress_callback("prediction", index + 1, total_batches)
    if not batches:
        raise TraditionalMLError("Dense prediction produced no probability batches.")
    probabilities = np.concatenate(batches).astype(np.float32, copy=False)
    predictions = probabilities.argmax(axis=1).astype(np.int16)
    confidence = probabilities.max(axis=1).astype(np.float32)
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1
    ).astype(np.float32)
    return DenseTraditionalPrediction(
        coordinates_xy=coordinates.astype(np.int32, copy=False),
        x_coordinates=x_values.astype(np.int32, copy=False),
        y_coordinates=y_values.astype(np.int32, copy=False),
        probabilities=probabilities,
        predictions=predictions,
        confidence=confidence,
        entropy=entropy,
    )


def analyze_traditional_ml(
    features: np.ndarray,
    *,
    coordinates_xy: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    classifier: str,
    parameters: Mapping[str, Any] | None = None,
    feature_mode: str = IMAGE_PLUS_SYMMETRY_MAPS_MODE,
    options: TraditionalMLOptions | None = None,
    feature_record: Mapping[str, Any] | None = None,
    input_shape: tuple[int, int] | None = None,
    input_sha256: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TraditionalMLResult:
    """Fit one conventional classifier and densely predict the whole image."""
    resolved_options = options if options is not None else TraditionalMLOptions()
    if not isinstance(resolved_options, TraditionalMLOptions):
        raise TypeError("options must be a TraditionalMLOptions instance.")
    specification = classifier_specification(classifier)
    resolved_parameters = specification.resolve(parameters)
    mode = validate_feature_mode(feature_mode)

    values, coordinates, resolved_labels, names = validate_features_and_support(
        features,
        coordinates_xy=coordinates_xy,
        labels=labels,
        class_names=class_names,
        options=resolved_options,
    )
    image_shape = (int(values.shape[1]), int(values.shape[2]))
    if input_shape is not None:
        requested_shape = (int(input_shape[0]), int(input_shape[1]))
        if requested_shape != image_shape:
            raise ValueError(
                "The input image and the feature artifact do not share a shape: "
                f"{requested_shape} != {image_shape}."
            )

    selected = select_feature_channels(values, mode)
    training_matrix = flatten_patches(
        extract_patches(
            selected,
            coordinates,
            resolved_options.classifier_patch_size,
            expected_channels=int(selected.shape[0]),
        )
    )

    if progress_callback is not None:
        progress_callback("training", 0, 1)
    fit_started = perf_counter()
    estimator = _fit_estimator(
        specification.identifier,
        resolved_parameters,
        seed=resolved_options.seed,
        matrix=training_matrix,
        labels=resolved_labels,
    )
    fit_seconds = perf_counter() - fit_started
    if progress_callback is not None:
        progress_callback("training", 1, 1)

    # Diagnostic only: this is measured on the user-selected support patches
    # themselves and is never held-out validation accuracy.
    support_training_accuracy = float(
        np.mean(np.asarray(estimator.predict(training_matrix)) == resolved_labels)
    )

    prediction_started = perf_counter()
    dense = _dense_traditional_prediction(
        estimator,
        selected,
        patch_size=resolved_options.classifier_patch_size,
        stride=resolved_options.stride,
        batch_size=resolved_options.batch_size,
        class_count=len(names),
        progress_callback=progress_callback,
    )
    prediction_seconds = perf_counter() - prediction_started

    grid_shape = (int(len(dense.y_coordinates)), int(len(dense.x_coordinates)))
    arrays = {
        "coordinates_xy": dense.coordinates_xy,
        "x_coordinates": dense.x_coordinates,
        "y_coordinates": dense.y_coordinates,
        "probabilities": dense.probabilities,
        "predictions": dense.predictions,
        "confidence": dense.confidence,
        "entropy": dense.entropy,
        "prediction_grid": dense.predictions.reshape(grid_shape),
        "confidence_grid": dense.confidence.reshape(grid_shape),
        "entropy_grid": dense.entropy.reshape(grid_shape),
    }

    support_counts = {
        name: int(count)
        for name, count in zip(
            names, np.bincount(resolved_labels, minlength=len(names)).tolist()
        )
    }
    record: dict[str, Any] = {
        "schema_version": TRADITIONAL_ML_RECORD_SCHEMA_VERSION,
        "operation": TRADITIONAL_ML_OPERATION,
        "feature_mode": mode,
        "channel_count": int(selected.shape[0]),
        "channel_names": channel_names_for_mode(mode),
        "classifier": {
            "identifier": specification.identifier,
            "parameters": {
                name: resolved_parameters[name] for name in specification.names
            },
            "supports_predict_proba": True,
        },
        "class_names": list(names),
        "class_count": len(names),
        "support_counts": support_counts,
        "support": {
            "coordinates_xy": coordinates.tolist(),
            "labels": resolved_labels.tolist(),
            "sha256": _array_sha256(
                np.column_stack((coordinates, resolved_labels)).astype(np.int64)
            ),
        },
        "seed": int(resolved_options.seed),
        "options": resolved_options.to_dict(),
        "classifier_patch_size": int(resolved_options.classifier_patch_size),
        "stride": int(resolved_options.stride),
        "batch_size": int(resolved_options.batch_size),
        "training_matrix_shape": [int(value) for value in training_matrix.shape],
        "input_shape": [int(value) for value in image_shape],
        "feature_shape": [int(value) for value in values.shape],
        "input_sha256": None if input_sha256 is None else str(input_sha256).lower(),
        "feature_sha256": _array_sha256(values),
        "feature_provenance": None if feature_record is None else dict(feature_record),
        "grid_shape": [int(value) for value in grid_shape],
        "sample_count": int(len(dense.coordinates_xy)),
        "timings_seconds": {
            "fit": fit_seconds,
            "prediction": prediction_seconds,
        },
        "support_training_accuracy": support_training_accuracy,
        "support_training_accuracy_is_validation": False,
        "runtime": _runtime_versions(),
    }
    return TraditionalMLResult(arrays=arrays, record=record)


__all__ = [
    "DenseTraditionalPrediction",
    "analyze_traditional_ml",
    "channel_names_for_mode",
    "flatten_patches",
    "select_feature_channels",
    "validate_feature_tensor",
    "validate_features_and_support",
]
