"""Strict scikit-learn estimator construction for the validation workflow.

Only Logistic Regression and Random Forest exist in v1. Unknown classifiers and
unknown parameter keys are errors; arbitrary keyword arguments are never passed
through.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from .contracts import (
    DEFAULT_SEED,
    LOGISTIC_REGRESSION,
    RANDOM_FOREST,
    classifier_specification,
)


def resolve_classifier_parameters(
    identifier: str,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return fully resolved scikit-learn parameters for one classifier."""
    return classifier_specification(identifier).resolve(parameters)


def build_estimator(
    identifier: str,
    parameters: Mapping[str, Any] | None = None,
    *,
    seed: int = DEFAULT_SEED,
):
    """Build one unfitted scikit-learn estimator.

    Logistic Regression always receives a training-only ``StandardScaler``
    inside a :class:`~sklearn.pipeline.Pipeline`. Random Forest always receives
    the configured seed and a single job for strict reproducibility.
    """
    specification = classifier_specification(identifier)
    resolved = specification.resolve(parameters)
    if identifier == LOGISTIC_REGRESSION:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=resolved["C"],
                        class_weight=resolved["class_weight"],
                        max_iter=resolved["max_iter"],
                        solver=resolved["solver"],
                    ),
                ),
            ]
        )
    if identifier == RANDOM_FOREST:
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=resolved["n_estimators"],
            max_depth=resolved["max_depth"],
            min_samples_leaf=resolved["min_samples_leaf"],
            max_features=resolved["max_features"],
            class_weight=resolved["class_weight"],
            criterion=resolved["criterion"],
            bootstrap=resolved["bootstrap"],
            random_state=int(seed),
            n_jobs=resolved["n_jobs"],
        )
    raise ValueError(f"Unsupported classifier {identifier!r}.")


def aligned_predict_proba(
    estimator: Any,
    matrix,
    *,
    class_count: int,
):
    """Return probabilities whose columns align with contiguous labels 0..K-1.

    The estimator class order is never assumed: it is read from ``classes_``,
    verified, and reordered when scikit-learn reports a different order.
    """
    import numpy as np

    probabilities = np.asarray(estimator.predict_proba(matrix), dtype=np.float64)
    classes = np.asarray(getattr(estimator, "classes_", None))
    if classes.ndim != 1 or classes.shape[0] == 0:
        raise ValueError("The estimator did not report fitted classes.")
    if probabilities.ndim != 2 or probabilities.shape[1] != classes.shape[0]:
        raise ValueError("Probability output does not align with the fitted classes.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probability output contains non-finite values.")
    if (probabilities < -1e-9).any():
        raise ValueError("Probability output contains negative values.")

    expected = np.arange(class_count)
    if not np.array_equal(classes, expected):
        aligned = np.zeros((probabilities.shape[0], class_count), dtype=np.float64)
        for column, label in enumerate(classes.tolist()):
            try:
                numeric = float(label)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Probability output reports a non-integer class {label!r}."
                ) from error
            if not math.isfinite(numeric) or numeric != int(numeric):
                raise ValueError(
                    f"Probability output reports a non-integer class {label!r}."
                )
            index = int(numeric)
            if not 0 <= index < class_count:
                raise ValueError(
                    f"Probability output reports an unexpected class {label!r}."
                )
            aligned[:, index] = probabilities[:, column]
        probabilities = aligned

    if class_count > 1:
        row_sums = probabilities.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5, rtol=0.0):
            raise ValueError(
                "Probability rows must sum approximately to one."
            )
    return np.clip(probabilities, 0.0, 1.0).astype(np.float32)


__all__ = [
    "aligned_predict_proba",
    "build_estimator",
    "resolve_classifier_parameters",
]
