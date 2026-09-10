"""Conventional machine-learning validation on the eight-channel representation.

This package is independent of the pretrained CNN. It offers two conventional
scikit-learn classifiers, exact feature-mode selection over the validated
eight-channel representation, and batched dense full-image prediction.

Importing this package does not import scikit-learn: the estimators are built
lazily when a classifier is actually constructed.
"""

from .contracts import (
    CHOICE,
    CLASSIFIER_IDENTIFIERS,
    CLASSIFIER_SPECIFICATIONS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLASSIFIER_PATCH_SIZE,
    DEFAULT_SEED,
    DEFAULT_STRIDE,
    FEATURE_MODES,
    FIXED,
    IMAGE_PLUS_SYMMETRY_MAPS_MODE,
    LOGISTIC_REGRESSION,
    MAXIMUM_PATCHES_PER_CLASS,
    MINIMUM_PATCHES_PER_CLASS,
    OPTIONAL_POSITIVE_INT,
    POSITIVE_FLOAT,
    POSITIVE_INT,
    RANDOM_FOREST,
    RAW_IMAGE_MODE,
    TRADITIONAL_ML_CAPABILITY_SCHEMA_VERSION,
    TRADITIONAL_ML_OPERATION,
    TRADITIONAL_ML_RECORD_SCHEMA_VERSION,
    ClassifierSpecification,
    ParameterSpec,
    TraditionalMLError,
    TraditionalMLOptions,
    classifier_specification,
    traditional_ml_capability,
    validate_feature_mode,
)
from .estimators import (
    aligned_predict_proba,
    build_estimator,
    resolve_classifier_parameters,
)
from .results import TraditionalMLResult
from .workflow import (
    DenseTraditionalPrediction,
    analyze_traditional_ml,
    channel_names_for_mode,
    flatten_patches,
    select_feature_channels,
    validate_feature_tensor,
    validate_features_and_support,
)

__all__ = [
    "CHOICE",
    "CLASSIFIER_IDENTIFIERS",
    "CLASSIFIER_SPECIFICATIONS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CLASSIFIER_PATCH_SIZE",
    "DEFAULT_SEED",
    "DEFAULT_STRIDE",
    "DenseTraditionalPrediction",
    "FEATURE_MODES",
    "FIXED",
    "IMAGE_PLUS_SYMMETRY_MAPS_MODE",
    "LOGISTIC_REGRESSION",
    "MAXIMUM_PATCHES_PER_CLASS",
    "MINIMUM_PATCHES_PER_CLASS",
    "OPTIONAL_POSITIVE_INT",
    "POSITIVE_FLOAT",
    "POSITIVE_INT",
    "RANDOM_FOREST",
    "RAW_IMAGE_MODE",
    "TRADITIONAL_ML_CAPABILITY_SCHEMA_VERSION",
    "TRADITIONAL_ML_OPERATION",
    "TRADITIONAL_ML_RECORD_SCHEMA_VERSION",
    "ClassifierSpecification",
    "ParameterSpec",
    "TraditionalMLError",
    "TraditionalMLOptions",
    "TraditionalMLResult",
    "aligned_predict_proba",
    "analyze_traditional_ml",
    "build_estimator",
    "channel_names_for_mode",
    "classifier_specification",
    "flatten_patches",
    "resolve_classifier_parameters",
    "select_feature_channels",
    "traditional_ml_capability",
    "validate_feature_mode",
    "validate_feature_tensor",
    "validate_features_and_support",
]
