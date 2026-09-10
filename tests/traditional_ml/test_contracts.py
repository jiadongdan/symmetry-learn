from __future__ import annotations

import numpy as np
import pytest

from symmlearn.traditional_ml import (
    CLASSIFIER_IDENTIFIERS,
    FEATURE_MODES,
    IMAGE_PLUS_SYMMETRY_MAPS_MODE,
    LOGISTIC_REGRESSION,
    MAXIMUM_PATCHES_PER_CLASS,
    MINIMUM_PATCHES_PER_CLASS,
    RANDOM_FOREST,
    RAW_IMAGE_MODE,
    TRADITIONAL_ML_CAPABILITY_SCHEMA_VERSION,
    TraditionalMLOptions,
    classifier_specification,
    resolve_classifier_parameters,
    traditional_ml_capability,
    validate_feature_mode,
)


classifier = pytest.mark.parametrize("identifier", CLASSIFIER_IDENTIFIERS)


def test_capability_block_matches_the_protocol_contract() -> None:
    block = traditional_ml_capability()
    assert block["schema_version"] == TRADITIONAL_ML_CAPABILITY_SCHEMA_VERSION
    assert block["feature_modes"] == [RAW_IMAGE_MODE, IMAGE_PLUS_SYMMETRY_MAPS_MODE]
    assert [entry["identifier"] for entry in block["classifiers"]] == [
        LOGISTIC_REGRESSION,
        RANDOM_FOREST,
    ]
    assert all(entry["supports_predict_proba"] is True for entry in block["classifiers"])
    assert block["classifiers"][0]["defaults"] == {
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 1000,
        "solver": "lbfgs",
    }
    assert block["classifiers"][1]["defaults"] == {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "criterion": "gini",
        "bootstrap": True,
        "n_jobs": 1,
    }


def test_protocol_limits_are_the_declared_constants() -> None:
    assert MINIMUM_PATCHES_PER_CLASS == 2
    assert MAXIMUM_PATCHES_PER_CLASS == 5000
    assert TraditionalMLOptions().maximum_patches_per_class == 5000


@classifier
def test_default_parameters_resolve_without_scikit_learn_types(identifier: str) -> None:
    resolved = resolve_classifier_parameters(identifier, None)
    assert resolved == classifier_specification(identifier).defaults()


def test_logistic_regression_maps_none_class_weight() -> None:
    resolved = resolve_classifier_parameters(
        LOGISTIC_REGRESSION, {"class_weight": "none"}
    )
    assert resolved["class_weight"] is None
    assert resolved["class_weight"] == resolve_classifier_parameters(
        LOGISTIC_REGRESSION, {"class_weight": None}
    )["class_weight"]


def test_random_forest_maps_all_and_optional_depth() -> None:
    resolved = resolve_classifier_parameters(
        RANDOM_FOREST,
        {"max_features": "all", "max_depth": None, "class_weight": "none"},
    )
    assert resolved["max_features"] is None
    assert resolved["max_depth"] is None
    assert resolved["class_weight"] is None
    assert resolve_classifier_parameters(
        RANDOM_FOREST, {"max_depth": 8}
    )["max_depth"] == 8
    # scikit-learn spells "all features" as None, so a null value must be
    # accepted and resolved to the same thing as the explicit "all" choice.
    assert resolve_classifier_parameters(
        RANDOM_FOREST, {"max_features": None}
    )["max_features"] is None


def test_unknown_classifier_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported classifier"):
        classifier_specification("support_vector_machine")


@classifier
def test_unknown_parameter_keys_are_rejected(identifier: str) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_classifier_parameters(identifier, {"gamma": 2.0})


@pytest.mark.parametrize(
    "parameters",
    [
        {"C": 0},
        {"C": -1.0},
        {"C": float("nan")},
        {"C": float("inf")},
        {"C": "not-a-number"},
        {"max_iter": 2.5},
        {"max_iter": 0},
        {"class_weight": "unbalanced"},
    ],
)
def test_logistic_regression_rejects_out_of_range_parameters(parameters) -> None:
    with pytest.raises(ValueError):
        resolve_classifier_parameters(LOGISTIC_REGRESSION, parameters)


@pytest.mark.parametrize(
    "parameters",
    [
        {"n_estimators": 0},
        {"min_samples_leaf": 1.5},
        {"max_features": "auto"},
        {"max_depth": -3},
        {"criterion": "entropy"},
        {"bootstrap": False},
        {"n_jobs": 4},
    ],
)
def test_random_forest_rejects_out_of_range_parameters(parameters) -> None:
    with pytest.raises(ValueError):
        resolve_classifier_parameters(RANDOM_FOREST, parameters)


def test_fixed_parameters_are_never_silently_overridden() -> None:
    assert resolve_classifier_parameters(
        LOGISTIC_REGRESSION, {"solver": "lbfgs"}
    )["solver"] == "lbfgs"
    with pytest.raises(ValueError, match="fixed"):
        resolve_classifier_parameters(LOGISTIC_REGRESSION, {"solver": "saga"})


def test_feature_modes_accept_only_the_two_documented_values() -> None:
    assert validate_feature_mode(RAW_IMAGE_MODE) == RAW_IMAGE_MODE
    assert (
        validate_feature_mode(IMAGE_PLUS_SYMMETRY_MAPS_MODE)
        == IMAGE_PLUS_SYMMETRY_MAPS_MODE
    )
    assert FEATURE_MODES == (RAW_IMAGE_MODE, IMAGE_PLUS_SYMMETRY_MAPS_MODE)
    with pytest.raises(ValueError, match="Unsupported feature mode"):
        validate_feature_mode("symmetry_only")


@pytest.mark.parametrize(
    "payload",
    [
        {"classifier_patch_size": 0},
        {"classifier_patch_size": 1.5},
        {"stride": 0},
        {"batch_size": -8},
        {"seed": -1},
        {"seed": 2**32},
        {"seed": 1.5},
        {"minimum_patches_per_class": 0},
        {"minimum_patches_per_class": 10, "maximum_patches_per_class": 5},
    ],
)
def test_options_reject_invalid_values(payload) -> None:
    with pytest.raises(ValueError):
        TraditionalMLOptions(**payload)


def test_options_round_trip_through_a_json_mapping() -> None:
    options = TraditionalMLOptions(
        classifier_patch_size=32, seed=7, stride=2, batch_size=16
    )
    assert TraditionalMLOptions.from_mapping(options.to_dict()) == options
    assert options.support_options() == {
        "classifier_patch_size": 32,
        "minimum_shots_per_class": MINIMUM_PATCHES_PER_CLASS,
        "maximum_shots_per_class": MAXIMUM_PATCHES_PER_CLASS,
    }
    assert TraditionalMLOptions().support_options()["maximum_shots_per_class"] == 5000
    with pytest.raises(ValueError, match="Unsupported traditional ML options"):
        TraditionalMLOptions.from_mapping({"epochs": 5})


def test_importing_traditional_ml_pulls_neither_torch_nor_scikit_learn() -> None:
    import subprocess
    import sys

    program = (
        "import sys; import symmlearn.traditional_ml; "
        "loaded = {name.split('.')[0] for name in sys.modules}; "
        "print('torch' in loaded, 'sklearn' in loaded)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "False False"
