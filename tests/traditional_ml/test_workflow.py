from __future__ import annotations

import json
import math

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from symmlearn.patches import dense_coordinate_grid, extract_patches
from symmlearn.traditional_ml import (
    IMAGE_PLUS_SYMMETRY_MAPS_MODE,
    LOGISTIC_REGRESSION,
    RANDOM_FOREST,
    RAW_IMAGE_MODE,
    TraditionalMLError,
    TraditionalMLOptions,
    aligned_predict_proba,
    analyze_traditional_ml,
    channel_names_for_mode,
    flatten_patches,
    select_feature_channels,
)


IMAGE_SHAPE = (128, 128)
PATCH_SIZE = 64
OPTIONS = TraditionalMLOptions(
    classifier_patch_size=PATCH_SIZE, seed=42, stride=16, batch_size=8
)

CLASS_A = [(40, 40), (44, 44), (48, 48), (52, 52)]
CLASS_B = [(88, 88), (84, 84), (80, 80), (76, 76)]
CLASS_C = [(40, 88), (44, 84), (48, 80), (52, 76)]


def _features(shape=(8, *IMAGE_SHAPE), seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(shape, dtype=np.float32)


def _support(*groups):
    coordinates, labels = [], []
    for label, group in enumerate(groups):
        for point in group:
            coordinates.append(point)
            labels.append(label)
    return (
        np.asarray(coordinates, dtype=np.int32),
        np.asarray(labels, dtype=np.int64),
    )


def _run(features, groups, *, classifier=RANDOM_FOREST, feature_mode=RAW_IMAGE_MODE,
         options=OPTIONS, parameters=None, names=None, **extra):
    coordinates, labels = _support(*groups)
    return analyze_traditional_ml(
        features,
        coordinates_xy=coordinates,
        labels=labels,
        class_names=names or [f"Class {index}" for index in range(len(groups))],
        classifier=classifier,
        parameters=parameters,
        feature_mode=feature_mode,
        options=options,
        **extra,
    )


# --- feature-mode selection --------------------------------------------------


def test_raw_mode_selects_exactly_channel_zero() -> None:
    features = _features()
    selected = select_feature_channels(features, RAW_IMAGE_MODE)
    assert selected.shape == (1, *IMAGE_SHAPE)
    assert selected.dtype == np.float32
    assert np.array_equal(selected[0], features[0])


def test_maps_mode_selects_all_eight_channels_in_contract_order() -> None:
    from symmlearn.features.eight_channel.contract import CHANNEL_NAMES

    features = _features()
    selected = select_feature_channels(features, IMAGE_PLUS_SYMMETRY_MAPS_MODE)
    assert selected.shape == (8, *IMAGE_SHAPE)
    assert np.array_equal(selected, features)
    assert channel_names_for_mode(RAW_IMAGE_MODE) == [CHANNEL_NAMES[0]]
    assert channel_names_for_mode(IMAGE_PLUS_SYMMETRY_MAPS_MODE) == list(CHANNEL_NAMES)


def test_channel_selection_is_contiguous_without_duplicating_the_tensor() -> None:
    features = _features()
    selected = select_feature_channels(features, IMAGE_PLUS_SYMMETRY_MAPS_MODE)
    assert selected.flags["C_CONTIGUOUS"]
    # The full eight-channel mode must not make a second unnecessary copy of a
    # large feature tensor; only the raw mode narrows to channel zero.
    assert np.shares_memory(selected, features)
    raw = select_feature_channels(features, RAW_IMAGE_MODE)
    assert raw.flags["C_CONTIGUOUS"]
    assert np.array_equal(raw[0], features[0])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda values: values[:7],
        lambda values: values[None],
        lambda values: values[:, 0, :],
    ],
)
def test_non_conforming_feature_tensors_are_rejected(mutate) -> None:
    with pytest.raises(ValueError):
        select_feature_channels(mutate(_features()), RAW_IMAGE_MODE)


def test_non_finite_features_are_rejected() -> None:
    features = _features()
    features[3, 10, 10] = np.nan
    with pytest.raises(ValueError, match="finite"):
        select_feature_channels(features, RAW_IMAGE_MODE)


# --- patch extraction and flattening ----------------------------------------


def test_flatten_patches_is_stable_c_order() -> None:
    patches = np.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2)
    flattened = flatten_patches(patches)
    assert flattened.shape == (2, 12)
    assert np.array_equal(flattened[1], patches[1].reshape(-1))
    assert flattened.dtype == np.float32


def test_flatten_patches_rejects_invalid_batches() -> None:
    with pytest.raises(ValueError, match="four-dimensional"):
        flatten_patches(np.zeros((2, 3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="empty"):
        flatten_patches(np.zeros((0, 3, 3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        flatten_patches(np.full((1, 2, 2, 2), np.nan, dtype=np.float32))


def test_training_matrix_uses_selected_channels_and_the_requested_patch() -> None:
    features = _features()
    coordinates, labels = _support(CLASS_A, CLASS_B)
    for feature_mode, expected_channels in (
        (RAW_IMAGE_MODE, 1),
        (IMAGE_PLUS_SYMMETRY_MAPS_MODE, 8),
    ):
        result = analyze_traditional_ml(
            features,
            coordinates_xy=coordinates,
            labels=labels,
            class_names=["a", "b"],
            classifier=LOGISTIC_REGRESSION,
            feature_mode=feature_mode,
            options=OPTIONS,
        )
        expected = expected_channels * PATCH_SIZE * PATCH_SIZE
        assert result.record["channel_count"] == expected_channels
        assert result.record["training_matrix_shape"] == [len(coordinates), expected]
        assert result.record["feature_mode"] == feature_mode


# --- estimator wiring --------------------------------------------------------


def test_logistic_regression_scales_only_the_support_patches(monkeypatch) -> None:
    captured: list[np.ndarray] = []
    original_fit = StandardScaler.fit

    def spy(self, matrix, y=None, **kwargs):
        captured.append(np.asarray(matrix).copy())
        return original_fit(self, matrix, y, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", spy)
    features = _features()
    coordinates, labels = _support(CLASS_A, CLASS_B)
    _run(features, (CLASS_A, CLASS_B), classifier=LOGISTIC_REGRESSION)

    expected = flatten_patches(
        extract_patches(
            select_feature_channels(features, RAW_IMAGE_MODE), coordinates, PATCH_SIZE
        )
    )
    assert len(captured) == 1
    assert captured[0].shape == expected.shape
    assert np.array_equal(captured[0], expected)


def test_random_forest_receives_the_unscaled_float32_matrix(monkeypatch) -> None:
    captured: list[np.ndarray] = []
    original_fit = RandomForestClassifier.fit

    def spy(self, matrix, y=None, **kwargs):
        captured.append(np.asarray(matrix).copy())
        return original_fit(self, matrix, y, **kwargs)

    monkeypatch.setattr(RandomForestClassifier, "fit", spy)
    features = _features()
    coordinates, _ = _support(CLASS_A, CLASS_B)
    _run(
        features,
        (CLASS_A, CLASS_B),
        classifier=RANDOM_FOREST,
        parameters={"n_estimators": 5},
    )

    expected = flatten_patches(
        extract_patches(
            select_feature_channels(features, RAW_IMAGE_MODE), coordinates, PATCH_SIZE
        )
    )
    assert len(captured) == 1
    assert captured[0].dtype == np.float32
    assert np.array_equal(captured[0], expected)


# --- dense prediction -------------------------------------------------------


def test_dense_outputs_use_the_shared_coordinate_grid() -> None:
    features = _features()
    result = _run(features, (CLASS_A, CLASS_B), parameters={"n_estimators": 5})
    arrays = result.arrays
    coordinates, x_values, y_values = dense_coordinate_grid(
        IMAGE_SHAPE, PATCH_SIZE, OPTIONS.stride
    )
    assert np.array_equal(arrays["coordinates_xy"], coordinates)
    assert np.array_equal(arrays["x_coordinates"], x_values)
    assert np.array_equal(arrays["y_coordinates"], y_values)

    grid_shape = (len(y_values), len(x_values))
    assert arrays["prediction_grid"].shape == grid_shape
    assert arrays["confidence_grid"].shape == grid_shape
    assert arrays["entropy_grid"].shape == grid_shape
    assert arrays["predictions"].dtype == np.int16
    assert arrays["probabilities"].dtype == np.float32
    assert arrays["probabilities"].shape == (len(coordinates), 2)
    assert np.array_equal(
        arrays["prediction_grid"].reshape(-1), arrays["predictions"]
    )
    assert not {"logits"} & set(arrays)


def test_confidence_and_entropy_follow_the_documented_formulas() -> None:
    features = _features()
    result = _run(features, (CLASS_A, CLASS_B), parameters={"n_estimators": 5})
    probabilities = result.arrays["probabilities"].astype(np.float64)
    expected_confidence = probabilities.max(axis=1)
    expected_entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1
    )
    assert np.allclose(result.arrays["confidence"], expected_confidence, atol=1e-6)
    assert np.allclose(result.arrays["entropy"], expected_entropy, atol=1e-6)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert result.arrays["entropy"].max() <= math.log(2) + 1e-6
    assert float(result.arrays["entropy"].min()) >= 0.0


def test_prediction_grid_is_ordered_y_then_x() -> None:
    features = _features()
    options = TraditionalMLOptions(classifier_patch_size=32, seed=3, stride=1, batch_size=64)
    image_shape = (96, 96)
    values = features[:, :96, :96]
    groups = (
        [(30, 30), (34, 34), (38, 38)],
        [(60, 60), (64, 64), (68, 68)],
    )
    result = _run(
        values,
        groups,
        classifier=RANDOM_FOREST,
        options=options,
        parameters={"n_estimators": 8},
    )
    lookup = {
        tuple(point): index
        for index, point in enumerate(result.arrays["coordinates_xy"].tolist())
    }
    coordinates, labels = _support(*groups)
    for point, label in zip(coordinates.tolist(), labels.tolist()):
        row = lookup[(int(point[0]), int(point[1]))]
        assert int(result.arrays["predictions"][row]) == label
    assert result.record["input_shape"] == list(image_shape)


@pytest.mark.parametrize("classifier", [LOGISTIC_REGRESSION, RANDOM_FOREST])
def test_batch_progress_events_cover_training_and_every_prediction_batch(
    classifier: str,
) -> None:
    features = _features()
    events: list[tuple[str, int, int]] = []
    _run(
        features,
        (CLASS_A, CLASS_B),
        classifier=classifier,
        parameters=None if classifier == LOGISTIC_REGRESSION else {"n_estimators": 5},
        progress_callback=lambda phase, current, total: events.append(
            (phase, current, total)
        ),
    )
    training = [event for event in events if event[0] == "training"]
    prediction = [event for event in events if event[0] == "prediction"]
    assert training == [("training", 0, 1), ("training", 1, 1)]

    coordinates, _, _ = dense_coordinate_grid(IMAGE_SHAPE, PATCH_SIZE, OPTIONS.stride)
    total_batches = math.ceil(len(coordinates) / OPTIONS.batch_size)
    assert total_batches > 1
    assert prediction[0] == ("prediction", 0, total_batches)
    assert prediction[-1] == ("prediction", total_batches, total_batches)
    assert [event[1] for event in prediction] == list(range(total_batches + 1))


# --- determinism ------------------------------------------------------------


@pytest.mark.parametrize("classifier", [LOGISTIC_REGRESSION, RANDOM_FOREST])
def test_repeated_runs_are_bitwise_reproducible(classifier: str) -> None:
    features = _features()
    parameters = None if classifier == LOGISTIC_REGRESSION else {"n_estimators": 12}
    first = _run(features, (CLASS_A, CLASS_B), classifier=classifier, parameters=parameters)
    second = _run(features, (CLASS_A, CLASS_B), classifier=classifier, parameters=parameters)
    assert np.array_equal(first.arrays["predictions"], second.arrays["predictions"])
    assert np.array_equal(first.arrays["probabilities"], second.arrays["probabilities"])
    assert np.array_equal(first.arrays["entropy"], second.arrays["entropy"])


def test_a_different_seed_changes_the_random_forest_result() -> None:
    features = _features()
    first = _run(
        features,
        (CLASS_A, CLASS_B),
        parameters={"n_estimators": 7},
        options=TraditionalMLOptions(classifier_patch_size=PATCH_SIZE, seed=1, stride=16, batch_size=8),
    )
    second = _run(
        features,
        (CLASS_A, CLASS_B),
        parameters={"n_estimators": 7},
        options=TraditionalMLOptions(classifier_patch_size=PATCH_SIZE, seed=2, stride=16, batch_size=8),
    )
    assert not np.array_equal(first.arrays["probabilities"], second.arrays["probabilities"])


# --- record -----------------------------------------------------------------


@pytest.mark.parametrize("classifier", [LOGISTIC_REGRESSION, RANDOM_FOREST])
def test_record_is_json_safe_and_carries_the_required_provenance(classifier: str) -> None:
    features = _features()
    feature_record = {"channel_names": ["image"], "n_max": 2}
    result = _run(
        features,
        (CLASS_A, CLASS_B),
        classifier=classifier,
        parameters=None if classifier == LOGISTIC_REGRESSION else {"n_estimators": 4},
        feature_record=feature_record,
        input_shape=IMAGE_SHAPE,
        input_sha256="A" * 64,
    )
    record = result.record
    text = json.dumps(record, sort_keys=True)
    assert json.loads(text) == record

    required = {
        "schema_version",
        "operation",
        "feature_mode",
        "channel_count",
        "channel_names",
        "classifier",
        "class_names",
        "support_counts",
        "support",
        "seed",
        "options",
        "classifier_patch_size",
        "stride",
        "batch_size",
        "input_shape",
        "feature_shape",
        "feature_sha256",
        "feature_provenance",
        "grid_shape",
        "sample_count",
        "timings_seconds",
        "support_training_accuracy",
        "runtime",
    }
    assert required <= set(record)
    assert record["operation"] == "traditional_ml_analyze"
    assert record["classifier"]["identifier"] == classifier
    assert record["class_names"] == ["Class 0", "Class 1"]
    assert record["support_counts"] == {"Class 0": 4, "Class 1": 4}
    assert record["seed"] == 42
    assert record["input_sha256"] == "a" * 64
    assert record["feature_provenance"] == feature_record
    assert record["support_training_accuracy_is_validation"] is False
    assert record["runtime"]["numpy"]
    assert record["runtime"]["scikit_learn"]
    assert record["runtime"]["scipy"] is not None
    assert record["runtime"]["symmetry_learn"]
    assert record["timings_seconds"]["fit"] >= 0
    assert record["timings_seconds"]["prediction"] >= 0
    assert len(record["support"]["coordinates_xy"]) == len(record["support"]["labels"])
    assert len(record["support"]["sha256"]) == 64
    assert record["channel_names"] == (
        ["image"] if record["feature_mode"] == RAW_IMAGE_MODE else record["channel_names"]
    )
    assert "model_identifier" not in record
    assert "checkpoint" not in json.dumps(record)


def test_input_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        _run(_features(), (CLASS_A, CLASS_B), input_shape=(64, 64))


def test_resolved_parameters_are_recorded_not_display_strings() -> None:
    result = _run(
        _features(),
        (CLASS_A, CLASS_B),
        classifier=RANDOM_FOREST,
        parameters={"n_estimators": 11, "max_features": "all", "max_depth": None,
                    "class_weight": "none"},
    )
    parameters = result.record["classifier"]["parameters"]
    assert parameters["n_estimators"] == 11
    assert parameters["max_features"] is None
    assert parameters["max_depth"] is None
    assert parameters["class_weight"] is None
    assert parameters["n_jobs"] == 1
    assert parameters["criterion"] == "gini"


# --- multiclass alignment ---------------------------------------------------


def test_multiclass_probability_columns_align_with_contiguous_labels() -> None:
    class _ReorderedEstimator:
        classes_ = np.asarray([2, 0, 1])

        def predict_proba(self, matrix):  # noqa: ARG002 - shape only
            return np.tile(np.asarray([[0.5, 0.3, 0.2]]), (2, 1))

    aligned = aligned_predict_proba(
        _ReorderedEstimator(), np.zeros((2, 3), dtype=np.float32), class_count=3
    )
    assert np.allclose(aligned[0], [0.3, 0.2, 0.5])
    assert aligned.dtype == np.float32


def test_multiclass_run_keeps_every_class_reachable() -> None:
    features = _features()
    options = TraditionalMLOptions(classifier_patch_size=32, seed=5, stride=1, batch_size=128)
    result = _run(
        features[:, :96, :96],
        ([(30, 30), (34, 34)], [(60, 60), (64, 64)], [(30, 64), (34, 68)]),
        classifier=RANDOM_FOREST,
        options=options,
        parameters={"n_estimators": 8},
    )
    assert result.record["class_count"] == 3
    assert result.arrays["probabilities"].shape[1] == 3
    assert set(np.unique(result.arrays["predictions"]).tolist()) <= {0, 1, 2}
    lookup = {
        tuple(point): index
        for index, point in enumerate(result.arrays["coordinates_xy"].tolist())
    }
    for point, label in zip(
        [(30, 30), (34, 34), (60, 60), (64, 64), (30, 64), (34, 68)],
        [0, 0, 1, 1, 2, 2],
    ):
        assert int(result.arrays["predictions"][lookup[point]]) == label


class _StubEstimator:
    def __init__(self, classes, probabilities) -> None:
        self.classes_ = np.asarray(classes)
        self._probabilities = np.asarray(probabilities)

    def predict_proba(self, matrix):  # noqa: ARG002 - shape only
        return self._probabilities


@pytest.mark.parametrize(
    "classes,probabilities,class_count,message",
    [
        ([0, 1, 5], [[0.2, 0.3, 0.5]], 3, "unexpected class"),
        ([0, 1], [[0.2, 0.2]], 2, "rows must sum"),
        ([0, 1], [[-0.1, 1.1]], 2, "negative"),
        ([0, 1], [[np.nan, 1.0]], 2, "non-finite"),
        ([0, 1], [[0.4, 0.4, 0.2]], 2, "align with the fitted classes"),
        ([], [], 2, "did not report fitted classes"),
        ([0.5], [[1.0]], 2, "non-integer class"),
    ],
)
def test_invalid_probability_outputs_are_rejected(
    classes, probabilities, class_count, message
) -> None:
    estimator = _StubEstimator(classes, probabilities)
    matrix = np.zeros((len(probabilities), 3), dtype=np.float32)
    with pytest.raises(ValueError, match=message):
        aligned_predict_proba(estimator, matrix, class_count=class_count)


# --- support validation errors ---------------------------------------------


def test_support_validation_errors_are_actionable() -> None:
    features = _features()
    cases = [
        ([(40, 40)], [0], ["only one class"], "two"),
        ([(40, 40), (44, 44)], [0, 0], ["a", "b"], "contiguous"),
        ([(40, 40), (44, 44), (48, 48)], [0, 0, 1], ["a", "b"], "minimum"),
        ([(40, 40), (44, 44), (40, 40), (48, 48)], [0, 0, 1, 1], ["a", "b"], "unique"),
        ([(40, 40), (44, 44), (5, 5), (48, 48)], [0, 0, 1, 1], ["a", "b"], "full patch"),
    ]
    for coordinates, labels, names, message in cases:
        with pytest.raises(ValueError, match=message):
            analyze_traditional_ml(
                features,
                coordinates_xy=np.asarray(coordinates, dtype=np.int32),
                labels=np.asarray(labels, dtype=np.int64),
                class_names=names,
                classifier=RANDOM_FOREST,
                feature_mode=RAW_IMAGE_MODE,
                options=OPTIONS,
            )


def test_maximum_patches_per_class_is_enforced() -> None:
    options = TraditionalMLOptions(
        classifier_patch_size=PATCH_SIZE, maximum_patches_per_class=3
    )
    with pytest.raises(ValueError, match="maximum support-point count"):
        _run(_features(), (CLASS_A, CLASS_B), options=options,
             parameters={"n_estimators": 4})


def test_image_smaller_than_the_classifier_patch_is_rejected() -> None:
    with pytest.raises(ValueError, match="smaller than the classifier patch"):
        _run(
            _features(shape=(8, 32, 32)),
            ([(16, 16), (18, 18)], [(20, 20), (22, 22)]),
            options=TraditionalMLOptions(classifier_patch_size=64),
        )


# --- failure handling -------------------------------------------------------


def test_non_convergence_is_reported_as_an_actionable_error() -> None:
    with pytest.raises(TraditionalMLError, match="did not converge"):
        _run(
            _features(),
            (CLASS_A, CLASS_B),
            classifier=LOGISTIC_REGRESSION,
            parameters={"C": 1.0, "max_iter": 1},
        )


def test_unknown_classifier_is_rejected_before_any_work() -> None:
    with pytest.raises(ValueError, match="Unsupported classifier"):
        _run(_features(), (CLASS_A, CLASS_B), classifier="gradient_boosting")


def test_options_must_be_a_validated_instance() -> None:
    with pytest.raises(TypeError, match="TraditionalMLOptions"):
        _run(_features(), (CLASS_A, CLASS_B), options={"classifier_patch_size": 64})


# --- no pretrained model involvement ---------------------------------------


@pytest.mark.parametrize("classifier", [LOGISTIC_REGRESSION, RANDOM_FOREST])
def test_workflow_never_touches_any_pretrained_checkpoint(
    monkeypatch, classifier: str
) -> None:
    import symmlearn.models.registry as registry
    import symmlearn.models.weights as weights

    def _forbidden(*args, **kwargs):  # noqa: ARG001 - signature compatibility
        raise AssertionError("the traditional workflow must not load a checkpoint")

    monkeypatch.setattr(weights, "load_pretrained_checkpoint", _forbidden)
    monkeypatch.setattr(weights, "resolve_model_weight", _forbidden)
    monkeypatch.setattr(registry, "build_registered_model", _forbidden)

    result = _run(
        _features(),
        (CLASS_A, CLASS_B),
        classifier=classifier,
        parameters=None if classifier == LOGISTIC_REGRESSION else {"n_estimators": 4},
    )
    assert result.arrays["prediction_grid"].size > 0
