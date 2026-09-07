from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from symmlearn.provider import (
    PROVIDER_CONTRACT_VERSION,
    WORKER_SCHEMA_VERSION,
    compute_features,
    model_capabilities,
)
from symmlearn.provider.registry import build_registered_model, provider_capabilities
from symmlearn.provider.model import add_task_adapters, trainable_state_dict
from symmlearn.provider.worker import (
    _dense_coordinate_grid,
    _extract_patches,
    _load_job,
    _validate_support,
    _validated_options,
)


def _unit_image(size: int = 64) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    image = np.sin(x / 4.0) + np.cos(y / 5.0) + 0.02 * x
    return ((image - image.min()) / (image.max() - image.min())).astype(
        np.float32
    )


def test_capabilities_define_one_versioned_model() -> None:
    capabilities = provider_capabilities()
    models = model_capabilities()
    assert capabilities["contract_version"] == "symmetry-learn-provider-v1"
    assert capabilities["provider"] == "symmetry-learn"
    assert PROVIDER_CONTRACT_VERSION == "symmetry-learn-provider-v1"
    assert WORKER_SCHEMA_VERSION == "scientific-symmetry-worker-v1"
    assert [model["identifier"] for model in models] == ["cnn_8ch_pg17"]
    assert models[0]["input_channels"] == 8
    assert models[0]["classifier_patch_size"] == 64
    assert capabilities["provider_version"] == "0.1.0"


def test_checkpoint_compatible_model_exposes_expected_state_keys() -> None:
    pytest.importorskip("torch")
    model = build_registered_model("cnn_8ch_pg17")
    state = model.state_dict()
    assert state["conv_blocks.0.0.weight"].shape == (32, 8, 3, 3)
    assert state["fc_blocks.0.0.weight"].shape == (512, 32768)
    assert state["classifier.weight"].shape == (17, 256)
    incompatible = model.load_state_dict(state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    adapted = add_task_adapters(model, bottleneck=4, task_classes=2)
    trainable = trainable_state_dict(adapted)
    assert len(trainable) == 12
    assert all(".adapter." in name or name.startswith("classifier.") for name in trainable)


def test_support_coordinates_and_dense_grid_use_xy_convention() -> None:
    options = _validated_options({"minimum_shots_per_class": 1})
    coordinates, labels, names = _validate_support(
        (96, 96),
        np.asarray([[32, 32], [64, 64]], dtype=np.int32),
        np.asarray([0, 1], dtype=np.int64),
        ["Phase A", "Phase B"],
        options,
    )
    features = np.zeros((8, 96, 96), dtype=np.float32)
    for y in range(96):
        for x in range(96):
            features[0, y, x] = y * 1000 + x
    patches = _extract_patches(features, coordinates, 64)
    assert patches[0, 0, 32, 32] == 32032
    assert labels.tolist() == [0, 1]
    assert names == ["Phase A", "Phase B"]
    dense, x_values, y_values = _dense_coordinate_grid((72, 80), 64, 4)
    assert dense[0].tolist() == [32, 32]
    assert x_values.tolist() == [32, 36, 40, 44, 48]
    assert y_values.tolist() == [32, 36, 40]


def test_direct_feature_api_builds_eight_channels() -> None:
    pytest.importorskip("torch")
    features, record = compute_features(
        _unit_image(),
        options={"n_max": 4, "symmetry_patch_size": 5},
        device="cpu",
    )
    assert features.shape == (8, 64, 64)
    assert np.isfinite(features).all()
    assert len(record["channel_names"]) == 8


def test_worker_rejects_an_incompatible_schema(tmp_path: Path) -> None:
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        _load_job(path)
    with pytest.raises(ValueError, match="Unsupported provider options"):
        _validated_options({"unknown": 1})
