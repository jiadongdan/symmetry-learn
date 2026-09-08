from __future__ import annotations

import json
from pathlib import Path

from symmlearn.models.cnn_8ch_pg17.specification import MODEL_SPECIFICATION
from symmlearn.models.registry import get_model_specification


def test_default_model_manifest_matches_the_registered_weight() -> None:
    manifest_path = (
        Path(__file__).parents[2]
        / "model_packages"
        / "symmetry_learn_default_model"
        / "src"
        / "symmlearn_default_model"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest["weights"][0]
    registered = get_model_specification("cnn_8ch_pg17").default_weight

    assert MODEL_SPECIFICATION.default_weight is registered
    assert declared["model_id"] == MODEL_SPECIFICATION.identifier
    assert declared["weight_id"] == registered.identifier
    assert declared["weight_version"] == registered.version
    assert declared["resource"] == registered.resource
    assert declared["sha256"] == registered.sha256
    assert declared["size_bytes"] == registered.size_bytes
