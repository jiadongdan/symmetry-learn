from __future__ import annotations

import pytest

from symmlearn.exceptions import WeightIntegrityError
from symmlearn.models.weights import file_sha256, load_pretrained_checkpoint


def test_checkpoint_loader_verifies_digest_and_strictly_restores_state(
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")
    source = torch.nn.Linear(3, 2)
    expected = {
        name: value.detach().clone() for name, value in source.state_dict().items()
    }
    path = tmp_path / "checkpoint.pth"
    torch.save({"model_state_dict": expected, "epoch": 7}, path)
    digest = file_sha256(path)

    target = torch.nn.Linear(3, 2)
    record = load_pretrained_checkpoint(target, path, digest)
    assert record["epoch"] == 7
    assert record["strict_load"] is True
    assert all(
        torch.equal(target.state_dict()[name], value)
        for name, value in expected.items()
    )

    with pytest.raises(WeightIntegrityError, match="SHA-256"):
        load_pretrained_checkpoint(target, path, "0" * 64)
