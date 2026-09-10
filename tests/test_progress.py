from __future__ import annotations

import numpy as np
import pytest

from symmlearn.finetuning.engine import fit_adapters
from symmlearn.inference.dense import predict_dense


def test_fine_tuning_reports_each_completed_epoch() -> None:
    torch = pytest.importorskip("torch")

    class TinyClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = torch.nn.Linear(4, 2)

        def forward(self, values):
            return self.classifier(values.flatten(start_dim=1))

    progress: list[tuple[int, int]] = []
    fit_adapters(
        TinyClassifier(),
        np.asarray(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
            ],
            dtype=np.float32,
        ),
        np.asarray([0, 1], dtype=np.int64),
        epochs=3,
        learning_rate=0.01,
        weight_decay=0.0,
        seed=7,
        device="cpu",
        progress_callback=lambda current, total: progress.append((current, total)),
    )

    assert progress == [(0, 3), (1, 3), (2, 3), (3, 3)]


def test_dense_prediction_reports_each_completed_patch_batch() -> None:
    torch = pytest.importorskip("torch")

    class MeanClassifier(torch.nn.Module):
        def forward(self, values):
            mean = values.mean(dim=(1, 2, 3))
            return torch.stack((-mean, mean), dim=1)

    progress: list[tuple[int, int]] = []
    result = predict_dense(
        MeanClassifier(),
        np.ones((1, 6, 6), dtype=np.float32),
        patch_size=2,
        stride=2,
        batch_size=4,
        device="cpu",
        expected_channels=1,
        progress_callback=lambda current, total: progress.append((current, total)),
    )

    assert result.prediction_grid.shape == (3, 3)
    assert progress == [(0, 3), (1, 3), (2, 3), (3, 3)]
