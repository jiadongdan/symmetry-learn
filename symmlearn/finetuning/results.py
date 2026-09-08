"""Structured results from task-specific fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FineTuningResult:
    """Training diagnostics and the selected support-loss state."""

    loss: list[float]
    support_accuracy: list[float]
    best_support_loss: float
    final_support_loss: float
    final_support_accuracy: float
    epochs: int
    runtime_seconds: float
    trainable_parameters: int
    warning: str

    def to_record(self) -> dict[str, object]:
        """Return the stable serializable training record."""
        return {
            "loss": self.loss,
            "support_accuracy": self.support_accuracy,
            "best_support_loss": self.best_support_loss,
            "final_support_loss": self.final_support_loss,
            "final_support_accuracy": self.final_support_accuracy,
            "epochs": self.epochs,
            "runtime_seconds": self.runtime_seconds,
            "trainable_parameters": self.trainable_parameters,
            "warning": self.warning,
        }
