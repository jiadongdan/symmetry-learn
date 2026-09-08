"""Runtime helpers shared by registered model implementations."""

from __future__ import annotations


def resolve_device(requested: str) -> str:
    """Resolve an automatic device while validating explicit requests."""
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is not available.")
    if requested != "cpu" and not requested.startswith("cuda"):
        raise ValueError("device must be auto, cpu, or a CUDA device.")
    return requested
