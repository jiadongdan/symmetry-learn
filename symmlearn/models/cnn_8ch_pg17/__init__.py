"""Eight-channel CNN pretrained for 17 plane-group classes."""

from .specification import MODEL_IDENTIFIER, MODEL_SPECIFICATION

__all__ = [
    "CheckpointCompatibleCNN",
    "MODEL_IDENTIFIER",
    "MODEL_SPECIFICATION",
    "build_model",
]


def __getattr__(name: str):
    if name in {"CheckpointCompatibleCNN", "build_model"}:
        from .architecture import CheckpointCompatibleCNN, build_model

        return {
            "CheckpointCompatibleCNN": CheckpointCompatibleCNN,
            "build_model": build_model,
        }[name]
    raise AttributeError(name)
