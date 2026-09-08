"""Reusable adapter modules, optimization, and task artifacts."""

__all__ = [
    "ADAPTER_SCHEMA_VERSION",
    "FineTuningResult",
    "add_task_adapters",
    "fit_adapters",
    "load_adapter_artifact",
    "save_adapter_artifact",
    "set_deterministic_seed",
    "trainable_state_dict",
]


def __getattr__(name: str):
    if name in {"ADAPTER_SCHEMA_VERSION", "load_adapter_artifact", "save_adapter_artifact"}:
        from .artifacts import (
            ADAPTER_SCHEMA_VERSION,
            load_adapter_artifact,
            save_adapter_artifact,
        )

        return {
            "ADAPTER_SCHEMA_VERSION": ADAPTER_SCHEMA_VERSION,
            "load_adapter_artifact": load_adapter_artifact,
            "save_adapter_artifact": save_adapter_artifact,
        }[name]
    if name in {"add_task_adapters", "trainable_state_dict"}:
        from .adapters import add_task_adapters, trainable_state_dict

        return {
            "add_task_adapters": add_task_adapters,
            "trainable_state_dict": trainable_state_dict,
        }[name]
    if name in {"fit_adapters", "set_deterministic_seed"}:
        from .engine import fit_adapters, set_deterministic_seed

        return {
            "fit_adapters": fit_adapters,
            "set_deterministic_seed": set_deterministic_seed,
        }[name]
    if name == "FineTuningResult":
        from .results import FineTuningResult

        return FineTuningResult
    raise AttributeError(name)
