"""Persist Provider arrays, adapter state, and provenance records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from symmlearn.finetuning.artifacts import save_adapter_artifact


def persist_provider_result(
    result: Any,
    *,
    output_path: str | Path,
    adapter_path: str | Path,
    record_path: str | Path,
) -> dict[str, Any]:
    """Write one Provider result using the existing NPZ, PT, and JSON contract."""
    output = Path(output_path)
    adapter = Path(adapter_path)
    record_file = Path(record_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    adapter.parent.mkdir(parents=True, exist_ok=True)
    record_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result.arrays)
    save_adapter_artifact(adapter, result.adapter_checkpoint)
    record = {
        **result.record,
        "output_path": str(output.resolve()),
        "adapter_path": str(adapter.resolve()),
        "record_path": str(record_file.resolve()),
    }
    record_file.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record
