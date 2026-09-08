"""Verify installed wheels and the actual default checkpoint, outside the checkout."""

import importlib.metadata as metadata
import json
from pathlib import Path

import symmlearn
import symmlearn_default_model
from symmlearn.provider.api import probe_model


def main():
    checkout = Path(__file__).resolve().parents[1]
    for module in (symmlearn, symmlearn_default_model):
        origin = Path(module.__file__).resolve()
        if checkout in origin.parents:
            raise RuntimeError(f"Smoke check imported source checkout: {origin}")
        print(f"Installed import: {origin}")
    if symmlearn.__version__ != metadata.version("symmetry-learn"):
        raise RuntimeError("Installed metadata and __version__ differ")
    report = probe_model(options={"device": "cpu"})
    if not report["available"] or not report["details"]["checkpoint"]["strict_load"]:
        raise RuntimeError("Installed default checkpoint did not load strictly")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
