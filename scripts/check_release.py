"""Check release archives without importing the source checkout (Python 3.11+)."""

import argparse
from email.parser import BytesParser
from hashlib import sha256
import json
from pathlib import Path
import tarfile
import tomllib
import zipfile


def check_release(root: Path, dist: Path, tag: str | None) -> None:
    projects = {
        "library": root,
        "model": root / "model_packages/symmetry_learn_default_model",
    }
    metadata = {
        key: tomllib.loads((path / "pyproject.toml").read_text())["project"]
        for key, path in projects.items()
    }
    if tag:
        expected = {
            "v" + metadata["library"]["version"],
            "model-v" + metadata["model"]["version"],
        }
        if tag not in expected:
            raise ValueError(f"Release tag {tag!r} must match one of {sorted(expected)}")
    dependency = f"symmetry-learn-default-model=={metadata['model']['version']}"
    if dependency not in metadata["library"]["dependencies"]:
        raise ValueError("Library dependency and model package version differ")
    manifest = json.loads((projects["model"] / "src/symmlearn_default_model/manifest.json").read_text())
    for kind, project in metadata.items():
        archives = sorted((dist / kind).glob("*"))
        wheels = [p for p in archives if p.suffix == ".whl"]
        sdists = [p for p in archives if p.name.endswith(".tar.gz")]
        if len(wheels) != 1 or len(sdists) != 1 or len(archives) != 2:
            raise ValueError(f"Expected exactly one wheel and sdist in {dist / kind}")
        for archive in archives:
            if archive.stat().st_size >= 100_000_000:
                raise ValueError(f"Archive exceeds the default PyPI size limit: {archive}")
            if archive.suffix == ".whl":
                with zipfile.ZipFile(archive) as handle:
                    contents = {n: handle.read(n) for n in handle.namelist() if not n.endswith("/")}
                meta_name = next(n for n in contents if n.endswith(".dist-info/METADATA"))
            else:
                with tarfile.open(archive) as handle:
                    contents = {m.name: handle.extractfile(m).read() for m in handle.getmembers() if m.isfile()}
                meta_name = min((n for n in contents if n.endswith("/PKG-INFO")), key=len)
            parsed = BytesParser().parsebytes(contents[meta_name])
            if parsed["Name"] != project["name"] or parsed["Version"] != project["version"]:
                raise ValueError(f"Wrong name/version in {archive}")
            if parsed["License-Expression"] != "MIT" or not any(n.endswith("/LICENSE") for n in contents):
                raise ValueError(f"Missing license in {archive}")
            if kind == "model":
                for weight in manifest["weights"]:
                    matches = [v for n, v in contents.items() if n.endswith("/" + weight["resource"])]
                    if len(matches) != 1 or len(matches[0]) != weight["size_bytes"] or sha256(matches[0]).hexdigest() != weight["sha256"]:
                        raise ValueError(f"Missing/corrupt checkpoint or Git LFS pointer in {archive}")
            elif any(n.endswith(".pth") for n in contents):
                raise ValueError(f"The library archive must not duplicate model weights: {archive}")
            print(f"Verified {archive.name}: {archive.stat().st_size:,} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--tag")
    args = parser.parse_args()
    check_release(Path(__file__).resolve().parents[1], args.dist, args.tag)
