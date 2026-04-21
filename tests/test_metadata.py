from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.md"


def test_readme_python_requirement_matches_pyproject():
    pyproject_text = PYPROJECT.read_text(encoding="utf-8")
    readme_text = README.read_text(encoding="utf-8")

    pyproject_match = re.search(r'requires-python\s*=\s*"([^"]+)"', pyproject_text)
    assert pyproject_match is not None, "requires-python not found in pyproject.toml"

    readme_match = re.search(r"- Python\s+([^\n]+)", readme_text)
    assert readme_match is not None, "Python requirement not found in README.md"

    readme_requirement = re.sub(r"\s+", "", readme_match.group(1).strip())
    pyproject_requirement = re.sub(r"\s+", "", pyproject_match.group(1).strip())

    assert readme_requirement == pyproject_requirement
