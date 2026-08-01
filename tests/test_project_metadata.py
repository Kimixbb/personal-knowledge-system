from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version


def test_project_supports_python_311_through_313() -> None:
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "pyproject.toml").open("rb") as handle:
        metadata = tomllib.load(handle)

    supported = SpecifierSet(metadata["project"]["requires-python"])

    assert Version("3.11") in supported
    assert Version("3.12") in supported
    assert Version("3.13") in supported
    assert Version("3.14") not in supported
