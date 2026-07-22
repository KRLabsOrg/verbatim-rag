"""Keep the public package versions aligned without importing heavy dependencies."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _project_version(path: Path) -> str:
    project_section = path.read_text().split("[project]", 1)[1].split("\n[", 1)[0]
    match = re.search(r'^version\s*=\s*"([^"]+)"', project_section, re.MULTILINE)
    assert match is not None, f"No [project] version found in {path}"
    return match.group(1)


def _module_version(path: Path) -> str:
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', path.read_text(), re.MULTILINE)
    assert match is not None, f"No __version__ found in {path}"
    return match.group(1)


def test_public_package_versions_match() -> None:
    root_version = _project_version(ROOT / "pyproject.toml")
    core_version = _project_version(ROOT / "packages" / "core" / "pyproject.toml")
    module_version = _module_version(ROOT / "verbatim_rag" / "__init__.py")

    assert root_version == core_version == module_version
