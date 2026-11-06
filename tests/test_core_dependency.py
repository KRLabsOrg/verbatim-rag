"""verbatim-rag's dependency on verbatim-core: the version range, and the core's private names it uses.

verbatim-rag and the API server use a few private names of the core. The core keeps them within a
minor version, and verbatim-rag requires the core's minor version, so a core release can't break an
installed verbatim-rag.
"""

import ast
import re
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent

SHARED_PRIVATE_NAMES = {
    ("verbatim_core._deprecation", "deprecated"),
    ("verbatim_core._deprecation", "warn_deprecated"),
    ("verbatim_core.response_builder", "_documents_and_citations"),
    ("verbatim_core._context", "_chunks_from_context"),
}


def _private_core_imports() -> set[tuple[str, str]]:
    found = set()
    for path in [*(ROOT / "verbatim_rag").rglob("*.py"), *(ROOT / "api").rglob("*.py")]:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            package, *modules = node.module.split(".")
            if package != "verbatim_core":
                continue
            private_module = any(module.startswith("_") for module in modules)
            found |= {
                (node.module, alias.name)
                for alias in node.names
                if private_module or alias.name.startswith("_")
            }
    return found


def _core_requirement() -> SpecifierSet:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return SpecifierSet(re.search(r'"verbatim-core([^"]*)"', pyproject)[1])


def _core_version() -> Version:
    pyproject = (ROOT / "packages" / "core" / "pyproject.toml").read_text(encoding="utf-8")
    return Version(re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)[1])


def test_verbatim_rag_uses_only_the_private_core_names_the_core_keeps():
    assert _private_core_imports() == SHARED_PRIVATE_NAMES


def test_verbatim_rag_requires_the_minor_version_of_the_core_in_the_repository():
    core, requirement = _core_version(), _core_requirement()
    next_minor = Version(f"{core.major}.{core.minor + 1}.0")

    assert core in requirement
    assert next_minor not in requirement
