"""Deprecation warnings for the public API of verbatim_core and verbatim_rag.

verbatim-rag uses `deprecated` and `warn_deprecated`: keep them within a minor version
(tests/test_core_dependency.py).
"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import deprecated as _deprecated

if TYPE_CHECKING:
    from typing_extensions import LiteralString

DEPRECATED_SINCE = "0.3.0"
# The release that removes the deprecated API: the 0.3.x and 0.4.x releases warn first.
# Every warning reads both versions from here.
REMOVED_IN = "0.5.0"


# Marks the deprecated arguments that the caller didn't pass.
_UNSET: Any = object()


class VerbatimDeprecationWarning(DeprecationWarning):
    """Emitted when code uses a part of the API that will be removed."""


def _message(name: str, use: str) -> str:
    return f"{name} is deprecated since {DEPRECATED_SINCE} and will be removed in {REMOVED_IN}. Use {use}."


def deprecated(name: str, *, use: str) -> _deprecated:
    """Mark a class, function or method as deprecated."""
    # The message is built at runtime. typing_extensions.deprecated asks for a literal only so
    # that type checkers can show the message; the decorator itself takes any string.
    return _deprecated(cast("LiteralString", _message(name, use)), category=VerbatimDeprecationWarning)


def warn_deprecated(name: str, *, use: str, stacklevel: int | None = 3) -> None:
    """Warn about a deprecated way of calling a function, such as passing a deprecated argument.

    Called from inside that function. The default stacklevel points at its caller. None points at
    the first caller outside verbatim_core and verbatim_rag, for a warning raised deep inside them.
    """
    if stacklevel is None:
        stacklevel = _first_level_outside_the_library()
    warnings.warn(_message(name, use), VerbatimDeprecationWarning, stacklevel=stacklevel)


def warn_contextual_without_client() -> None:
    """Warn that contextual template mode is used without an LLM client.

    Contextual template mode needs an LLM client. Until the deprecation window ends, the static
    templates stand in for it.
    """
    warn_deprecated(
        "Contextual template mode without an LLM client",
        use="TemplateManager(llm_client=...)",
        stacklevel=None,
    )


def _first_level_outside_the_library() -> int:
    # Frame 2 is the caller of warn_deprecated, stacklevel 2 for warnings.warn there.
    level, frame = 2, sys._getframe(2)  # noqa: SLF001  # the cheap way to walk the caller's frames
    while frame is not None and frame.f_globals.get("__name__", "").split(".")[0] in (
        "verbatim_core",
        "verbatim_rag",
    ):
        level, frame = level + 1, frame.f_back
    return level
