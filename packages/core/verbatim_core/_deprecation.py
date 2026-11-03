"""Deprecation warnings for the public API of verbatim_core and verbatim_rag."""

from __future__ import annotations

import warnings

from typing_extensions import deprecated as _deprecated

DEPRECATED_SINCE = "0.3.0"
# The release that removes the deprecated API. Provisional until the length of the
# deprecation window is decided. Every warning reads both versions from here.
REMOVED_IN = "0.5.0"


class VerbatimDeprecationWarning(DeprecationWarning):
    """Emitted when code uses a part of the API that will be removed."""


def _message(name: str, use: str) -> str:
    return (
        f"{name} is deprecated since {DEPRECATED_SINCE} and will be removed in {REMOVED_IN}. "
        f"Use {use}."
    )


def deprecated(name: str, *, use: str):
    """Mark a class, function or method as deprecated."""
    return _deprecated(_message(name, use), category=VerbatimDeprecationWarning)


def warn_deprecated(name: str, *, use: str, stacklevel: int = 3) -> None:
    """Warn about a deprecated way of calling a function, such as passing a deprecated argument.

    Called from inside that function. The default stacklevel points at its caller.
    """
    warnings.warn(_message(name, use), VerbatimDeprecationWarning, stacklevel=stacklevel)
