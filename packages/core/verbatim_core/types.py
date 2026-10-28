"""Shared typing protocols for verbatim_core."""

from __future__ import annotations

from typing import Protocol


class HasText(Protocol):
    """Protocol for objects exposing a ``text`` attribute."""

    text: str
