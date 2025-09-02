from __future__ import annotations

from typing import Protocol


class HasText(Protocol):
    text: str
