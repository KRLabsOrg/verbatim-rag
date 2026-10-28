"""Universal document container used across verbatim_core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UniversalDocument:
    """Minimal document with content, title, source and metadata."""

    content: str
    title: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_text(
        cls,
        text: str,
        title: str = "",
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> UniversalDocument:
        """Create a document from raw text."""
        return cls(content=text, title=title, source=source, metadata=metadata or {})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UniversalDocument:
        """Create a document from a dict with ``content`` (or ``text``)."""
        if not isinstance(data, dict):
            msg = "UniversalDocument.from_dict expects a dict"
            raise TypeError(msg)
        content = data.get("content") or data.get("text")
        if not isinstance(content, str) or not content:
            msg = "UniversalDocument requires 'content' (or 'text') as non-empty string"
            raise ValueError(msg)
        return cls(
            content=content,
            title=data.get("title", ""),
            source=data.get("source", ""),
            metadata=data.get("metadata") or {},
        )

    def to_context(self) -> dict[str, Any]:
        """Convert the document to a context dict."""
        return {
            "content": self.content,
            "title": self.title,
            "source": self.source,
            "metadata": self.metadata,
        }
