from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class UniversalDocument:
    content: str
    title: str = ""
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_text(
        cls,
        text: str,
        title: str = "",
        source: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> "UniversalDocument":
        return cls(content=text, title=title, source=source, metadata=metadata or {})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniversalDocument":
        if not isinstance(data, dict):
            raise TypeError("UniversalDocument.from_dict expects a dict")
        content = data.get("content") or data.get("text")
        if not isinstance(content, str) or not content:
            raise ValueError(
                "UniversalDocument requires 'content' (or 'text') as non-empty string"
            )
        return cls(
            content=content,
            title=data.get("title", ""),
            source=data.get("source", ""),
            metadata=data.get("metadata") or {},
        )

    def to_context(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "title": self.title,
            "source": self.source,
            "metadata": self.metadata,
        }
