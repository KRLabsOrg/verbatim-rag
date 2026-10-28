from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any


class RAGProvider(ABC):
    """Abstract provider interface for RAG-agnostic retrieval/generation."""

    @abstractmethod
    def retrieve(
        self, question: str, k: int = 5, filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return list of context dicts: {'content', 'title'?, 'source'?, 'metadata'?}."""
        raise NotImplementedError

    def generate(self, question: str, context: list[dict[str, Any]]) -> str:
        """Optionally generate an initial answer from context (not required)."""
        raise NotImplementedError

    async def retrieve_async(
        self, question: str, k: int = 5, filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Async variant. Default delegates to sync via thread to avoid blocking."""
        return await asyncio.to_thread(self.retrieve, question, k, filter)
