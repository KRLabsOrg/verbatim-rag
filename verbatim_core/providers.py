from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import asyncio


class RAGProvider(ABC):
    """Abstract provider interface for RAG-agnostic retrieval/generation."""

    @abstractmethod
    def retrieve(
        self, question: str, k: int = 5, filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return list of context dicts: {'content', 'title'?, 'source'?, 'metadata'?}."""
        raise NotImplementedError

    def generate(self, question: str, context: List[Dict[str, Any]]) -> str:
        """Optionally generate an initial answer from context (not required)."""
        raise NotImplementedError

    async def retrieve_async(
        self, question: str, k: int = 5, filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Async variant. Default delegates to sync via thread to avoid blocking."""
        return await asyncio.to_thread(self.retrieve, question, k, filter)
