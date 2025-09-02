"""
RAG-agnostic provider interfaces and lightweight adapters.

RAGProvider abstracts how context is retrieved (and optionally generated).
IndexProvider adapts the in-package VerbatimIndex to the provider interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import asyncio

from verbatim_rag.index import VerbatimIndex
from verbatim_rag.core import VerbatimRAG
from verbatim_rag.universal_document import UniversalDocument


class RAGProvider(ABC):
    """Abstract provider interface for RAG-agnostic retrieval/generation."""

    @abstractmethod
    def retrieve(
        self, question: str, k: int = 5, filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return a list of context items (dicts) with 'content', optional 'title', 'source', 'metadata'."""
        raise NotImplementedError

    def generate(self, question: str, context: List[Dict[str, Any]]) -> str:
        """Optionally generate an initial answer from context (not required)."""
        raise NotImplementedError

    async def retrieve_async(
        self, question: str, k: int = 5, filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Async variant. Default delegates to sync via thread to avoid blocking."""
        return await asyncio.to_thread(self.retrieve, question, k, filter)


class IndexProvider(RAGProvider):
    """Adapter for the built-in VerbatimIndex as a RAGProvider."""

    def __init__(self, index: VerbatimIndex):
        self.index = index

    def retrieve(
        self, question: str, k: int = 5, filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        results = self.index.query(text=question, k=k, filter=filter)
        # Convert SearchResult -> UniversalDocument -> context dict
        context: List[Dict[str, Any]] = []
        for r in results:
            doc = UniversalDocument.from_text(
                text=r.text,
                title=r.metadata.get("title", ""),
                source=r.metadata.get("source", ""),
                metadata={
                    k: v
                    for k, v in (r.metadata or {}).items()
                    if k not in {"title", "source"}
                },
            )
            context.append(doc.to_context())
        return context

    async def retrieve_async(
        self, question: str, k: int = 5, filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.retrieve, question, k, filter)


class VerbatimRAGProvider(RAGProvider):
    """Adapter that treats VerbatimRAG as a retrieval provider.

    It delegates retrieval to the underlying index; generation is intentionally
    not implemented here (verbatim transformation is handled by VerbatimTransform).
    """

    def __init__(self, rag: VerbatimRAG):
        self.rag = rag

    def retrieve(
        self, question: str, k: int = 5, filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        results = self.rag.index.query(text=question, k=k, filter=filter)
        context: List[Dict[str, Any]] = []
        for r in results:
            context.append(
                {
                    "content": r.text,
                    "title": r.metadata.get("title", ""),
                    "source": r.metadata.get("source", ""),
                    "metadata": {
                        k: v
                        for k, v in (r.metadata or {}).items()
                        if k not in {"title", "source"}
                    },
                }
            )
        return context

    async def retrieve_async(
        self, question: str, k: int = 5, filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.retrieve, question, k, filter)
