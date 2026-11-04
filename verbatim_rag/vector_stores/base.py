"""
Base classes for vector stores.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from verbatim_core import Chunk


@dataclass
class SearchResult:
    """Result from vector search."""

    id: str
    score: float
    metadata: Dict[str, Any]
    text: str  # Original clean text for display
    enhanced_text: str = ""  # Enhanced text used for vectorization

    def to_chunk(self) -> Chunk:
        """The chunk of this row, with the row id as its id (ADR 0002), its text (not the enhanced
        text used for embedding) and its metadata. The index doesn't store source offsets.

        A chunk id stays the same while its document stays indexed, and changes when the document
        is indexed again.
        """
        return Chunk(id=self.id, text=self.text, metadata=dict(self.metadata))

    def __repr__(self) -> str:
        return f"SearchResult(id={self.id}, score={self.score}, metadata={self.metadata}, text={self.text}, enhanced_text={self.enhanced_text})"

    def __str__(self) -> str:
        return self.__repr__()

    def __format__(self, format_spec: str) -> str:
        return self.__repr__()

    def __gt__(self, other: "SearchResult") -> bool:
        return self.score > other.score

    def __lt__(self, other: "SearchResult") -> bool:
        return self.score < other.score

    def __eq__(self, other: "SearchResult") -> bool:
        return self.score == other.score

    def __hash__(self) -> int:
        return hash((self.id, self.score, self.metadata, self.text, self.enhanced_text))


class VectorStore(ABC):
    """Base vector store interface."""

    @abstractmethod
    def add_vectors(
        self,
        ids: List[str],
        dense_vectors: Optional[List[List[float]]],
        sparse_vectors: Optional[List[Dict[int, float]]],
        texts: List[str],
        enhanced_texts: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """Add vectors with metadata, original text, and enhanced text."""
        pass

    @abstractmethod
    def query(
        self,
        dense_query: Optional[List[float]] = None,
        sparse_query: Optional[Dict[int, float]] = None,
        text_query: Optional[str] = None,
        top_k: int = 5,
        search_type: str = "hybrid",
        filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Query for similar vectors using hybrid search."""
        pass

    @abstractmethod
    def delete(self, ids: List[str]):
        """Delete vectors by IDs."""
        pass
