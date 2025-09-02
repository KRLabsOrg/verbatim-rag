"""
Chunking service for text processing in Verbatim RAG.

Separates chunking logic from the index for better organization and reusability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from verbatim_rag.document import Document, DocumentType
from verbatim_rag.schema import DocumentSchema


class ChunkingStrategy(Enum):
    """Available chunking strategies."""

    RECURSIVE = "recursive"
    TOKEN = "token"
    SENTENCE = "sentence"
    WORD = "word"
    SDPM = "sdpm"


@dataclass
class ChunkingConfig:
    """Configuration for chunking operations."""

    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    recipe: str = "markdown"  # Used by recursive chunker
    lang: str = "en"
    chunk_size: int = 512
    chunk_overlap: int = 50
    merge_threshold: float = 0.7  # SDPM only
    split_threshold: float = 0.3  # SDPM only

    @classmethod
    def from_document_metadata(
        cls, metadata: Dict[str, Any], document_type: DocumentType
    ) -> "ChunkingConfig":
        """Create chunking config from document metadata with fallbacks."""
        return cls(
            strategy=ChunkingStrategy(
                str(metadata.get("chunker_type", "recursive")).lower()
            ),
            recipe=str(
                metadata.get(
                    "chunker_recipe",
                    "default" if document_type == DocumentType.TXT else "markdown",
                )
            ),
            lang=str(metadata.get("lang", "en")),
            chunk_size=int(metadata.get("chunk_size", 512)),
            chunk_overlap=int(metadata.get("chunk_overlap", 50)),
            merge_threshold=float(metadata.get("merge_threshold", 0.7)),
            split_threshold=float(metadata.get("split_threshold", 0.3)),
        )


class ChunkerInterface(ABC):
    """Abstract interface for text chunkers."""

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """Chunk text into a list of text segments."""
        pass


class ChonkieChunker(ChunkerInterface):
    """Chunker implementation using the Chonkie library."""

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self._chunker = self._create_chunker()

    def _create_chunker(self):
        """Create the appropriate chonkie chunker based on config."""
        try:
            import chonkie
        except ImportError as e:
            raise ImportError(
                "Text chunking requires chonkie. Install with: pip install chonkie"
            ) from e

        if self.config.strategy == ChunkingStrategy.RECURSIVE:
            return chonkie.RecursiveChunker.from_recipe(
                self.config.recipe, lang=self.config.lang
            )
        elif self.config.strategy == ChunkingStrategy.TOKEN:
            return chonkie.TokenChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
        elif self.config.strategy == ChunkingStrategy.SENTENCE:
            return chonkie.SentenceChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
        elif self.config.strategy == ChunkingStrategy.WORD:
            return chonkie.WordChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
        elif self.config.strategy == ChunkingStrategy.SDPM:
            return chonkie.SDPMChunker(
                chunk_size=self.config.chunk_size,
                merge_threshold=self.config.merge_threshold,
                split_threshold=self.config.split_threshold,
            )
        else:
            # Default fallback to recursive
            return chonkie.RecursiveChunker.from_recipe(
                self.config.recipe, lang=self.config.lang
            )

    def chunk(self, text: str) -> List[str]:
        """Chunk text using the configured chonkie chunker."""
        chunks = self._chunker(text)
        return [c.text for c in chunks if getattr(c, "text", "").strip()]


class ChunkingService:
    """Service for handling all text chunking operations."""

    def __init__(self, default_config: Optional[ChunkingConfig] = None):
        """Initialize chunking service with default configuration."""
        self.default_config = default_config or ChunkingConfig()

    def chunk_text(
        self, text: str, config: Optional[ChunkingConfig] = None
    ) -> List[str]:
        """Chunk text using the specified or default configuration."""
        if not text.strip():
            return []

        chunking_config = config or self.default_config
        chunker = ChonkieChunker(chunking_config)
        chunks = chunker.chunk(text)

        # Fallback to original text if no chunks produced
        if not chunks:
            chunks = [text]

        return chunks

    def chunk_document(
        self,
        document: Union[Document, DocumentSchema],
        config: Optional[ChunkingConfig] = None,
    ) -> List[str]:
        """Chunk a document using its metadata for configuration."""
        # Both Document and DocumentSchema now have content_type
        metadata = document.metadata or {}
        content = getattr(document, "raw_content", None) or getattr(document, "content")
        doc_type = document.content_type

        # Extract chunking config from document metadata
        config = config or ChunkingConfig.from_document_metadata(metadata, doc_type)

        print(config)

        return self.chunk_text(content, config)

    def chunk_with_metadata(
        self, text: str, metadata: Dict[str, Any], document_type: DocumentType
    ) -> List[str]:
        """Chunk text using metadata-based configuration."""
        config = ChunkingConfig.from_document_metadata(metadata, document_type)
        return self.chunk_text(text, config)


# Default service instance
default_chunking_service = ChunkingService()


def chunk_text(text: str, config: Optional[ChunkingConfig] = None) -> List[str]:
    """Convenience function for chunking text."""
    return default_chunking_service.chunk_text(text, config)


def chunk_document(document: Union[Document, DocumentSchema]) -> List[str]:
    """Convenience function for chunking a document."""
    return default_chunking_service.chunk_document(document)
