"""
Chunking service for text processing in Verbatim RAG.

DEPRECATED: This module is deprecated in favor of verbatim_rag.chunker_providers.
Use ChunkerProvider interface and implementations (MarkdownChunkerProvider,
ChonkieChunkerProvider, SimpleChunkerProvider) instead.

This module is kept for backward compatibility but may be removed in a future version.
"""

import warnings

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
    preserve_headings: bool = True  # Auto-enabled for markdown
    include_metadata: bool = True  # Add metadata to enhanced content

    @classmethod
    def from_document_metadata(
        cls, metadata: Dict[str, Any], document_type: DocumentType
    ) -> "ChunkingConfig":
        """Create chunking config from document metadata with fallbacks."""
        recipe = str(
            metadata.get(
                "chunker_recipe",
                "default" if document_type == DocumentType.TXT else "markdown",
            )
        )

        # Auto-enable hierarchical chunking for markdown documents
        preserve_headings = True if recipe == "markdown" else False

        return cls(
            strategy=ChunkingStrategy(
                str(metadata.get("chunker_type", "recursive")).lower()
            ),
            recipe=recipe,
            lang=str(metadata.get("lang", "en")),
            chunk_size=int(metadata.get("chunk_size", 512)),
            chunk_overlap=int(metadata.get("chunk_overlap", 50)),
            merge_threshold=float(metadata.get("merge_threshold", 0.7)),
            split_threshold=float(metadata.get("split_threshold", 0.3)),
            preserve_headings=preserve_headings,
            include_metadata=bool(metadata.get("include_metadata", True)),
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
            chunker = chonkie.RecursiveChunker.from_recipe(
                self.config.recipe, lang=self.config.lang
            )

            # Wrap with hierarchical tracking if needed
            if self.config.recipe == "markdown" and self.config.preserve_headings:
                try:
                    from verbatim_rag.ingestion.hierarchical_chunker import (
                        HierarchicalWrapper,
                    )

                    chunker = HierarchicalWrapper(chunker)
                except ImportError:
                    print(
                        "Warning: HierarchicalWrapper not found, using regular chunker"
                    )

            return chunker
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

    def chunk_with_metadata(self, text: str):
        """Chunk text and return full chunk objects with metadata."""
        return self._chunker(text)


class ChunkingService:
    """
    Service for handling all text chunking operations.

    DEPRECATED: Use verbatim_rag.chunker_providers.ChunkerProvider instead.
    """

    def __init__(self, default_config: Optional[ChunkingConfig] = None):
        """
        Initialize chunking service with default configuration.

        DEPRECATED: Use verbatim_rag.chunker_providers instead.
        """
        warnings.warn(
            "ChunkingService is deprecated. Use verbatim_rag.chunker_providers "
            "(MarkdownChunkerProvider, ChonkieChunkerProvider, etc.) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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

        return self.chunk_text(content, config)

    def chunk_with_metadata(
        self, text: str, metadata: Dict[str, Any], document_type: DocumentType
    ) -> List[str]:
        """Chunk text using metadata-based configuration."""
        config = ChunkingConfig.from_document_metadata(metadata, document_type)
        return self.chunk_text(text, config)

    def chunk_document_enhanced(
        self,
        document: Union[Document, DocumentSchema],
        config: Optional[ChunkingConfig] = None,
    ) -> List[tuple]:
        """
        Chunk document and return (original_text, enhanced_text) tuples.

        Returns list of (chunk_text, enhanced_content) pairs where:
        - chunk_text: Original chunk text
        - enhanced_content: Text with hierarchical headings + metadata
        """
        # Extract content and metadata
        metadata = document.metadata or {}
        content = getattr(document, "raw_content", None) or getattr(document, "content")
        doc_type = document.content_type

        # Get chunking config
        config = config or ChunkingConfig.from_document_metadata(metadata, doc_type)
        chunker = ChonkieChunker(config)

        # Get chunk objects with metadata
        chunk_objects = chunker.chunk_with_metadata(content)

        # Build enhanced content for each chunk
        enhanced_chunks = []
        for i, chunk in enumerate(chunk_objects):
            original_text = chunk.text

            # Build enhanced content if configured
            if config.preserve_headings or config.include_metadata:
                enhanced_text = self._build_enhanced_content(
                    chunk_text=original_text,
                    heading_path=getattr(chunk, "heading_path", None),
                    document=document,
                    chunk_number=i,
                    config=config,
                )
            else:
                enhanced_text = original_text

            enhanced_chunks.append((original_text, enhanced_text))

        return enhanced_chunks

    def _build_enhanced_content(
        self,
        chunk_text: str,
        heading_path: List[str] = None,
        document: Union[Document, DocumentSchema] = None,
        chunk_number: int = 0,
        config: ChunkingConfig = None,
    ) -> str:
        """Build enhanced content with hierarchical headings and metadata."""
        parts = []

        # Add heading hierarchy if available
        if heading_path and config.preserve_headings:
            parts.extend(heading_path)
            parts.append("")  # Blank line after headings

        # Add original content
        parts.append(chunk_text)

        # Add metadata if enabled
        if config.include_metadata and document:
            parts.append("")  # Blank line before metadata
            parts.append("---")  # Metadata separator

            # Basic metadata
            title = getattr(document, "title", "") or "Unknown Document"
            source = getattr(document, "source", "") or "Unknown Source"

            metadata_lines = [
                f"Document: {title}",
                f"Source: {source}",
            ]

            # Add document metadata
            if document.metadata:
                for key, value in document.metadata.items():
                    # Skip internal chunking metadata and sensitive identifiers
                    skip_keys = {
                        "chunker_type",
                        "chunker_recipe",
                        "lang",
                        "chunk_size",
                        "chunk_overlap",
                        "preserve_headings",
                        "include_metadata",
                        # Sensitive identifiers that should never be embedded
                        "user_id",
                        "userId",
                        "dataset_id",
                    }
                    if key not in skip_keys:
                        formatted_key = key.replace("_", " ").title()
                        metadata_lines.append(f"{formatted_key}: {value}")

            parts.extend(metadata_lines)

        return "\n".join(parts)


# Default service instance
default_chunking_service = ChunkingService()


def chunk_text(text: str, config: Optional[ChunkingConfig] = None) -> List[str]:
    """Convenience function for chunking text."""
    return default_chunking_service.chunk_text(text, config)


def chunk_document(document: Union[Document, DocumentSchema]) -> List[str]:
    """Convenience function for chunking a document."""
    return default_chunking_service.chunk_document(document)
