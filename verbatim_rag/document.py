"""
Document classes for the Verbatim RAG system.

This module provides a hierarchical document structure:
- Document: Original user-uploaded content (PDF, text, etc.)
- Chunk: Raw portions of the document after chunking
- ProcessedChunk: Enhanced chunks with metadata and embeddings for retrieval
"""

from datetime import datetime
from enum import Enum
from typing import Any, List, Optional, Dict
from dataclasses import dataclass, field
from pathlib import Path
import uuid


class DocumentType(Enum):
    """Supported document types."""

    PDF = "pdf"
    TXT = "txt"
    HTML = "html"
    MARKDOWN = "markdown"
    DOCX = "docx"
    CSV = "csv"
    JSON = "json"
    WEB_PAGE = "web_page"
    UNKNOWN = "unknown"


class ChunkType(Enum):
    """Types of chunks based on content structure."""

    PARAGRAPH = "paragraph"
    SECTION = "section"
    HEADER = "header"
    TABLE = "table"
    LIST = "list"
    CODE = "code"
    FIGURE = "figure"
    ABSTRACT = "abstract"
    REFERENCE = "reference"
    UNKNOWN = "unknown"


@dataclass
class Document:
    """
    Represents an original document uploaded by the user.

    This is the root-level document that contains all metadata about the original
    source and can be chunked into smaller pieces for processing.
    """

    # Core attributes
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    source: str = ""  # File path, URL, or identifier
    content_type: DocumentType = DocumentType.UNKNOWN
    raw_content: str = ""

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Processing info
    processing_config: Dict[str, Any] = field(default_factory=dict)

    # Relationships
    chunks: List["Chunk"] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Post-initialization processing."""
        if not self.title and self.source:
            self.title = Path(self.source).name

        # Infer content type from source if not provided
        if self.content_type == DocumentType.UNKNOWN and self.source:
            self._infer_content_type()

    def _infer_content_type(self):
        """Infer content type from source path/URL."""
        if not self.source:
            return

        extension = Path(self.source).suffix.lower()
        type_mapping = {
            ".pdf": DocumentType.PDF,
            ".txt": DocumentType.TXT,
            ".html": DocumentType.HTML,
            ".htm": DocumentType.HTML,
            ".md": DocumentType.MARKDOWN,
            ".markdown": DocumentType.MARKDOWN,
            ".docx": DocumentType.DOCX,
            ".csv": DocumentType.CSV,
            ".json": DocumentType.JSON,
        }

        if extension in type_mapping:
            self.content_type = type_mapping[extension]
        elif self.source.startswith(("http://", "https://")):
            self.content_type = DocumentType.WEB_PAGE

    def add_chunk(self, chunk: "Chunk"):
        """Add a chunk to this document."""
        chunk.document_id = self.id
        self.chunks.append(chunk)
        self.updated_at = datetime.now()

    def get_chunk_by_id(self, chunk_id: str) -> Optional["Chunk"]:
        """Get a specific chunk by ID."""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None

    def get_chunks_by_type(self, chunk_type: ChunkType) -> List["Chunk"]:
        """Get all chunks of a specific type."""
        return [chunk for chunk in self.chunks if chunk.chunk_type == chunk_type]

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "content_type": self.content_type.value,
            "raw_content": self.raw_content,
            "metadata": self.metadata,
            "processing_config": self.processing_config,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create document from dictionary representation."""
        # Convert timestamps
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        data["content_type"] = DocumentType(data["content_type"])

        # Convert chunks
        chunks_data = data.pop("chunks", [])
        doc = cls(**data)
        doc.chunks = [Chunk.from_dict(chunk_data) for chunk_data in chunks_data]

        return doc


@dataclass
class Chunk:
    """
    Represents a chunk of text from a document.

    This is the raw chunk after splitting the document, before any processing
    for embeddings or retrieval.
    """

    # Core attributes
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    content: str = ""

    # Position in document
    start_index: int = 0
    end_index: int = 0
    chunk_number: int = 0

    # Content classification
    chunk_type: ChunkType = ChunkType.UNKNOWN

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Relationships
    processed_chunks: List["ProcessedChunk"] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)

    def add_processed_chunk(self, processed_chunk: "ProcessedChunk"):
        """Add a processed version of this chunk."""
        processed_chunk.chunk_id = self.id
        self.processed_chunks.append(processed_chunk)

    def get_processed_chunk_by_id(
        self, processed_id: str
    ) -> Optional["ProcessedChunk"]:
        """Get a specific processed chunk by ID."""
        for pc in self.processed_chunks:
            if pc.id == processed_id:
                return pc
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary representation."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "content": self.content,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "chunk_number": self.chunk_number,
            "chunk_type": self.chunk_type.value,
            "metadata": self.metadata,
            "processed_chunks": [pc.to_dict() for pc in self.processed_chunks],
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create chunk from dictionary representation."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["chunk_type"] = ChunkType(data["chunk_type"])

        # Convert processed chunks
        processed_chunks_data = data.pop("processed_chunks", [])
        chunk = cls(**data)
        chunk.processed_chunks = [
            ProcessedChunk.from_dict(pc_data) for pc_data in processed_chunks_data
        ]

        return chunk


@dataclass
class ProcessedChunk:
    """
    Represents a processed chunk ready for embedding and retrieval.

    This contains the enhanced content with headers, context, and metadata
    optimized for semantic search and retrieval.
    """

    # Core attributes
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    chunk_id: str = ""
    enhanced_content: str = ""  # Content with headers, context, etc.

    # Structure and context
    page_number: Optional[int] = None
    section_title: Optional[str] = None

    # Processing metadata
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

    # Retrieval metadata (for display and citations)
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)

    def get_display_content(self) -> str:
        """Get content optimized for display to users."""
        # For display, we might want to show the original content
        # or enhanced content depending on the use case
        return self.enhanced_content

    def get_retrieval_context(self) -> Dict[str, Any]:
        """Get context information for retrieval and citations."""
        return {
            "page_number": self.page_number,
            "section_title": self.section_title,
            **self.retrieval_metadata,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert processed chunk to dictionary representation."""
        return {
            "id": self.id,
            "chunk_id": self.chunk_id,
            "enhanced_content": self.enhanced_content,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "processing_metadata": self.processing_metadata,
            "retrieval_metadata": self.retrieval_metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessedChunk":
        """Create processed chunk from dictionary representation."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)
