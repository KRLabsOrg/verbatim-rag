"""
Document schema system for Verbatim RAG.

Provides a flexible document schema with Pydantic validation and automatic
metadata handling for unknown fields. Content is used for processing but
not stored in document metadata to avoid redundancy.

Users can create their own domain-specific schemas by inheriting from DocumentSchema.
"""

from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Optional, Any, Dict
from datetime import datetime
import uuid


class DocumentSchema(BaseModel):
    """Base document schema - simple and extensible."""

    # Pydantic v2 config
    model_config = ConfigDict(extra="allow")

    # Core fields
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(
        ..., description="Document text content"
    )  # For processing, not stored

    # Common metadata
    title: Optional[str] = Field(None, max_length=500)
    source: Optional[str] = Field(None, description="URL or file path")
    doc_type: Optional[str] = Field(None, description="Document type identifier")
    created_at: datetime = Field(default_factory=datetime.now)

    # Flexible extension point
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def move_unknown_to_metadata(cls, data):
        """Route unknown keys into metadata before validation (Pydantic v2)."""
        if not isinstance(data, dict):
            return data
        known = set(cls.model_fields.keys())
        meta = dict(data.get("metadata") or {})
        for key in list(data.keys()):
            if key not in known:
                meta[key] = data.pop(key)
        if meta:
            data["metadata"] = meta
        return data

    def to_storage_dict(self) -> Dict[str, Any]:
        """Convert to dict for storage (WITHOUT content to avoid redundancy)."""
        data = self.model_dump()
        # Remove content - it goes to chunks collection
        data.pop("content", None)

        # Convert datetime to timestamp for Milvus
        created = data.get("created_at")
        if isinstance(created, datetime):
            data["created_at"] = int(created.timestamp())

        return data

    @classmethod
    def from_url(cls, url: str, title: Optional[str] = None, **kwargs):
        """Create instance with content extracted from URL.

        Args:
            url: The URL to process
            title: Optional title for the document
            **kwargs: Additional fields for the schema (custom fields)

        Returns:
            Instance of the schema class with content from URL
        """
        from verbatim_rag.ingestion.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        content = processor.extract_content_from_url(url)

        return cls(content=content, source=url, title=title, **kwargs)

    @classmethod
    def from_file(cls, file_path: str, title: Optional[str] = None, **kwargs):
        """Create instance with content extracted from file.

        Args:
            file_path: Path to the file to process
            title: Optional title for the document
            **kwargs: Additional fields for the schema (custom fields)

        Returns:
            Instance of the schema class with content from file
        """
        from verbatim_rag.ingestion.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        content = processor.extract_content_from_file(file_path)

        return cls(content=content, source=file_path, title=title, **kwargs)
