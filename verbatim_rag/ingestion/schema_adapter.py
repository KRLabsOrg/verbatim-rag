"""
Schema adapter: convert DocumentSchema to pre-chunked Document objects.

Uses the centralized ChunkingService for all chunking operations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from verbatim_rag.schema import DocumentSchema
from verbatim_rag.document import (
    Document,
    DocumentType,
    Chunk,
    ChunkType,
    ProcessedChunk,
)
from verbatim_rag.chunking import ChunkingService


def schema_to_document(
    schema: DocumentSchema,
    document_type: DocumentType = DocumentType.MARKDOWN,
) -> Document:
    """Convert a DocumentSchema into a pre-chunked Document using ChunkingService."""
    # Flatten metadata (exclude core fields), convert datetimes to ISO strings
    base_metadata = schema.model_dump(
        exclude={"id", "title", "source", "content", "metadata"}
    )
    custom_metadata = schema.metadata or {}
    flattened: Dict[str, Any] = {**base_metadata, **custom_metadata}
    for k, v in list(flattened.items()):
        if isinstance(v, datetime):
            flattened[k] = v.isoformat()

    document = Document(
        id=schema.id,
        title=schema.title or "",
        source=schema.source or "",
        content_type=document_type,
        raw_content=schema.content,
        metadata=flattened,
    )

    # Use ChunkingService for all chunking operations
    chunking_service = ChunkingService()
    enhanced_chunks = chunking_service.chunk_document_enhanced(document)

    for i, (original_text, enhanced_content) in enumerate(enhanced_chunks):
        doc_chunk = Chunk(
            document_id=document.id,
            content=original_text,
            chunk_number=i,
            chunk_type=ChunkType.PARAGRAPH,
            metadata=document.metadata.copy(),
        )

        processed = ProcessedChunk(
            chunk_id=doc_chunk.id, enhanced_content=enhanced_content
        )
        doc_chunk.add_processed_chunk(processed)
        document.add_chunk(doc_chunk)

    return document
