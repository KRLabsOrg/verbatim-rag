"""
Schema adapter: convert DocumentSchema to pre-chunked Document objects.

Uses the chunker_providers system for all chunking operations.
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
from verbatim_rag.chunker_providers import MarkdownChunkerProvider


def schema_to_document(
    schema: DocumentSchema,
    document_type: DocumentType = DocumentType.MARKDOWN,
) -> Document:
    """Convert a DocumentSchema into a pre-chunked Document using chunker providers."""
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

    # Use chunker provider for structural chunking
    chunker = MarkdownChunkerProvider()
    chunk_tuples = chunker.chunk(schema.content)

    for i, (raw_text, struct_enhanced) in enumerate(chunk_tuples):
        # Add document metadata to enhanced content
        enhanced_content = _add_document_metadata(struct_enhanced, document)

        doc_chunk = Chunk(
            document_id=document.id,
            content=raw_text,
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


def _add_document_metadata(text: str, doc: Document) -> str:
    """Add document metadata footer to enhanced text."""
    parts = [text, "", "---"]
    parts.append(f"Document: {doc.title or 'Unknown'}")
    parts.append(f"Source: {doc.source or 'Unknown'}")

    if doc.metadata:
        skip_keys = {"user_id", "dataset_id", "userId"}
        for key, value in doc.metadata.items():
            if key not in skip_keys:
                formatted_key = key.replace("_", " ").title()
                parts.append(f"{formatted_key}: {value}")

    return "\n".join(parts)
