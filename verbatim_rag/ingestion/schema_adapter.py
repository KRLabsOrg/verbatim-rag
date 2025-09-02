"""
Schema adapter: convert DocumentSchema to pre-chunked Document objects.

Uses chonkie for text chunking. Docling is NOT required here because content
is provided as raw text. For file/URL parsing use DocumentProcessor instead.
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


def schema_to_document(
    schema: DocumentSchema,
    document_type: DocumentType = DocumentType.MARKDOWN,
) -> Document:
    """Convert a DocumentSchema into a pre-chunked Document using chonkie.

    Per-document chunking overrides can be supplied via `schema.metadata`:
    - chunker_type: recursive | token | sentence | word | sdpm
    - chunker_recipe: used by recursive (e.g., "markdown")
    - lang, chunk_size, chunk_overlap, merge_threshold, split_threshold
    """
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

    # Build chunker from metadata overrides
    try:
        import chonkie
    except ImportError as e:
        raise ImportError(
            "Text chunking requires chonkie. Install with: pip install chonkie"
        ) from e

    meta = document.metadata or {}
    chunker_type = str(meta.get("chunker_type", "recursive")).lower()
    chunker_recipe = str(
        meta.get(
            "chunker_recipe",
            "markdown" if document_type == DocumentType.MARKDOWN else "default",
        )
    )
    lang = str(meta.get("lang", "en"))
    chunk_size = int(meta.get("chunk_size", 512))
    chunk_overlap = int(meta.get("chunk_overlap", 50))
    merge_threshold = float(meta.get("merge_threshold", 0.7))
    split_threshold = float(meta.get("split_threshold", 0.3))

    if chunker_type == "recursive":
        chunker = chonkie.RecursiveChunker.from_recipe(chunker_recipe, lang=lang)
    elif chunker_type == "token":
        chunker = chonkie.TokenChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    elif chunker_type == "sentence":
        chunker = chonkie.SentenceChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    elif chunker_type == "word":
        chunker = chonkie.WordChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    elif chunker_type == "sdpm":
        chunker = chonkie.SDPMChunker(
            chunk_size=chunk_size,
            merge_threshold=merge_threshold,
            split_threshold=split_threshold,
        )
    else:
        chunker = chonkie.RecursiveChunker.from_recipe(chunker_recipe, lang=lang)

    chunks = chunker(schema.content)

    for i, c in enumerate(chunks):
        doc_chunk = Chunk(
            document_id=document.id,
            content=c.text,
            chunk_number=i,
            chunk_type=ChunkType.PARAGRAPH,
            metadata=document.metadata.copy(),
        )
        processed = ProcessedChunk(chunk_id=doc_chunk.id, enhanced_content=c.text)
        doc_chunk.add_processed_chunk(processed)
        document.add_chunk(doc_chunk)

    return document
