"""
Document ingestion module for Verbatim RAG.

Simple integration of docling + chonkie for document processing.
"""

from .document_processor import DocumentProcessor
from .context_enriched_processor import ContextEnrichedProcessor

__all__ = ["DocumentProcessor", "ContextEnrichedProcessor"]
