"""
Document ingestion module for Verbatim RAG.

Simple integration of docling + chonkie for document processing.
"""

from .document_processor import DocumentProcessor
from .schema_adapter import schema_to_document

__all__ = ["DocumentProcessor", "schema_to_document"]
