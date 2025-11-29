"""
Verbatim RAG - A minimalistic RAG system that prevents hallucination by ensuring all generated content
is explicitly derived from source documents.
"""

__version__ = "0.1.0"

from verbatim_rag.core import VerbatimRAG
from verbatim_rag.document import Document
from verbatim_rag.extractors import LLMSpanExtractor, SpanExtractor
from verbatim_rag.index import VerbatimIndex
from verbatim_rag.models import (
    Citation,
    DocumentWithHighlights,
    Highlight,
    QueryRequest,
    QueryResponse,
    StreamingResponse,
    StreamingResponseType,
    StructuredAnswer,
)
from verbatim_rag.streaming import StreamingRAG
from verbatim_rag.templates import TemplateManager
from verbatim_rag.verbatim_doc import VerbatimDOC
from verbatim_rag.schema import DocumentSchema
from verbatim_rag.transform import VerbatimTransform
from verbatim_rag.transform import verbatim_query, verbatim_query_async
from verbatim_rag.providers import RAGProvider, IndexProvider, VerbatimRAGProvider
from verbatim_rag.universal_document import UniversalDocument

# Optional ingestion module (requires docling + chonkie)
try:
    from verbatim_rag.ingestion import DocumentProcessor

    INGESTION_AVAILABLE = True
except ImportError:
    DocumentProcessor = None
    INGESTION_AVAILABLE = False
