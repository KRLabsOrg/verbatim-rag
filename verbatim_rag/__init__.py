"""
Verbatim RAG - A minimalistic RAG system that prevents hallucination by ensuring all generated content
is explicitly derived from source documents.
"""

__version__ = "0.1.0"

from verbatim_rag.core import VerbatimRAG
from verbatim_rag.document import Document
from verbatim_rag.extractors import LLMSpanExtractor, SpanExtractor
from verbatim_rag.index import VerbatimIndex
from verbatim_rag.loader import DocumentLoader
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
from verbatim_rag.template_manager import TemplateManager
from verbatim_rag.text_splitter import TextSplitter
