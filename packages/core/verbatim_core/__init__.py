"""
Lightweight, RAG-agnostic core for verbatim transformation.

This subpackage provides:
- VerbatimTransform: question + context -> cited, grounded answer (sync/async)
- RAGProvider (interface): minimal retrieval contract (no index/Milvus deps)
- UniversalDocument: simple context container
- @verbatim_enhance decorator: drop-in enhancement for existing RAG fns
- CLI: `verbatim-enhance` for batch processing JSON(L)
"""

from ._deprecation import VerbatimDeprecationWarning
from .enhance import verbatim_enhance
from .errors import ConfigurationError, InvalidInputError, UpstreamError, VerbatimError
from .providers import RAGProvider
from .transform import VerbatimTransform, verbatim_query, verbatim_query_async
from .types import (
    Answer,
    AnswerCitation,
    Chunk,
    ExtractedSpan,
    Extraction,
    ResultWarning,
    Span,
    Usage,
)
from .universal_document import UniversalDocument

__all__ = [
    "VerbatimTransform",
    "verbatim_query",
    "verbatim_query_async",
    "RAGProvider",
    "UniversalDocument",
    "verbatim_enhance",
    "VerbatimError",
    "InvalidInputError",
    "ConfigurationError",
    "UpstreamError",
    "Chunk",
    "Span",
    "ExtractedSpan",
    "Usage",
    "ResultWarning",
    "Extraction",
    "AnswerCitation",
    "Answer",
    "VerbatimDeprecationWarning",
]
