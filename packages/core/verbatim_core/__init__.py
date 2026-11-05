"""Lightweight, RAG-agnostic core for verbatim transformation.

This subpackage provides:
- Transformer: question + chunks -> cited answer built from the chunks' text (sync/async/streaming)
- VerbatimTransform: deprecated, question + context dicts -> QueryResponse over a Transformer
- RAGProvider (interface): minimal retrieval contract (no index/Milvus deps)
- UniversalDocument: simple context container
- @verbatim_enhance decorator: drop-in enhancement for existing RAG fns
- CLI: `verbatim-enhance` for batch processing JSON(L)
"""

from verbatim_core._deprecation import VerbatimDeprecationWarning
from verbatim_core.enhance import verbatim_enhance
from verbatim_core.errors import ConfigurationError, InvalidInputError, UpstreamError, VerbatimError
from verbatim_core.providers import RAGProvider
from verbatim_core.transform import VerbatimTransform, verbatim_query, verbatim_query_async
from verbatim_core.transformer import AnswerEvent, SpansEvent, TransformConfig, Transformer, TransformEvent
from verbatim_core.types import Answer, AnswerCitation, Chunk, ExtractedSpan, Extraction, ResultWarning, Span, Usage
from verbatim_core.universal_document import UniversalDocument

__all__ = [
    "Answer",
    "AnswerCitation",
    "AnswerEvent",
    "Chunk",
    "ConfigurationError",
    "ExtractedSpan",
    "Extraction",
    "InvalidInputError",
    "RAGProvider",
    "ResultWarning",
    "Span",
    "SpansEvent",
    "TransformConfig",
    "TransformEvent",
    "Transformer",
    "UniversalDocument",
    "UpstreamError",
    "Usage",
    "VerbatimDeprecationWarning",
    "VerbatimError",
    "VerbatimTransform",
    "verbatim_enhance",
    "verbatim_query",
    "verbatim_query_async",
]
