"""Lightweight, RAG-agnostic core for verbatim transformation.

This subpackage provides:
- VerbatimTransform: question + context -> cited, grounded answer (sync/async)
- RAGProvider (interface): minimal retrieval contract (no index/Milvus deps)
- UniversalDocument: simple context container
- @verbatim_enhance decorator: drop-in enhancement for existing RAG fns
- CLI: `verbatim-enhance` for batch processing JSON(L)
"""

from .enhance import verbatim_enhance
from .providers import RAGProvider
from .transform import VerbatimTransform, verbatim_query, verbatim_query_async
from .universal_document import UniversalDocument

__all__ = [
    "RAGProvider",
    "UniversalDocument",
    "VerbatimTransform",
    "verbatim_enhance",
    "verbatim_query",
    "verbatim_query_async",
]
