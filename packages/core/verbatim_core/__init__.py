"""Lightweight, RAG-agnostic core for verbatim transformation.

This subpackage provides:
- VerbatimTransform: question + context -> cited, grounded answer (sync/async)
- RAGProvider (interface): minimal retrieval contract (no index/Milvus deps)
- UniversalDocument: simple context container
- @verbatim_enhance decorator: drop-in enhancement for existing RAG fns
- CLI: `verbatim-enhance` for batch processing JSON(L)
"""

from verbatim_core.enhance import verbatim_enhance
from verbatim_core.providers import RAGProvider
from verbatim_core.transform import VerbatimTransform, verbatim_query, verbatim_query_async
from verbatim_core.universal_document import UniversalDocument

__all__ = [
    "RAGProvider",
    "UniversalDocument",
    "VerbatimTransform",
    "verbatim_enhance",
    "verbatim_query",
    "verbatim_query_async",
]
