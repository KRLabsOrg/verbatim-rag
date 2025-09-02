"""
Lightweight, RAG-agnostic core for verbatim transformation.

This subpackage provides:
- VerbatimTransform: question + context -> cited, grounded answer (sync/async)
- RAGProvider (interface): minimal retrieval contract (no index/Milvus deps)
- UniversalDocument: simple context container
- @verbatim_enhance decorator: drop-in enhancement for existing RAG fns
- CLI: `verbatim-enhance` for batch processing JSON(L)
"""

from .transform import VerbatimTransform, verbatim_query, verbatim_query_async
from .providers import RAGProvider
from .universal_document import UniversalDocument
from .enhance import verbatim_enhance

__all__ = [
    "VerbatimTransform",
    "verbatim_query",
    "verbatim_query_async",
    "RAGProvider",
    "UniversalDocument",
    "verbatim_enhance",
]
