"""Lightweight, RAG-agnostic core for verbatim transformation.

This subpackage provides:
- VerbatimTransform: question + context -> cited, grounded answer (sync/async)
- UniversalDocument: simple context container
- @verbatim_enhance decorator: drop-in enhancement for existing RAG fns
- CLI: `verbatim-enhance` for batch processing JSON(L)
"""

from verbatim_core.enhance import verbatim_enhance
from verbatim_core.transform import VerbatimTransform
from verbatim_core.universal_document import UniversalDocument

__all__ = [
    "UniversalDocument",
    "VerbatimTransform",
    "verbatim_enhance",
]
