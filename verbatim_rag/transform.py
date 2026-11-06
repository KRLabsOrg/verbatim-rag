"""
Moved to verbatim_core.transform. This module re-exports its names and warns when they are used.
"""

from typing import Any

import verbatim_core.transform as _core_transform
from verbatim_core._deprecation import warn_deprecated

# What replaces each name, which is itself deprecated
_REPLACEMENTS = {
    "VerbatimTransform": "Transformer",
    "verbatim_query": "RAGProvider.retrieve, then Transformer.transform",
    "verbatim_query_async": "RAGProvider.retrieve_async, then Transformer.atransform",
}

# A star import gives the names that the module defined, which __getattr__ serves
__all__ = list(_REPLACEMENTS)


def __getattr__(name: str) -> Any:
    # The module also imported SearchResult and verbatim_core's classes, which code may import
    # from it.
    if name == "SearchResult":
        from verbatim_rag.vector_stores import SearchResult

        warn_deprecated(
            "verbatim_rag.transform.SearchResult", use="verbatim_rag.vector_stores.SearchResult"
        )
        return SearchResult
    if name.startswith("_") or not hasattr(_core_transform, name):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warn_deprecated(
        f"verbatim_rag.transform.{name}",
        use=_REPLACEMENTS.get(name, f"verbatim_core.transform.{name}"),
    )
    return getattr(_core_transform, name)
