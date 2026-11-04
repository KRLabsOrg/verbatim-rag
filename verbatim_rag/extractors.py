"""
Compatibility layer: expose extractors from verbatim_core.
"""

from __future__ import annotations

from verbatim_core.extractors import (
    ExtractConfig,
    LLMExtractConfig,
    LLMSpanExtractor,
    ModelExtractConfig,
    ModelSpanExtractor,
    SemanticHighlightConfig,
    SemanticHighlightExtractor,
    SpanExtractor,
)

__all__ = [
    "SpanExtractor",
    "ExtractConfig",
    "ModelSpanExtractor",
    "ModelExtractConfig",
    "LLMSpanExtractor",
    "LLMExtractConfig",
    "SemanticHighlightExtractor",
    "SemanticHighlightConfig",
]
