"""
Compatibility layer: expose extractors from verbatim_core.
"""

from __future__ import annotations

from verbatim_core.extractors import SpanExtractor, LLMSpanExtractor, ModelSpanExtractor

__all__ = ["SpanExtractor", "ModelSpanExtractor", "LLMSpanExtractor"]
