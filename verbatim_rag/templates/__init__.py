"""
Template system for the Verbatim RAG system.

This module provides a strategy pattern implementation for generating and
filling templates with extracted spans. Supports static templates,
contextual LLM-generated templates, and random template selection.
"""

from .base import TemplateStrategy
from .static import StaticTemplate
from .contextual import ContextualTemplate
from .random import RandomTemplate
from .manager import TemplateManager
from .filler import TemplateFiller

__all__ = [
    "TemplateStrategy",
    "StaticTemplate",
    "ContextualTemplate",
    "RandomTemplate",
    "TemplateManager",
    "TemplateFiller",
]
