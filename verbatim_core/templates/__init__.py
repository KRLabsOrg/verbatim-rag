"""
Template system (RAG-agnostic copy) mirroring verbatim_rag.templates.
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
