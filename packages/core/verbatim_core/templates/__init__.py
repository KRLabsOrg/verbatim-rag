"""Template system (RAG-agnostic copy) mirroring verbatim_rag.templates."""

from verbatim_core.templates.base import TemplateStrategy
from verbatim_core.templates.contextual import ContextualTemplate
from verbatim_core.templates.filler import TemplateFiller
from verbatim_core.templates.manager import TemplateManager
from verbatim_core.templates.question_specific import QuestionSpecificTemplate
from verbatim_core.templates.random import RandomTemplate
from verbatim_core.templates.static import StaticTemplate

__all__ = [
    "ContextualTemplate",
    "QuestionSpecificTemplate",
    "RandomTemplate",
    "StaticTemplate",
    "TemplateFiller",
    "TemplateManager",
    "TemplateStrategy",
]
