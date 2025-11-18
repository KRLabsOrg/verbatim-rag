"""
Thin compatibility re-exports for templates from verbatim_core.
"""

from verbatim_core.templates import (
    TemplateStrategy,
    StaticTemplate,
    ContextualTemplate,
    RandomTemplate,
    QuestionSpecificTemplate,
    TemplateManager,
    TemplateFiller,
)

__all__ = [
    "TemplateStrategy",
    "StaticTemplate",
    "ContextualTemplate",
    "RandomTemplate",
    "QuestionSpecificTemplate",
    "TemplateManager",
    "TemplateFiller",
]
