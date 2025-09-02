"""
Static template strategy for the Verbatim RAG system.

Provides a simple, static template approach that uses predefined templates
without LLM generation. Supports customization and persistence.
"""

from typing import List, Dict, Any
from .base import TemplateStrategy
from .filler import TemplateFiller


class StaticTemplate(TemplateStrategy):
    """
    Static template strategy using predefined templates.

    This strategy provides fast, deterministic template generation without
    requiring LLM calls. Users can customize the template and it supports
    persistence across sessions.
    """

    DEFAULT_TEMPLATE = """## Response

Based on the available documents, here are the key findings:

[DISPLAY_SPANS]

---
*These excerpts are taken verbatim from the source documents to ensure accuracy.*"""

    def __init__(self, template: str = None):
        """
        Initialize static template strategy.

        :param template: Custom template string (uses default if None)
        """
        self.template = template or self.DEFAULT_TEMPLATE
        self.filler = TemplateFiller()
        self.validate_template(self.template)

    def generate(self, question: str, spans: List[str], citation_count: int = 0) -> str:
        """
        Return the static template (no generation needed).

        :param question: User's question (not used for static templates)
        :param spans: List of spans (not used for static templates)
        :param citation_count: Number of citation spans (not used for static templates)
        :return: The static template string
        """
        return self.template

    def fill(
        self,
        template: str,
        display_spans: List[Dict[str, Any]],
        citation_spans: List[Dict[str, Any]],
    ) -> str:
        """
        Fill the template with actual span content.

        :param template: Template string with placeholders
        :param display_spans: Spans to display with full text
        :param citation_spans: Spans for citation reference only
        :return: Filled template
        """
        return self.filler.fill(template, display_spans, citation_spans)

    def save_state(self) -> Dict[str, Any]:
        """
        Save the current template configuration.

        :return: Dictionary with template state
        """
        return {"type": "static", "template": self.template}

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load template configuration from saved state.

        :param state: Dictionary with template configuration
        """
        if "template" in state:
            self.template = state["template"]
            self.validate_template(self.template)

    def set_template(self, template: str) -> None:
        """
        Set a new template string.

        :param template: New template string
        :raises ValueError: If template is invalid
        """
        self.validate_template(template)
        self.template = template

    def get_template(self) -> str:
        """
        Get the current template string.

        :return: Current template
        """
        return self.template

    @classmethod
    def create_simple(cls, intro: str = None, outro: str = None) -> "StaticTemplate":
        """
        Create a simple static template with custom intro/outro.

        :param intro: Optional introduction text
        :param outro: Optional conclusion text
        :return: New StaticTemplate instance
        """
        intro = intro or "Based on the available documents:"
        parts = [intro, "", "[DISPLAY_SPANS]"]

        if outro:
            parts.extend(["", outro])

        template = "\n".join(parts)
        return cls(template)

    @classmethod
    def create_academic(cls) -> "StaticTemplate":
        """
        Create an academic-style template.

        :return: New StaticTemplate instance with academic formatting
        """
        template = """## Literature Review

Based on the available literature:

[DISPLAY_SPANS]

### Summary

These findings provide evidence relevant to the research question."""

        return cls(template)

    @classmethod
    def create_brief(cls) -> "StaticTemplate":
        """
        Create a brief, minimal template.

        :return: New StaticTemplate instance with minimal formatting
        """
        template = """**Key Points:**

[DISPLAY_SPANS]"""

        return cls(template)
