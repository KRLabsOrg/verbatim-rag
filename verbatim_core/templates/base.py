"""
Base classes for template strategies in the Verbatim RAG system.

Defines the abstract interface that all template strategies must implement,
including methods for generation, filling, and persistence.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class TemplateStrategy(ABC):
    """
    Abstract base class for all template generation strategies.

    Template strategies are responsible for:
    1. Generating templates with placeholders based on context
    2. Filling templates with actual span content
    3. Saving and loading their configuration state
    """

    @abstractmethod
    def generate(self, question: str, spans: List[str], citation_count: int = 0) -> str:
        """
        Generate a template with placeholders for the given context.

        :param question: The user's question
        :param spans: List of text spans that will fill the template
        :param citation_count: Number of citation-only spans
        :return: Template string with placeholders
        """
        pass

    @abstractmethod
    def fill(
        self,
        template: str,
        display_spans: List[Dict[str, Any]],
        citation_spans: List[Dict[str, Any]],
    ) -> str:
        """
        Fill a template with actual span content.

        :param template: Template string with placeholders
        :param display_spans: Spans to display with full text
        :param citation_spans: Spans for citation reference only
        :return: Filled template with actual content
        """
        pass

    @abstractmethod
    def save_state(self) -> Dict[str, Any]:
        """
        Save the current state of this template strategy.

        :return: Dictionary containing the strategy's configuration
        """
        pass

    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load configuration state into this template strategy.

        :param state: Dictionary containing strategy configuration
        """
        pass

    def validate_template(self, template: str) -> None:
        """
        Validate that a template has required placeholders.

        :param template: Template to validate
        :raises ValueError: If template is invalid
        """
        if not template or not template.strip():
            raise ValueError("Template cannot be empty")

        # Check for at least one acceptable placeholder
        acceptable_placeholders = [
            "[RELEVANT_SENTENCES]",
            "[DISPLAY_SPANS]",
            "[FACT_1]",
        ]

        has_placeholder = any(
            placeholder in template for placeholder in acceptable_placeholders
        )
        if not has_placeholder:
            raise ValueError(
                "Template must contain at least one of: "
                "[RELEVANT_SENTENCES], [DISPLAY_SPANS], or [FACT_1]"
            )
