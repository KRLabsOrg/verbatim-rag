"""Base classes for template strategies in the Verbatim RAG system.

Defines the abstract interface that all template strategies must implement,
including methods for generation, filling, and persistence.
"""

from abc import ABC, abstractmethod
from typing import Any


class TemplateStrategy(ABC):
    """Abstract base class for all template generation strategies.

    Template strategies are responsible for:
    1. Generating templates with placeholders based on context
    2. Filling templates with actual span content
    3. Saving and loading their configuration state
    """

    @abstractmethod
    def generate(self, question: str, spans: list[str], citation_count: int = 0) -> str:
        """Generate a template with placeholders for the given context.

        :param question: The user's question
        :param spans: List of text spans that will fill the template
        :param citation_count: Number of citation-only spans
        :return: Template string with placeholders
        """

    @abstractmethod
    def fill(
        self,
        template: str,
        display_spans: list[dict[str, Any]],
        citation_spans: list[dict[str, Any]],
    ) -> str:
        """Fill a template with actual span content.

        :param template: Template string with placeholders
        :param display_spans: Spans to display with full text
        :param citation_spans: Spans for citation reference only
        :return: Filled template with actual content
        """

    @abstractmethod
    def save_state(self) -> dict[str, Any]:
        """Save the current state of this template strategy.

        :return: Dictionary containing the strategy's configuration
        """

    @abstractmethod
    def load_state(self, state: dict[str, Any]) -> None:
        """Load configuration state into this template strategy.

        :param state: Dictionary containing strategy configuration
        """

    def validate_template(self, template: str) -> None:
        """Validate that a template has required placeholders.

        :param template: Template to validate
        :raises ValueError: If template is invalid
        """
        if not template or not template.strip():
            msg = "Template cannot be empty"
            raise ValueError(msg)

        # Check for at least one acceptable placeholder
        acceptable_placeholders = [
            "[RELEVANT_SENTENCES]",
            "[DISPLAY_SPANS]",
            "[SPAN_1]",
            "[FACT_1]",
        ]

        has_placeholder = any(placeholder in template for placeholder in acceptable_placeholders)
        if not has_placeholder:
            msg = (
                "Template must contain at least one of: "
                "[RELEVANT_SENTENCES], [DISPLAY_SPANS], or [SPAN_1]"
            )
            raise ValueError(
                msg,
            )
