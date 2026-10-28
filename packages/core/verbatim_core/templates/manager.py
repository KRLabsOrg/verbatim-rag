"""Template manager for the Verbatim RAG system.

Provides a unified interface for managing different template strategies,
including mode selection, persistence, and processing workflows.
"""

import contextlib
import json
import os
from typing import TYPE_CHECKING, Any

from verbatim_core.llm_client import LLMClient

from .contextual import ContextualTemplate
from .question_specific import QuestionSpecificTemplate
from .random import RandomTemplate
from .static import StaticTemplate
from .structured import StructuredTemplate

if TYPE_CHECKING:
    from .base import TemplateStrategy


class TemplateManager:
    """Template manager with strategy pattern and mode selection.

    Manages different template strategies and provides a unified interface
    for template generation and filling. Supports persistence of configuration
    across sessions.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        default_mode: str = "static",
        rag_system=None,
    ) -> None:
        """Initialize template manager.

        :param llm_client: Optional LLM client for contextual and random modes
        :param default_mode: Default template mode ("static", "contextual", "random", "question_specific", "structured")
        :param rag_system: Optional RAG system for structured mode
        """
        self.llm_client = llm_client
        self.rag_system = rag_system
        self.current_mode = default_mode
        self.citation_mode = "inline"

        # Initialize strategies
        self.strategies: dict[str, TemplateStrategy] = {
            "static": StaticTemplate(citation_mode=self.citation_mode),
            "contextual": ContextualTemplate(llm_client, citation_mode=self.citation_mode)
            if llm_client
            else None,
            "random": RandomTemplate(llm_client=llm_client, citation_mode=self.citation_mode),
            "question_specific": QuestionSpecificTemplate(citation_mode=self.citation_mode),
            "structured": StructuredTemplate(
                rag_system=rag_system, citation_mode=self.citation_mode,
            ),
        }

        # Validate initial mode
        if self.current_mode not in self.strategies:
            self.current_mode = "static"

        if self.strategies[self.current_mode] is None:
            self.current_mode = "static"

    def set_mode(self, mode: str) -> bool:
        """Switch to a different template mode.

        :param mode: Template mode to switch to
        :return: True if mode was switched successfully
        """
        if mode not in self.strategies:
            return False

        if self.strategies[mode] is None:
            return False

        self.current_mode = mode
        return True

    def get_current_mode(self) -> str:
        """Get the current template mode.

        :return: Current mode name
        """
        return self.current_mode

    def get_available_modes(self) -> list[str]:
        """Get list of available template modes.

        :return: List of mode names that are available
        """
        return [mode for mode, strategy in self.strategies.items() if strategy is not None]

    def process(
        self,
        question: str,
        display_spans: list[dict[str, Any]],
        citation_spans: list[dict[str, Any]],
    ) -> str:
        """Generate and fill a template in one operation.

        :param question: The user's question
        :param display_spans: Spans to display with full text
        :param citation_spans: Spans for citation reference only
        :return: Completed response text
        """
        all_spans, citation_count = self._get_template_inputs(display_spans, citation_spans)

        # Generate template
        strategy = self.strategies[self.current_mode]
        template = strategy.generate(question, all_spans, citation_count)

        # Fill template
        return strategy.fill(template, display_spans, citation_spans)

    async def process_async(
        self,
        question: str,
        display_spans: list[dict[str, Any]],
        citation_spans: list[dict[str, Any]],
    ) -> str:
        """Async version of process for contextual templates.

        :param question: The user's question
        :param display_spans: Spans to display with full text
        :param citation_spans: Spans for citation reference only
        :return: Completed response text
        """
        all_spans, citation_count = self._get_template_inputs(display_spans, citation_spans)

        strategy = self.strategies[self.current_mode]

        # Use async generation if available
        if hasattr(strategy, "generate_async") and self.current_mode == "contextual":
            template = await strategy.generate_async(question, all_spans, citation_count)
        else:
            template = strategy.generate(question, all_spans, citation_count)

        return strategy.fill(template, display_spans, citation_spans)

    @staticmethod
    def _get_template_inputs(
        display_spans: list[dict[str, Any]],
        citation_spans: list[dict[str, Any]],
    ) -> tuple[list[str], int]:
        has_linked_citations = any(span.get("citation_ids") for span in display_spans)
        if has_linked_citations:
            return [span["text"] for span in display_spans], 0

        return [span["text"] for span in display_spans + citation_spans], len(citation_spans)

    def get_template(
        self, question: str = "", spans: list[str] | None = None, citation_count: int = 0,
    ) -> str:
        """Generate a template without filling it.

        :param question: The user's question
        :param spans: List of spans that will fill the template
        :param citation_count: Number of citation-only spans
        :return: Template string with placeholders
        """
        spans = spans or []
        strategy = self.strategies[self.current_mode]
        return strategy.generate(question, spans, citation_count)

    def fill_template(
        self,
        template: str,
        display_spans: list[dict[str, Any]],
        citation_spans: list[dict[str, Any]],
    ) -> str:
        """Fill a template with content.

        :param template: Template string with placeholders
        :param display_spans: Spans to display with full text
        :param citation_spans: Spans for citation reference only
        :return: Filled template
        """
        strategy = self.strategies[self.current_mode]
        return strategy.fill(template, display_spans, citation_spans)

    def save(self, filepath: str) -> None:
        """Save all template configurations to file.

        :param filepath: Path to save configuration
        """
        data = {"current_mode": self.current_mode, "strategies": {}}

        # Save state for each available strategy
        for mode, strategy in self.strategies.items():
            if strategy is not None:
                data["strategies"][mode] = strategy.save_state()

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> bool:
        """Load template configurations from file.

        :param filepath: Path to load configuration from
        :return: True if loaded successfully
        """
        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath) as f:
                data = json.load(f)

            # Load mode
            if "current_mode" in data:
                mode = data["current_mode"]
                if self.strategies.get(mode) is not None:
                    self.current_mode = mode

            # Load strategy states
            strategies_data = data.get("strategies", {})
            for mode, state in strategies_data.items():
                if mode in self.strategies and self.strategies[mode] is not None:
                    with contextlib.suppress(Exception):
                        self.strategies[mode].load_state(state)

            return True

        except Exception:
            return False

    def info(self) -> dict[str, Any]:
        """Get current template manager state information.

        :return: Dictionary with current state info
        """
        info_data = {
            "current_mode": self.current_mode,
            "available_modes": self.get_available_modes(),
            "has_llm_client": self.llm_client is not None,
        }

        # Add mode-specific info
        if self.current_mode == "random":
            random_strategy = self.strategies["random"]
            if hasattr(random_strategy, "get_template_count"):
                info_data["random_template_count"] = random_strategy.get_template_count()

        return info_data

    # Convenience methods for specific modes

    def use_static_mode(self, template: str | None = None) -> None:
        """Switch to static mode with optional custom template.

        :param template: Optional custom template
        """
        if template:
            static_strategy = StaticTemplate(template)
            self.strategies["static"] = static_strategy

        self.set_mode("static")

    def use_contextual_mode(
        self,
        use_per_fact: bool = True,
        template_preview_chars: int | None = 100,
        preserve_span_newlines: bool = False,
        template_prompt: str | None = None,
        system_prompt: str | None = None,
    ) -> bool:
        """Switch to contextual mode with configuration.

        :param use_per_fact: Whether to use per-span placeholders (≤8 spans).
        :param template_preview_chars: Max chars per span in the template-gen prompt.
            None = full span. Default 100 (pre-0.2.8 behaviour).
        :param preserve_span_newlines: Keep newlines in spans shown to the template LLM.
        :param template_prompt: Custom Jinja2 prompt string (see ContextualTemplate).
        :param system_prompt: System message for the template-generation call.
        :return: True if switched successfully
        """
        if not self.llm_client:
            return False

        if self.strategies["contextual"] is None:
            self.strategies["contextual"] = ContextualTemplate(
                self.llm_client, citation_mode=self.citation_mode,
            )

        contextual_strategy = self.strategies["contextual"]
        contextual_strategy.set_per_fact_mode(use_per_fact)
        contextual_strategy.template_preview_chars = template_preview_chars
        contextual_strategy.preserve_span_newlines = preserve_span_newlines
        contextual_strategy.template_prompt = template_prompt
        contextual_strategy.system_prompt = system_prompt

        return self.set_mode("contextual")

    def use_random_mode(self, templates: list[str] | None = None) -> bool:
        """Switch to random mode with optional template pool.

        :param templates: Optional list of templates
        :return: True if switched successfully
        """
        if templates:
            random_strategy = RandomTemplate(templates, self.llm_client)
            self.strategies["random"] = random_strategy

        return self.set_mode("random")

    def generate_random_templates(self, count: int = 10) -> bool:
        """Generate diverse random templates if in random mode.

        :param count: Number of templates to generate
        :return: True if generation was attempted
        """
        if self.current_mode != "random":
            return False

        random_strategy = self.strategies["random"]
        if hasattr(random_strategy, "generate_diverse_templates"):
            try:
                random_strategy.generate_diverse_templates(count)
                return True
            except Exception:
                pass

        return False

    def use_question_specific_mode(
        self, templates: dict[str, dict[str, Any]] | None = None,
    ) -> bool:
        """Switch to question-specific mode with optional template definitions.

        :param templates: Optional dictionary mapping category names to template configs
                          Format: {
                              "category_name": {
                                  "template": "Template with [RELEVANT_SENTENCES]",
                                  "examples": ["Example question 1", "Example question 2"]
                              }
                          }
        :return: True if switched successfully
        """
        if templates:
            question_specific_strategy = QuestionSpecificTemplate()
            question_specific_strategy.set_question_templates(templates)
            self.strategies["question_specific"] = question_specific_strategy

        return self.set_mode("question_specific")

    def use_structured_mode(
        self,
        template: str | None = None,
        placeholder_mappings: dict[str, str] | None = None,
    ) -> bool:
        r"""Switch to structured mode with semantic placeholders.

        :param template: Template with semantic placeholders like [METHODOLOGY], [RESULTS]
        :param placeholder_mappings: Custom placeholder → query mappings
        :return: True if switched successfully

        Example:
            manager.use_structured_mode(
                template="# Analysis\\n## Method\\n[METHODOLOGY]\\n## Results\\n[RESULTS]",
                placeholder_mappings={"THEIR_METHOD": "what method did the baseline use"}
            )

        """
        structured_strategy = self.strategies.get("structured")

        if structured_strategy is None:
            structured_strategy = StructuredTemplate(
                rag_system=self.rag_system, citation_mode=self.citation_mode,
            )
            self.strategies["structured"] = structured_strategy
        else:
            structured_strategy.set_citation_mode(self.citation_mode)

        # Update RAG system if not set
        if self.rag_system and not structured_strategy.rag_system:
            structured_strategy.set_rag_system(self.rag_system)

        # Set template if provided
        if template:
            structured_strategy.set_template(template)

        # Add custom mappings if provided
        if placeholder_mappings:
            for placeholder, query in placeholder_mappings.items():
                structured_strategy.add_placeholder_mapping(placeholder, query)

        return self.set_mode("structured")

    def set_rag_system(self, rag_system) -> None:
        """Set the RAG system for modes that need it (structured).

        :param rag_system: RAG system instance
        """
        self.rag_system = rag_system

        # Update strategies that need RAG system
        if self.strategies.get("structured"):
            self.strategies["structured"].set_rag_system(rag_system)

    async def process_structured_async(
        self,
        question: str,
        template: str | None = None,
        placeholder_mappings: dict[str, str] | None = None,
    ) -> str:
        """Convenience helper to run structured extraction.

        Note: Prefer calling rag.query_async() directly after setting up
        structured mode. This method is kept for convenience.
        """
        if not self.use_structured_mode(
            template=template, placeholder_mappings=placeholder_mappings,
        ):
            msg = "Structured mode unavailable"
            raise ValueError(msg)

        if not self.rag_system:
            msg = "RAG system not set"
            raise ValueError(msg)

        # Delegate to RAG query which handles structured mode
        response = await self.rag_system.query_async(question)
        return response.answer

    def set_citation_mode(self, mode: str) -> None:
        """Configure how citations are rendered inside filled templates ("inline" or "hidden")."""
        allowed = {"inline", "hidden"}
        if mode not in allowed:
            msg = f"Unsupported citation mode: {mode}"
            raise ValueError(msg)

        self.citation_mode = mode

        for strategy in self.strategies.values():
            if strategy and hasattr(strategy, "set_citation_mode"):
                strategy.set_citation_mode(mode)

    def set_citation_format(self, citation_format: str) -> None:
        """Configure citation marker format for all strategies.

        :param citation_format: str.format template with {number} and {span_id} variables.
            Default "[{number}]". Example: "[{span_id}]" renders "[cite1]" when
            span data includes {"span_id": "cite1"}.
        """
        for strategy in self.strategies.values():
            if strategy and hasattr(strategy, "set_citation_format"):
                strategy.set_citation_format(citation_format)
