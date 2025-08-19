"""
Template manager for the Verbatim RAG system.

Provides a unified interface for managing different template strategies,
including mode selection, persistence, and processing workflows.
"""

import json
import os
from typing import Dict, Any, List, Optional
from .base import TemplateStrategy
from .static import StaticTemplate
from .contextual import ContextualTemplate
from .random import RandomTemplate
from ..llm_client import LLMClient


class TemplateManager:
    """
    Template manager with strategy pattern and mode selection.

    Manages different template strategies and provides a unified interface
    for template generation and filling. Supports persistence of configuration
    across sessions.
    """

    def __init__(
        self, llm_client: Optional[LLMClient] = None, default_mode: str = "static"
    ):
        """
        Initialize template manager.

        :param llm_client: Optional LLM client for contextual and random modes
        :param default_mode: Default template mode ("static", "contextual", "random")
        """
        self.llm_client = llm_client
        self.current_mode = default_mode

        # Initialize strategies
        self.strategies: Dict[str, TemplateStrategy] = {
            "static": StaticTemplate(),
            "contextual": ContextualTemplate(llm_client) if llm_client else None,
            "random": RandomTemplate(llm_client=llm_client),
        }

        # Validate initial mode
        if self.current_mode not in self.strategies:
            self.current_mode = "static"

        if self.strategies[self.current_mode] is None:
            print(
                f"Warning: {self.current_mode} mode requires LLM client, falling back to static"
            )
            self.current_mode = "static"

    def set_mode(self, mode: str) -> bool:
        """
        Switch to a different template mode.

        :param mode: Template mode to switch to
        :return: True if mode was switched successfully
        """
        if mode not in self.strategies:
            print(f"Unknown template mode: {mode}")
            return False

        if self.strategies[mode] is None:
            print(f"Mode {mode} is not available (requires LLM client)")
            return False

        self.current_mode = mode
        return True

    def get_current_mode(self) -> str:
        """
        Get the current template mode.

        :return: Current mode name
        """
        return self.current_mode

    def get_available_modes(self) -> List[str]:
        """
        Get list of available template modes.

        :return: List of mode names that are available
        """
        return [
            mode for mode, strategy in self.strategies.items() if strategy is not None
        ]

    def process(
        self,
        question: str,
        display_spans: List[Dict[str, Any]],
        citation_spans: List[Dict[str, Any]],
    ) -> str:
        """
        Generate and fill a template in one operation.

        :param question: The user's question
        :param display_spans: Spans to display with full text
        :param citation_spans: Spans for citation reference only
        :return: Completed response text
        """
        # Extract span texts for template generation
        all_spans = [span["text"] for span in display_spans + citation_spans]
        citation_count = len(citation_spans)

        # Generate template
        strategy = self.strategies[self.current_mode]
        template = strategy.generate(question, all_spans, citation_count)

        # Fill template
        return strategy.fill(template, display_spans, citation_spans)

    async def process_async(
        self,
        question: str,
        display_spans: List[Dict[str, Any]],
        citation_spans: List[Dict[str, Any]],
    ) -> str:
        """
        Async version of process for contextual templates.

        :param question: The user's question
        :param display_spans: Spans to display with full text
        :param citation_spans: Spans for citation reference only
        :return: Completed response text
        """
        all_spans = [span["text"] for span in display_spans + citation_spans]
        citation_count = len(citation_spans)

        strategy = self.strategies[self.current_mode]

        # Use async generation if available
        if hasattr(strategy, "generate_async") and self.current_mode == "contextual":
            template = await strategy.generate_async(
                question, all_spans, citation_count
            )
        else:
            template = strategy.generate(question, all_spans, citation_count)

        return strategy.fill(template, display_spans, citation_spans)

    def get_template(
        self, question: str = "", spans: List[str] = None, citation_count: int = 0
    ) -> str:
        """
        Generate a template without filling it.

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
        display_spans: List[Dict[str, Any]],
        citation_spans: List[Dict[str, Any]],
    ) -> str:
        """
        Fill a template with content.

        :param template: Template string with placeholders
        :param display_spans: Spans to display with full text
        :param citation_spans: Spans for citation reference only
        :return: Filled template
        """
        strategy = self.strategies[self.current_mode]
        return strategy.fill(template, display_spans, citation_spans)

    def save(self, filepath: str) -> None:
        """
        Save all template configurations to file.

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
        """
        Load template configurations from file.

        :param filepath: Path to load configuration from
        :return: True if loaded successfully
        """
        if not os.path.exists(filepath):
            print(f"Template config file not found: {filepath}")
            return False

        try:
            with open(filepath, "r") as f:
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
                    try:
                        self.strategies[mode].load_state(state)
                    except Exception as e:
                        print(f"Warning: Failed to load state for {mode} strategy: {e}")

            return True

        except Exception as e:
            print(f"Failed to load template config: {e}")
            return False

    def info(self) -> Dict[str, Any]:
        """
        Get current template manager state information.

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
                info_data["random_template_count"] = (
                    random_strategy.get_template_count()
                )

        return info_data

    # Convenience methods for specific modes

    def use_static_mode(self, template: str = None) -> None:
        """
        Switch to static mode with optional custom template.

        :param template: Optional custom template
        """
        if template:
            static_strategy = StaticTemplate(template)
            self.strategies["static"] = static_strategy

        self.set_mode("static")

    def use_contextual_mode(self, use_per_fact: bool = True) -> bool:
        """
        Switch to contextual mode with configuration.

        :param use_per_fact: Whether to use per-fact placeholders
        :return: True if switched successfully
        """
        if not self.llm_client:
            print("Contextual mode requires LLM client")
            return False

        if self.strategies["contextual"] is None:
            self.strategies["contextual"] = ContextualTemplate(self.llm_client)

        contextual_strategy = self.strategies["contextual"]
        contextual_strategy.set_per_fact_mode(use_per_fact)

        return self.set_mode("contextual")

    def use_random_mode(self, templates: List[str] = None) -> bool:
        """
        Switch to random mode with optional template pool.

        :param templates: Optional list of templates
        :return: True if switched successfully
        """
        if templates:
            random_strategy = RandomTemplate(templates, self.llm_client)
            self.strategies["random"] = random_strategy

        return self.set_mode("random")

    def generate_random_templates(self, count: int = 10) -> bool:
        """
        Generate diverse random templates if in random mode.

        :param count: Number of templates to generate
        :return: True if generation was attempted
        """
        if self.current_mode != "random":
            print("Must be in random mode to generate templates")
            return False

        random_strategy = self.strategies["random"]
        if hasattr(random_strategy, "generate_diverse_templates"):
            try:
                random_strategy.generate_diverse_templates(count)
                return True
            except Exception as e:
                print(f"Template generation failed: {e}")

        return False
