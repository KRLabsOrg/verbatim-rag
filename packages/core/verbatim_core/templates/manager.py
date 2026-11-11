"""Template manager for the Verbatim RAG system.

Provides a unified interface for managing different template strategies,
including mode selection, persistence, and processing workflows.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, cast

from verbatim_core._checks import _config_from_settings
from verbatim_core._deprecation import deprecated, warn_contextual_without_client, warn_deprecated
from verbatim_core._transform_config import TransformConfig
from verbatim_core.errors import ConfigurationError, VerbatimError
from verbatim_core.llm_client import LLMClient
from verbatim_core.templates.base import TemplateStrategy
from verbatim_core.templates.contextual import ContextualTemplate
from verbatim_core.templates.question_specific import QuestionSpecificTemplate
from verbatim_core.templates.random import RandomTemplate
from verbatim_core.templates.static import StaticTemplate
from verbatim_core.templates.structured import StructuredTemplate

logger = logging.getLogger(__name__)


class TemplateManager:
    """Template manager with strategy pattern and mode selection.

    Manages different template strategies and provides a unified interface
    for template generation and filling. Supports persistence of configuration
    across sessions.

    Its settings, `config`, are the default settings of a `Transformer` built with it.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        default_mode: str = "static",
        rag_system: Any = None,  # noqa: ANN401  # duck-typed RAG system; avoids circular import
    ) -> None:
        """Initialize template manager.

        :param llm_client: Optional LLM client for contextual and random modes
        :param default_mode: Default template mode ("static", "contextual", "random", "question_specific", "structured")
        :param rag_system: Optional RAG system for structured mode
        """
        self.llm_client = llm_client
        self.rag_system = rag_system

        # Initialize strategies
        self.strategies: dict[str, TemplateStrategy | None] = {
            "static": StaticTemplate(),
            "contextual": ContextualTemplate(llm_client) if llm_client else None,
            "random": RandomTemplate(llm_client=llm_client),
            "question_specific": QuestionSpecificTemplate(),
            "structured": StructuredTemplate(rag_system=rag_system),
        }

        # Validate initial mode
        mode = default_mode if default_mode in self.strategies else "static"
        if self.strategies[mode] is None:
            warn_contextual_without_client()
            mode = "static"

        self.config = TransformConfig.model_validate({"template_mode": mode})

    @property
    def current_mode(self) -> str:
        """The template mode of `config`."""
        return self.config.template_mode

    @current_mode.setter
    def current_mode(self, mode: str) -> None:
        warn_deprecated("Assigning TemplateManager.current_mode", use=_mode_replacement(mode))
        self._switch_mode(mode)

    @property
    def citation_mode(self) -> str:
        """The citation mode of `config`."""
        return self.config.citation_mode

    @citation_mode.setter
    def citation_mode(self, mode: str) -> None:
        warn_deprecated("Assigning TemplateManager.citation_mode", use="TransformConfig(citation_mode=...)")
        self._set_citations(citation_mode=mode)

    def set_mode(self, mode: str) -> bool:
        """Switch to a different template mode.

        Deprecated: pass `TransformConfig(template_mode=...)` to the `Transformer` or its call.

        :param mode: Template mode to switch to
        :return: True if mode was switched successfully
        """
        warn_deprecated("TemplateManager.set_mode", use=_mode_replacement(mode))
        return self._switch_mode(mode)

    def _switch_mode(self, mode: str) -> bool:
        if mode not in self.strategies:
            logger.warning(
                "Unknown template mode %s. The template mode stays %s.",
                mode,
                self.config.template_mode,
            )
            return False

        if self.strategies[mode] is None:
            warn_contextual_without_client()
            return False

        self._replace_config(template_mode=mode)
        return True

    def _replace_config(self, **settings: object) -> None:
        """Replace settings of `config`.

        :raises ConfigurationError: If a setting is invalid
        """
        # Only the settings set before stay set, so a call can tell them from the defaults.
        self.config = _config_from_settings(TransformConfig, {**self.config.model_dump(exclude_unset=True), **settings})

    def get_current_mode(self) -> str:
        """Get the current template mode.

        :return: Current mode name
        """
        return self.config.template_mode

    def strategy(self, mode: str) -> TemplateStrategy | None:
        """Return the strategy of a template mode, without switching to it.

        :param mode: Template mode
        :return: The strategy, or None if the mode needs an LLM client the manager doesn't have
        :raises KeyError: If there is no such mode
        """
        return self.strategies[mode]

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
        strategy = self._current_strategy()
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

        strategy = self._current_strategy()

        # Use async generation if available
        if hasattr(strategy, "generate_async") and self.config.template_mode == "contextual":
            template = await cast("ContextualTemplate", strategy).generate_async(question, all_spans, citation_count)
        else:
            template = strategy.generate(question, all_spans, citation_count)

        return strategy.fill(template, display_spans, citation_spans)

    def _current_strategy(self) -> TemplateStrategy:
        """Return the strategy of the template mode of `config`.

        Switching modes skips a mode without a strategy, so the mode of `config` has one.
        """
        return cast("TemplateStrategy", self.strategies[self.config.template_mode])

    @staticmethod
    def _get_template_inputs(
        display_spans: list[dict[str, Any]],
        citation_spans: list[dict[str, Any]],
    ) -> tuple[list[str], int]:
        has_linked_citations = any(span.get("citation_ids") for span in display_spans)
        if has_linked_citations:
            return [span["text"] for span in display_spans], 0

        return [span["text"] for span in display_spans + citation_spans], len(citation_spans)

    def get_template(self, question: str = "", spans: list[str] | None = None, citation_count: int = 0) -> str:
        """Generate a template without filling it.

        :param question: The user's question
        :param spans: List of spans that will fill the template
        :param citation_count: Number of citation-only spans
        :return: Template string with placeholders
        """
        spans = spans or []
        strategy = self._current_strategy()
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
        strategy = self._current_strategy()
        return strategy.fill(template, display_spans, citation_spans)

    def save(self, filepath: str) -> None:
        """Save all template configurations to file.

        :param filepath: Path to save configuration
        """
        data = {
            "current_mode": self.config.template_mode,
            "citation_mode": self.config.citation_mode,
            "citation_format": self.config.citation_format,
            "strategies": {},
        }

        # Save state for each available strategy
        for mode, strategy in self.strategies.items():
            if strategy is not None:
                data["strategies"][mode] = strategy.save_state()

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # noqa: PTH103, PTH120  # for a bare file name, Path.parent is "." where dirname is ""

        with Path(filepath).open("w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> bool:
        """Load template configurations from file.

        :param filepath: Path to load configuration from
        :return: True if loaded successfully
        """
        if not os.path.exists(filepath):  # noqa: PTH110  # unlike Path.exists, an unreadable path is "not found"
            logger.warning("Template config file not found: %s", filepath)
            return False

        try:
            with Path(filepath).open() as f:
                data = json.load(f)
        # A file that can't be read, or isn't JSON
        except (OSError, ValueError):
            logger.warning("Could not load the template config %s", filepath, exc_info=True)
            return False

        # Read and check the whole file before any of it reaches the manager, so a file with
        # an invalid shape or setting leaves the manager as it was.
        try:
            settings, strategies_data = _read_saved_settings(data)
            mode = settings.pop("template_mode", None)
            if mode is not None and self.strategies.get(mode) is not None:
                settings["template_mode"] = mode
            new_config = self._settings_config(settings)
        except (VerbatimError, AttributeError, TypeError, ValueError):
            logger.warning("Could not load the template config %s", filepath, exc_info=True)
            return False

        previous_config = self.config
        previous_states = {mode: strategy.save_state() for mode, strategy in self.strategies.items() if strategy}

        try:
            self.config = new_config

            # Load strategy states
            for mode, state in strategies_data.items():
                strategy = self.strategies.get(mode)
                if strategy is not None:
                    # A saved state of the wrong shape doesn't stop the other strategies.
                    try:
                        strategy.load_state(state)
                    except (KeyError, TypeError, ValueError):
                        logger.warning(
                            "Could not load the saved state of the %s templates",
                            mode,
                            exc_info=True,
                        )
                        # Restore this strategy, so a state that failed halfway leaves no part of itself.
                        strategy.load_state(previous_states[mode])

            # Before 0.3.0 only the contextual templates saved the citation format.
            citations = settings
            if citations:
                self._set_citations(**citations)

        # A missing dependency or embedder is a deployment problem, which a different file fixes no more
        # than this one did: the manager keeps its state and the caller sees the error.
        except ConfigurationError:
            self._restore(previous_config, previous_states)
            raise
        except Exception:
            self._restore(previous_config, previous_states)
            logger.warning("Could not load the template config %s", filepath, exc_info=True)
            return False
        else:
            return True

    def _restore(self, config: TransformConfig, states: dict[str, dict[str, Any]]) -> None:
        """Put back the settings and the strategy states a failed `load` found."""
        self.config = config
        for mode, state in states.items():
            strategy = self.strategies.get(mode)
            if strategy is not None:
                try:
                    strategy.load_state(state)
                except (KeyError, TypeError, ValueError):
                    logger.warning("Could not restore the %s templates after a failed load", mode, exc_info=True)

    def _settings_config(self, settings: dict[str, Any]) -> TransformConfig:
        """Return the config the settings of a saved file describe.

        :raises ConfigurationError: If a setting is invalid
        """
        if not settings:
            return self.config
        return _config_from_settings(TransformConfig, {**self.config.model_dump(exclude_unset=True), **settings})

    def info(self) -> dict[str, Any]:
        """Get current template manager state information.

        :return: Dictionary with current state info
        """
        info_data = {
            "current_mode": self.config.template_mode,
            "available_modes": self.get_available_modes(),
            "has_llm_client": self.llm_client is not None,
        }

        # Add mode-specific info
        if self.config.template_mode == "random":
            random_strategy = self.strategies["random"]
            if hasattr(random_strategy, "get_template_count"):
                info_data["random_template_count"] = cast("RandomTemplate", random_strategy).get_template_count()

        return info_data

    # Convenience methods for specific modes

    def use_static_mode(self, template: str | None = None) -> None:
        """Switch to static mode with optional custom template.

        :param template: Optional custom template
        """
        if template:
            static_strategy = StaticTemplate(template)
            self.strategies["static"] = static_strategy

        self._switch_mode("static")

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
            warn_contextual_without_client()
            return False

        if self.strategies["contextual"] is None:
            self.strategies["contextual"] = ContextualTemplate(self.llm_client, citation_mode=self.config.citation_mode)

        contextual_strategy = cast("ContextualTemplate", self.strategies["contextual"])
        contextual_strategy.set_per_fact_mode(use_per_fact)
        contextual_strategy.template_preview_chars = template_preview_chars
        contextual_strategy.preserve_span_newlines = preserve_span_newlines
        contextual_strategy.template_prompt = template_prompt
        contextual_strategy.system_prompt = system_prompt

        return self._switch_mode("contextual")

    def use_random_mode(self, templates: list[str] | None = None) -> bool:
        """Switch to random mode with optional template pool.

        :param templates: Optional list of templates
        :return: True if switched successfully
        """
        if templates:
            random_strategy = RandomTemplate(templates, self.llm_client)
            self.strategies["random"] = random_strategy

        return self._switch_mode("random")

    def generate_random_templates(self, count: int = 10) -> bool:
        """Generate diverse random templates if in random mode.

        :param count: Number of templates to generate
        :return: True if generation was attempted
        """
        if self.config.template_mode != "random":
            logger.warning(
                "Generating templates needs random template mode, not %s",
                self.config.template_mode,
            )
            return False

        random_strategy = self.strategies["random"]
        if hasattr(random_strategy, "generate_diverse_templates"):
            try:
                cast("RandomTemplate", random_strategy).generate_diverse_templates(count)
            except ValueError as e:
                logger.warning("Could not generate templates: %s", e)
            else:
                return True

        return False

    def use_question_specific_mode(
        self,
        templates: list[dict[str, Any]] | None = None,
        *,
        embedder: Any = None,  # noqa: ANN401  # duck-typed embedder with encode(texts), such as a SentenceTransformer
    ) -> bool:
        """Switch to question-specific mode with optional template definitions.

        :param templates: Optional list of template configs, each with a "template" and its
                          "examples": [{
                              "template": "Template with [RELEVANT_SENTENCES]",
                              "examples": ["Example question 1", "Example question 2"]
                          }]
        :param embedder: A loaded embedding model, such as a SentenceTransformer, that embeds the
            examples and questions. See `QuestionSpecificTemplate`.
        :return: True if switched successfully
        """
        if templates or embedder is not None:
            self.strategies["question_specific"] = QuestionSpecificTemplate(templates, embedder=embedder)

        return self._switch_mode("question_specific")

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
        structured_strategy = cast("StructuredTemplate | None", self.strategies.get("structured"))

        if structured_strategy is None:
            structured_strategy = StructuredTemplate(
                rag_system=self.rag_system, citation_mode=self.config.citation_mode
            )
            self.strategies["structured"] = structured_strategy
        else:
            structured_strategy.set_citation_mode(self.config.citation_mode)

        # Update RAG system if not set
        if self.rag_system and not structured_strategy.rag_system:
            structured_strategy.rag_system = self.rag_system

        # Set template if provided
        if template:
            structured_strategy.set_template(template)

        # Add custom mappings if provided
        if placeholder_mappings:
            for placeholder, query in placeholder_mappings.items():
                structured_strategy.add_placeholder_mapping(placeholder, query)

        return self._switch_mode("structured")

    @deprecated("TemplateManager.set_rag_system", use="VerbatimRAG.query_async")
    def set_rag_system(self, rag_system: Any) -> None:  # noqa: ANN401  # duck-typed RAG system
        """Set the RAG system for modes that need it (structured).

        :param rag_system: RAG system instance
        """
        self._attach_rag_system(rag_system)

    def _attach_rag_system(self, rag_system: Any) -> None:  # noqa: ANN401  # duck-typed RAG system
        """Let structured mode call back into the RAG system, until #74 replaces it (ADR 0004)."""
        self.rag_system = rag_system

        # Update strategies that need RAG system
        structured_strategy = cast("StructuredTemplate | None", self.strategies.get("structured"))
        if structured_strategy:
            structured_strategy.rag_system = rag_system

    async def process_structured_async(
        self,
        question: str,
        template: str | None = None,
        placeholder_mappings: dict[str, str] | None = None,
    ) -> str:
        """Run structured extraction, as a convenience helper.

        Note: Prefer calling rag.query_async() directly after setting up
        structured mode. This method is kept for convenience.
        """
        if not self.use_structured_mode(template=template, placeholder_mappings=placeholder_mappings):
            msg = "Structured mode unavailable"
            raise ValueError(msg)

        if not self.rag_system:
            msg = "RAG system not set"
            raise ValueError(msg)

        # Delegate to RAG query which handles structured mode
        response = await self.rag_system.query_async(question)
        return response.answer

    def set_citation_mode(self, mode: str) -> None:
        """Configure how citations are rendered inside filled templates ("inline" or "hidden").

        Deprecated: pass `TransformConfig(citation_mode=...)` to the `Transformer` or its call.
        """
        warn_deprecated("TemplateManager.set_citation_mode", use="TransformConfig(citation_mode=...)")
        allowed = {"inline", "hidden"}
        if mode not in allowed:
            msg = f"Unsupported citation mode: {mode}"
            raise ValueError(msg)

        self._set_citations(citation_mode=mode)

    def set_citation_format(self, citation_format: str) -> None:
        """Configure citation marker format for all strategies.

        Deprecated: pass `TransformConfig(citation_format=...)` to the `Transformer` or its call.

        :param citation_format: str.format template with {number} and {span_id} variables.
            Default "[{number}]". Example: "[{span_id}]" renders "[cite1]" when
            span data includes {"span_id": "cite1"}.
        :raises ConfigurationError: If the format uses other variables
        """
        warn_deprecated("TemplateManager.set_citation_format", use="TransformConfig(citation_format=...)")
        self._set_citations(citation_format=citation_format)

    def _set_citations(self, **settings: str) -> None:
        """Replace the citation settings of `config` and of every strategy.

        `process` fills templates with the settings of the strategies.
        """
        self._replace_config(**settings)

        for strategy in self.strategies.values():
            # Each strategy takes the settings it has a setter for.
            if strategy and "citation_mode" in settings and hasattr(strategy, "set_citation_mode"):
                cast("Any", strategy).set_citation_mode(settings["citation_mode"])
            if strategy and "citation_format" in settings and hasattr(strategy, "set_citation_format"):
                cast("Any", strategy).set_citation_format(settings["citation_format"])


def _read_saved_settings(data: Any) -> tuple[dict[str, Any], dict[str, Any]]:  # noqa: ANN401  # the decoded JSON, checked here
    """Return the settings and the strategy states of a saved file, checked before either is applied.

    :param data: What `json.load` returned for the file
    :return: The settings of `config`, and the state of each template strategy
    :raises ValueError: If the file has a shape `save` never writes
    """
    if not isinstance(data, dict):
        msg = f"a template config must be a JSON object, not {type(data).__name__}"
        raise TypeError(msg)

    strategies_data = data.get("strategies", {})
    if not isinstance(strategies_data, dict):
        msg = "the strategies of a template config must be a JSON object, not "
        raise TypeError(msg + type(strategies_data).__name__)
    for mode, state in strategies_data.items():
        if not isinstance(state, dict):
            msg = f"the saved state of the {mode} templates must be a JSON object, not "
            raise TypeError(msg + type(state).__name__)

    settings: dict[str, Any] = {}
    if "current_mode" in data:
        mode = data["current_mode"]
        if not isinstance(mode, str):
            msg = f"current_mode must be a template mode name, not {type(mode).__name__}"
            raise TypeError(msg)
        settings["template_mode"] = mode

    # Before 0.3.0 only the contextual templates saved the citation format.
    citations = {key: data[key] for key in ("citation_mode", "citation_format") if key in data}
    contextual_state = strategies_data.get("contextual") or {}
    if "citation_format" not in citations and "citation_format" in contextual_state:
        citations["citation_format"] = contextual_state["citation_format"]
    settings.update(citations)
    return settings, strategies_data


def _mode_replacement(mode: str) -> str:
    """Return what replaces switching the templates to a mode."""
    if mode == "structured":
        return "TemplateManager.use_structured_mode()"
    return "TransformConfig(template_mode=...)"
