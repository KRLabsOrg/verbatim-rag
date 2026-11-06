"""Contextual template strategy for the Verbatim RAG system.

Provides LLM-powered template generation that creates contextually appropriate
templates based on the question and available spans. Supports both per-span
and aggregate placeholders.
"""

import logging
from typing import Any

from typing_extensions import override

from verbatim_core._logging import content_logger
from verbatim_core.errors import UpstreamError
from verbatim_core.llm_client import Completion, LLMClient, _acall_old_client, _call_old_client, _replaces
from verbatim_core.templates.base import TemplateResult, TemplateStrategy
from verbatim_core.templates.filler import TemplateFiller
from verbatim_core.types import ResultWarning, Usage

logger = logging.getLogger(__name__)

# Per-span placeholders are used for at most this many spans, aggregate placeholders for more.
_MAX_PER_FACT_SPANS = 8


class ContextualTemplate(TemplateStrategy):
    """Contextual template strategy using LLM generation.

    This strategy generates templates dynamically based on the question,
    available spans, and citation count.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        use_per_fact: bool = True,
        citation_mode: str = "inline",
        citation_format: str = "[{number}]",
        template_preview_chars: int | None = 100,
        preserve_span_newlines: bool = False,
        template_prompt: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize the contextual template strategy.

        Args:
            llm_client: LLM client for template generation.
            use_per_fact: Whether to prefer per-span placeholders (≤8 spans).
            citation_mode: "inline" or "hidden".
            citation_format: Marker format string; see TemplateFiller.
            template_preview_chars: Max chars shown per span in the
                template-generation prompt. None = full span, no truncation.
                Default 100 reproduces pre-0.2.8 behaviour.
            preserve_span_newlines: When False (default), newlines inside
                spans are collapsed to a space before the template LLM sees them.
                Set True to let the template LLM see multi-line structure.
            template_prompt: Custom Jinja2 template string used instead of
                the bundled per_fact.txt / aggregate.txt prompt. Receives the same
                variables: question, n_spans, citation_count, and either
                spans_block (per-fact) or span_preview (aggregate) depending on
                use_per_fact. None → use bundled prompts.
            system_prompt: Optional system message passed to the LLM during
                template generation. None → no system message.

        """
        self.llm_client = llm_client
        self.use_per_fact = use_per_fact
        self.citation_mode = citation_mode
        self.citation_format = citation_format
        self.template_preview_chars = template_preview_chars
        self.preserve_span_newlines = preserve_span_newlines
        self.template_prompt = template_prompt
        self.system_prompt = system_prompt
        self.filler = TemplateFiller(citation_mode=citation_mode, citation_format=citation_format)

        self._template_cache: dict[str, str] = {}
        self._max_cache_size = 100

    def set_citation_mode(self, citation_mode: str) -> None:
        """Set how the filled templates render citations.

        Args:
            citation_mode: "inline" or "hidden".

        """
        self.citation_mode = citation_mode
        self.filler.set_citation_mode(citation_mode)

    def set_citation_format(self, citation_format: str) -> None:
        """Set the citation marker of the filled templates.

        Args:
            citation_format: Marker format string; see TemplateFiller.

        """
        self.citation_format = citation_format
        self.filler.citation_format = citation_format

    def generate(self, question: str, spans: list[str], citation_count: int = 0) -> str:
        """Generate a template with the LLM, cached by the question and the spans.

        Without spans, or when the LLM call fails with an `UpstreamError`, the template is the
        fallback template.

        Args:
            question: The user's question.
            spans: The span texts that will fill the template.
            citation_count: Number of citation-only spans.

        Returns:
            Template string with placeholders.

        """
        if not spans:
            return self._get_fallback_template(citation_count > 0)

        cache_key = self._create_cache_key(question, spans, citation_count)
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]

        try:
            template = self.llm_client.generate_template(**self._template_request(question, spans, citation_count))

            template = self._post_process_template(template, citation_count)
            self._cache_template(cache_key, template)

        except UpstreamError as e:
            _log_fallback(e.code)
            return self._get_fallback_template(citation_count > 0)
        else:
            return template

    async def generate_async(self, question: str, spans: list[str], citation_count: int = 0) -> str:
        """Async version of `generate`."""
        if not spans:
            return self._get_fallback_template(citation_count > 0)

        cache_key = self._create_cache_key(question, spans, citation_count)
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]

        try:
            template = await self.llm_client.generate_template_async(
                **self._template_request(question, spans, citation_count)
            )

            template = self._post_process_template(template, citation_count)
            self._cache_template(cache_key, template)

        except UpstreamError as e:
            _log_fallback(e.code)
            return self._get_fallback_template(citation_count > 0)
        else:
            return template

    @override
    def generate_result(
        self, question: str, spans: list[str], citation_count: int = 0, *, seed: int | None = None
    ) -> TemplateResult:
        """Ask the LLM for a template, without the cache.

        When the LLM call fails with an `UpstreamError`, or the LLM sends an empty template, the
        result is the fallback template with the warning `template_fallback`. An exception of a
        client written before 0.3.0, which replaces `generate_template`, is such a failure. Other
        errors propagate.

        :raises ConfigurationError: If the LLM service rejects the key, the model or the quota
        """
        if not spans:
            return TemplateResult(text=self._get_fallback_template(citation_count > 0))
        request = self._template_request(question, spans, citation_count)
        if _replaces(self.llm_client, "generate_template", "template_completion"):
            try:
                text = _call_old_client(self.llm_client, "generate_template", **request)
            except UpstreamError as error:
                return self._fallback_result(citation_count, error.code)
            return TemplateResult(text=self._post_process_template(text, citation_count))
        try:
            completion = self.llm_client.template_completion(**request)
        except UpstreamError as error:
            return self._fallback_result(citation_count, error.code)
        return self._template_result(completion, spans, citation_count)

    @override
    async def generate_result_async(
        self, question: str, spans: list[str], citation_count: int = 0, *, seed: int | None = None
    ) -> TemplateResult:
        """Async version of `generate_result`."""
        if not spans:
            return TemplateResult(text=self._get_fallback_template(citation_count > 0))
        request = self._template_request(question, spans, citation_count)
        if _replaces(self.llm_client, "generate_template_async", "template_completion_async"):
            try:
                text = await _acall_old_client(self.llm_client, "generate_template_async", **request)
            except UpstreamError as error:
                return self._fallback_result(citation_count, error.code)
            return TemplateResult(text=self._post_process_template(text, citation_count))
        try:
            completion = await self.llm_client.template_completion_async(**request)
        except UpstreamError as error:
            return self._fallback_result(citation_count, error.code)
        return self._template_result(completion, spans, citation_count)

    def _template_request(self, question: str, spans: list[str], citation_count: int) -> dict[str, Any]:
        return {
            "question": question,
            "spans": spans,
            "citation_count": citation_count,
            "use_per_fact": self.use_per_fact and len(spans) <= _MAX_PER_FACT_SPANS,
            "template_preview_chars": self.template_preview_chars,
            "preserve_span_newlines": self.preserve_span_newlines,
            "template_prompt": self.template_prompt,
            "system_prompt": self.system_prompt,
        }

    def _template_result(self, completion: Completion, spans: list[str], citation_count: int) -> TemplateResult:
        units = {
            "llm_calls": 1,
            "spans": len(spans),
            "characters": sum(len(span) for span in spans),
        }
        if completion.input_tokens is not None and completion.output_tokens is not None:
            units.update(input_tokens=completion.input_tokens, output_tokens=completion.output_tokens)
        usage = Usage(
            model_id=getattr(self.llm_client, "model", None),
            model_version=completion.model,
            units=units,
        )
        if not completion.text.strip():
            return self._fallback_result(citation_count, "empty_reply", usage)

        template = self._post_process_template(completion.text, citation_count)
        logger.debug("The LLM wrote a template of %d characters", len(template))
        content_logger.debug("Template: %r", template)
        return TemplateResult(text=template, usage=[usage])

    def _fallback_result(self, citation_count: int, error_code: str, usage: Usage | None = None) -> TemplateResult:
        _log_fallback(error_code)
        warning = ResultWarning(
            code="template_fallback",
            message="The LLM could not write a template, so the answer uses the fallback template.",
        )
        return TemplateResult(
            text=self._get_fallback_template(citation_count > 0),
            usage=[usage] if usage else [],
            warnings=[warning],
        )

    def fill(
        self,
        template: str,
        display_spans: list[dict[str, Any]],
        citation_spans: list[dict[str, Any]],
    ) -> str:
        """Fill the template's placeholders with the spans and their citations.

        Args:
            template: Template string with placeholders.
            display_spans: Spans to display with full text.
            citation_spans: Spans for citation reference only.

        Returns:
            The filled template.

        """
        return self.filler.fill(template, display_spans, citation_spans)

    def save_state(self) -> dict[str, Any]:
        """Return the settings to save, which `load_state` restores.

        The state also names the model and temperature of the LLM client, which `load_state`
        doesn't change.
        """
        return {
            "type": "contextual",
            "use_per_fact": self.use_per_fact,
            "model": self.llm_client.model,
            "temperature": self.llm_client.temperature,
            "citation_format": self.citation_format,
            "template_preview_chars": self.template_preview_chars,
            "preserve_span_newlines": self.preserve_span_newlines,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore the settings of a saved state.

        The model and temperature of the saved state are only logged: they belong to the LLM
        client, which the saved state doesn't change.

        Args:
            state: A state that `save_state` returned.

        """
        self.use_per_fact = state.get("use_per_fact", True)
        if "citation_format" in state:
            self.set_citation_format(state["citation_format"])
        if "template_preview_chars" in state:
            self.template_preview_chars = state["template_preview_chars"]
        if "preserve_span_newlines" in state:
            self.preserve_span_newlines = state["preserve_span_newlines"]
        if "model" in state or "temperature" in state:
            logger.debug(
                "The saved state names the model %s and temperature %s. Set them on the LLM "
                "client, which the saved state doesn't change.",
                state.get("model"),
                state.get("temperature"),
            )

    def set_per_fact_mode(self, use_per_fact: bool) -> None:
        """Choose between per-span and aggregate placeholders, and clear the template cache.

        Args:
            use_per_fact: Whether to prefer per-span placeholders (≤8 spans).

        """
        self.use_per_fact = use_per_fact
        self._template_cache.clear()

    def clear_cache(self) -> None:
        """Clear the cache of generated templates."""
        self._template_cache.clear()

    def _create_cache_key(self, question: str, spans: list[str], citation_count: int) -> str:
        import hashlib

        span_sample = " | ".join(span[:30] for span in spans[:3])
        key_string = (
            f"{question[:100]}|{span_sample}|{len(spans)}|{citation_count}|{self.use_per_fact}"
            f"|{self.template_preview_chars}|{self.preserve_span_newlines}"
        )
        return hashlib.md5(key_string.encode()).hexdigest()[:12]  # noqa: S324  # template cache key, not for security

    def _cache_template(self, key: str, template: str) -> None:
        if len(self._template_cache) >= self._max_cache_size:
            oldest_key = next(iter(self._template_cache))
            del self._template_cache[oldest_key]
        self._template_cache[key] = template

    def _post_process_template(self, template: str, citation_count: int) -> str:
        if not template or not template.strip():
            return self._get_fallback_template(citation_count > 0)

        try:
            self.validate_template(template)
        except ValueError:
            template = self.filler.ensure_placeholder(template)

        if citation_count > 0 and "[CITATION_REFS]" not in template:
            template += "\n\nAdditional relevant information can be found in [CITATION_REFS]."
        elif citation_count == 0 and "[CITATION_REFS]" in template:
            template = template.replace("[CITATION_REFS]", "").strip()

        return template

    def _get_fallback_template(self, has_citations: bool) -> str:
        template = """## Response

Based on the available documents:

[DISPLAY_SPANS]"""

        if has_citations:
            template += "\n\n**Additional References:** [CITATION_REFS]"

        template += "\n\n---\n*These excerpts are taken verbatim from the source documents to ensure accuracy.*"

        return template


def _log_fallback(error_code: str | None) -> None:
    logger.warning(
        "Using the fallback template, because the LLM wrote no template (%s)",
        error_code,
        extra={"event": "fallback_used", "warning_code": "template_fallback"},
    )
