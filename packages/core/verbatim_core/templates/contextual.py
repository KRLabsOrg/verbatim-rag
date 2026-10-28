"""Contextual template strategy for the Verbatim RAG system.

Provides LLM-powered template generation that creates contextually appropriate
templates based on the question and available spans. Supports both per-span
and aggregate placeholders.
"""

from typing import Any

from verbatim_core.llm_client import LLMClient

from .base import TemplateStrategy
from .filler import TemplateFiller

MAX_PER_FACT_SPANS = 8


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

        :param llm_client: LLM client for template generation.
        :param use_per_fact: Whether to prefer per-span placeholders (≤8 spans).
        :param citation_mode: "inline" or "hidden".
        :param citation_format: Marker format string; see TemplateFiller.
        :param template_preview_chars: Max chars shown per span in the
            template-generation prompt. None = full span, no truncation.
            Default 100 reproduces pre-0.2.8 behaviour.
        :param preserve_span_newlines: When False (default), newlines inside
            spans are collapsed to a space before the template LLM sees them.
            Set True to let the template LLM see multi-line structure.
        :param template_prompt: Custom Jinja2 template string used instead of
            the bundled per_fact.txt / aggregate.txt prompt. Receives the same
            variables: question, n_spans, citation_count, and either
            spans_block (per-fact) or span_preview (aggregate) depending on
            use_per_fact. None → use bundled prompts.
        :param system_prompt: Optional system message passed to the LLM during
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
        """Set the citation mode and propagate it to the filler."""
        self.citation_mode = citation_mode
        self.filler.set_citation_mode(citation_mode)

    def set_citation_format(self, citation_format: str) -> None:
        """Set the citation marker format and propagate it to the filler."""
        self.citation_format = citation_format
        self.filler.citation_format = citation_format

    def generate(self, question: str, spans: list[str], citation_count: int = 0) -> str:
        """Generate a template with the LLM, falling back to a default on failure."""
        if not spans:
            return self._get_fallback_template(citation_count > 0)

        cache_key = self._create_cache_key(question, spans, citation_count)
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]

        try:
            template = self.llm_client.generate_template(
                question=question,
                spans=spans,
                citation_count=citation_count,
                use_per_fact=self.use_per_fact and len(spans) <= MAX_PER_FACT_SPANS,
                template_preview_chars=self.template_preview_chars,
                preserve_span_newlines=self.preserve_span_newlines,
                template_prompt=self.template_prompt,
                system_prompt=self.system_prompt,
            )

            template = self._post_process_template(template, citation_count)
            self._cache_template(cache_key, template)
        except Exception:  # noqa: BLE001  # any LLM/provider failure falls back to a default template
            return self._get_fallback_template(citation_count > 0)
        else:
            return template

    async def generate_async(self, question: str, spans: list[str], citation_count: int = 0) -> str:
        """Generate a template asynchronously, falling back to a default on failure."""
        if not spans:
            return self._get_fallback_template(citation_count > 0)

        cache_key = self._create_cache_key(question, spans, citation_count)
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]

        try:
            template = await self.llm_client.generate_template_async(
                question=question,
                spans=spans,
                citation_count=citation_count,
                use_per_fact=self.use_per_fact and len(spans) <= MAX_PER_FACT_SPANS,
                template_preview_chars=self.template_preview_chars,
                preserve_span_newlines=self.preserve_span_newlines,
                template_prompt=self.template_prompt,
                system_prompt=self.system_prompt,
            )

            template = self._post_process_template(template, citation_count)
            self._cache_template(cache_key, template)
        except Exception:  # noqa: BLE001  # any LLM/provider failure falls back to a default template
            return self._get_fallback_template(citation_count > 0)
        else:
            return template

    def fill(
        self,
        template: str,
        display_spans: list[dict[str, Any]],
        citation_spans: list[dict[str, Any]],
    ) -> str:
        """Fill the template with display and citation spans."""
        return self.filler.fill(template, display_spans, citation_spans)

    def save_state(self) -> dict[str, Any]:
        """Return the serializable state of this strategy."""
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
        """Restore strategy state from a saved dictionary."""
        self.use_per_fact = state.get("use_per_fact", True)
        if "citation_format" in state:
            self.set_citation_format(state["citation_format"])
        if "template_preview_chars" in state:
            self.template_preview_chars = state["template_preview_chars"]
        if "preserve_span_newlines" in state:
            self.preserve_span_newlines = state["preserve_span_newlines"]
        if "model" in state or "temperature" in state:
            pass

    def set_per_fact_mode(self, use_per_fact: bool) -> None:
        """Toggle per-span placeholders and clear the template cache."""
        self.use_per_fact = use_per_fact
        self._template_cache.clear()

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._template_cache.clear()

    def _create_cache_key(self, question: str, spans: list[str], citation_count: int) -> str:
        import hashlib

        span_sample = " | ".join(span[:30] for span in spans[:3])
        key_string = (
            f"{question[:100]}|{span_sample}|{len(spans)}|{citation_count}|{self.use_per_fact}"
            f"|{self.template_preview_chars}|{self.preserve_span_newlines}"
        )
        return hashlib.md5(key_string.encode(), usedforsecurity=False).hexdigest()[:12]

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
