"""
Contextual template strategy for the Verbatim RAG system.

Provides LLM-powered template generation that creates contextually appropriate
templates based on the question and available spans. Supports both per-fact
and aggregate placeholders.
"""

from typing import List, Dict, Any
from .base import TemplateStrategy
from .filler import TemplateFiller
from verbatim_core.llm_client import LLMClient


class ContextualTemplate(TemplateStrategy):
    """
    Contextual template strategy using LLM generation.

    This strategy generates templates dynamically based on the question,
    available spans, and citation count. It can use either per-fact
    placeholders for better integration or aggregate placeholders for
    larger span sets.
    """

    def __init__(self, llm_client: LLMClient, use_per_fact: bool = True):
        """
        Initialize contextual template strategy.

        :param llm_client: LLM client for template generation
        :param use_per_fact: Whether to prefer per-fact placeholders
        """
        self.llm_client = llm_client
        self.use_per_fact = use_per_fact
        self.filler = TemplateFiller()

        # Cache for generated templates (optional optimization)
        self._template_cache: Dict[str, str] = {}
        self._max_cache_size = 100

    def generate(self, question: str, spans: List[str], citation_count: int = 0) -> str:
        """
        Generate a contextual template based on question and spans.

        :param question: The user's question
        :param spans: List of text spans that will fill the template
        :param citation_count: Number of citation-only spans
        :return: Generated template string
        """
        if not spans:
            return self._get_fallback_template(citation_count > 0)

        # Create cache key
        cache_key = self._create_cache_key(question, spans, citation_count)
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]

        try:
            template = self.llm_client.generate_template(
                question=question,
                spans=spans,
                citation_count=citation_count,
                use_per_fact=self.use_per_fact and len(spans) <= 8,
            )

            # Validate and clean the generated template
            template = self._post_process_template(template, citation_count)

            # Cache the result
            self._cache_template(cache_key, template)

            return template

        except Exception as e:
            print(f"Contextual template generation failed: {e}")
            return self._get_fallback_template(citation_count > 0)

    async def generate_async(
        self, question: str, spans: List[str], citation_count: int = 0
    ) -> str:
        """
        Async version of template generation.

        :param question: The user's question
        :param spans: List of text spans that will fill the template
        :param citation_count: Number of citation-only spans
        :return: Generated template string
        """
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
                use_per_fact=self.use_per_fact and len(spans) <= 8,
            )

            template = self._post_process_template(template, citation_count)
            self._cache_template(cache_key, template)

            return template

        except Exception as e:
            print(f"Async contextual template generation failed: {e}")
            return self._get_fallback_template(citation_count > 0)

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
        Save the current strategy configuration.

        :return: Dictionary with strategy state
        """
        return {
            "type": "contextual",
            "use_per_fact": self.use_per_fact,
            "model": self.llm_client.model,
            "temperature": self.llm_client.temperature,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load strategy configuration from saved state.

        :param state: Dictionary with strategy configuration
        """
        self.use_per_fact = state.get("use_per_fact", True)

        # Note: LLM client configuration is typically set at initialization
        # and not changed during load_state, but we could recreate if needed
        if "model" in state or "temperature" in state:
            print(
                f"Note: LLM client config in saved state (model: {state.get('model')}, "
                f"temp: {state.get('temperature')}) - client must be updated separately"
            )

    def set_per_fact_mode(self, use_per_fact: bool) -> None:
        """
        Enable or disable per-fact placeholder generation.

        :param use_per_fact: Whether to use per-fact placeholders
        """
        self.use_per_fact = use_per_fact
        # Clear cache when mode changes
        self._template_cache.clear()

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._template_cache.clear()

    def _create_cache_key(
        self, question: str, spans: List[str], citation_count: int
    ) -> str:
        """Create a cache key for the given template parameters."""
        # Use a hash of key parameters to create a reasonable cache key
        import hashlib

        # Truncate spans for cache key to avoid huge keys
        span_sample = " | ".join(span[:30] for span in spans[:3])
        key_string = f"{question[:100]}|{span_sample}|{len(spans)}|{citation_count}|{self.use_per_fact}"

        return hashlib.md5(key_string.encode()).hexdigest()[:12]

    def _cache_template(self, key: str, template: str) -> None:
        """Cache a generated template with size limits."""
        if len(self._template_cache) >= self._max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._template_cache))
            del self._template_cache[oldest_key]

        self._template_cache[key] = template

    def _post_process_template(self, template: str, citation_count: int) -> str:
        """
        Post-process generated template to ensure it's valid.

        :param template: Raw generated template
        :param citation_count: Number of citation spans
        :return: Processed template
        """
        if not template or not template.strip():
            return self._get_fallback_template(citation_count > 0)

        # Validate the template has required placeholders
        try:
            self.validate_template(template)
        except ValueError:
            # Add missing placeholder
            template = self.filler.ensure_placeholder(template)

        # Handle citation references
        if citation_count > 0 and "[CITATION_REFS]" not in template:
            template += (
                "\n\nAdditional relevant information can be found in [CITATION_REFS]."
            )
        elif citation_count == 0 and "[CITATION_REFS]" in template:
            template = template.replace("[CITATION_REFS]", "").strip()

        return template

    def _get_fallback_template(self, has_citations: bool) -> str:
        """
        Get a fallback template when generation fails.

        :param has_citations: Whether there are citation-only spans
        :return: Fallback template string
        """
        template = """## Response

Based on the available documents:

[DISPLAY_SPANS]"""

        if has_citations:
            template += "\n\n**Additional References:** [CITATION_REFS]"

        template += "\n\n---\n*These excerpts are taken verbatim from the source documents to ensure accuracy.*"

        return template
