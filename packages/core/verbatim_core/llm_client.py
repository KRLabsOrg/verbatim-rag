"""
Centralized LLM client for all OpenAI interactions in the Verbatim RAG system.

This module provides a unified interface for both synchronous and asynchronous
LLM calls, with specialized methods for span extraction and template generation.
"""

import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:
    raise ImportError("OpenAI package required: pip install openai")


class LLMClient:
    """
    Centralized LLM interaction handler with async support.

    Provides a unified interface for all OpenAI API calls used throughout
    the Verbatim RAG system, including span extraction and template generation.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        api_base: str = "https://api.openai.com/v1",
        api_key: str | None = None,
    ):
        """
        Initialize the LLM client.

        :param model: The OpenAI model to use
        :param temperature: Default temperature for completions
        :param api_base: The base URL for the OpenAI API (can be used with custom models and with VLLM)
        :param api_key: Optional API key. Falls back to OPENAI_API_KEY when unset.
        """
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        self.client = openai.OpenAI(base_url=api_base, api_key=self.api_key)

        self.async_client = openai.AsyncOpenAI(base_url=api_base, api_key=self.api_key)

    def complete(
        self,
        prompt: str,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        system_prompt: str | None = None,
    ) -> str:
        """
        Synchronous text completion.

        :param prompt: The prompt to send
        :param json_mode: Whether to request JSON output format
        :param temperature: Override default temperature
        :param system_prompt: Optional system message to prepend
        :return: The completion text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        if not response.choices or response.choices[0].message is None:
            raise ValueError("LLM returned empty or filtered response")
        return response.choices[0].message.content

    async def complete_async(
        self,
        prompt: str,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        system_prompt: str | None = None,
    ) -> str:
        """
        Asynchronous text completion.

        :param prompt: The prompt to send
        :param json_mode: Whether to request JSON output format
        :param temperature: Override default temperature
        :param system_prompt: Optional system message to prepend
        :return: The completion text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.async_client.chat.completions.create(**kwargs)
        if not response.choices or response.choices[0].message is None:
            raise ValueError("LLM returned empty or filtered response")
        return response.choices[0].message.content

    def extract_spans(self, question: str, documents: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Specialized method for span extraction from documents.

        :param question: The user's question
        :param documents: Dictionary mapping doc IDs to document text
        :return: Dictionary mapping doc IDs to lists of extracted spans
        """
        prompt = self._build_extraction_prompt(question, documents)
        try:
            response = self.complete(prompt, json_mode=True)
            return json.loads(response)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Span extraction failed: %s", e)
            # Return empty results for all documents on failure
            return {doc_id: [] for doc_id in documents.keys()}

    async def extract_spans_async(
        self, question: str, documents: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """
        Async span extraction from documents.

        :param question: The user's question
        :param documents: Dictionary mapping doc IDs to document text
        :return: Dictionary mapping doc IDs to lists of extracted spans
        """
        prompt = self._build_extraction_prompt(question, documents)
        try:
            response = await self.complete_async(prompt, json_mode=True)
            return json.loads(response)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Async span extraction failed: %s", e)
            return {doc_id: [] for doc_id in documents.keys()}

    def extract_structured(
        self,
        question: str,
        template: str,
        placeholders: Dict[str, str],
        documents: List[str],
    ) -> Dict[str, List[Dict[str, any]]]:
        """
        Extract spans organized by template placeholders with document attribution.

        :param question: The user's question
        :param template: Template with placeholders like [METHODOLOGY]
        :param placeholders: Dict mapping placeholder names to hints
        :param documents: List of document texts
        :return: Dict mapping placeholder names to lists of {text, doc} objects
        """
        prompt = self._build_structured_extraction_prompt(
            question, template, placeholders, documents
        )
        try:
            response = self.complete(prompt, json_mode=True)
            return self._normalize_structured_response(json.loads(response), placeholders)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Structured extraction failed: %s", e)
            return {name: [] for name in placeholders.keys()}

    async def extract_structured_async(
        self,
        question: str,
        template: str,
        placeholders: Dict[str, str],
        documents: List[str],
    ) -> Dict[str, List[Dict[str, any]]]:
        """
        Async version of structured extraction with document attribution.

        :param question: The user's question
        :param template: Template with placeholders like [METHODOLOGY]
        :param placeholders: Dict mapping placeholder names to hints
        :param documents: List of document texts
        :return: Dict mapping placeholder names to lists of {text, doc} objects
        """
        prompt = self._build_structured_extraction_prompt(
            question, template, placeholders, documents
        )
        try:
            response = await self.complete_async(prompt, json_mode=True)
            return self._normalize_structured_response(json.loads(response), placeholders)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Structured extraction failed: %s", e)
            return {name: [] for name in placeholders.keys()}

    def _normalize_structured_response(
        self, response: Dict, placeholders: Dict[str, str]
    ) -> Dict[str, List[Dict[str, any]]]:
        """
        Normalize LLM response to ensure consistent format.

        Handles both old format (list of strings) and new format (list of {text, doc}).
        """
        result = {}
        for name in placeholders.keys():
            items = response.get(name, [])
            normalized = []
            for item in items:
                if isinstance(item, str):
                    # Old format - just text, no doc attribution
                    normalized.append({"text": item, "doc": 0})
                elif isinstance(item, dict) and "text" in item:
                    # New format with doc attribution
                    normalized.append({"text": item["text"], "doc": item.get("doc", 0)})
            result[name] = normalized
        return result

    def _build_structured_extraction_prompt(
        self,
        question: str,
        template: str,
        placeholders: Dict[str, str],
        documents: List[str],
    ) -> str:
        """Build prompt for structured extraction with document attribution."""
        from .prompts import load_prompt

        placeholder_spec = "\n".join(f"- {name}: {hint}" for name, hint in placeholders.items())
        docs_text = "\n\n---\n\n".join(f"[Document {i}]\n{doc}" for i, doc in enumerate(documents))

        return load_prompt(
            "extraction/structured",
            question=question,
            template=template,
            placeholder_spec=placeholder_spec,
            docs_text=docs_text,
        )

    def generate_template(
        self,
        question: str,
        spans: List[str],
        citation_count: int,
        use_per_fact: bool = True,
    ) -> str:
        """
        Generate a contextual template for the given question and spans.

        :param question: The user's question
        :param spans: List of spans that will fill the template
        :param citation_count: Number of citation-only spans
        :param use_per_fact: Whether to use per-fact placeholders
        :return: Generated template string
        """
        if use_per_fact and len(spans) <= 8:
            prompt = self._build_per_fact_template_prompt(question, spans, citation_count)
        else:
            prompt = self._build_aggregate_template_prompt(question, spans, citation_count)

        try:
            return self.complete(prompt, temperature=self.temperature)
        except Exception as e:
            logger.error("Template generation failed: %s", e)
            return self._fallback_template(citation_count > 0)

    async def generate_template_async(
        self,
        question: str,
        spans: List[str],
        citation_count: int,
        use_per_fact: bool = True,
    ) -> str:
        """
        Async template generation.

        :param question: The user's question
        :param spans: List of spans that will fill the template
        :param citation_count: Number of citation-only spans
        :param use_per_fact: Whether to use per-fact placeholders
        :return: Generated template string
        """
        if use_per_fact and len(spans) <= 8:
            prompt = self._build_per_fact_template_prompt(question, spans, citation_count)
        else:
            prompt = self._build_aggregate_template_prompt(question, spans, citation_count)

        try:
            return await self.complete_async(prompt, temperature=self.temperature)
        except Exception as e:
            logger.error("Async template generation failed: %s", e)
            return self._fallback_template(citation_count > 0)

    def _build_extraction_prompt(self, question: str, documents: Dict[str, str]) -> str:
        """Build the prompt for batch span extraction."""
        from .prompts import load_prompt

        return load_prompt(
            "extraction/default",
            question=question,
            documents=json.dumps(documents, indent=2),
        )

    def _build_per_fact_template_prompt(
        self, question: str, spans: List[str], citation_count: int
    ) -> str:
        """Build prompt for per-fact template generation."""
        from .prompts import load_prompt

        span_lines = []
        for i, span in enumerate(spans, start=1):
            clean = span.replace("\n", " ").strip()[:100]  # Truncate for prompt size
            span_lines.append(f"{i}. {clean}...")
        spans_block = "\n".join(span_lines)

        return load_prompt(
            "template/per_fact",
            question=question,
            n_spans=len(spans),
            spans_block=spans_block,
            citation_count=citation_count,
        )

    def _build_aggregate_template_prompt(
        self, question: str, spans: List[str], citation_count: int
    ) -> str:
        """Build prompt for aggregate template generation."""
        from .prompts import load_prompt

        span_preview = " | ".join(span[:50] + "..." for span in spans[:3])

        return load_prompt(
            "template/aggregate",
            question=question,
            n_spans=len(spans),
            span_preview=span_preview,
            citation_count=citation_count,
        )

    def _fallback_template(self, has_citations: bool = False) -> str:
        """Return a simple fallback template when generation fails."""
        from .prompts import load_prompt

        return load_prompt("template/fallback", has_citations=has_citations)

    # Batch span extraction API (for compatibility)
    def extract_relevant_spans_batch(
        self, question: str, documents: Dict[str, str]
    ) -> Dict[str, List[str]]:
        return self.extract_spans(question, documents)

    async def extract_relevant_spans_batch_async(
        self, question: str, documents: Dict[str, str]
    ) -> Dict[str, List[str]]:
        return await self.extract_spans_async(question, documents)

    # Single-doc convenience
    def extract_relevant_spans(self, question: str, document_text: str) -> List[str]:
        result = self.extract_relevant_spans_batch(question, {"doc": document_text})
        return result.get("doc", [])

    async def extract_relevant_spans_async(self, question: str, document_text: str) -> List[str]:
        result = await self.extract_relevant_spans_batch_async(question, {"doc": document_text})
        return result.get("doc", [])

    # Template generation (simple compatibility methods)
    def simple_complete(self, prompt: str) -> str:
        return self.complete(prompt)

    async def simple_complete_async(self, prompt: str) -> str:
        return await self.complete_async(prompt)
