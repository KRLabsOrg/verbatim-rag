"""
LLM clients for all language-model interactions in the Verbatim RAG system.

This module defines :class:`BaseLLMClient`, the provider-agnostic interface that
holds all span-extraction and template-generation logic, and :class:`LLMClient`,
the OpenAI-backed implementation. Provider-specific clients (e.g. AWS Bedrock)
only need to implement :meth:`BaseLLMClient.complete` and
:meth:`BaseLLMClient.complete_async`.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:
    raise ImportError("OpenAI package required: pip install openai")


class BaseLLMClient(ABC):
    """
    Provider-agnostic LLM interaction interface.

    Concrete subclasses implement the two completion primitives (:meth:`complete`
    and :meth:`complete_async`); every higher-level method (span extraction,
    template generation) is built on top of them and shared across providers.
    """

    def __init__(self, model: str, temperature: float = 0.7):
        """
        :param model: The model identifier to use.
        :param temperature: Default temperature for completions.
        """
        self.model = model
        self.temperature = temperature

    @abstractmethod
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

    @abstractmethod
    async def complete_async(
        self,
        prompt: str,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        system_prompt: str | None = None,
    ) -> str:
        """
        Asynchronous text completion. See :meth:`complete` for parameter docs.
        """

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
        template_preview_chars: Optional[int] = 100,
        preserve_span_newlines: bool = False,
        template_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a contextual template for the given question and spans.

        :param question: The user's question
        :param spans: List of spans that will fill the template
        :param citation_count: Number of citation-only spans
        :param use_per_fact: Whether to use per-span placeholders
        :param template_preview_chars: Max chars per span shown to the template LLM.
            None = no truncation. Default 100 reproduces pre-0.2.8 behaviour.
        :param preserve_span_newlines: Keep newlines in spans shown to the template
            LLM. Default False collapses them (pre-0.2.8 behaviour).
        :param template_prompt: Custom Jinja2 prompt string. Variables available:
            question, n_spans, citation_count, spans_block (per-fact) or
            span_preview (aggregate). None uses the bundled prompt.
        :param system_prompt: System message for the template-generation call.
        :return: Generated template string
        """
        if use_per_fact and len(spans) <= 8:
            prompt = self._build_per_fact_template_prompt(
                question,
                spans,
                citation_count,
                template_preview_chars,
                preserve_span_newlines,
                template_prompt,
            )
        else:
            prompt = self._build_aggregate_template_prompt(
                question,
                spans,
                citation_count,
                template_preview_chars,
                preserve_span_newlines,
                template_prompt,
            )

        try:
            return self.complete(prompt, temperature=self.temperature, system_prompt=system_prompt)
        except Exception as e:
            logger.error("Template generation failed: %s", e)
            return self._fallback_template(citation_count > 0)

    async def generate_template_async(
        self,
        question: str,
        spans: List[str],
        citation_count: int,
        use_per_fact: bool = True,
        template_preview_chars: Optional[int] = 100,
        preserve_span_newlines: bool = False,
        template_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Async template generation. See generate_template for parameter docs.
        """
        if use_per_fact and len(spans) <= 8:
            prompt = self._build_per_fact_template_prompt(
                question,
                spans,
                citation_count,
                template_preview_chars,
                preserve_span_newlines,
                template_prompt,
            )
        else:
            prompt = self._build_aggregate_template_prompt(
                question,
                spans,
                citation_count,
                template_preview_chars,
                preserve_span_newlines,
                template_prompt,
            )

        try:
            return await self.complete_async(
                prompt, temperature=self.temperature, system_prompt=system_prompt
            )
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
        self,
        question: str,
        spans: List[str],
        citation_count: int,
        template_preview_chars: Optional[int] = 100,
        preserve_span_newlines: bool = False,
        template_prompt: Optional[str] = None,
    ) -> str:
        from .prompts import load_prompt, render_prompt

        span_lines = []
        for i, span in enumerate(spans, start=1):
            text = span if preserve_span_newlines else span.replace("\n", " ")
            text = text.strip()
            if template_preview_chars is not None:
                text = text[:template_preview_chars]
            suffix = "..." if template_preview_chars is not None else ""
            span_lines.append(f"{i}. {text}{suffix}")
        spans_block = "\n".join(span_lines)

        ctx = dict(
            question=question,
            n_spans=len(spans),
            spans_block=spans_block,
            citation_count=citation_count,
        )
        if template_prompt is not None:
            return render_prompt(template_prompt, **ctx)
        return load_prompt("template/per_fact", **ctx)

    def _build_aggregate_template_prompt(
        self,
        question: str,
        spans: List[str],
        citation_count: int,
        template_preview_chars: Optional[int] = 100,
        preserve_span_newlines: bool = False,
        template_prompt: Optional[str] = None,
    ) -> str:
        from .prompts import load_prompt, render_prompt

        preview_chars = 50 if template_preview_chars is None else min(50, template_preview_chars)
        preview_spans = []
        for span in spans[:3]:
            text = span if preserve_span_newlines else span.replace("\n", " ")
            preview_spans.append(text[:preview_chars] + "...")
        span_preview = " | ".join(preview_spans)

        ctx = dict(
            question=question,
            n_spans=len(spans),
            span_preview=span_preview,
            citation_count=citation_count,
        )
        if template_prompt is not None:
            return render_prompt(template_prompt, **ctx)
        return load_prompt("template/aggregate", **ctx)

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


class LLMClient(BaseLLMClient):
    """
    Centralized OpenAI interaction handler with async support.

    Provides a unified interface for all OpenAI API calls used throughout
    the Verbatim RAG system, including span extraction and template generation.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        api_base: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        extra_body: dict | None = None,
        max_tokens: int | None = None,
    ):
        """
        Initialize the LLM client.

        :param model: The OpenAI model to use
        :param temperature: Default temperature for completions
        :param api_base: The base URL for the OpenAI API (can be used with custom models and with VLLM)
        :param api_key: Optional API key. Falls back to OPENAI_API_KEY when unset.
        :param extra_body: Provider-specific request fields passed through to the
            chat-completions call (e.g. vLLM ``chat_template_kwargs``).
        :param max_tokens: Optional max completion tokens. Useful for reasoning
            models that otherwise exhaust the context window while thinking.
        """
        super().__init__(model=model, temperature=temperature)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        self.extra_body = extra_body
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(base_url=api_base, api_key=self.api_key)

        self.async_client = openai.AsyncOpenAI(base_url=api_base, api_key=self.api_key)

    def _build_kwargs(
        self,
        messages: list,
        json_mode: bool,
        temperature: Optional[float],
    ) -> dict:
        """Assemble chat-completion kwargs shared by sync and async paths."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        extra_body = dict(self.extra_body or {})
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
            # Guided JSON decoding and Qwen3-style thinking conflict: the grammar
            # forces JSON from the first token while thinking wants a <think>
            # block, leaving message.content empty. Disable thinking for JSON.
            chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
            chat_template_kwargs.setdefault("enable_thinking", False)
            extra_body["chat_template_kwargs"] = chat_template_kwargs
        if extra_body:
            kwargs["extra_body"] = extra_body

        return kwargs

    @staticmethod
    def _content_or_raise(response) -> str:
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise ValueError("LLM returned empty or filtered response")
        return content

    def complete(
        self,
        prompt: str,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        system_prompt: str | None = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs = self._build_kwargs(messages, json_mode, temperature)

        response = self.client.chat.completions.create(**kwargs)
        return self._content_or_raise(response)

    async def complete_async(
        self,
        prompt: str,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        system_prompt: str | None = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs = self._build_kwargs(messages, json_mode, temperature)

        response = await self.async_client.chat.completions.create(**kwargs)
        return self._content_or_raise(response)
