"""
Centralized LLM client for all OpenAI interactions in the Verbatim RAG system.

This module provides a unified interface for both synchronous and asynchronous
LLM calls, with specialized methods for span extraction and template generation.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from .errors import ConfigurationError, UpstreamError, VerbatimError

logger = logging.getLogger(__name__)

try:
    import openai
    from openai import (
        APIConnectionError,
        APIError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        NotFoundError,
        PermissionDeniedError,
        RateLimitError,
    )
except ImportError:
    raise ImportError("OpenAI package required: pip install openai")


@dataclass(frozen=True)
class Completion:
    """The text of an LLM reply, with what the provider reported about the call."""

    text: str
    model: str | None = None  # the model name the provider returned
    input_tokens: int | None = None
    output_tokens: int | None = None


def _translate_error(error: APIError) -> VerbatimError:
    """The library's error for an error of the OpenAI client.

    The message leaves out the provider's message, which can quote the API key.
    """
    if isinstance(error, APITimeoutError):
        return UpstreamError(
            "The LLM service did not answer in time. Retry later.",
            code="timeout",
            service="llm",
            retryable=True,
        )
    if isinstance(error, APIConnectionError):
        return UpstreamError(
            "Could not connect to the LLM service. Check the base URL of the LLM client.",
            code="connection_failed",
            service="llm",
            retryable=True,
        )
    if isinstance(error, AuthenticationError):
        return ConfigurationError(
            "The LLM service rejected the API key. Give the LLM client a valid key.",
            code="authentication_failed",
        )
    if isinstance(error, PermissionDeniedError):
        return ConfigurationError(
            "The API key may not use this model. Use a key with access to it, or another model.",
            code="permission_denied",
        )
    if isinstance(error, NotFoundError):
        return ConfigurationError(
            "The LLM service doesn't know the model. Check the model name and the base URL of "
            "the LLM client.",
            code="model_not_found",
        )
    if isinstance(error, RateLimitError) and error.code == "insufficient_quota":
        return ConfigurationError(
            "The account of the API key has no quota left. Add quota or use another key.",
            code="quota_exceeded",
        )
    if isinstance(error, RateLimitError):
        return UpstreamError(
            "The LLM service is limiting the rate of requests. Retry later.",
            code="rate_limited",
            service="llm",
            retryable=True,
            retry_after=_retry_after(error),
        )
    if isinstance(error, APIStatusError) and error.status_code >= 500:
        return UpstreamError(
            f"The LLM service failed with status {error.status_code}. Retry later.",
            code="server_error",
            service="llm",
            retryable=True,
        )
    if isinstance(error, APIStatusError):
        return UpstreamError(
            f"The LLM service rejected the request with status {error.status_code}. Check that "
            "the model supports JSON replies and that the input fits its context window.",
            code="request_rejected",
            service="llm",
        )
    return UpstreamError(
        "The LLM service sent a reply that can't be read. Retry, and check that the base URL of "
        "the LLM client points at an OpenAI-compatible service.",
        code="invalid_reply",
        service="llm",
        retryable=True,
    )


def _retry_after(error: APIStatusError) -> float | None:
    try:
        return float(error.response.headers["retry-after"])
    except (KeyError, ValueError):
        return None


def _completion(response) -> Completion:
    """The completion in a chat completion response."""
    if not response.choices or response.choices[0].message is None:
        raise UpstreamError(
            "The LLM service sent an empty reply. A content filter may have removed it. Retry, "
            "or check the content filter of the LLM service.",
            code="empty_reply",
            service="llm",
        )
    text = response.choices[0].message.content
    if text is None:
        raise UpstreamError(
            "The LLM service sent a reply without text. A content filter may have removed it. "
            "Retry, or check the content filter of the LLM service.",
            code="empty_reply",
            service="llm",
        )
    usage = getattr(response, "usage", None)
    return Completion(
        text=text,
        model=getattr(response, "model", None),
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
    )


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
        :raises UpstreamError: If the call fails or the reply is empty
        :raises ConfigurationError: If the service rejects the key, the model or the quota
        """
        return self.completion(
            prompt, json_mode=json_mode, temperature=temperature, system_prompt=system_prompt
        ).text

    async def complete_async(
        self,
        prompt: str,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        system_prompt: str | None = None,
    ) -> str:
        """
        Asynchronous text completion. See `complete`.
        """
        completion = await self.completion_async(
            prompt, json_mode=json_mode, temperature=temperature, system_prompt=system_prompt
        )
        return completion.text

    def completion(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        system_prompt: str | None = None,
    ) -> Completion:
        """
        Like `complete`, and also returns the model name and the token counts the provider sent.
        """
        request = self._request(prompt, json_mode, temperature, system_prompt)
        try:
            response = self.client.chat.completions.create(**request)
        except APIError as e:
            raise _translate_error(e) from e
        return _completion(response)

    async def completion_async(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        system_prompt: str | None = None,
    ) -> Completion:
        """
        Asynchronous version of `completion`.
        """
        request = self._request(prompt, json_mode, temperature, system_prompt)
        try:
            response = await self.async_client.chat.completions.create(**request)
        except APIError as e:
            raise _translate_error(e) from e
        return _completion(response)

    def _request(
        self,
        prompt: str,
        json_mode: bool,
        temperature: Optional[float],
        system_prompt: str | None,
    ) -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        request = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        return request

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
