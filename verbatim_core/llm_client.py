"""
Centralized LLM client for all OpenAI interactions in the Verbatim RAG system.

This module provides a unified interface for both synchronous and asynchronous
LLM calls, with specialized methods for span extraction and template generation.
"""

import json
import os
from typing import Optional, Dict, List

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
    ):
        """
        Initialize the LLM client.

        :param model: The OpenAI model to use
        :param temperature: Default temperature for completions
        :param api_base: The base URL for the OpenAI API (can be used with custom models and with VLLM)
        """
        self.model = model
        self.temperature = temperature
        self.api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"
        self.client = openai.OpenAI(base_url=api_base, api_key=self.api_key)

        self.async_client = openai.AsyncOpenAI(base_url=api_base, api_key=self.api_key)

    def complete(
        self, prompt: str, json_mode: bool = False, temperature: Optional[float] = None
    ) -> str:
        """
        Synchronous text completion.

        :param prompt: The prompt to send
        :param json_mode: Whether to request JSON output format
        :param temperature: Override default temperature
        :return: The completion text
        """
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    async def complete_async(
        self, prompt: str, json_mode: bool = False, temperature: Optional[float] = None
    ) -> str:
        """
        Asynchronous text completion.

        :param prompt: The prompt to send
        :param json_mode: Whether to request JSON output format
        :param temperature: Override default temperature
        :return: The completion text
        """
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.async_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def extract_spans(
        self, question: str, documents: Dict[str, str]
    ) -> Dict[str, List[str]]:
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
            print(f"Span extraction failed: {e}")
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
            print(f"Async span extraction failed: {e}")
            return {doc_id: [] for doc_id in documents.keys()}

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
            prompt = self._build_per_fact_template_prompt(
                question, spans, citation_count
            )
        else:
            prompt = self._build_aggregate_template_prompt(
                question, spans, citation_count
            )

        try:
            return self.complete(prompt, temperature=self.temperature)
        except Exception as e:
            print(f"Template generation failed: {e}")
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
            prompt = self._build_per_fact_template_prompt(
                question, spans, citation_count
            )
        else:
            prompt = self._build_aggregate_template_prompt(
                question, spans, citation_count
            )

        try:
            return await self.complete_async(prompt, temperature=self.temperature)
        except Exception as e:
            print(f"Async template generation failed: {e}")
            return self._fallback_template(citation_count > 0)

    def _build_extraction_prompt(self, question: str, documents: Dict[str, str]) -> str:
        """Build the prompt for batch span extraction."""
        return f"""Extract EXACT verbatim text spans from multiple documents that answer the question.

# Rules
1. Extract **only** text that explicitly addresses the question
2. Never paraphrase, modify, or add to the original text
3. Preserve original wording, capitalization, and punctuation
4. Order spans within each document by relevance - MOST RELEVANT FIRST
5. Include complete sentences or paragraphs for context

# Output Format
Return a JSON object mapping document IDs to span arrays ordered by relevance:
{{
  "doc_0": ["most relevant span", "next most relevant span"],
  "doc_1": ["most relevant from doc 1"],
  "doc_2": []
}}

If no relevant information in a document, use empty array.

# Your Task
Question: {question}

Documents:
{json.dumps(documents, indent=2)}

Extract verbatim spans from each document:"""

    def _build_per_fact_template_prompt(
        self, question: str, spans: List[str], citation_count: int
    ) -> str:
        """Build prompt for per-fact template generation."""
        span_lines = []
        for i, span in enumerate(spans, start=1):
            clean = span.replace("\n", " ").strip()[:100]  # Truncate for prompt size
            span_lines.append(f"{i}. {clean}...")
        spans_block = "\n".join(span_lines)

        return f"""Generate a response template for this Q&A scenario:

Question: {question}

Content that will be inserted into the template:
- Total verbatim facts to show (display facts): {len(spans)}
- Full list of verbatim facts:
{spans_block}
- Additional citation-only facts (only numbers, no text shown): {citation_count}

Template strategy rules:
- Use per-fact placeholders [FACT_1]..[FACT_{len(spans)}] each exactly once.
- If citation-only facts exist, you MAY place [CITATION_REFS] exactly once where their numbers should appear, otherwise omit it.

Instructions:
- Intro: 1 concise sentence tying question to facts.
- Then present each fact in a structured way (bulleted list or numbered list). Each list item should contain exactly one placeholder at the start after a bold label you infer or a generic label (e.g. Fact 3) if unsure.
- DO NOT invent content beyond connective phrases; never summarize or paraphrase inside placeholders.
- No duplicate placeholders; no placeholder inside a heading alone.
- Avoid leading a bullet list with another nested bullet list.

Template requirements:
- Use only placeholders plus minimal connective prose (no actual span text).
- {"Include [CITATION_REFS] once" if citation_count > 0 else "Do NOT include [CITATION_REFS]"}.
- End without extra commentary like "Hope this helps".

Return ONLY the template text (no explanation)."""

    def _build_aggregate_template_prompt(
        self, question: str, spans: List[str], citation_count: int
    ) -> str:
        """Build prompt for aggregate template generation."""
        span_preview = " | ".join(span[:50] + "..." for span in spans[:3])

        return f"""Generate a response template for this Q&A scenario (OUTPUT MUST BE VALID GITHUB-FLAVORED MARKDOWN):

Question: {question}

Content that will be inserted into the template:
- Total verbatim facts to show (display facts): {len(spans)}
- Preview of content: {span_preview}
- Additional citation-only facts (only numbers, no text shown): {citation_count}

Template strategy rules (Markdown correctness is critical):
- Use [DISPLAY_SPANS] exactly once for the aggregate of all verbatim spans.
- If citation-only facts exist, you MAY place [CITATION_REFS] exactly once where their numbers should appear, otherwise omit it.

Markdown formatting requirements:
- Use only GitHub-Flavored Markdown (GFM): headings (##, ###), paragraphs, bullet/numbered lists, bold/italic, blockquotes, and tables.
- Do NOT wrap the entire template in code fences.
- Every heading must be followed by a blank line unless immediately followed by a list.
- Placeholders must not be inside backticks, code blocks, or HTML tags.

Instructions:
- Intro: 1 concise sentence tying question to spans.
- Provide a section header then include the aggregate placeholder.
- Do NOT invent or paraphrase span content; placeholders stand in for verbatim content only.
- Avoid nested lists; keep structure shallow and clean.

Template requirements:
- Must contain [DISPLAY_SPANS].
- {"Include [CITATION_REFS] once" if citation_count > 0 else "Do NOT include [CITATION_REFS]"}.
- End without extra commentary like "Hope this helps".

Return ONLY the template text (no explanation)."""

    def _fallback_template(self, has_citations: bool = False) -> str:
        """Return a simple fallback template when generation fails."""
        template = """## Response

Based on the available documents:

[DISPLAY_SPANS]"""

        if has_citations:
            template += "\n\n**Additional References:** [CITATION_REFS]"

        template += "\n\n---\n*These excerpts are taken verbatim from the source documents to ensure accuracy.*"

        return template

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

    async def extract_relevant_spans_async(
        self, question: str, document_text: str
    ) -> List[str]:
        result = await self.extract_relevant_spans_batch_async(
            question, {"doc": document_text}
        )
        return result.get("doc", [])

    # Template generation (simple compatibility methods)
    def simple_complete(self, prompt: str) -> str:
        return self.complete(prompt)

    async def simple_complete_async(self, prompt: str) -> str:
        return await self.complete_async(prompt)
