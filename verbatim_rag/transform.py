"""
RAG-agnostic verbatim transform.

Takes a question and arbitrary context documents and returns a grounded,
cited answer using the existing extractor + template manager + response builder.

Note: The optional `answer` is currently ignored (planned for retroactive
verbatim conversion in a later iteration).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from verbatim_rag.extractors import LLMSpanExtractor, SpanExtractor
from verbatim_rag.templates import TemplateManager
from verbatim_rag.response_builder import ResponseBuilder
from verbatim_rag.vector_stores import SearchResult
from verbatim_rag.llm_client import LLMClient
from verbatim_rag.providers import RAGProvider


def _coerce_context_to_results(context: Iterable[Dict[str, Any]]) -> List[SearchResult]:
    """Convert a list of lightweight context dicts to SearchResult objects.

    Expected dict keys (flexible):
    - content or text (required)
    - title (optional)
    - source (optional)
    - metadata (optional)
    """
    results: List[SearchResult] = []
    for i, item in enumerate(context):
        if not isinstance(item, dict):
            raise TypeError(
                "Each context item must be a dict with 'content' or 'text'."
            )
        text = item.get("content") or item.get("text")
        if not text or not isinstance(text, str):
            raise ValueError("Context item missing 'content' (or 'text') string field.")
        metadata = {
            "title": item.get("title", ""),
            "source": item.get("source", ""),
            **(item.get("metadata") or {}),
        }
        results.append(
            SearchResult(id=f"ctx_{i}", score=1.0, metadata=metadata, text=text)
        )
    return results


class VerbatimTransform:
    """Stateless transform that produces a verbatim, cited answer from context."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        extractor: SpanExtractor | None = None,
        template_manager: TemplateManager | None = None,
        max_display_spans: int = 5,
        extraction_mode: str = "auto",  # batch | individual | auto
        template_mode: str = "contextual",  # static | contextual | random
    ):
        self.llm_client = llm_client or LLMClient()
        self.extractor = extractor or LLMSpanExtractor(
            llm_client=self.llm_client,
            extraction_mode=extraction_mode,
            max_display_spans=max_display_spans,
        )
        self.template_manager = template_manager or TemplateManager(
            llm_client=self.llm_client, default_mode=template_mode
        )
        self.response_builder = ResponseBuilder()
        self.max_display_spans = max_display_spans

    def transform(
        self,
        question: str,
        context: Iterable[Dict[str, Any]],
        answer: str | None = None,  # Ignored for now
    ):
        """Produce a verbatim answer from question + context (answer ignored)."""
        # 1) Coerce context to SearchResult list
        search_results = _coerce_context_to_results(list(context))

        # 2) Extract spans
        relevant_spans = self.extractor.extract_spans(question, search_results)

        # 3) Split spans into display vs citation-only (preserve extractor order)
        all_spans = []
        for doc_text, spans in relevant_spans.items():
            for span in spans:
                all_spans.append({"text": span, "doc_text": doc_text})
        display_spans = all_spans[: self.max_display_spans]
        citation_spans = all_spans[self.max_display_spans :]

        # 4) Generate answer via template manager
        answer_text = self.template_manager.process(
            question, display_spans, citation_spans
        )
        answer_text = self.response_builder.clean_answer(answer_text)

        # 5) Build structured response
        return self.response_builder.build_response(
            question=question,
            answer=answer_text,
            search_results=search_results,
            relevant_spans=relevant_spans,
            display_span_count=len(display_spans),
        )

    async def transform_async(
        self,
        question: str,
        context: Iterable[Dict[str, Any]],
        answer: str | None = None,
    ):
        """Async version using the extractor/template async APIs."""
        search_results = _coerce_context_to_results(list(context))

        relevant_spans = await self.extractor.extract_spans_async(
            question, search_results
        )

        all_spans = []
        for doc_text, spans in relevant_spans.items():
            for span in spans:
                all_spans.append({"text": span, "doc_text": doc_text})
        display_spans = all_spans[: self.max_display_spans]
        citation_spans = all_spans[self.max_display_spans :]

        answer_text = await self.template_manager.process_async(
            question, display_spans, citation_spans
        )
        answer_text = self.response_builder.clean_answer(answer_text)

        return self.response_builder.build_response(
            question=question,
            answer=answer_text,
            search_results=search_results,
            relevant_spans=relevant_spans,
            display_span_count=len(display_spans),
        )


def verbatim_query(
    provider: RAGProvider,
    question: str,
    k: int = 5,
    filter: str | None = None,
    answer: str | None = None,
) -> Any:
    """Convenience: retrieve via provider and apply verbatim transform.

    Returns the same QueryResponse structure as VerbatimRAG.query.
    """
    context = provider.retrieve(question, k=k, filter=filter)
    vt = VerbatimTransform()
    return vt.transform(question=question, context=context, answer=answer)


async def verbatim_query_async(
    provider: RAGProvider,
    question: str,
    k: int = 5,
    filter: str | None = None,
    answer: str | None = None,
) -> Any:
    """Async convenience: provider.retrieve_async + transform_async."""
    context = await provider.retrieve_async(question, k=k, filter=filter)
    vt = VerbatimTransform()
    return await vt.transform_async(question=question, context=context, answer=answer)
