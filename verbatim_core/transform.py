from __future__ import annotations

from typing import Any, Dict, Iterable, List

# Reuse existing components from verbatim_rag without importing vector/index types
from .extractors import LLMSpanExtractor, SpanExtractor
from .templates import TemplateManager
from .response_builder import ResponseBuilder
from .llm_client import LLMClient
from .providers import RAGProvider


class _ResultView:
    """Minimal view object with the fields extractors expect (text, metadata)."""

    def __init__(
        self,
        text: str,
        metadata: Dict[str, Any] | None = None,
        rid: str = "ctx",
        score: float = 1.0,
    ):
        self.id = rid
        self.text = text
        self.metadata = metadata or {}
        self.score = score


def _coerce_context_to_results(
    context: Iterable[Dict[str, Any] | Any],
) -> List[_ResultView]:
    results: List[_ResultView] = []
    for i, item in enumerate(context):
        # Accept objects with a .text attribute
        if hasattr(item, "text") and isinstance(getattr(item, "text"), str):
            text = getattr(item, "text")
            meta = getattr(item, "metadata", {}) or {}
            results.append(_ResultView(text=text, metadata=meta, rid=f"ctx_{i}"))
            continue
        if not isinstance(item, dict):
            raise TypeError(
                "Each context item must be a dict with 'content' (or 'text')."
            )
        text = item.get("content") or item.get("text")
        if not text or not isinstance(text, str):
            raise ValueError("Context item missing 'content' (or 'text') string field.")
        meta = {
            "title": item.get("title", ""),
            "source": item.get("source", ""),
            **(item.get("metadata") or {}),
        }
        results.append(_ResultView(text=text, metadata=meta, rid=f"ctx_{i}"))
    return results


class VerbatimTransform:
    """RAG-agnostic verbatim transform using existing components (sync/async)."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        extractor: SpanExtractor | None = None,
        template_manager: TemplateManager | None = None,
        max_display_spans: int = 5,
        extraction_mode: str = "auto",
        template_mode: str = "contextual",
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
        search_results = _coerce_context_to_results(list(context))

        relevant_spans = self.extractor.extract_spans(question, search_results)  # type: ignore[arg-type]

        all_spans = []
        for doc_text, spans in relevant_spans.items():
            for span in spans:
                all_spans.append({"text": span, "doc_text": doc_text})
        display_spans = all_spans[: self.max_display_spans]
        citation_spans = all_spans[self.max_display_spans :]

        answer_text = self.template_manager.process(
            question, display_spans, citation_spans
        )
        answer_text = self.response_builder.clean_answer(answer_text)

        return self.response_builder.build_response(
            question=question,
            answer=answer_text,
            search_results=search_results,  # compatible with response builder expectations
            relevant_spans=relevant_spans,
            display_span_count=len(display_spans),
        )

    async def transform_async(
        self,
        question: str,
        context: Iterable[Dict[str, Any]],
        answer: str | None = None,
    ):
        search_results = _coerce_context_to_results(list(context))

        relevant_spans = await self.extractor.extract_spans_async(  # type: ignore[attr-defined]
            question,
            search_results,  # type: ignore[arg-type]
        )

        all_spans = []
        for doc_text, spans in relevant_spans.items():
            for span in spans:
                all_spans.append({"text": span, "doc_text": doc_text})
        display_spans = all_spans[: self.max_display_spans]
        citation_spans = all_spans[self.max_display_spans :]

        answer_text = await self.template_manager.process_async(  # type: ignore[attr-defined]
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
    context = await provider.retrieve_async(question, k=k, filter=filter)
    vt = VerbatimTransform()
    return await vt.transform_async(question=question, context=context, answer=answer)
