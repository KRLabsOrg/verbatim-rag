"""
Extractors for identifying relevant spans in documents (RAG-agnostic).

This copy avoids importing vector-store specific types; accepts any objects
with a `.text` attribute as search results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Dict

from .llm_client import LLMClient


class SpanExtractor(ABC):
    @abstractmethod
    def extract_spans(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        pass


class ModelSpanExtractor(SpanExtractor):
    """Not available in verbatim_core. Use verbatim_rag.extractors.ModelSpanExtractor."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ModelSpanExtractor is only available in verbatim_rag.extractors."
        )


class LLMSpanExtractor(SpanExtractor):
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        model: str = "gpt-4o-mini",
        extraction_mode: str = "auto",
        max_display_spans: int = 5,
        batch_size: int = 5,
    ):
        self.llm_client = llm_client or LLMClient(model)
        self.extraction_mode = extraction_mode
        self.max_display_spans = max_display_spans
        self.batch_size = batch_size

    def extract_spans(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        if not search_results:
            return {}
        should_batch = self.extraction_mode == "batch" or (
            self.extraction_mode == "auto" and len(search_results) <= self.batch_size
        )
        if should_batch:
            return self._extract_spans_batch(question, search_results)
        else:
            return self._extract_spans_individual(question, search_results)

    async def extract_spans_async(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        if not search_results:
            return {}
        should_batch = self.extraction_mode == "batch" or (
            self.extraction_mode == "auto" and len(search_results) <= self.batch_size
        )
        if should_batch:
            return await self._extract_spans_batch_async(question, search_results)
        else:
            return await self._extract_spans_individual_async(question, search_results)

    def _extract_spans_batch(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        print("Extracting spans (batch mode)...")
        top_results = search_results[: self.batch_size]
        documents_text = {}
        for i, result in enumerate(top_results):
            documents_text[f"doc_{i}"] = result.text
        try:
            result = self.llm_client.extract_relevant_spans_batch(
                question, documents_text
            )
            return result
        except Exception as e:
            print(f"Batch extraction failed, falling back to individual: {e}")
            return self._extract_spans_individual(question, search_results)

    async def _extract_spans_batch_async(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        print("Extracting spans (batch mode, async)...")
        top_results = search_results[: self.batch_size]
        documents_text = {}
        for i, result in enumerate(top_results):
            documents_text[f"doc_{i}"] = result.text
        try:
            result = await self.llm_client.extract_relevant_spans_batch_async(
                question, documents_text
            )
            return result
        except Exception as e:
            print(f"Batch extraction failed, falling back to individual: {e}")
            return await self._extract_spans_individual_async(question, search_results)

    def _extract_spans_individual(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        results: Dict[str, List[str]] = {}
        for result in search_results:
            doc_text = result.text
            try:
                spans = self.llm_client.extract_relevant_spans(question, doc_text)
                results[doc_text] = spans
            except Exception as e:
                print(f"Failed to extract spans for a document: {e}")
                results[doc_text] = []
        return results

    async def _extract_spans_individual_async(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        results: Dict[str, List[str]] = {}
        for result in search_results:
            doc_text = result.text
            try:
                spans = await self.llm_client.extract_relevant_spans_async(
                    question, doc_text
                )
                results[doc_text] = spans
            except Exception as e:
                print(f"Failed to extract spans for a document: {e}")
                results[doc_text] = []
        return results
