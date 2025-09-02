"""
Response builder (RAG-agnostic copy) — uses verbatim_rag.models for output types.
"""

from __future__ import annotations

from typing import List, Dict, Any, Set, Tuple
from verbatim_core.models import (
    QueryResponse,
    DocumentWithHighlights,
    Highlight,
    Citation,
    StructuredAnswer,
)


class ResponseBuilder:
    def build_response(
        self,
        question: str,
        answer: str,
        search_results: List[Any],
        relevant_spans: Dict[str, List[str]],
        display_span_count: int | None = None,
    ) -> QueryResponse:
        documents_with_highlights = []
        all_citations = []

        current_citation_number = 1
        for result_index, result in enumerate(search_results):
            result_content = getattr(result, "text", "")
            highlights = []
            spans_for_doc = relevant_spans.get(result_content, [])
            if spans_for_doc:
                highlights = self._create_highlights(result_content, spans_for_doc)
                for highlight_index, highlight in enumerate(highlights):
                    is_display = (
                        display_span_count is None
                        or current_citation_number <= display_span_count
                    )
                    all_citations.append(
                        Citation(
                            text=highlight.text,
                            doc_index=result_index,
                            highlight_index=highlight_index,
                            number=current_citation_number,
                            type="display" if is_display else "reference",
                        )
                    )
                    current_citation_number += 1

            documents_with_highlights.append(
                DocumentWithHighlights(content=result_content, highlights=highlights)
            )

        structured_answer = StructuredAnswer(text=answer, citations=all_citations)
        return QueryResponse(
            question=question,
            answer=answer,
            structured_answer=structured_answer,
            documents=documents_with_highlights,
        )

    def _create_highlights(self, doc_content: str, spans: List[str]) -> List[Highlight]:
        highlights: List[Highlight] = []
        highlighted_regions: Set[Tuple[int, int]] = set()
        for span in spans:
            start = 0
            while True:
                start = doc_content.find(span, start)
                if start == -1:
                    break
                end = start + len(span)
                if not self._has_overlap(start, end, highlighted_regions):
                    highlights.append(Highlight(text=span, start=start, end=end))
                    highlighted_regions.add((start, end))
                start = end
        return highlights

    def _has_overlap(self, start: int, end: int, regions: Set[Tuple[int, int]]) -> bool:
        for s, e in regions:
            if start <= e and end >= s:
                return True
        return False

    def clean_answer(self, answer: str) -> str:
        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        return answer.replace("\\n", "\n")
