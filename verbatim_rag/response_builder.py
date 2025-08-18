"""
Response builder for structured RAG responses
Centralizes all response formatting logic in the package
"""

from typing import List, Dict, Any, Set, Tuple
from .models import (
    QueryResponse,
    DocumentWithHighlights,
    Highlight,
    Citation,
    StructuredAnswer,
)


class ResponseBuilder:
    """
    Centralizes response building logic in the package
    Prevents duplication between API and package
    """

    def __init__(self):
        pass

    def build_response(
        self,
        question: str,
        answer: str,
        search_results: List[Any],
        relevant_spans: Dict[str, List[str]],
        display_span_count: int | None = None,
    ) -> QueryResponse:
        """
        Build a complete QueryResponse with proper highlighting and citations

        Args:
            question: The user's question
            answer: The generated answer
            search_results: List of search results
            relevant_spans: Dictionary mapping result text to relevant spans

        Returns:
            Complete QueryResponse with highlights and citations
        """
        documents_with_highlights = []
        all_citations = []

        # Process each search result to create highlights and citations
        current_citation_number = 1
        for result_index, result in enumerate(search_results):
            result_content = result.text
            highlights = []

            # Get spans for this result
            result_spans = relevant_spans.get(result_content, [])

            if result_spans:
                # Process spans to create non-overlapping highlights
                highlights = self._create_highlights(result_content, result_spans)

                # Create citations for each highlight
                for highlight_index, highlight in enumerate(highlights):
                    # Determine if this citation should be considered a display fact
                    is_display = (
                        display_span_count is None
                        or current_citation_number <= display_span_count
                    )
                    citation = Citation(
                        text=highlight.text,
                        doc_index=result_index,
                        highlight_index=highlight_index,
                        number=current_citation_number,
                        type="display" if is_display else "reference",
                    )
                    all_citations.append(citation)
                    current_citation_number += 1

            # Create document with highlights
            document_with_highlights = DocumentWithHighlights(
                content=result_content, highlights=highlights
            )
            documents_with_highlights.append(document_with_highlights)

        # Create structured answer
        structured_answer = StructuredAnswer(text=answer, citations=all_citations)

        # Return complete response
        return QueryResponse(
            question=question,
            answer=answer,
            structured_answer=structured_answer,
            documents=documents_with_highlights,
        )

    def _create_highlights(self, doc_content: str, spans: List[str]) -> List[Highlight]:
        """
        Create non-overlapping highlights from spans

        Args:
            doc_content: The document content
            spans: List of text spans to highlight

        Returns:
            List of Highlight objects
        """
        highlights: List[Highlight] = []
        highlighted_regions: Set[Tuple[int, int]] = set()

        # Preserve incoming span order so numbering stays aligned with template assignment
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
                start = end  # advance regardless to avoid infinite loop on zero-length

        return highlights

    def _has_overlap(self, start: int, end: int, regions: Set[Tuple[int, int]]) -> bool:
        """Check if a region overlaps with any existing regions"""
        for region_start, region_end in regions:
            if start <= region_end and end >= region_start:
                return True
        return False

    def clean_answer(self, answer: str) -> str:
        """Clean up the generated answer"""
        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        answer = answer.replace("\\n", "\n")
        return answer
