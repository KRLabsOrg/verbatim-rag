"""
Response builder for constructing QueryResponse objects with highlights and citations.

This module provides the ResponseBuilder class which takes extracted spans and
search results and constructs a complete QueryResponse with proper highlighting
and citation numbering.
"""

from typing import List, Dict, Any, Set, Tuple
from verbatim_core.models import (
    QueryResponse,
    DocumentWithHighlights,
    Highlight,
    Citation,
    StructuredAnswer,
)


class ResponseBuilder:
    """
    Builds structured query responses with highlights and citations.

    Takes search results and extracted spans and creates a complete QueryResponse
    object with proper document highlighting and citation tracking.
    """

    def __init__(self):
        """Initialize the response builder."""
        pass

    def build_response(
        self,
        question: str,
        answer: str,
        search_results: List[Any],
        relevant_spans: Dict[str, List[str]],
        display_span_count: int = None,
    ) -> QueryResponse:
        """
        Build a complete QueryResponse from components.

        :param question: The original question
        :param answer: The generated answer text
        :param search_results: List of search results from the index
        :param relevant_spans: Dict mapping document text to extracted spans
        :param display_span_count: Number of spans to display vs cite-only
        :return: Complete QueryResponse object
        """
        documents_with_highlights = []
        all_citations = []

        current_citation_number = 1

        for result_index, result in enumerate(search_results):
            result_content = getattr(result, "text", "")
            highlights = []

            # Find spans for this document
            spans_for_doc = relevant_spans.get(result_content, [])

            if spans_for_doc:
                # Create highlights for this document
                highlights = self._create_highlights(result_content, spans_for_doc)

                # Create citations for each highlight
                for highlight_index, highlight in enumerate(highlights):
                    # Determine if this should be a display citation or reference-only
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

            # Add document with highlights
            documents_with_highlights.append(
                DocumentWithHighlights(
                    content=result_content,
                    highlights=highlights,
                    title=getattr(result, "title", "")
                    or result.metadata.get("title", ""),
                    source=getattr(result, "source", "")
                    or result.metadata.get("source", ""),
                    metadata=getattr(result, "metadata", {}),
                )
            )

        # Create structured answer with citations
        structured_answer = StructuredAnswer(text=answer, citations=all_citations)

        return QueryResponse(
            question=question,
            answer=answer,
            structured_answer=structured_answer,
            documents=documents_with_highlights,
        )

    def _create_highlights(self, doc_content: str, spans: List[str]) -> List[Highlight]:
        """
        Create highlight objects for spans in document content.

        Uses sophisticated overlap detection to avoid conflicting highlights.

        :param doc_content: The full document text
        :param spans: List of text spans to highlight
        :return: List of Highlight objects
        """
        highlights: List[Highlight] = []
        highlighted_regions: Set[Tuple[int, int]] = set()

        for span in spans:
            # Find all occurrences of this span in the document
            start = 0
            while True:
                start = doc_content.find(span, start)
                if start == -1:
                    break

                end = start + len(span)

                # Check for overlap with existing highlights
                if not self._has_overlap(start, end, highlighted_regions):
                    highlights.append(Highlight(text=span, start=start, end=end))
                    highlighted_regions.add((start, end))

                # Continue searching from the end of current match
                start = end

        return highlights

    def _has_overlap(self, start: int, end: int, regions: Set[Tuple[int, int]]) -> bool:
        """
        Check if a text region overlaps with existing highlighted regions.

        :param start: Start position of new region
        :param end: End position of new region
        :param regions: Set of existing (start, end) tuples
        :return: True if there's overlap, False otherwise
        """
        for region_start, region_end in regions:
            # Check for overlap: new region starts before old ends and ends after old starts
            if start < region_end and end > region_start:
                return True
        return False

    def clean_answer(self, answer: str) -> str:
        """
        Clean up generated answer text.

        Removes common formatting issues and artifacts from LLM generation.

        :param answer: Raw answer text from generation
        :return: Cleaned answer text
        """
        if not answer:
            return ""

        # Remove surrounding quotes if present
        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        elif answer.startswith("'") and answer.endswith("'"):
            answer = answer[1:-1]

        # Convert literal newlines
        answer = answer.replace("\\n", "\n")

        # Clean up multiple spaces
        import re

        answer = re.sub(r" {2,}", " ", answer)

        # Clean up multiple newlines (but preserve paragraph breaks)
        answer = re.sub(r"\n{3,}", "\n\n", answer)

        return answer.strip()
