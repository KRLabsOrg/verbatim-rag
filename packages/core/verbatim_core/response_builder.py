"""Response builder for constructing QueryResponse objects with highlights and citations.

This module provides the ResponseBuilder class which takes extracted spans and
search results and constructs a complete QueryResponse with proper highlighting
and citation numbering.
"""

import re
from collections.abc import Sequence
from typing import Any

from verbatim_core._checks import _check_unique_ids
from verbatim_core.errors import InvalidInputError
from verbatim_core.models import (
    Citation,
    DocumentWithHighlights,
    Highlight,
    QueryResponse,
    StructuredAnswer,
)
from verbatim_core.types import Answer, AnswerCitation, Chunk


class ResponseBuilder:
    """Builds structured query responses with highlights and citations.

    Takes search results and extracted spans and creates a complete QueryResponse
    object with proper document highlighting and citation tracking.
    """

    def __init__(self) -> None:
        """Initialize the response builder."""

    def build_response(
        self,
        question: str,
        answer: str,
        search_results: list[Any],
        relevant_spans: dict[str, list[str]],
        display_span_count: int | None = None,
    ) -> QueryResponse:
        """Build a complete QueryResponse from components.

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
                    is_display = display_span_count is None or current_citation_number <= display_span_count

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
                    title=getattr(result, "title", "") or result.metadata.get("title", ""),
                    source=getattr(result, "source", "") or result.metadata.get("source", ""),
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

    def build_from_answer(self, question: str, answer: Answer, chunks: Sequence[Chunk]) -> QueryResponse:
        """Build a QueryResponse from an answer and the chunks it was written from.

        Each citation's span is highlighted at its offsets, without searching for its text. When
        it overlaps a span cited before it in the same chunk, the earlier one keeps the highlight,
        and the later one gets neither a highlight nor a citation in the response.

        :param question: The original question
        :param answer: The answer, whose citations point at spans of the chunks
        :param chunks: The chunks the answer was written from, in the order of the documents
        :return: Complete QueryResponse object
        :raises InvalidInputError: If two chunks share an id, or a citation points outside them
        """
        documents, citations = _documents_and_citations(answer.citations, chunks)
        return QueryResponse(
            question=question,
            answer=answer.text,
            structured_answer=StructuredAnswer(text=answer.text, citations=citations),
            documents=documents,
            warnings=answer.warnings,
        )

    def _create_highlights(self, doc_content: str, spans: list[str]) -> list[Highlight]:
        """Create highlight objects for spans in document content.

        Uses sophisticated overlap detection to avoid conflicting highlights.

        :param doc_content: The full document text
        :param spans: List of text spans to highlight
        :return: List of Highlight objects
        """
        highlights: list[Highlight] = []
        highlighted_regions: set[tuple[int, int]] = set()

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

    def _has_overlap(self, start: int, end: int, regions: set[tuple[int, int]]) -> bool:
        """Check if a text region overlaps with existing highlighted regions.

        :param start: Start position of new region
        :param end: End position of new region
        :param regions: Set of existing (start, end) tuples
        :return: True if there's overlap, False otherwise
        """
        return any(start < region_end and end > region_start for region_start, region_end in regions)

    def clean_answer(self, answer: str) -> str:
        """Clean up generated answer text.

        Removes common formatting issues and artifacts from LLM generation.

        :param answer: Raw answer text from generation
        :return: Cleaned answer text
        """
        if not answer:
            return ""

        # Remove surrounding quotes if present
        if (answer.startswith('"') and answer.endswith('"')) or (answer.startswith("'") and answer.endswith("'")):
            answer = answer[1:-1]

        # Convert literal newlines
        answer = answer.replace("\\n", "\n")

        # Clean up multiple spaces
        answer = re.sub(r" {2,}", " ", answer)

        # Clean up multiple newlines (but preserve paragraph breaks)
        answer = re.sub(r"\n{3,}", "\n\n", answer)

        return answer.strip()


def _documents_and_citations(
    answer_citations: Sequence[AnswerCitation], chunks: Sequence[Chunk]
) -> tuple[list[DocumentWithHighlights], list[Citation]]:
    """Return the chunks as the documents of a response, and the citations of the response.

    Each cited span is highlighted at its offsets. A citation that overlaps one before it in its
    chunk gets neither a highlight nor a citation.

    :raises InvalidInputError: If two chunks share an id, or a citation points outside them
    """
    _check_unique_ids(chunks)
    positions = {chunk.id: position for position, chunk in enumerate(chunks)}
    highlights: list[list[Highlight]] = [[] for _ in chunks]
    citations = []
    for index, citation in enumerate(answer_citations):
        position = positions.get(citation.chunk_id)
        if position is None:
            msg = f"answer.citations[{index}] points at a chunk that is not in chunks."
            raise InvalidInputError(
                msg,
                code="invalid_value",
                param=f"answer.citations[{index}].chunk_id",
            )
        if not citation.start < citation.end <= len(chunks[position].text):
            msg = f"answer.citations[{index}] has offsets outside its chunk."
            raise InvalidInputError(
                msg,
                code="invalid_value",
                param=f"answer.citations[{index}].end",
            )
        chunk_highlights = highlights[position]
        if any(citation.start < highlight.end and citation.end > highlight.start for highlight in chunk_highlights):
            continue
        text = chunks[position].text[citation.start : citation.end]
        chunk_highlights.append(Highlight(text=text, start=citation.start, end=citation.end))
        citations.append(
            Citation(
                text=text,
                doc_index=position,
                highlight_index=len(chunk_highlights) - 1,
                number=citation.number,
                type=citation.type,
            )
        )

    documents = [
        DocumentWithHighlights(
            content=chunk.text,
            highlights=chunk_highlights,
            title=chunk.metadata.get("title", ""),
            source=chunk.metadata.get("source", ""),
            metadata=chunk.metadata,
        )
        for chunk, chunk_highlights in zip(chunks, highlights, strict=False)
    ]
    return documents, citations
