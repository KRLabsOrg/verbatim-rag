"""Tests for VerbatimRAG.query against an in-memory index. They need the full verbatim-rag package."""

from unittest.mock import MagicMock

import pytest
from verbatim_core.extractors import SpanExtractor
from verbatim_core.templates import TemplateManager

verbatim_rag = pytest.importorskip("verbatim_rag")

CHUNK_SPANS = {
    "Alpha one. Alpha two. Alpha three.": ["Alpha one.", "Alpha two.", "Alpha three."],
    "Beta one. Beta two. Beta three.": ["Beta one.", "Beta two.", "Beta three."],
    "Gamma one. Gamma two.": ["Gamma one.", "Gamma two."],
}
# Three chunks hold eight spans, and the answer shows the first five of them.
SHOWN_SPANS = ["Alpha one.", "Alpha two.", "Alpha three.", "Beta one.", "Beta two."]
CITED_ONLY_SPANS = ["Beta three.", "Gamma one.", "Gamma two."]


class FixedSpanExtractor(SpanExtractor):
    def extract_spans(self, question, search_results):
        return {result.text: CHUNK_SPANS[result.text] for result in search_results}


class InMemoryIndex:
    def query(self, text=None, k=5, filter=None, hybrid_weights=None, rrf_k=60, search_params=None):
        from verbatim_rag.vector_stores.base import SearchResult

        results = [
            SearchResult(id=f"row-{i}", score=1.0, metadata={"title": f"Doc {i}"}, text=chunk)
            for i, chunk in enumerate(CHUNK_SPANS)
        ]
        return results[:k]


@pytest.fixture
def rag():
    return verbatim_rag.VerbatimRAG(
        InMemoryIndex(),
        k=3,
        extractor=FixedSpanExtractor(),
        template_manager=TemplateManager(default_mode="static"),
        max_display_spans=5,
        llm_client=MagicMock(),
    )


def citation_texts(response, citation_type):
    citations = response.structured_answer.citations
    return [citation.text for citation in citations if citation.type == citation_type]


class TestDisplayCitations:
    def test_query_displays_the_citations_of_the_spans_the_answer_shows(self, rag):
        response = rag.query("What is listed?")

        assert all(span in response.answer for span in SHOWN_SPANS)
        assert citation_texts(response, "display") == SHOWN_SPANS
        assert citation_texts(response, "reference") == CITED_ONLY_SPANS

    async def test_query_async_displays_the_citations_of_the_spans_the_answer_shows(self, rag):
        response = await rag.query_async("What is listed?")

        assert citation_texts(response, "display") == SHOWN_SPANS
        assert citation_texts(response, "reference") == CITED_ONLY_SPANS

    def test_query_and_the_streamed_answer_mark_the_same_citations(self, rag):
        streamed = verbatim_rag.StreamingRAG(rag).stream_query_sync("What is listed?")
        [final] = [stage for stage in streamed if stage["type"] == "answer"]
        streamed_types = [c["type"] for c in final["data"]["structured_answer"]["citations"]]

        response = rag.query("What is listed?")

        assert [c.type for c in response.structured_answer.citations] == streamed_types
