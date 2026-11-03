"""Tests for verbatim_core.response_builder."""

from unittest.mock import MagicMock

import pytest
from verbatim_core import Answer, AnswerCitation, Chunk, InvalidInputError
from verbatim_core.response_builder import ResponseBuilder


class TestCreateHighlights:
    def setup_method(self):
        self.builder = ResponseBuilder()

    def test_single_span(self):
        highlights = self.builder._create_highlights("The cat sat on the mat.", ["cat"])
        assert len(highlights) == 1
        assert highlights[0].text == "cat"
        assert highlights[0].start == 4
        assert highlights[0].end == 7

    def test_multiple_non_overlapping_spans(self):
        highlights = self.builder._create_highlights("The cat sat on the mat.", ["cat", "mat"])
        assert len(highlights) == 2
        texts = {h.text for h in highlights}
        assert texts == {"cat", "mat"}

    def test_span_not_found(self):
        highlights = self.builder._create_highlights("The cat sat.", ["dog"])
        assert len(highlights) == 0

    def test_overlapping_spans_prevented(self):
        text = "The big cat sat."
        highlights = self.builder._create_highlights(text, ["big cat", "cat"])
        # "cat" overlaps with "big cat", so only one should be created
        assert len(highlights) == 1
        assert highlights[0].text == "big cat"


class TestHasOverlap:
    def setup_method(self):
        self.builder = ResponseBuilder()

    def test_no_overlap(self):
        assert not self.builder._has_overlap(0, 5, {(10, 15)})

    def test_overlap(self):
        assert self.builder._has_overlap(3, 8, {(5, 10)})

    def test_contained(self):
        assert self.builder._has_overlap(6, 8, {(5, 10)})

    def test_empty_regions(self):
        assert not self.builder._has_overlap(0, 5, set())


class TestCleanAnswer:
    def setup_method(self):
        self.builder = ResponseBuilder()

    def test_removes_surrounding_double_quotes(self):
        assert self.builder.clean_answer('"Hello world"') == "Hello world"

    def test_removes_surrounding_single_quotes(self):
        assert self.builder.clean_answer("'Hello world'") == "Hello world"

    def test_converts_literal_newlines(self):
        assert "line1\nline2" == self.builder.clean_answer("line1\\nline2")

    def test_collapses_multiple_spaces(self):
        assert "a b" == self.builder.clean_answer("a   b")

    def test_collapses_many_newlines(self):
        result = self.builder.clean_answer("a\n\n\n\nb")
        assert result == "a\n\nb"

    def test_empty_string(self):
        assert self.builder.clean_answer("") == ""


class TestBuildResponse:
    def setup_method(self):
        self.builder = ResponseBuilder()

    def _make_result(self, text, title="", source=""):
        r = MagicMock()
        r.text = text
        r.metadata = {"title": title, "source": source}
        r.title = title
        r.source = source
        return r

    def test_basic_response(self):
        results = [self._make_result("The cat sat on the mat.")]
        spans = {"The cat sat on the mat.": ["cat"]}

        response = self.builder.build_response(
            question="What animal?",
            answer="A cat.",
            search_results=results,
            relevant_spans=spans,
        )

        assert response.question == "What animal?"
        assert response.answer == "A cat."
        assert len(response.documents) == 1
        assert len(response.documents[0].highlights) == 1
        assert response.structured_answer.citations[0].text == "cat"

    def test_citation_numbering(self):
        results = [
            self._make_result("Doc one has alpha and beta."),
            self._make_result("Doc two has gamma."),
        ]
        spans = {
            "Doc one has alpha and beta.": ["alpha", "beta"],
            "Doc two has gamma.": ["gamma"],
        }

        response = self.builder.build_response(
            question="Q",
            answer="A",
            search_results=results,
            relevant_spans=spans,
            display_span_count=2,
        )

        citations = response.structured_answer.citations
        assert len(citations) == 3
        assert citations[0].number == 1
        assert citations[0].type == "display"
        assert citations[1].number == 2
        assert citations[1].type == "display"
        assert citations[2].number == 3
        assert citations[2].type == "reference"

    def test_no_spans(self):
        results = [self._make_result("Some text.")]
        spans = {"Some text.": []}

        response = self.builder.build_response(
            question="Q", answer="A", search_results=results, relevant_spans=spans
        )

        assert len(response.documents) == 1
        assert len(response.documents[0].highlights) == 0
        assert len(response.structured_answer.citations) == 0


class TestBuildFromAnswer:
    CHUNKS = [
        Chunk(
            id="opened",
            text="Tolls apply. The bridge opened in 1932. Tolls apply.",
            metadata={"title": "History", "source": "bridge.md", "year": 1932},
        ),
        Chunk(id="design", text="Bradfield designed the bridge."),
    ]

    def answer(self, *citations):
        return Answer(
            text="The answer.",
            citations=[
                AnswerCitation(number=number, type=kind, chunk_id=chunk_id, start=start, end=end)
                for number, (kind, chunk_id, start, end) in enumerate(citations, start=1)
            ],
        )

    def test_each_citation_highlights_its_span_at_its_offsets(self):
        answer = self.answer(("display", "opened", 13, 39), ("reference", "design", 0, 30))

        response = ResponseBuilder().build_from_answer("When?", answer, self.CHUNKS)

        opened, design = response.documents
        assert [(h.start, h.end, h.text) for h in opened.highlights] == [
            (13, 39, "The bridge opened in 1932.")
        ]
        assert [h.text for h in design.highlights] == ["Bradfield designed the bridge."]
        assert [
            (c.number, c.type, c.doc_index, c.highlight_index, c.text)
            for c in response.structured_answer.citations
        ] == [
            (1, "display", 0, 0, "The bridge opened in 1932."),
            (2, "reference", 1, 0, "Bradfield designed the bridge."),
        ]
        assert (response.question, response.answer) == ("When?", "The answer.")

    def test_repeated_text_is_highlighted_only_where_it_is_cited(self):
        answer = self.answer(("display", "opened", 40, 52))

        response = ResponseBuilder().build_from_answer("When?", answer, self.CHUNKS)

        assert [(h.start, h.end) for h in response.documents[0].highlights] == [(40, 52)]

    def test_a_span_overlapping_an_earlier_one_gets_no_highlight_or_citation(self):
        answer = self.answer(
            ("display", "opened", 13, 39),
            ("display", "opened", 24, 52),
            ("display", "design", 0, 9),
        )

        response = ResponseBuilder().build_from_answer("When?", answer, self.CHUNKS)

        assert [(h.start, h.end) for h in response.documents[0].highlights] == [(13, 39)]
        assert [c.number for c in response.structured_answer.citations] == [1, 3]

    def test_a_citation_of_a_chunk_that_is_not_given_is_an_input_error(self):
        answer = self.answer(("display", "private-chunk-id", 0, 5))

        with pytest.raises(InvalidInputError) as raised:
            ResponseBuilder().build_from_answer("When?", answer, self.CHUNKS)

        assert raised.value.param == "answer.citations[0].chunk_id"
        assert "private-chunk-id" not in str(raised.value)

    def test_offsets_outside_the_chunk_are_an_input_error(self):
        answer = self.answer(("display", "design", 0, 30), ("display", "design", 20, 99))

        with pytest.raises(InvalidInputError) as raised:
            ResponseBuilder().build_from_answer("When?", answer, self.CHUNKS)

        assert raised.value.param == "answer.citations[1].end"

    def test_two_chunks_with_the_same_id_are_an_input_error(self):
        chunks = [*self.CHUNKS, Chunk(id="opened", text="Another text.")]

        with pytest.raises(InvalidInputError) as raised:
            ResponseBuilder().build_from_answer("When?", self.answer(), chunks)

        assert raised.value.param == "chunks[2].id"

    def test_documents_carry_the_chunk_metadata(self):
        response = ResponseBuilder().build_from_answer("When?", self.answer(), self.CHUNKS)

        document = response.documents[0]
        assert (document.content, document.title, document.source) == (
            self.CHUNKS[0].text,
            "History",
            "bridge.md",
        )
        assert document.metadata == {"title": "History", "source": "bridge.md", "year": 1932}
