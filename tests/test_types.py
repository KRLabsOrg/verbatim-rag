"""Tests for the shared types exported by verbatim_core."""

import pytest
from pydantic import ValidationError
from verbatim_core import (
    Answer,
    AnswerCitation,
    Chunk,
    ExtractedSpan,
    Extraction,
    ResultWarning,
    Span,
    Usage,
)

# "café 🙂" spans code points 6 to 12. In UTF-16 the emoji takes two units, so the end would be 13.
CHUNK_TEXT = "naïve café \U0001f642 ok"
CAFE_EMOJI = "café \U0001f642"


class TestSpan:
    def test_offsets_count_unicode_code_points(self):
        span = Span(start=6, end=12, text=CAFE_EMOJI, score=0.8)
        assert CHUNK_TEXT[span.start : span.end] == span.text

    def test_rejects_text_whose_length_disagrees_with_its_offsets(self):
        with pytest.raises(ValidationError):
            Span(start=6, end=13, text=CAFE_EMOJI, score=0.8)

    @pytest.mark.parametrize(("start", "end", "text"), [(5, 5, ""), (5, 3, "ab"), (-1, 1, "ab")])
    def test_rejects_empty_reversed_or_negative_offsets(self, start, end, text):
        with pytest.raises(ValidationError):
            Span(start=start, end=end, text=text, score=0.5)

    @pytest.mark.parametrize("score", [-0.1, 1.2])
    def test_score_is_a_probability(self, score):
        with pytest.raises(ValidationError):
            Span(start=0, end=2, text="na", score=score)

    def test_is_frozen(self):
        span = Span(start=0, end=2, text="na", score=1.0)
        with pytest.raises(ValidationError):
            span.start = 1


class TestChunk:
    def test_metadata_and_source_offsets_are_optional(self):
        chunk = Chunk(id="row-1", text=CHUNK_TEXT)
        assert chunk.metadata == {}
        assert chunk.source_start is None
        assert chunk.source_end is None

    def test_id_must_not_be_empty(self):
        with pytest.raises(ValidationError):
            Chunk(id="", text=CHUNK_TEXT)

    def test_source_end_cannot_precede_source_start(self):
        with pytest.raises(ValidationError):
            Chunk(id="row-1", text=CHUNK_TEXT, source_start=40, source_end=20)

    def test_is_frozen(self):
        chunk = Chunk(id="row-1", text=CHUNK_TEXT)
        with pytest.raises(ValidationError):
            chunk.text = "other"


class TestExtractedSpan:
    def test_is_a_span_that_names_its_chunk(self):
        chunk = Chunk(id="row-1", text=CHUNK_TEXT)
        span = ExtractedSpan(chunk_id="row-1", start=6, end=12, text=CAFE_EMOJI, score=1.0)
        assert isinstance(span, Span)
        assert span.chunk_id == chunk.id
        assert chunk.text[span.start : span.end] == span.text

    def test_keeps_the_span_invariants(self):
        with pytest.raises(ValidationError):
            ExtractedSpan(chunk_id="row-1", start=6, end=13, text=CAFE_EMOJI, score=1.0)


class TestUsage:
    def test_work_without_a_model_has_no_model_id(self):
        usage = Usage(model_id=None, model_version=None, units={"chunks": 3})
        assert usage.model_id is None
        assert usage.units == {"chunks": 3}

    def test_units_cannot_be_negative(self):
        with pytest.raises(ValidationError):
            Usage(model_id="gpt-4o-mini", model_version="gpt-4o-mini", units={"llm_calls": -1})


class TestExtraction:
    def test_an_empty_extraction_has_no_spans_usage_or_warnings(self):
        extraction = Extraction()
        assert extraction.spans == []
        assert extraction.usage == []
        assert extraction.warnings == []

    def test_round_trips_through_json_unchanged(self):
        extraction = Extraction(
            spans=[ExtractedSpan(chunk_id="row-1", start=6, end=12, text=CAFE_EMOJI, score=1.0)],
            usage=[
                Usage(
                    model_id="gpt-4o-mini",
                    model_version="gpt-4o-mini-2024-07-18",
                    units={"chunks": 1, "llm_calls": 1},
                )
            ],
            warnings=[
                ResultWarning(code="quote_not_found", message="A quote could not be located.")
            ],
        )
        assert Extraction.model_validate_json(extraction.model_dump_json()) == extraction


class TestAnswerCitation:
    @pytest.mark.parametrize("citation_type", ["display", "reference"])
    def test_is_displayed_or_reference_only(self, citation_type):
        citation = AnswerCitation(number=1, type=citation_type, chunk_id="row-1", start=6, end=12)
        assert citation.type == citation_type

    def test_rejects_other_kinds(self):
        with pytest.raises(ValidationError):
            AnswerCitation(number=1, type="footnote", chunk_id="row-1", start=6, end=12)

    def test_numbers_start_at_one(self):
        with pytest.raises(ValidationError):
            AnswerCitation(number=0, type="display", chunk_id="row-1", start=6, end=12)


class TestAnswer:
    def test_round_trips_through_json_unchanged(self):
        answer = Answer(
            text="The menu says: café \U0001f642 [1]",
            citations=[AnswerCitation(number=1, type="display", chunk_id="row-1", start=6, end=12)],
            spans=[ExtractedSpan(chunk_id="row-1", start=6, end=12, text=CAFE_EMOJI, score=1.0)],
            usage=[Usage(model_id=None, model_version=None, units={"chunks": 1})],
        )
        assert answer.warnings == []
        assert Answer.model_validate_json(answer.model_dump_json()) == answer
