"""Tests for SpanExtractor's stateless interface and its compatibility with old subclasses."""

import logging
import warnings
from types import SimpleNamespace

import pytest
from extractor_contract import ExtractorContract
from verbatim_core import (
    Chunk,
    ExtractedSpan,
    Extraction,
    InvalidInputError,
    ResultWarning,
    Usage,
    VerbatimDeprecationWarning,
)
from verbatim_core._deprecation import REMOVED_IN
from verbatim_core.extractors import LLMSpanExtractor, SpanExtractor, _check_unique_ids

QUESTION = "How tall is the arch?"


class QuestionWordExtractor(SpanExtractor):
    """Marks the first occurrence in each chunk of each question word longer than three letters."""

    def extract(self, question, chunks, *, config=None):
        _check_unique_ids(chunks)
        words = [word.strip("?.,").lower() for word in question.split()]
        spans = []
        for chunk in chunks:
            for word in words:
                start = chunk.text.lower().find(word)
                if len(word) > 3 and start >= 0:
                    end = start + len(word)
                    span = ExtractedSpan(
                        chunk_id=chunk.id,
                        start=start,
                        end=end,
                        text=chunk.text[start:end],
                        score=0.5,
                    )
                    spans.append(span)
        usage = Usage(model_id="question-words", model_version="1", units={"chunks": len(chunks)})
        return Extraction(spans=spans, usage=[usage])


class TestQuestionWordExtractor(ExtractorContract):
    @pytest.fixture
    def extractor(self):
        return QuestionWordExtractor()


def old_extractor(span_texts):
    """An extractor written against the old interface, returning fixed span texts per chunk text."""

    class OldExtractor(SpanExtractor):
        def extract_spans(self, question, search_results):
            return {result.text: span_texts.get(result.text, []) for result in search_results}

    return OldExtractor()


def chunk_with_text_for(text, chunk_text, chunk_id="c1"):
    chunk = Chunk(id=chunk_id, text=chunk_text)
    return old_extractor({chunk_text: [text]}), chunk


class TestOldInterfaceSubclass:
    """The old interface keeps working. Its calls warn, which TestOldInterfaceWarning checks."""

    @pytest.fixture(autouse=True)
    def old_interface_warnings(self):
        """Record the deprecation warnings: only the old interface's may be among them."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", VerbatimDeprecationWarning)
            yield
        others = [
            str(w.message)
            for w in caught
            if issubclass(w.category, VerbatimDeprecationWarning)
            and not str(w.message).startswith("Implementing only extract_spans()")
        ]
        assert not others

    def test_extract_locates_each_span_text_in_its_chunk(self):
        height = Chunk(id="height", text="The arch rises 134 metres above the water.")
        tolls = Chunk(id="tolls", text="Tolls are collected in both directions.")
        extractor = old_extractor(
            {height.text: ["134 metres", "The arch"], tolls.text: ["both directions"]}
        )

        extraction = extractor.extract(QUESTION, [height, tolls])

        assert extraction.spans == [
            ExtractedSpan(chunk_id="height", start=15, end=25, text="134 metres", score=1.0),
            ExtractedSpan(chunk_id="height", start=0, end=8, text="The arch", score=1.0),
            ExtractedSpan(chunk_id="tolls", start=23, end=38, text="both directions", score=1.0),
        ]
        assert extraction.warnings == []

    def test_a_span_text_is_located_without_its_surrounding_whitespace(self):
        extractor, chunk = chunk_with_text_for(" 134 metres ", "The arch rises 134 metres above.")

        extraction = extractor.extract(QUESTION, [chunk])

        assert extraction.spans == [
            ExtractedSpan(chunk_id="c1", start=15, end=25, text="134 metres", score=1.0)
        ]

    def test_a_repeated_span_text_points_once_at_its_first_occurrence(self):
        chunk = Chunk(id="c1", text="Stop, look, and stop. Stop again.")
        extractor = old_extractor({chunk.text: ["Stop", "Stop"]})

        extraction = extractor.extract(QUESTION, [chunk])

        assert extraction.spans == [
            ExtractedSpan(chunk_id="c1", start=0, end=4, text="Stop", score=1.0)
        ]

    def test_a_span_text_that_matches_only_fuzzily_becomes_the_chunks_text(self):
        extractor, chunk = chunk_with_text_for("THE CAT, SAT.", "Before the cat , sat . after")

        extraction = extractor.extract(QUESTION, [chunk])

        assert extraction.spans == [
            ExtractedSpan(chunk_id="c1", start=7, end=22, text="the cat , sat .", score=1.0)
        ]

    def test_a_fuzzy_match_scoring_the_threshold_of_llm_quotes_is_kept(self):
        # The fuzzy matcher scores "short" against "shirt" 0.8, LLMSpanExtractor's default.
        extractor, chunk = chunk_with_text_for("short", "a shirt")

        [span] = extractor.extract(QUESTION, [chunk]).spans

        assert span.text == "shirt"

    def test_a_fuzzy_match_scoring_below_the_threshold_is_dropped(self):
        # Scores 0.71.
        extractor, chunk = chunk_with_text_for(
            "a long quote that is longer than the text", "short text"
        )

        assert extractor.extract(QUESTION, [chunk]).spans == []

    def test_the_best_fuzzy_match_wins(self):
        extractor, chunk = chunk_with_text_for(
            "quick brown fox", "a quick brawn fax, then a quick brown fax"
        )

        [span] = extractor.extract(QUESTION, [chunk]).spans

        assert (span.start, span.end) == (26, 41)

    def test_equally_good_fuzzy_matches_resolve_to_the_earliest(self):
        extractor, chunk = chunk_with_text_for(
            "quick brown fox", "the quick brown fax and the quick brown fax"
        )

        [span] = extractor.extract(QUESTION, [chunk]).spans

        assert (span.start, span.end) == (4, 19)

    def test_a_span_text_that_cannot_be_located_becomes_a_warning(self):
        extractor, chunk = chunk_with_text_for(
            "Ferries run every ten minutes.", "Tolls are collected in both directions."
        )

        extraction = extractor.extract(QUESTION, [chunk])

        assert extraction.spans == []
        assert extraction.warnings == [
            ResultWarning(
                code="span_text_not_found", message="A span text for chunks[0] is not in the chunk."
            )
        ]

    def test_dropped_span_texts_are_counted_in_the_log_and_shown_only_on_the_content_logger(
        self, caplog
    ):
        extractor, chunk = chunk_with_text_for(
            "Ferries run every ten minutes.", "Tolls are collected in both directions."
        )
        caplog.set_level(logging.DEBUG)

        extractor.extract(QUESTION, [chunk])

        [count, texts] = caplog.records
        assert (count.name, count.levelno) == ("verbatim_core.extractors", logging.WARNING)
        assert count.getMessage() == "Dropped 1 span texts that are not in their chunks"
        assert (count.event, count.warning_code) == ("fallback_used", "span_text_not_found")
        assert (texts.name, texts.levelno) == ("verbatim_core.content", logging.DEBUG)
        assert "Ferries run every ten minutes." in texts.getMessage()

    def test_usage_counts_the_chunks_and_their_characters(self):
        extractor = old_extractor({})
        chunks = [Chunk(id="a", text="four"), Chunk(id="b", text="three")]

        extraction = extractor.extract(QUESTION, chunks)

        assert extraction.usage == [
            Usage(model_id=None, model_version=None, units={"chunks": 2, "characters": 9})
        ]

    def test_chunks_with_the_same_text_get_their_own_spans(self):
        extractor = old_extractor({"The arch rises.": ["arch"]})
        chunks = [Chunk(id="a", text="The arch rises."), Chunk(id="b", text="The arch rises.")]

        extraction = extractor.extract(QUESTION, chunks)

        assert [span.chunk_id for span in extraction.spans] == ["a", "b"]

    async def test_two_chunks_with_the_same_id_are_rejected(self):
        extractor = old_extractor({})
        chunks = [Chunk(id="report.pdf", text="One."), Chunk(id="report.pdf", text="Two.")]

        with pytest.raises(InvalidInputError) as raised:
            extractor.extract(QUESTION, chunks)
        with pytest.raises(InvalidInputError):
            await extractor.aextract(QUESTION, chunks)

        assert (raised.value.code, raised.value.param) == ("invalid_value", "chunks[1].id")
        assert str(raised.value) == (
            "chunks[1] has the same id as chunks[0]. Chunk ids must be unique within a call."
        )

    async def test_a_config_is_rejected(self):
        extractor = old_extractor({})
        chunks = [Chunk(id="a", text="One.")]

        with pytest.raises(InvalidInputError) as raised:
            extractor.extract(QUESTION, chunks, config=object())
        with pytest.raises(InvalidInputError):
            await extractor.aextract(QUESTION, chunks, config=object())

        assert raised.value.param == "config"

    async def test_aextract_returns_what_extract_returns(self):
        chunk = Chunk(id="c1", text="The arch rises 134 metres above the water.")
        extractor = old_extractor({chunk.text: ["134 metres", "Ferries run."]})

        assert await extractor.aextract(QUESTION, [chunk]) == extractor.extract(QUESTION, [chunk])

    async def test_aextract_calls_the_subclass_async_method(self):
        class AsyncOldExtractor(SpanExtractor):
            def extract_spans(self, question, search_results):
                raise AssertionError("the async method should be called")

            async def extract_spans_async(self, question, search_results):
                return {result.text: ["arch"] for result in search_results}

        extraction = await AsyncOldExtractor().aextract(QUESTION, [Chunk(id="c1", text="An arch.")])

        assert [span.text for span in extraction.spans] == ["arch"]

    async def test_extract_spans_async_returns_the_subclass_output_unchanged(self):
        extractor = old_extractor({"The arch.": ["arch", "arch", "Ferries run."]})

        with pytest.warns(VerbatimDeprecationWarning, match="extract_spans_async is deprecated"):
            span_texts = await extractor.extract_spans_async(
                QUESTION, [Chunk(id="c1", text="The arch.")]
            )

        assert span_texts == {"The arch.": ["arch", "arch", "Ferries run."]}


class TestOldInterfaceWarning:
    MESSAGE = (
        "Implementing only extract_spans() in old_extractor.<locals>.OldExtractor is "
        f"deprecated since 0.3.0 and will be removed in {REMOVED_IN}. Use extract()."
    )

    def test_extract_warns_where_it_is_called(self):
        with pytest.warns(VerbatimDeprecationWarning) as record:
            old_extractor({}).extract(QUESTION, [Chunk(id="c1", text="The arch.")])

        assert [str(warning.message) for warning in record] == [self.MESSAGE]
        assert record[0].filename == __file__

    async def test_aextract_warns_where_it_is_awaited(self):
        with pytest.warns(VerbatimDeprecationWarning) as record:
            await old_extractor({}).aextract(QUESTION, [Chunk(id="c1", text="The arch.")])

        assert [str(warning.message) for warning in record] == [self.MESSAGE]
        assert record[0].filename == __file__

    def test_the_warning_names_the_class_that_implements_extract_spans(self):
        class Improved(type(old_extractor({}))):
            def extract(self, question, chunks, *, config=None):
                return super().extract(question, chunks)

        with pytest.warns(VerbatimDeprecationWarning, match="extract_spans\\(\\) in old_extractor"):
            Improved().extract(QUESTION, [Chunk(id="c1", text="The arch.")])


class TestNewInterfaceSubclass:
    def test_extract_spans_groups_the_span_texts_by_chunk_text(self):
        results = [SimpleNamespace(text="The arch rises."), SimpleNamespace(text="Tolls.")]

        with pytest.warns(VerbatimDeprecationWarning, match="extract_spans is deprecated"):
            span_texts = QuestionWordExtractor().extract_spans(QUESTION, results)

        assert span_texts == {"The arch rises.": ["arch"], "Tolls.": []}

    async def test_extract_spans_async_groups_the_span_texts_by_chunk_text(self):
        results = [SimpleNamespace(text="The arch rises."), SimpleNamespace(text="Tolls.")]

        with pytest.warns(VerbatimDeprecationWarning, match="extract_spans_async is deprecated"):
            span_texts = await QuestionWordExtractor().extract_spans_async(QUESTION, results)

        assert span_texts == {"The arch rises.": ["arch"], "Tolls.": []}

    async def test_calling_the_base_extract_raises_instead_of_recursing(self):
        class Delegating(QuestionWordExtractor):
            def extract(self, question, chunks, *, config=None):
                return super(QuestionWordExtractor, self).extract(question, chunks)

        chunks = [Chunk(id="c1", text="The arch.")]
        with pytest.raises(NotImplementedError):
            Delegating().extract(QUESTION, chunks)
        with pytest.raises(NotImplementedError):
            await SpanExtractor().aextract(QUESTION, chunks)


class TestSubclassing:
    def test_a_subclass_implementing_neither_method_is_rejected(self):
        with pytest.raises(TypeError, match=r"must implement extract\(\) or extract_spans\(\)"):

            class NoExtractor(SpanExtractor):
                pass

    def test_defining_subclasses_never_warns(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            class NewExtractor(SpanExtractor):
                def extract(self, question, chunks, *, config=None):
                    return Extraction()

            class CustomLLMSpanExtractor(LLMSpanExtractor):
                pass

            old_extractor({})

        assert not hasattr(SpanExtractor, "__deprecated__")
