"""VerbatimRAG on a Transformer, against an in-memory index. They need the full verbatim-rag package."""

import logging
import re
import sys
import time
import types

import pytest
from verbatim_core import (
    ConfigurationError,
    ExtractedSpan,
    Extraction,
    InvalidInputError,
    TransformConfig,
    Transformer,
    UpstreamError,
    VerbatimDeprecationWarning,
)
from verbatim_core.extractors import LLMSpanExtractor, SpanExtractor
from verbatim_core.templates import TemplateManager

verbatim_rag = pytest.importorskip("verbatim_rag")

from verbatim_rag.vector_stores.base import SearchResult  # noqa: E402

QUESTION = "When did the bridge open?"


def rows():
    return [
        SearchResult(
            id="row-7",
            score=0.9,
            metadata={"title": "Bridge", "source": "bridge.md"},
            text="The bridge opened in 1932. It has eight lanes.",
            enhanced_text="Title: Bridge\nThe bridge opened in 1932. It has eight lanes.",
        ),
        SearchResult(
            id="row-3",
            score=0.8,
            metadata={"title": "Tunnel", "source": "tunnel.md"},
            text="The tunnel opened in 1992.",
        ),
        SearchResult(id="row-5", score=0.7, metadata={}, text="Tolls are electronic."),
    ]


class RecordingIndex:
    def __init__(self):
        self.queries = []

    def query(self, text=None, k=5, filter=None, hybrid_weights=None, rrf_k=60, search_params=None):
        self.queries.append(
            {
                "k": k,
                "filter": filter,
                "hybrid_weights": hybrid_weights,
                "rrf_k": rrf_k,
                "search_params": search_params,
            }
        )
        return rows()[:k]


class OpenedExtractor(SpanExtractor):
    """Extracts every sentence with the word "opened", and records the chunks it was given."""

    def __init__(self, error=None):
        self.error = error
        self.chunks = []

    def extract(self, question, chunks, *, config=None):
        if self.error:
            raise self.error
        self.chunks = list(chunks)
        spans = [
            ExtractedSpan(
                chunk_id=chunk.id, start=m.start(), end=m.end(), text=m.group(), score=1.0
            )
            for chunk in chunks
            for m in re.finditer(r"[^.\s][^.]*opened[^.]*\.", chunk.text)
        ]
        return Extraction(spans=spans)


def build(index=None, extractor=None, **settings):
    transformer = Transformer(extractor or OpenedExtractor(), TemplateManager())
    return verbatim_rag.VerbatimRAG(index or RecordingIndex(), transformer=transformer, **settings)


CALLS = pytest.mark.parametrize("call", ["query", "query_async", "query_stream"])


async def query(rag, call, *args, **settings):
    """The response of one of VerbatimRAG's calls. For query_stream, the last event's."""
    if call == "query":
        return rag.query(*args, **settings)
    if call == "query_async":
        return await rag.query_async(*args, **settings)
    events = [event async for event in rag.query_stream(*args, **settings)]
    return events[-1].response


class TestConstruction:
    def test_with_a_transformer_nothing_warns(self):
        transformer = Transformer(OpenedExtractor(), TemplateManager())

        rag = verbatim_rag.VerbatimRAG(RecordingIndex(), transformer=transformer)

        assert (rag.transformer, rag.extractor, rag.template_manager) == (
            transformer,
            transformer.extractor,
            transformer.templates,
        )

    def test_the_transformer_may_be_passed_by_position(self):
        transformer = Transformer(OpenedExtractor(), TemplateManager())

        assert verbatim_rag.VerbatimRAG(RecordingIndex(), transformer).transformer is transformer

    def test_the_arguments_may_follow_the_transformer_by_position(self):
        transformer = Transformer(OpenedExtractor(), TemplateManager())
        llm_client, detector, reranker = StructuredLLM(), object(), ReversingReranker()

        rag = verbatim_rag.VerbatimRAG(
            RecordingIndex(), transformer, llm_client, detector, reranker
        )

        assert (rag.llm_client, rag.intent_detector, rag.reranker, rag.k) == (
            llm_client,
            detector,
            reranker,
            5,
        )

    def test_without_a_transformer_it_builds_todays_and_warns(self):
        llm_client = types.SimpleNamespace(model="fake-llm", temperature=0.0)

        with pytest.warns(VerbatimDeprecationWarning, match="without a transformer") as warned:
            rag = verbatim_rag.VerbatimRAG(
                RecordingIndex(), template_mode="static", llm_client=llm_client
            )

        assert isinstance(rag.extractor, LLMSpanExtractor)
        assert rag.extractor.llm_client is llm_client
        assert rag.template_manager.current_mode == "static"
        assert warned[0].filename == __file__

    def test_k_in_the_constructor_warns(self):
        with pytest.warns(VerbatimDeprecationWarning, match="query"):
            build(k=2)

    def test_assigning_the_extractor_or_templates_warns_and_applies(self):
        rag = build()
        extractor = OpenedExtractor()

        with pytest.warns(VerbatimDeprecationWarning, match="VerbatimRAG.extractor"):
            rag.extractor = extractor
        with pytest.warns(VerbatimDeprecationWarning, match="VerbatimRAG.max_display_spans"):
            rag.max_display_spans = 1
        response = rag.query(QUESTION)

        assert [chunk.id for chunk in extractor.chunks] == ["row-7", "row-3", "row-5"]
        assert [c.type for c in response.structured_answer.citations] == ["display", "reference"]

    def test_assigning_the_extractor_leaves_the_transformer_given_alone(self):
        transformer = Transformer(OpenedExtractor(), TemplateManager())
        rag = verbatim_rag.VerbatimRAG(RecordingIndex(), transformer=transformer)
        extractor = transformer.extractor

        with pytest.warns(VerbatimDeprecationWarning):
            rag.extractor = OpenedExtractor()

        assert transformer.extractor is extractor

    def test_assigning_max_display_spans_keeps_the_other_settings(self):
        transformer = Transformer(
            OpenedExtractor(), TemplateManager(), config=TransformConfig(citation_mode="hidden")
        )
        rag = verbatim_rag.VerbatimRAG(RecordingIndex(), transformer=transformer)

        with pytest.warns(VerbatimDeprecationWarning):
            rag.max_display_spans = 1
        response = rag.query(QUESTION)

        assert "[1]" not in response.answer
        assert [c.type for c in response.structured_answer.citations] == ["display", "reference"]

    def test_assigning_the_templates_warns_and_applies(self):
        rag = build()
        templates = TemplateManager()
        templates.use_static_mode(template="Found: [DISPLAY_SPANS]")

        with pytest.warns(VerbatimDeprecationWarning, match="VerbatimRAG.template_manager"):
            rag.template_manager = templates

        assert rag.query(QUESTION).answer.startswith("Found:")
        assert templates.strategies["structured"].rag_system is rag

    def test_an_argument_by_position_and_by_keyword_is_a_configuration_error(self):
        transformer = Transformer(OpenedExtractor(), TemplateManager())

        with pytest.raises(ConfigurationError):
            verbatim_rag.VerbatimRAG(RecordingIndex(), transformer, object(), llm_client=object())

    def test_old_arguments_next_to_a_transformer_are_a_configuration_error(self):
        with pytest.raises(ConfigurationError):
            build(template_mode="static")


@CALLS
class TestQuery:
    async def test_the_chunks_are_the_rows_of_the_index(self, call):
        extractor = OpenedExtractor()

        await query(build(extractor=extractor), call, QUESTION)

        assert [(c.id, c.text, c.metadata) for c in extractor.chunks] == [
            (row.id, row.text, row.metadata) for row in rows()
        ]
        assert all(c.source_start is None and c.source_end is None for c in extractor.chunks)

    async def test_retrieval_settings_apply_to_one_call(self, call):
        index = RecordingIndex()
        rag = build(index)
        settings = {
            "k": 2,
            "filter": "year > 1900",
            "hybrid_weights": {"dense": 1.0},
            "rrf_k": 10,
            "search_params": {"ef": 64},
        }

        await query(rag, call, QUESTION, **settings)
        await query(rag, call, QUESTION)

        assert index.queries == [
            settings,
            {"k": 5, "filter": None, "hybrid_weights": None, "rrf_k": 60, "search_params": None},
        ]

    async def test_the_answer_cites_the_spans_at_their_offsets(self, call):
        response = await query(build(), call, QUESTION)

        assert "[1] The bridge opened in 1932." in response.answer
        assert [(d.title, [h.text for h in d.highlights]) for d in response.documents] == [
            ("Bridge", ["The bridge opened in 1932."]),
            ("Tunnel", ["The tunnel opened in 1992."]),
            ("", []),
        ]

    async def test_the_settings_of_the_transformer_apply(self, call):
        transformer = Transformer(
            OpenedExtractor(), TemplateManager(), config=TransformConfig(max_display_spans=1)
        )
        rag = verbatim_rag.VerbatimRAG(RecordingIndex(), transformer=transformer)

        response = await query(rag, call, QUESTION)

        assert [c.type for c in response.structured_answer.citations] == ["display", "reference"]

    async def test_a_failed_extraction_raises(self, call):
        error = UpstreamError("The LLM service failed.", code="server_error", service="llm")

        with pytest.raises(UpstreamError):
            await query(build(extractor=OpenedExtractor(error=error)), call, QUESTION)

    async def test_the_steps_are_logged_at_debug_and_nothing_is_printed(self, call, caplog, capsys):
        caplog.set_level(logging.DEBUG, logger="verbatim_rag")

        await query(build(), call, QUESTION)

        assert capsys.readouterr() == ("", "")
        assert [r.levelno for r in caplog.records if r.name == "verbatim_rag.core"] == [
            logging.DEBUG
        ] * 3


class FailingReranker:
    def __init__(self, error):
        self.error = error

    def rerank(self, question, results):
        raise self.error

    async def rerank_async(self, question, results):
        raise self.error


class ReversingReranker:
    def rerank(self, question, results):
        return list(reversed(results))


@CALLS
class TestReranker:
    async def test_the_reranked_order_is_the_order_of_the_chunks(self, call):
        extractor = OpenedExtractor()

        await query(build(extractor=extractor, reranker=ReversingReranker()), call, QUESTION)

        assert [chunk.id for chunk in extractor.chunks] == ["row-5", "row-3", "row-7"]

    async def test_a_failed_service_falls_back_to_the_retrieval_order(self, call, caplog):
        error = UpstreamError("The reranker failed.", code="timeout", service="reranker")
        extractor = OpenedExtractor()
        rag = build(extractor=extractor, reranker=FailingReranker(error))

        response = await query(rag, call, QUESTION)

        assert [chunk.id for chunk in extractor.chunks] == ["row-7", "row-3", "row-5"]
        assert [warning.code for warning in response.warnings] == ["reranker_fallback"]
        [record] = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert (record.name, record.event, record.warning_code) == (
            "verbatim_rag.core",
            "fallback_used",
            "reranker_fallback",
        )

    async def test_another_error_propagates(self, call):
        rag = build(reranker=FailingReranker(KeyError("bug")))

        with pytest.raises(KeyError):
            await query(rag, call, QUESTION)


class StructuredLLM:
    model = "fake-llm"
    temperature = 0.0

    def extract_structured(self, question, template, placeholders, documents):
        return {"OPENING": [{"text": "The bridge opened in 1932.", "doc": 0}]}

    async def extract_structured_async(self, question, template, placeholders, documents):
        return self.extract_structured(question, template, placeholders, documents)


@CALLS
class TestStructuredMode:
    async def test_the_client_of_verbatim_rag_answers(self, call):
        rag = build(llm_client=StructuredLLM())
        rag.template_manager.use_structured_mode(template="## Opening\n[OPENING]")

        response = await query(rag, call, QUESTION)

        assert "The bridge opened in 1932." in response.answer

    async def test_a_reranker_fallback_is_a_warning_of_the_response(self, call):
        error = UpstreamError("The reranker failed.", code="timeout", service="reranker")
        rag = build(llm_client=StructuredLLM(), reranker=FailingReranker(error))
        rag.template_manager.use_structured_mode(template="## Opening\n[OPENING]")

        response = await query(rag, call, QUESTION)

        assert [warning.code for warning in response.warnings] == ["reranker_fallback"]

    async def test_without_a_client_it_is_a_configuration_error(self, call):
        rag = build()
        rag.template_manager.use_structured_mode(template="## Opening\n[OPENING]")

        with pytest.raises(ConfigurationError):
            await query(rag, call, QUESTION)


class GreetingDetector:
    def detect(self, question):
        return {"route": "predefined", "answer": "Hello! Ask me about the bridge."}


async def stream(rag, *args, **settings):
    return [event async for event in rag.query_stream(*args, **settings)]


class TestQueryStream:
    async def test_the_chunks_then_their_spans_then_the_response_of_query(self):
        rag = build()

        events = await stream(rag, QUESTION)

        assert [event.type for event in events] == ["documents", "spans", "response"]
        documents, spans, response = events
        assert [chunk.id for chunk in documents.chunks] == ["row-7", "row-3", "row-5"]
        assert [(span.chunk_id, span.text) for span in spans.spans] == [
            ("row-7", "The bridge opened in 1932."),
            ("row-3", "The tunnel opened in 1992."),
        ]
        assert response.response == rag.query(QUESTION)

    async def test_concurrent_streams_keep_their_own_k(self):
        rag = build()
        two, three = rag.query_stream(QUESTION, k=2), rag.query_stream(QUESTION, k=3)

        first_of_two, first_of_three = await anext(two), await anext(three)
        rest_of_two = [event async for event in two]

        assert len(first_of_two.chunks) == 2
        assert len(first_of_three.chunks) == 3
        assert len(rest_of_two[-1].response.documents) == 2

    async def test_a_question_the_intent_detector_answers_is_the_response_alone(self):
        rag = build(intent_detector=GreetingDetector())

        events = await stream(rag, QUESTION)

        assert [event.type for event in events] == ["response"]
        assert events[0].response == rag.query(QUESTION)
        assert events[0].response.answer == "Hello! Ask me about the bridge."

    async def test_a_failure_raises_after_the_documents(self):
        error = UpstreamError("The LLM service failed.", code="server_error", service="llm")
        events = []

        with pytest.raises(UpstreamError):
            async for event in build(extractor=OpenedExtractor(error=error)).query_stream(QUESTION):
                events.append(event)

        assert [event.type for event in events] == ["documents"]

    async def test_structured_mode_sends_the_documents_then_the_response(self):
        rag = build(llm_client=StructuredLLM())
        rag.template_manager.use_structured_mode(template="## Opening\n[OPENING]")

        events = await stream(rag, QUESTION)

        assert [event.type for event in events] == ["documents", "response"]
        assert events[-1].response == rag.query(QUESTION)


def timeout_from(cause):
    """An UpstreamError raised from the cause."""
    try:
        raise UpstreamError("The LLM timed out.", code="timeout", service="llm") from cause
    except UpstreamError as error:
        return error


def logged_traceback(caplog):
    [record] = [r for r in caplog.records if r.levelno >= logging.WARNING]
    return logging.Formatter().formatException(record.exc_info)


def dict_events(rag, *args, **settings):
    with pytest.warns(VerbatimDeprecationWarning):
        return verbatim_rag.StreamingRAG(rag).stream_query_sync(*args, **settings)


class OverlappingExtractor(SpanExtractor):
    """Extracts two overlapping spans of the first chunk."""

    def extract(self, question, chunks, *, config=None):
        chunk = chunks[0]
        spans = [
            ExtractedSpan(chunk_id=chunk.id, start=s, end=e, text=chunk.text[s:e], score=1.0)
            for s, e in [(4, 20), (0, 10)]
        ]
        return Extraction(spans=spans)


class TestStreamingRAG:
    def test_it_warns_and_so_do_its_methods(self):
        with pytest.warns(VerbatimDeprecationWarning, match="^StreamingRAG is deprecated"):
            streaming = verbatim_rag.StreamingRAG(build())
        with pytest.warns(VerbatimDeprecationWarning, match=r"^StreamingRAG\.stream_query is"):
            streaming.stream_query(QUESTION)
        with pytest.warns(VerbatimDeprecationWarning, match=r"^StreamingRAG\.stream_query_sync"):
            streaming.stream_query_sync(QUESTION)

    def test_the_events_of_query_stream_as_dicts(self):
        rag = build()

        events = dict_events(rag, QUESTION)

        assert [event["type"] for event in events] == [
            "documents",
            "progress",
            "highlights",
            "answer",
        ]
        documents, progress, highlights, answer = events
        assert [(d["content"], d["title"], d["highlights"]) for d in documents["data"]] == [
            (row.text, row.metadata.get("title", ""), []) for row in rows()
        ]
        assert progress["stage"] == "extraction_complete"
        assert isinstance(progress["elapsed_ms"], int) and progress["elapsed_ms"] >= 0
        assert highlights["data"] == answer["data"]["documents"]
        assert answer == {"type": "answer", "data": rag.query(QUESTION).model_dump(), "done": True}

    def test_the_progress_event_times_the_extraction_alone(self):
        class SlowIndex(RecordingIndex):
            def query(self, **retrieval):
                time.sleep(0.3)
                return super().query(**retrieval)

        class SlowExtractor(OpenedExtractor):
            def extract(self, question, chunks, *, config=None):
                time.sleep(0.1)
                return super().extract(question, chunks, config=config)

        events = dict_events(build(SlowIndex(), SlowExtractor()), QUESTION)

        [progress] = [event for event in events if event["type"] == "progress"]
        assert 100 <= progress["elapsed_ms"] < 300

    def test_the_highlights_are_the_responses_where_spans_overlap(self):
        events = dict_events(build(extractor=OverlappingExtractor()), QUESTION)

        [highlights] = [event for event in events if event["type"] == "highlights"]
        [answer] = [event for event in events if event["type"] == "answer"]
        assert [h["text"] for h in highlights["data"][0]["highlights"]] == ["bridge opened in"]
        assert highlights["data"] == answer["data"]["documents"]

    def test_num_docs_and_filter_apply_to_the_call(self):
        index = RecordingIndex()

        dict_events(build(index), QUESTION, num_docs=2, filter="year > 1900")

        assert [(q["k"], q["filter"]) for q in index.queries] == [(2, "year > 1900")]

    async def test_stream_query_passes_the_retrieval_settings(self):
        index = RecordingIndex()
        settings = {
            "filter": "year > 1900",
            "hybrid_weights": {"dense": 1.0},
            "rrf_k": 10,
            "search_params": {"ef": 64},
        }

        with pytest.warns(VerbatimDeprecationWarning):
            events = verbatim_rag.StreamingRAG(build(index)).stream_query(QUESTION, 2, **settings)
        [event async for event in events]

        assert index.queries == [{"k": 2, **settings}]

    @pytest.mark.parametrize(
        "error, sent",
        [
            (UpstreamError(f"Failed on {QUESTION}", code="timeout", service="llm"), "timeout"),
            (
                ConfigurationError(f"Rejected {QUESTION}", code="authentication_failed"),
                "authentication_failed",
            ),
            (KeyError(QUESTION), "internal_error"),
        ],
    )
    def test_a_failure_is_the_last_event_with_its_code_alone(self, error, sent):
        events = dict_events(build(extractor=OpenedExtractor(error=error)), QUESTION)

        assert [event["type"] for event in events] == ["documents", "error"]
        assert events[-1] == {"type": "error", "error": sent, "code": sent, "done": True}

    def test_an_invalid_input_error_is_sent_with_its_message(self):
        error = InvalidInputError(
            "chunks[1].id is not unique.", code="invalid_value", param="chunks[1].id"
        )

        events = dict_events(build(extractor=OpenedExtractor(error=error)), QUESTION)

        assert events[-1] == {
            "type": "error",
            "error": "chunks[1].id is not unique.",
            "code": "invalid_value",
            "done": True,
        }

    def test_a_span_outside_the_chunks_fails_as_in_query(self):
        class StrayExtractor(SpanExtractor):
            def extract(self, question, chunks, *, config=None):
                span = ExtractedSpan(chunk_id="row-1", start=0, end=3, text="The", score=1.0)
                return Extraction(spans=[span])

        rag = build(extractor=StrayExtractor())

        events = dict_events(rag, QUESTION)

        with pytest.raises(InvalidInputError) as raised:
            rag.query(QUESTION)
        assert events[-1] == {
            "type": "error",
            "error": str(raised.value),
            "code": "invalid_value",
            "done": True,
        }

    @pytest.mark.parametrize(
        "error, code, shown",
        [
            (KeyError(QUESTION), "internal_error", "KeyError"),
            (timeout_from(TimeoutError(QUESTION)), "timeout", "UpstreamError: The LLM timed out."),
        ],
    )
    def test_a_failure_is_logged_once_with_its_traceback_without_customer_text(
        self, caplog, error, code, shown
    ):
        dict_events(build(extractor=OpenedExtractor(error=error)), QUESTION)

        [record] = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert (record.name, record.levelno, record.getMessage()) == (
            "verbatim_rag.streaming",
            logging.ERROR,
            f"Streamed query failed: {code}",
        )
        traceback = logging.Formatter().formatException(record.exc_info)
        assert ", in extract\n" in traceback
        assert traceback.endswith(shown)
        assert QUESTION not in traceback

    def test_the_log_shows_the_class_of_a_cause(self, caplog):
        dict_events(
            build(extractor=OpenedExtractor(error=timeout_from(TimeoutError(QUESTION)))), QUESTION
        )

        assert "TimeoutError\n\nThe above exception was the direct cause" in logged_traceback(
            caplog
        )

    def test_the_log_leaves_out_a_suppressed_context(self, caplog):
        try:
            try:
                raise TimeoutError(QUESTION)
            except TimeoutError:
                raise UpstreamError("The LLM timed out.", code="timeout", service="llm") from None
        except UpstreamError as raised:
            error = raised

        dict_events(build(extractor=OpenedExtractor(error=error)), QUESTION)

        assert "TimeoutError" not in logged_traceback(caplog)

    def test_an_error_whose_causes_loop_is_logged(self, caplog):
        error, cause = KeyError("first"), KeyError("second")
        error.__context__, cause.__context__ = cause, error

        events = dict_events(build(extractor=OpenedExtractor(error=error)), QUESTION)

        assert events[-1]["code"] == "internal_error"
        assert logged_traceback(caplog).count("KeyError") == 2


class TestVerbatimDOC:
    async def test_each_query_is_filled_with_the_spans_of_its_chunks(self):
        from verbatim_rag.verbatim_doc import VerbatimDOC

        extractor = OpenedExtractor()
        template = "## Opening\n[!query=When did they open?]\n"

        response = await VerbatimDOC(build(extractor=extractor)).process(
            template, auto_approve=True
        )

        assert response.answer == (
            "## Opening\n[1] The bridge opened in 1932.\n\n[2] The tunnel opened in 1992.\n"
        )
        assert [(c.text, c.doc_index) for c in response.structured_answer.citations] == [
            ("The bridge opened in 1932.", 0),
            ("The tunnel opened in 1992.", 1),
        ]
        assert [chunk.id for chunk in extractor.chunks] == ["row-7", "row-3", "row-5"]

    async def test_the_spans_are_in_the_order_of_their_chunks(self):
        from verbatim_rag.verbatim_doc import VerbatimDOC

        class LastChunkFirst(OpenedExtractor):
            def extract(self, question, chunks, *, config=None):
                extraction = super().extract(question, chunks, config=config)
                return Extraction(spans=list(reversed(extraction.spans)))

        doc = VerbatimDOC(build(extractor=LastChunkFirst()))

        _, [result] = await doc.process_interactive("[!query=When did they open?]")

        assert [(span.text, span.doc_index) for span in result.spans] == [
            ("The bridge opened in 1932.", 0),
            ("The tunnel opened in 1992.", 1),
        ]

    async def test_documents_sharing_a_rag_retrieve_with_their_own_k(self):
        from verbatim_rag.verbatim_doc import VerbatimDOC

        index = RecordingIndex()
        rag = build(index)
        template = "[!query=When did they open?]"

        await VerbatimDOC(rag, k=2).process(template)
        await VerbatimDOC(rag).process(template)

        assert [query["k"] for query in index.queries] == [2, 5]

    async def test_without_k_the_deprecated_k_of_the_rag_applies(self):
        from verbatim_rag.verbatim_doc import VerbatimDOC

        index = RecordingIndex()
        with pytest.warns(VerbatimDeprecationWarning):
            rag = build(index, k=3)

        await VerbatimDOC(rag).process("[!query=When did they open?]")

        assert [query["k"] for query in index.queries] == [3]

    @pytest.mark.parametrize("k", [0, -1, 2.5, "3", True])
    def test_k_must_be_a_positive_integer(self, k):
        from verbatim_rag.verbatim_doc import VerbatimDOC

        with pytest.raises(ConfigurationError):
            VerbatimDOC(build(), k=k)

    async def test_a_failed_query_raises(self):
        from verbatim_rag.verbatim_doc import VerbatimDOC

        error = UpstreamError("The LLM service failed.", code="server_error", service="llm")
        doc = VerbatimDOC(build(extractor=OpenedExtractor(error=error)))

        with pytest.raises(UpstreamError):
            await doc.process("## Opening\n[!query=When did they open?]\n", auto_approve=True)


def test_a_hybrid_merge_logs_its_step_at_debug(caplog):
    from verbatim_rag.vector_stores.hybrid_search import merge_hybrid_results

    caplog.set_level(logging.DEBUG, logger="verbatim_rag")
    results = {"dense": [{"id": "row-1"}, {"id": "row-2"}], "sparse": [{"id": "row-2"}]}

    merged = merge_hybrid_results(results, 2, {"dense": 1.0, "sparse": 1.0}, log_label="Store")

    assert [hit["id"] for hit in merged] == ["row-2", "row-1"]
    assert [r.levelno for r in caplog.records if r.name.startswith("verbatim_rag")] == [
        logging.DEBUG
    ]


def test_a_search_result_becomes_a_chunk_of_its_row():
    [row] = rows()[:1]

    chunk = row.to_chunk()

    assert (chunk.id, chunk.text, chunk.metadata) == (row.id, row.text, row.metadata)
    assert (chunk.source_start, chunk.source_end) == (None, None)


class TestDocumentChunk:
    def test_chunk_from_the_document_module_is_document_chunk_and_warns(self):
        from verbatim_rag import document

        with pytest.warns(VerbatimDeprecationWarning, match="DocumentChunk") as warned:
            old = document.Chunk

        assert old is document.DocumentChunk
        assert warned[0].filename == __file__

    def test_a_document_with_chunks_round_trips(self):
        from verbatim_rag.document import Document, DocumentChunk

        doc = Document(title="Bridge")
        doc.add_chunk(DocumentChunk(content="The bridge opened in 1932."))

        assert Document.from_dict(doc.to_dict()).chunks[0].content == "The bridge opened in 1932."

    @pytest.mark.parametrize("old_name", [False, True])
    def test_pre_chunked_documents_are_indexed_with_either_name(self, old_name):
        from verbatim_rag import document
        from verbatim_rag.document import Document, ProcessedChunk
        from verbatim_rag.index import VerbatimIndex

        if old_name:
            with pytest.warns(VerbatimDeprecationWarning):
                chunk_class = document.Chunk
        else:
            chunk_class = document.DocumentChunk
        chunk = chunk_class(content="The bridge opened in 1932.")
        chunk.add_processed_chunk(ProcessedChunk(id="row-1", enhanced_content="Bridge: opened."))
        doc = Document(title="Bridge")
        doc.add_chunk(chunk)
        store = MemoryVectorStore()
        index = VerbatimIndex(vector_store=store, dense_provider=OneDimensionEmbedder())

        index.add_documents([doc])

        assert [(row.id, row.text, row.enhanced_text) for row in store.rows] == [
            ("row-1", "The bridge opened in 1932.", "Bridge: opened.")
        ]


class MemoryVectorStore:
    """A vector store that keeps its rows in a list and returns them all."""

    def __init__(self):
        self.rows = []

    def add_vectors(self, ids, dense_vectors, sparse_vectors, texts, enhanced_texts, metadatas):
        self.rows += [
            SearchResult(id=i, score=1.0, metadata=m, text=t, enhanced_text=e)
            for i, t, e, m in zip(ids, texts, enhanced_texts, metadatas)
        ]

    def query(self, *args, **settings):
        return list(self.rows)

    def delete(self, ids):
        self.rows = [row for row in self.rows if row.id not in ids]


class OneDimensionEmbedder:
    def embed_batch(self, texts):
        return [[1.0] for _ in texts]

    def embed_text(self, text):
        return [1.0]

    def get_dimension(self):
        return 1


class TestRerankerServices:
    """The API rerankers raise UpstreamError for a failed service, so VerbatimRAG falls back."""

    @pytest.mark.parametrize(
        "status, error_class, code",
        [
            (None, UpstreamError, "timeout"),
            (401, ConfigurationError, "authentication_failed"),
            (404, ConfigurationError, "model_not_found"),
            (429, UpstreamError, "rate_limited"),
            (503, UpstreamError, "server_error"),
            (400, UpstreamError, "request_rejected"),
        ],
    )
    def test_jina(self, monkeypatch, status, error_class, code):
        import requests

        from verbatim_rag.rerankers import JinaReranker

        def post(url, **settings):
            if status is None:
                raise requests.Timeout("timed out")
            response = requests.Response()
            response.status_code = status
            return response

        monkeypatch.setattr(requests, "post", post)

        with pytest.raises(error_class) as raised:
            JinaReranker(api_key="key").rerank(QUESTION, rows())  # pragma: allowlist secret

        assert raised.value.code == code

    def test_an_unreadable_jina_reply_is_an_upstream_error(self, monkeypatch):
        import requests

        from verbatim_rag.rerankers import JinaReranker

        def post(url, **settings):
            response = requests.Response()
            response.status_code = 200
            response._content = b"<html>"
            return response

        monkeypatch.setattr(requests, "post", post)

        with pytest.raises(UpstreamError) as raised:
            JinaReranker(api_key="key").rerank(QUESTION, rows())  # pragma: allowlist secret

        assert raised.value.code == "invalid_reply"

    def test_a_bug_in_calling_cohere_propagates(self, monkeypatch):
        class Client:
            def __init__(self, api_key):
                pass

            def rerank(self, **request):
                raise TypeError("rerank() got an unexpected keyword argument")

        monkeypatch.setitem(sys.modules, "cohere", types.SimpleNamespace(Client=Client))
        from verbatim_rag.rerankers import CohereReranker

        with pytest.raises(TypeError):
            CohereReranker(api_key="key").rerank(QUESTION, rows())  # pragma: allowlist secret

    def test_an_unreadable_cohere_reply_is_an_upstream_error(self, monkeypatch):
        class Client:
            def __init__(self, api_key):
                pass

            def rerank(self, **request):
                return types.SimpleNamespace()

        monkeypatch.setitem(sys.modules, "cohere", types.SimpleNamespace(Client=Client))
        from verbatim_rag.rerankers import CohereReranker

        with pytest.raises(UpstreamError) as raised:
            CohereReranker(api_key="key").rerank(QUESTION, rows())  # pragma: allowlist secret

        assert raised.value.code == "invalid_reply"

    def test_a_reply_that_cannot_be_read(self, monkeypatch):
        import requests

        from verbatim_rag.rerankers import JinaReranker

        def post(url, **settings):
            response = requests.Response()
            response.status_code = 200
            response._content = b"not json"
            return response

        monkeypatch.setattr(requests, "post", post)

        with pytest.raises(UpstreamError) as raised:
            JinaReranker(api_key="key").rerank(QUESTION, rows())  # pragma: allowlist secret

        assert (raised.value.code, raised.value.retryable) == ("invalid_reply", True)

    def test_cohere(self, monkeypatch):
        class ApiError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code

        class Client:
            def __init__(self, api_key):
                pass

            def rerank(self, **request):
                raise ApiError(503)

        monkeypatch.setitem(sys.modules, "cohere", types.SimpleNamespace(Client=Client))
        from verbatim_rag.rerankers import CohereReranker

        with pytest.raises(UpstreamError) as raised:
            CohereReranker(api_key="key").rerank(QUESTION, rows())  # pragma: allowlist secret

        assert (raised.value.code, raised.value.service) == ("server_error", "reranker")
