"""VerbatimRAG on a Transformer, against an in-memory index. They need the full verbatim-rag package."""

import logging
import re
import sys
import types
import warnings

import pytest
from verbatim_core import (
    ConfigurationError,
    ExtractedSpan,
    Extraction,
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


@pytest.fixture
def no_deprecation_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", VerbatimDeprecationWarning)
        yield


async def query(rag, asynchronous, *args, **settings):
    if asynchronous:
        return await rag.query_async(*args, **settings)
    return rag.query(*args, **settings)


class TestConstruction:
    def test_with_a_transformer_nothing_warns(self, no_deprecation_warnings):
        transformer = Transformer(OpenedExtractor(), TemplateManager())

        rag = verbatim_rag.VerbatimRAG(RecordingIndex(), transformer=transformer)

        assert (rag.transformer, rag.extractor, rag.template_manager) == (
            transformer,
            transformer.extractor,
            transformer.templates,
        )

    def test_the_transformer_may_be_passed_by_position(self, no_deprecation_warnings):
        transformer = Transformer(OpenedExtractor(), TemplateManager())

        assert verbatim_rag.VerbatimRAG(RecordingIndex(), transformer).transformer is transformer

    def test_the_arguments_may_follow_the_transformer_by_position(self, no_deprecation_warnings):
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


@pytest.mark.parametrize("asynchronous", [False, True])
class TestQuery:
    async def test_the_chunks_are_the_rows_of_the_index(self, asynchronous):
        extractor = OpenedExtractor()

        await query(build(extractor=extractor), asynchronous, QUESTION)

        assert [(c.id, c.text, c.metadata) for c in extractor.chunks] == [
            (row.id, row.text, row.metadata) for row in rows()
        ]
        assert all(c.source_start is None and c.source_end is None for c in extractor.chunks)

    async def test_retrieval_settings_apply_to_one_call(self, asynchronous):
        index = RecordingIndex()
        rag = build(index)
        settings = {
            "k": 2,
            "filter": "year > 1900",
            "hybrid_weights": {"dense": 1.0},
            "rrf_k": 10,
            "search_params": {"ef": 64},
        }

        await query(rag, asynchronous, QUESTION, **settings)
        await query(rag, asynchronous, QUESTION)

        assert index.queries == [
            settings,
            {"k": 5, "filter": None, "hybrid_weights": None, "rrf_k": 60, "search_params": None},
        ]

    async def test_the_answer_cites_the_spans_at_their_offsets(self, asynchronous):
        response = await query(build(), asynchronous, QUESTION)

        assert "[1] The bridge opened in 1932." in response.answer
        assert [(d.title, [h.text for h in d.highlights]) for d in response.documents] == [
            ("Bridge", ["The bridge opened in 1932."]),
            ("Tunnel", ["The tunnel opened in 1992."]),
            ("", []),
        ]

    async def test_the_settings_of_the_transformer_apply(self, asynchronous):
        transformer = Transformer(
            OpenedExtractor(), TemplateManager(), config=TransformConfig(max_display_spans=1)
        )
        rag = verbatim_rag.VerbatimRAG(RecordingIndex(), transformer=transformer)

        response = await query(rag, asynchronous, QUESTION)

        assert [c.type for c in response.structured_answer.citations] == ["display", "reference"]

    async def test_a_failed_extraction_raises(self, asynchronous):
        error = UpstreamError("The LLM service failed.", code="server_error", service="llm")

        with pytest.raises(UpstreamError):
            await query(build(extractor=OpenedExtractor(error=error)), asynchronous, QUESTION)

    async def test_the_steps_are_logged_at_debug_and_nothing_is_printed(
        self, asynchronous, caplog, capsys
    ):
        caplog.set_level(logging.DEBUG, logger="verbatim_rag")

        await query(build(), asynchronous, QUESTION)

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


@pytest.mark.parametrize("asynchronous", [False, True])
class TestReranker:
    async def test_the_reranked_order_is_the_order_of_the_chunks(self, asynchronous):
        extractor = OpenedExtractor()

        await query(
            build(extractor=extractor, reranker=ReversingReranker()), asynchronous, QUESTION
        )

        assert [chunk.id for chunk in extractor.chunks] == ["row-5", "row-3", "row-7"]

    async def test_a_failed_service_falls_back_to_the_retrieval_order(self, asynchronous, caplog):
        error = UpstreamError("The reranker failed.", code="timeout", service="reranker")
        extractor = OpenedExtractor()
        rag = build(extractor=extractor, reranker=FailingReranker(error))

        response = await query(rag, asynchronous, QUESTION)

        assert [chunk.id for chunk in extractor.chunks] == ["row-7", "row-3", "row-5"]
        assert [warning.code for warning in response.warnings] == ["reranker_fallback"]
        [record] = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert (record.name, record.event, record.warning_code) == (
            "verbatim_rag.core",
            "fallback_used",
            "reranker_fallback",
        )

    async def test_another_error_propagates(self, asynchronous):
        rag = build(reranker=FailingReranker(KeyError("bug")))

        with pytest.raises(KeyError):
            await query(rag, asynchronous, QUESTION)


class StructuredLLM:
    model = "fake-llm"
    temperature = 0.0

    def extract_structured(self, question, template, placeholders, documents):
        return {"OPENING": [{"text": "The bridge opened in 1932.", "doc": 0}]}

    async def extract_structured_async(self, question, template, placeholders, documents):
        return self.extract_structured(question, template, placeholders, documents)


@pytest.mark.parametrize("asynchronous", [False, True])
class TestStructuredMode:
    async def test_the_client_of_verbatim_rag_answers(self, asynchronous):
        rag = build(llm_client=StructuredLLM())
        rag.template_manager.use_structured_mode(template="## Opening\n[OPENING]")

        response = await query(rag, asynchronous, QUESTION)

        assert "The bridge opened in 1932." in response.answer

    async def test_a_reranker_fallback_is_a_warning_of_the_response(self, asynchronous):
        error = UpstreamError("The reranker failed.", code="timeout", service="reranker")
        rag = build(llm_client=StructuredLLM(), reranker=FailingReranker(error))
        rag.template_manager.use_structured_mode(template="## Opening\n[OPENING]")

        response = await query(rag, asynchronous, QUESTION)

        assert [warning.code for warning in response.warnings] == ["reranker_fallback"]

    async def test_without_a_client_it_is_a_configuration_error(self, asynchronous):
        rag = build()
        rag.template_manager.use_structured_mode(template="## Opening\n[OPENING]")

        with pytest.raises(ConfigurationError):
            await query(rag, asynchronous, QUESTION)


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

    def test_a_document_with_chunks_round_trips(self, no_deprecation_warnings):
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
            JinaReranker(api_key="key").rerank(QUESTION, rows())

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
            JinaReranker(api_key="key").rerank(QUESTION, rows())

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
            CohereReranker(api_key="key").rerank(QUESTION, rows())

    def test_an_unreadable_cohere_reply_is_an_upstream_error(self, monkeypatch):
        class Client:
            def __init__(self, api_key):
                pass

            def rerank(self, **request):
                return types.SimpleNamespace()

        monkeypatch.setitem(sys.modules, "cohere", types.SimpleNamespace(Client=Client))
        from verbatim_rag.rerankers import CohereReranker

        with pytest.raises(UpstreamError) as raised:
            CohereReranker(api_key="key").rerank(QUESTION, rows())

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
            JinaReranker(api_key="key").rerank(QUESTION, rows())

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
            CohereReranker(api_key="key").rerank(QUESTION, rows())

        assert (raised.value.code, raised.value.service) == ("server_error", "reranker")
