"""VerbatimTransform, the deprecated adapter over the Transformer, with a stub extractor."""

import re
from unittest.mock import MagicMock

import pytest
from verbatim_core import (
    ConfigurationError,
    ExtractedSpan,
    Extraction,
    InvalidInputError,
    ResultWarning,
    VerbatimDeprecationWarning,
    VerbatimTransform,
)
from verbatim_core.extractors import SpanExtractor
from verbatim_core.providers import RAGProvider
from verbatim_core.templates import TemplateManager
from verbatim_core.universal_document import UniversalDocument

QUESTION = "When did the bridge open?"


class SentenceExtractor(SpanExtractor):
    """Extracts every sentence with the word "opened", and records the chunks it was given."""

    def __init__(self):
        self.chunks = []

    def extract(self, question, chunks, *, config=None):
        self.chunks = list(chunks)
        spans = [
            ExtractedSpan(
                chunk_id=chunk.id,
                start=match.start(),
                end=match.end(),
                text=match.group(),
                score=1.0,
            )
            for chunk in chunks
            for match in re.finditer(r"[^.\s][^.]*opened[^.]*\.", chunk.text)
        ]
        return Extraction(spans=spans)


def build_transform(extractor=None, **settings):
    with pytest.warns(VerbatimDeprecationWarning, match="VerbatimTransform"):
        return VerbatimTransform(
            llm_client=MagicMock(),
            extractor=extractor or SentenceExtractor(),
            template_manager=TemplateManager(default_mode="static"),
            **settings,
        )


class TestContext:
    def test_items_become_chunks_with_positional_ids(self):
        extractor = SentenceExtractor()
        context = [
            {"content": "The bridge opened in 1932.", "title": "Bridge", "source": "bridge.md"},
            {"text": "Tolls are electronic.", "metadata": {"page": 3}},
        ]

        build_transform(extractor).transform(QUESTION, context)

        assert [(chunk.id, chunk.text, chunk.metadata) for chunk in extractor.chunks] == [
            ("ctx_0", "The bridge opened in 1932.", {"title": "Bridge", "source": "bridge.md"}),
            ("ctx_1", "Tolls are electronic.", {"title": "", "source": "", "page": 3}),
        ]

    def test_an_object_with_text_is_a_chunk(self):
        extractor = SentenceExtractor()
        item = MagicMock(text="The bridge opened in 1932.", metadata={"title": "Bridge"})

        build_transform(extractor).transform(QUESTION, [item])

        assert [(chunk.text, chunk.metadata) for chunk in extractor.chunks] == [
            ("The bridge opened in 1932.", {"title": "Bridge"})
        ]

    @pytest.mark.parametrize(
        "item, code, param",
        [
            (42, "invalid_value", "context[1]"),
            ({"title": "The bridge"}, "missing_field", "context[1]"),
            ({"text": ""}, "invalid_value", "context[1]"),
            ({"content": ["The bridge opened in 1932."]}, "invalid_value", "context[1]"),
            (MagicMock(text="", metadata={}), "invalid_value", "context[1]"),
            (
                {"content": "The bridge opened.", "metadata": "bridge"},
                "invalid_value",
                "context[1].metadata",
            ),
            (
                {"content": "The bridge opened.", "metadata": {1: "bridge"}},
                "invalid_value",
                "context[1].metadata",
            ),
            ({"content": "The bridge opened.", "title": None}, "invalid_value", "context[1].title"),
            (
                {"content": "The bridge opened.", "metadata": {"source": ["bridge.md"]}},
                "invalid_value",
                "context[1].source",
            ),
        ],
    )
    async def test_an_item_of_the_wrong_shape_is_an_input_error_before_extraction(
        self, item, code, param
    ):
        extractor = SentenceExtractor()
        transform = build_transform(extractor)
        context = [{"content": "Tolls are electronic."}, item]

        with pytest.raises(InvalidInputError) as raised:
            transform.transform(QUESTION, context)
        with pytest.raises(InvalidInputError):
            await transform.transform_async(QUESTION, context)

        assert (raised.value.code, raised.value.param) == (code, param)
        assert "bridge" not in str(raised.value).lower()
        assert extractor.chunks == []


class TestVerbatimTransform:
    def test_warns_at_the_line_that_built_it(self):
        with pytest.warns(VerbatimDeprecationWarning, match="Use Transformer") as warned:
            VerbatimTransform(llm_client=MagicMock(), template_manager=TemplateManager())

        assert warned[0].filename == __file__

    async def test_answers_with_the_spans_of_the_extractor(self):
        transform = build_transform()
        context = [{"content": "Tolls are electronic. The bridge opened in 1932."}]

        response = transform.transform(QUESTION, context)

        assert response.question == QUESTION
        assert "[1] The bridge opened in 1932." in response.answer
        assert [h.text for h in response.documents[0].highlights] == ["The bridge opened in 1932."]
        assert await transform.transform_async(QUESTION, context) == response

    def test_the_warnings_of_the_extraction_reach_the_response(self):
        warning = ResultWarning(code="quote_not_found", message="A quote of chunks[0] is missing.")

        class WarningExtractor(SentenceExtractor):
            def extract(self, question, chunks, *, config=None):
                extraction = super().extract(question, chunks)
                return extraction.model_copy(update={"warnings": [warning]})

        response = build_transform(WarningExtractor()).transform(
            QUESTION, [{"content": "The bridge opened in 1932."}]
        )

        assert response.warnings == [warning]

    def test_max_display_spans_applies(self):
        context = [{"content": "The bridge opened in 1932. The tunnel opened in 1992."}]

        response = build_transform(max_display_spans=1).transform(QUESTION, context)

        assert [c.type for c in response.structured_answer.citations] == ["display", "reference"]

    def test_the_template_settings_of_the_template_manager_apply(self):
        transform = build_transform()
        with pytest.warns(VerbatimDeprecationWarning):
            transform.template_manager.set_citation_mode("hidden")

        response = transform.transform(QUESTION, [{"content": "The bridge opened in 1932."}])

        assert "[1]" not in response.answer

    def test_structured_template_mode_is_a_configuration_error(self):
        with pytest.warns(VerbatimDeprecationWarning):
            transform = VerbatimTransform(llm_client=MagicMock(), template_mode="structured")

        with pytest.raises(ConfigurationError):
            transform.transform(QUESTION, [{"content": "The bridge opened in 1932."}])

    def test_without_spans_the_answer_says_so(self):
        response = build_transform().transform(QUESTION, [{"content": "Tolls are electronic."}])

        assert "No relevant information" in response.answer

    async def test_an_extractor_that_does_not_subclass_span_extractor_still_works(self):
        class DuckExtractor:
            def extract_spans(self, question, search_results):
                return {result.text: ["opened in 1932"] for result in search_results}

        transform = build_transform(DuckExtractor())
        context = [{"content": "The bridge opened in 1932."}]

        with pytest.warns(VerbatimDeprecationWarning, match="DuckExtractor") as w:
            response = transform.transform(QUESTION, context)
        with pytest.warns(VerbatimDeprecationWarning):
            assert await transform.transform_async(QUESTION, context) == response

        assert "opened in 1932" in response.answer
        assert w[0].filename == __file__

    async def test_the_async_extract_spans_of_such_an_extractor_is_used(self):
        class DuckExtractor:
            def extract_spans(self, question, search_results):
                return {result.text: ["opened in 1932"] for result in search_results}

            async def extract_spans_async(self, question, search_results):
                return {result.text: ["The bridge opened"] for result in search_results}

        transform = build_transform(DuckExtractor())

        with pytest.warns(VerbatimDeprecationWarning):
            response = await transform.transform_async(
                QUESTION, [{"content": "The bridge opened in 1932."}]
            )

        assert "[1] The bridge opened" in response.answer

    def test_an_extractor_that_implements_only_extract_spans_still_works(self):
        class OldExtractor(SpanExtractor):
            def extract_spans(self, question, search_results):
                return {result.text: ["opened in 1932"] for result in search_results}

        transform = build_transform(OldExtractor())

        with pytest.warns(VerbatimDeprecationWarning, match="Implementing only extract_spans") as w:
            response = transform.transform(QUESTION, [{"content": "The bridge opened in 1932."}])

        assert "opened in 1932" in response.answer
        assert w[0].filename == __file__


class TestVerbatimQuery:
    class Provider(RAGProvider):
        def retrieve(self, question, k=5, filter=None):
            return []

    class Client:
        model = "fake-llm"
        temperature = 0.0

    async def test_warns_at_the_call(self, monkeypatch):
        from verbatim_core import transform

        monkeypatch.setattr(transform, "LLMClient", self.Client)
        with pytest.warns(VerbatimDeprecationWarning, match="verbatim_query is") as warned:
            transform.verbatim_query(self.Provider(), QUESTION)
        with pytest.warns(VerbatimDeprecationWarning, match="verbatim_query_async") as warned_async:
            answering = transform.verbatim_query_async(self.Provider(), QUESTION)
        await answering

        assert warned[0].filename == warned_async[0].filename == __file__


class TestTheModuleOfVerbatimRag:
    """verbatim_rag.transform held a copy of VerbatimTransform. It now re-exports the core one."""

    @pytest.fixture
    def moved(self):
        return pytest.importorskip("verbatim_rag.transform")

    def test_its_names_are_the_core_names_and_warn(self, moved):
        with pytest.warns(VerbatimDeprecationWarning, match="Use Transformer") as w:
            moved_class = moved.VerbatimTransform

        assert moved_class is VerbatimTransform
        assert w[0].filename == __file__

    def test_the_other_names_it_had_still_work_and_warn(self, moved):
        from verbatim_core.llm_client import LLMClient

        from verbatim_rag.vector_stores import SearchResult

        with pytest.warns(VerbatimDeprecationWarning, match="verbatim_rag.transform.LLMClient"):
            assert moved.LLMClient is LLMClient
        with pytest.warns(VerbatimDeprecationWarning, match="verbatim_rag.vector_stores"):
            assert moved.SearchResult is SearchResult

    def test_a_star_import_gives_its_three_names(self, moved):
        namespace = {}
        with pytest.warns(VerbatimDeprecationWarning):
            exec("from verbatim_rag.transform import *", namespace)

        assert {"VerbatimTransform", "verbatim_query", "verbatim_query_async"} <= set(namespace)

    def test_another_name_is_an_attribute_error(self, moved):
        with pytest.raises(AttributeError):
            moved.Nothing

    def test_the_package_root_exports_the_core_class_without_a_warning(self, moved, recwarn):
        import verbatim_rag

        assert verbatim_rag.VerbatimTransform is VerbatimTransform
        assert not [w for w in recwarn if issubclass(w.category, VerbatimDeprecationWarning)]


class TestUniversalDocument:
    def test_from_text(self):
        doc = UniversalDocument.from_text("Hello", title="T", source="S")
        assert doc.content == "Hello"
        assert doc.title == "T"

    def test_from_dict_content_key(self):
        doc = UniversalDocument.from_dict({"content": "Hello", "title": "T"})
        assert doc.content == "Hello"

    def test_from_dict_text_key(self):
        doc = UniversalDocument.from_dict({"text": "Hello"})
        assert doc.content == "Hello"

    def test_from_dict_missing_content_raises(self):
        with pytest.raises(ValueError):
            UniversalDocument.from_dict({"title": "no content"})

    def test_from_dict_bad_type_raises(self):
        with pytest.raises(TypeError):
            UniversalDocument.from_dict("not a dict")

    def test_to_context_roundtrip(self):
        doc = UniversalDocument(content="Hello", title="T", source="S", metadata={"k": "v"})
        ctx = doc.to_context()
        assert ctx["content"] == "Hello"
        doc2 = UniversalDocument.from_dict(ctx)
        assert doc2.content == doc.content
