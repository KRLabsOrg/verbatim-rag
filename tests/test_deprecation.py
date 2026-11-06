"""Tests for the deprecation warnings of verbatim_core and verbatim_rag."""

import asyncio
import importlib
import inspect
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest
from verbatim_core import (
    Chunk,
    ConfigurationError,
    Extraction,
    TransformConfig,
    Transformer,
    VerbatimDeprecationWarning,
)
from verbatim_core._deprecation import REMOVED_IN, deprecated, warn_deprecated
from verbatim_core.extractors import (
    LLMSpanExtractor,
    ModelSpanExtractor,
    SemanticHighlightExtractor,
    SpanExtractor,
)
from verbatim_core.llm_client import LLMClient
from verbatim_core.providers import RAGProvider
from verbatim_core.templates import TemplateManager
from verbatim_core.templates.question_specific import QuestionSpecificTemplate
from verbatim_core.templates.structured import StructuredTemplate
from verbatim_core.transform import VerbatimTransform, verbatim_query, verbatim_query_async

try:
    import verbatim_rag
except ImportError:
    verbatim_rag = None

OLD_EXTRACT_MESSAGE = (
    f"old_extract is deprecated since 0.3.0 and will be removed in {REMOVED_IN}. Use extract()."
)

_UNSET = object()


def build_extractor(model=_UNSET):
    """Stands in for a library function with a deprecated argument."""
    if model is not _UNSET:
        warn_deprecated("model", use="llm_client")
    return "extractor"


def test_is_a_deprecation_warning():
    assert issubclass(VerbatimDeprecationWarning, DeprecationWarning)


class TestDeprecatedFunction:
    def test_warns_with_the_library_category_and_message_when_called(self):
        @deprecated("old_extract", use="extract()")
        def old_extract():
            return "spans"

        with pytest.warns(VerbatimDeprecationWarning) as record:
            assert old_extract() == "spans"
        assert [str(w.message) for w in record] == [OLD_EXTRACT_MESSAGE]

    def test_decorating_does_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            @deprecated("old_extract", use="extract()")
            def old_extract():
                return "spans"

        assert caught == []

    def test_is_marked_for_type_checkers(self):
        @deprecated("old_extract", use="extract()")
        def old_extract():
            return "spans"

        assert old_extract.__deprecated__ == OLD_EXTRACT_MESSAGE


class TestDeprecatedClass:
    def test_warns_when_used_not_when_its_module_is_imported(self, tmp_path, monkeypatch):
        (tmp_path / "legacy_extractors.py").write_text(
            "from verbatim_core._deprecation import deprecated\n"
            "\n"
            "\n"
            '@deprecated("LegacyExtractor", use="NewExtractor")\n'
            "class LegacyExtractor:\n"
            "    pass\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "legacy_extractors", raising=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            module = importlib.import_module("legacy_extractors")
        assert not [w for w in caught if issubclass(w.category, VerbatimDeprecationWarning)]

        with pytest.warns(VerbatimDeprecationWarning, match="^LegacyExtractor is deprecated"):
            module.LegacyExtractor()


def test_the_test_suite_turns_a_deprecation_warning_into_an_error():
    with pytest.raises(VerbatimDeprecationWarning):
        build_extractor(model="gpt-4o-mini")


class TestDeprecatedArgument:
    def test_does_not_warn_when_the_argument_is_left_out(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert build_extractor() == "extractor"
        assert caught == []

    def test_warns_at_the_line_that_passed_it(self):
        with pytest.warns(VerbatimDeprecationWarning) as record:
            call_line = inspect.currentframe().f_lineno + 1
            build_extractor(model="gpt-4o-mini")

        [warning] = record
        assert str(warning.message) == (
            f"model is deprecated since 0.3.0 and will be removed in {REMOVED_IN}. Use llm_client."
        )
        assert (warning.filename, warning.lineno) == (__file__, call_line)


# Every old entry point: a call of it, and the name and replacement its warning gives


def deprecation(name: str, use: str) -> str:
    return f"{name} is deprecated since 0.3.0 and will be removed in {REMOVED_IN}. Use {use}."


QUESTION = "When did the bridge open?"
CHUNKS = [Chunk(id="c1", text="The bridge opened in 1932.")]
USE_TRANSFORMER = "VerbatimRAG(index, transformer=Transformer(extractor, templates))"
CONTEXTUAL = TransformConfig(template_mode="contextual")
needs_verbatim_rag = pytest.mark.skipif(
    verbatim_rag is None, reason="needs the full verbatim-rag package"
)


@pytest.fixture(autouse=True)
def nothing_is_downloaded_or_sent(monkeypatch):
    """Model packages can't be imported, so loading a model fails after its warning, and LLM
    clients get a key without calling their service."""
    for package in ["torch", "transformers", "sentence_transformers"]:
        monkeypatch.setitem(sys.modules, package, None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


class SpansOnlyExtractor(SpanExtractor):
    """An extractor written before 0.3.0."""

    def extract_spans(self, question, search_results):
        return {}


class CurrentExtractor(SpanExtractor):
    def extract(self, question, chunks, *, config=None):
        return Extraction()


class StubHighlighter:
    def to(self, device):
        return self

    def eval(self):
        return self

    def process(self, question, context, **settings):
        return {"spans": []}


class Retrieved(Exception):
    """Ends a call when it retrieves, after it has warned."""


class StoppingProvider(RAGProvider):
    def retrieve(self, question, k=5, filter=None):
        raise Retrieved

    async def retrieve_async(self, question, k=5, filter=None):
        raise Retrieved


def awaited(make_awaitable):
    """Await in a coroutine of this module, the caller the warning points at."""

    async def call():
        return await make_awaitable()

    return asyncio.run(call())


def fails(error, call):
    with pytest.raises(error):
        call()


def llm_client():
    return LLMClient("gpt-4o-mini")


class EmptyIndex:
    def query(self, text=None, k=5, **retrieval):
        return []


def rag():
    transformer = Transformer(CurrentExtractor(), TemplateManager())
    return verbatim_rag.VerbatimRAG(EmptyIndex(), transformer=transformer)


def old_module_attribute(module, name):
    return getattr(importlib.import_module(module), name)


ENTRY_POINTS = [
    # Extractors
    pytest.param(
        lambda: SpansOnlyExtractor().extract(QUESTION, CHUNKS),
        "Implementing only extract_spans() in SpansOnlyExtractor",
        "extract()",
        id="extract of an extract_spans extractor",
    ),
    pytest.param(
        lambda: awaited(lambda: SpansOnlyExtractor().aextract(QUESTION, CHUNKS)),
        "Implementing only extract_spans() in SpansOnlyExtractor",
        "extract()",
        id="aextract of an extract_spans extractor",
    ),
    pytest.param(
        lambda: CurrentExtractor().extract_spans(QUESTION, CHUNKS),
        "SpanExtractor.extract_spans",
        "extract()",
        id="SpanExtractor.extract_spans",
    ),
    pytest.param(
        lambda: awaited(lambda: CurrentExtractor().extract_spans_async(QUESTION, CHUNKS)),
        "SpanExtractor.extract_spans_async",
        "aextract()",
        id="SpanExtractor.extract_spans_async",
    ),
    pytest.param(
        lambda: LLMSpanExtractor(llm_client()).extract_spans(QUESTION, []),
        "LLMSpanExtractor.extract_spans",
        "extract()",
        id="LLMSpanExtractor.extract_spans",
    ),
    pytest.param(
        lambda: awaited(lambda: LLMSpanExtractor(llm_client()).extract_spans_async(QUESTION, [])),
        "LLMSpanExtractor.extract_spans_async",
        "aextract()",
        id="LLMSpanExtractor.extract_spans_async",
    ),
    pytest.param(
        lambda: LLMSpanExtractor(),
        "LLMSpanExtractor without an llm_client",
        "LLMSpanExtractor(LLMClient(model))",
        id="LLMSpanExtractor without a client",
    ),
    pytest.param(
        lambda: LLMSpanExtractor(llm_client(), model="gpt-4o"),
        "Passing model to LLMSpanExtractor",
        "the model of the llm_client",
        id="LLMSpanExtractor(model=...)",
    ),
    pytest.param(
        lambda: LLMSpanExtractor(llm_client(), batch_size=4, system_prompt="Quote."),
        "Passing batch_size, system_prompt to LLMSpanExtractor",
        "config=LLMExtractConfig(...)",
        id="LLMSpanExtractor settings",
    ),
    pytest.param(
        lambda: LLMSpanExtractor(llm_client(), max_display_spans=5),
        "Passing max_display_spans to LLMSpanExtractor",
        "TransformConfig(max_display_spans=...)",
        id="LLMSpanExtractor(max_display_spans=...)",
    ),
    pytest.param(
        lambda: fails(ConfigurationError, lambda: ModelSpanExtractor(model_path="org/model")),
        "Loading a model in the ModelSpanExtractor constructor",
        "ModelSpanExtractor.from_pretrained(model_id)",
        id="ModelSpanExtractor(model_path=...)",
    ),
    pytest.param(
        lambda: fails(ConfigurationError, lambda: SemanticHighlightExtractor("org/model")),
        "Loading a model in the SemanticHighlightExtractor constructor",
        "SemanticHighlightExtractor.from_pretrained(model_id)",
        id="SemanticHighlightExtractor(model_name)",
    ),
    pytest.param(
        lambda: ModelSpanExtractor(StubHighlighter(), threshold=0.5),
        "Passing threshold to ModelSpanExtractor",
        "config=ModelExtractConfig(...)",
        id="ModelSpanExtractor settings",
    ),
    pytest.param(
        lambda: ModelSpanExtractor(StubHighlighter(), extraction_mode="individual"),
        "Passing extraction_mode to ModelSpanExtractor",
        "nothing in its place, as it has no effect",
        id="ModelSpanExtractor(extraction_mode=...)",
    ),
    pytest.param(
        lambda: ModelSpanExtractor(StubHighlighter(), max_display_spans=5),
        "Passing max_display_spans to ModelSpanExtractor",
        "TransformConfig(max_display_spans=...)",
        id="ModelSpanExtractor(max_display_spans=...)",
    ),
    pytest.param(
        lambda: SemanticHighlightExtractor(StubHighlighter(), output_mode="spans"),
        "Passing output_mode to SemanticHighlightExtractor",
        "config=SemanticHighlightConfig(...)",
        id="SemanticHighlightExtractor settings",
    ),
    # Templates
    pytest.param(
        lambda: TemplateManager(default_mode="contextual"),
        "Contextual template mode without an LLM client",
        "TemplateManager(llm_client=...)",
        id="TemplateManager contextual without a client",
    ),
    pytest.param(
        lambda: TemplateManager().use_contextual_mode(),
        "Contextual template mode without an LLM client",
        "TemplateManager(llm_client=...)",
        id="use_contextual_mode without a client",
    ),
    pytest.param(
        lambda: Transformer(CurrentExtractor(), TemplateManager()).transform(
            QUESTION, CHUNKS, config=CONTEXTUAL
        ),
        "Contextual template mode without an LLM client",
        "TemplateManager(llm_client=...)",
        id="Transformer call contextual without a client",
    ),
    pytest.param(
        lambda: Transformer(CurrentExtractor(), TemplateManager(), config=CONTEXTUAL),
        "Contextual template mode without an LLM client",
        "TemplateManager(llm_client=...)",
        id="Transformer default config contextual without a client",
    ),
    pytest.param(
        lambda: TemplateManager().set_mode("random"),
        "TemplateManager.set_mode",
        "TransformConfig(template_mode=...)",
        id="TemplateManager.set_mode",
    ),
    pytest.param(
        lambda: TemplateManager().set_mode("structured"),
        "TemplateManager.set_mode",
        "TemplateManager.use_structured_mode()",
        id="TemplateManager.set_mode structured",
    ),
    pytest.param(
        lambda: setattr(TemplateManager(), "current_mode", "random"),
        "Assigning TemplateManager.current_mode",
        "TransformConfig(template_mode=...)",
        id="TemplateManager.current_mode =",
    ),
    pytest.param(
        lambda: setattr(TemplateManager(), "citation_mode", "hidden"),
        "Assigning TemplateManager.citation_mode",
        "TransformConfig(citation_mode=...)",
        id="TemplateManager.citation_mode =",
    ),
    pytest.param(
        lambda: TemplateManager().set_citation_mode("hidden"),
        "TemplateManager.set_citation_mode",
        "TransformConfig(citation_mode=...)",
        id="TemplateManager.set_citation_mode",
    ),
    pytest.param(
        lambda: TemplateManager().set_citation_format("({number})"),
        "TemplateManager.set_citation_format",
        "TransformConfig(citation_format=...)",
        id="TemplateManager.set_citation_format",
    ),
    pytest.param(
        lambda: TemplateManager().set_rag_system(None),
        "TemplateManager.set_rag_system",
        "VerbatimRAG.query_async",
        id="TemplateManager.set_rag_system",
    ),
    pytest.param(
        lambda: StructuredTemplate().set_rag_system(None),
        "StructuredTemplate.set_rag_system",
        "VerbatimRAG.query_async",
        id="StructuredTemplate.set_rag_system",
    ),
    pytest.param(
        lambda: fails(
            ValueError, lambda: awaited(lambda: StructuredTemplate().fill_async(QUESTION))
        ),
        "StructuredTemplate.fill_async",
        "VerbatimRAG.query_async",
        id="StructuredTemplate.fill_async",
    ),
    pytest.param(
        lambda: QuestionSpecificTemplate(model_name="org/embedder"),
        "QuestionSpecificTemplate(model_name=..., device=...)",
        "QuestionSpecificTemplate.from_pretrained(model_name, device)",
        id="QuestionSpecificTemplate(model_name=...)",
    ),
    pytest.param(
        lambda: fails(
            ConfigurationError,
            lambda: QuestionSpecificTemplate(
                templates=[{"template": "[DISPLAY_SPANS]", "examples": [QUESTION]}]
            ),
        ),
        "QuestionSpecificTemplate without an embedder",
        "QuestionSpecificTemplate.from_pretrained(model_name)",
        id="QuestionSpecificTemplate without an embedder",
    ),
    # The transform helpers
    pytest.param(
        lambda: VerbatimTransform(llm_client=llm_client()),
        "VerbatimTransform",
        "Transformer",
        id="VerbatimTransform",
    ),
    pytest.param(
        lambda: fails(Retrieved, lambda: verbatim_query(StoppingProvider(), QUESTION)),
        "verbatim_query",
        "RAGProvider.retrieve, then Transformer.transform",
        id="verbatim_query",
    ),
    pytest.param(
        lambda: fails(
            Retrieved, lambda: awaited(lambda: verbatim_query_async(StoppingProvider(), QUESTION))
        ),
        "verbatim_query_async",
        "RAGProvider.retrieve_async, then Transformer.atransform",
        id="verbatim_query_async",
    ),
    # verbatim-rag
    pytest.param(
        lambda: verbatim_rag.VerbatimRAG(EmptyIndex()),
        "VerbatimRAG without a transformer",
        USE_TRANSFORMER,
        id="VerbatimRAG without a transformer",
        marks=needs_verbatim_rag,
    ),
    pytest.param(
        lambda: verbatim_rag.VerbatimRAG(EmptyIndex(), model="gpt-4o", template_mode="static"),
        "VerbatimRAG without a transformer",
        USE_TRANSFORMER,
        id="VerbatimRAG old arguments",
        marks=needs_verbatim_rag,
    ),
    pytest.param(
        lambda: verbatim_rag.VerbatimRAG(EmptyIndex(), k=3, transformer=rag().transformer),
        "VerbatimRAG(k=...)",
        "query(question, k=...)",
        id="VerbatimRAG(k=...)",
        marks=needs_verbatim_rag,
    ),
    pytest.param(
        lambda: setattr(rag(), "extractor", CurrentExtractor()),
        "Assigning VerbatimRAG.extractor",
        USE_TRANSFORMER,
        id="VerbatimRAG.extractor =",
        marks=needs_verbatim_rag,
    ),
    pytest.param(
        lambda: setattr(rag(), "template_manager", TemplateManager()),
        "Assigning VerbatimRAG.template_manager",
        USE_TRANSFORMER,
        id="VerbatimRAG.template_manager =",
        marks=needs_verbatim_rag,
    ),
    pytest.param(
        lambda: setattr(rag(), "max_display_spans", 3),
        "Assigning VerbatimRAG.max_display_spans",
        "Transformer(..., config=TransformConfig(max_display_spans=...))",
        id="VerbatimRAG.max_display_spans =",
        marks=needs_verbatim_rag,
    ),
    pytest.param(
        lambda: old_module_attribute("verbatim_rag.document", "Chunk"),
        "verbatim_rag.document.Chunk",
        "DocumentChunk",
        id="verbatim_rag.document.Chunk",
        marks=needs_verbatim_rag,
    ),
    pytest.param(
        lambda: old_module_attribute("verbatim_rag.transform", "VerbatimTransform"),
        "verbatim_rag.transform.VerbatimTransform",
        "Transformer",
        id="verbatim_rag.transform.VerbatimTransform",
        marks=needs_verbatim_rag,
    ),
    pytest.param(
        lambda: old_module_attribute("verbatim_rag.transform", "SearchResult"),
        "verbatim_rag.transform.SearchResult",
        "verbatim_rag.vector_stores.SearchResult",
        id="verbatim_rag.transform.SearchResult",
        marks=needs_verbatim_rag,
    ),
    pytest.param(
        lambda: verbatim_rag.StreamingRAG(rag()),
        "StreamingRAG",
        "VerbatimRAG.query_stream",
        id="StreamingRAG",
        marks=needs_verbatim_rag,
    ),
    pytest.param(
        lambda: verbatim_rag.StreamingRAG(rag()).stream_query(QUESTION),
        "StreamingRAG.stream_query",
        "VerbatimRAG.query_stream",
        id="StreamingRAG.stream_query",
        marks=needs_verbatim_rag,
    ),
    pytest.param(
        lambda: verbatim_rag.StreamingRAG(rag()).stream_query_sync(QUESTION),
        "StreamingRAG.stream_query_sync",
        "VerbatimRAG.query_stream",
        id="StreamingRAG.stream_query_sync",
        marks=needs_verbatim_rag,
    ),
]


@pytest.mark.parametrize("call, name, use", ENTRY_POINTS)
def test_an_old_entry_point_warns_with_its_message_in_the_module_that_calls_it(call, name, use):
    with pytest.warns(VerbatimDeprecationWarning) as record:
        call()

    [warning] = [w for w in record if str(w.message) == deprecation(name, use)]
    assert warning.category is VerbatimDeprecationWarning
    assert warning.filename == __file__


REPO_ROOT = Path(__file__).resolve().parents[1]
MISSING_DEPENDENCY = 77


def run_in_fresh_interpreter(script: str) -> subprocess.CompletedProcess:
    """Run a script from the repository root, skipping when a third-party dependency is missing."""
    guarded = (
        "try:\n"
        + textwrap.indent(textwrap.dedent(script), "    ")
        + f"\nexcept ModuleNotFoundError:\n    raise SystemExit({MISSING_DEPENDENCY})\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", guarded], cwd=REPO_ROOT, capture_output=True, text=True
    )
    if result.returncode == MISSING_DEPENDENCY:
        pytest.skip("the package's dependencies are not installed")
    return result


@pytest.mark.parametrize("package", ["verbatim_core", "verbatim_rag"])
def test_importing_in_a_fresh_interpreter_warns_configures_and_prints_nothing(package):
    result = run_in_fresh_interpreter(
        f"""
        import logging
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import {package}
        from verbatim_core import VerbatimDeprecationWarning
        ours = [w for w in caught if issubclass(w.category, VerbatimDeprecationWarning)]
        assert not ours, [str(w.message) for w in ours]
        assert (logging.root.level, logging.root.handlers) == (logging.WARNING, [])
        # Only ours: the packages verbatim_rag imports, such as torch, configure their own loggers
        configured = {{
            name: (logger.level, logger.handlers, logger.propagate)
            for name, logger in logging.Logger.manager.loggerDict.items()
            if name.split(".")[0] in ("verbatim_core", "verbatim_rag")
            and isinstance(logger, logging.Logger)
            and (logger.level, logger.handlers, logger.propagate) != (logging.NOTSET, [], True)
        }}
        assert not configured, configured
        """
    )
    assert result.returncode == 0, result.stderr
    assert (result.stdout, result.stderr) == ("", "")


def test_verbatim_rag_exports_the_same_warning_class():
    result = run_in_fresh_interpreter(
        """
        import verbatim_core
        import verbatim_rag
        assert verbatim_rag.VerbatimDeprecationWarning is verbatim_core.VerbatimDeprecationWarning
        """
    )
    assert result.returncode == 0, result.stderr
