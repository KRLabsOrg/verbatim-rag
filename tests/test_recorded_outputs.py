"""Today's output of the public entry points, compared with recordings in tests/recordings/.

The stateless-core work rebuilds or deprecates these entry points, and their adapters must keep
producing the same output. Run with RECORD_OUTPUTS=1 to rewrite the recordings after an
intended change, then review the diff.

The LLM is a fake OpenAI client that returns fixed quotes and templates, the model extractors
load fake torch and transformers packages, and VerbatimRAG searches an in-memory index.
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from verbatim_core import Transformer, UpstreamError, VerbatimDeprecationWarning
from verbatim_core.extractors import (
    LLMSpanExtractor,
    ModelSpanExtractor,
    SemanticHighlightExtractor,
    SpanExtractor,
)
from verbatim_core.llm_client import LLMClient
from verbatim_core.providers import RAGProvider
from verbatim_core.templates import TemplateManager
from verbatim_core.templates.random import RandomTemplate

RECORDINGS = Path(__file__).parent / "recordings"

QUESTION = "When did the Harbour Bridge open, and how many vehicles cross it?"

CHUNKS = [
    {
        "text": (
            "The Harbour Bridge opened in March 1932 after eight years of construction. "
            "It carries eight lanes of road traffic and two railway lines. "
            "Around 160,000 vehicles cross it each day."
        ),
        "title": "Harbour Bridge",
        "source": "bridge.md",
    },
    {
        # Spaced punctuation, as in OCR output: its first quote matches only in fuzzy mode.
        "text": (
            "Crews repaint the steel arch continuously ; one full coat takes about ten years . "
            "The arch rises 134 metres above the water."
        ),
        "title": "Maintenance",
        "source": "maintenance.md",
    },
    {
        "text": "Tolls are collected electronically in both directions.",
        "title": "Tolls",
        "source": "tolls.md",
    },
    {
        "text": (
            "The bridge was designed by a team led by John Bradfield. "
            "Its granite pylons carry no load."
        ),
        "title": "Design",
        "source": "design.md",
    },
]

# What the fake LLM quotes from each chunk. "Ferries run every ten minutes." is in no chunk.
QUOTES = {
    CHUNKS[0]["text"]: [
        "The Harbour Bridge opened in March 1932 after eight years of construction.",
        "Around 160,000 vehicles cross it each day.",
    ],
    CHUNKS[1]["text"]: [
        "Crews repaint the steel arch continuously; one full coat takes about ten years.",
        "The arch rises 134 metres above the water.",
    ],
    CHUNKS[2]["text"]: [],
    CHUNKS[3]["text"]: [
        "The bridge was designed by a team led by John Bradfield.",
        "Ferries run every ten minutes.",
    ],
}

# What the fake highlighter models mark in each chunk.
HIGHLIGHTS = {
    CHUNKS[0]["text"]: [
        "The Harbour Bridge opened in March 1932 after eight years of construction.",
        "Around 160,000 vehicles cross it each day.",
    ],
    CHUNKS[1]["text"]: ["The arch rises 134 metres above the water."],
    CHUNKS[2]["text"]: [],
    CHUNKS[3]["text"]: ["Its granite pylons carry no load."],
}


def assert_recorded(name, output):
    path = RECORDINGS / f"{name}.json"
    output = json.loads(json.dumps(output))
    if os.environ.get("RECORD_OUTPUTS"):
        path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    assert output == json.loads(path.read_text(encoding="utf-8"))


def spans_by_chunk(relevant_spans):
    return [{"chunk": text, "spans": spans} for text, spans in relevant_spans.items()]


def without_timing(events):
    return [{**event, "elapsed_ms": None} if "elapsed_ms" in event else event for event in events]


def search_results():
    return [SimpleNamespace(text=chunk["text"], metadata={}) for chunk in CHUNKS]


def test_no_chunk_repeats_the_text_of_its_spans():
    # Locating one occurrence per span (ADR 0001) then leaves the recordings unchanged.
    for chunk, texts in [*QUOTES.items(), *HIGHLIGHTS.items()]:
        assert all(chunk.count(text) <= 1 for text in texts)


# The fake OpenAI client


# What the fake LLM writes when asked for random templates
# One per line, as the prompt asks
GENERATED_TEMPLATES = ["**What the sources say:** [DISPLAY_SPANS]", "### In short: [DISPLAY_SPANS]"]

DOCUMENT_ENTRY = re.compile(r'"(doc(?:_\d+)?)": ("(?:[^"\\]|\\.)*")')
SPAN_PLACEHOLDERS = re.compile(r"\[SPAN_1\]\.\.\[SPAN_(\d+)\]")


def complete(messages, response_format=None, **settings):
    prompt = messages[-1]["content"]
    if response_format:
        documents = DOCUMENT_ENTRY.findall(prompt)
        content = json.dumps({key: QUOTES[json.loads(text)] for key, text in documents})
    elif "diverse answer templates" in prompt:
        content = "\n".join(GENERATED_TEMPLATES)
    else:
        count = int(SPAN_PLACEHOLDERS.search(prompt).group(1))
        content = "The sources say:\n" + "\n".join(f"- [SPAN_{i}]" for i in range(1, count + 1))
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


async def complete_async(**request):
    return complete(**request)


class FakeOpenAI:
    def __init__(self, **settings):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=complete))


class FakeAsyncOpenAI:
    def __init__(self, **settings):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=complete_async))


@pytest.fixture(autouse=True)
def fake_openai(monkeypatch):
    openai = SimpleNamespace(OpenAI=FakeOpenAI, AsyncOpenAI=FakeAsyncOpenAI)
    monkeypatch.setattr("verbatim_core.llm_client.openai", openai)


# The fake model packages


def token_offsets(text):
    return [(match.start(), match.end()) for match in re.finditer(r"\w+|[^\w\s]", text)]


class Probabilities(list):
    """The array operations SemanticHighlightExtractor applies to token probabilities."""

    def __getitem__(self, index):
        item = super().__getitem__(index)
        return Probabilities(item) if isinstance(index, slice) else item

    def __gt__(self, threshold):
        return [probability > threshold for probability in self]


class FakeHighlighter:
    def to(self, device):
        return self

    def eval(self):
        return self

    def process(self, question, context, **settings):
        spans = []
        for text in HIGHLIGHTS[context]:
            start = context.index(text)
            spans.append({"start": start, "end": start + len(text), "text": text, "score": 0.9})
        return {"spans": spans}


class FakeSemanticHighlighter:
    def to(self, device):
        return self

    def eval(self):
        return self

    def tokenizer(self, text, **settings):
        return {"offset_mapping": token_offsets(text)}

    def process(self, question, context, **settings):
        return {"highlighted_sentences": [f" {text} " for text in HIGHLIGHTS[context]]}

    def get_raw_predictions(self, question, contexts):
        context = contexts[0]
        regions = [
            (context.index(text), context.index(text) + len(text)) for text in HIGHLIGHTS[context]
        ]
        token_probabilities = [
            0.9 if any(start <= token_start < end for start, end in regions) else 0.1
            for token_start, _ in token_offsets(context)
        ]
        # Wrapped in the probabilities of the question and separator tokens.
        probabilities = Probabilities([0.0, *token_probabilities, 0.0])
        return SimpleNamespace(
            pruning_probs=probabilities, context_ranges=[(1, len(probabilities) - 1)]
        )


MODELS = {
    ModelSpanExtractor.DEFAULT_MODEL: FakeHighlighter,
    "zilliz/semantic-highlight-bilingual-v1": FakeSemanticHighlighter,
}


@pytest.fixture
def fake_model_packages(monkeypatch):
    unavailable = SimpleNamespace(is_available=lambda: False)
    torch = SimpleNamespace(cuda=unavailable, backends=SimpleNamespace(mps=unavailable))
    transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(
            from_pretrained=lambda name, **settings: SimpleNamespace(
                auto_map={"AutoModel": "modeling.Highlighter"}
            )
        ),
        AutoModel=SimpleNamespace(from_pretrained=lambda name, **settings: MODELS[name]()),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)


class TestLLMSpanExtractor:
    @pytest.mark.parametrize("extraction_mode", ["batch", "individual"])
    @pytest.mark.parametrize("span_match_mode", ["exact", "fuzzy"])
    def test_extract_spans(self, span_match_mode, extraction_mode):
        with pytest.warns(VerbatimDeprecationWarning):
            extractor = LLMSpanExtractor(
                llm_client=LLMClient(),
                extraction_mode=extraction_mode,
                span_match_mode=span_match_mode,
            )
            relevant_spans = extractor.extract_spans(QUESTION, search_results())

        assert_recorded(f"llm_span_extractor_{span_match_mode}", spans_by_chunk(relevant_spans))

    @pytest.mark.parametrize("extraction_mode", ["batch", "individual"])
    @pytest.mark.parametrize("span_match_mode", ["exact", "fuzzy"])
    async def test_extract_spans_async(self, span_match_mode, extraction_mode):
        with pytest.warns(VerbatimDeprecationWarning):
            extractor = LLMSpanExtractor(
                llm_client=LLMClient(),
                extraction_mode=extraction_mode,
                span_match_mode=span_match_mode,
            )
            relevant_spans = await extractor.extract_spans_async(QUESTION, search_results())

        assert_recorded(f"llm_span_extractor_{span_match_mode}", spans_by_chunk(relevant_spans))


@pytest.mark.usefixtures("fake_model_packages")
class TestModelExtractors:
    def test_highlighter_extract_spans(self):
        with pytest.warns(VerbatimDeprecationWarning):
            extractor = ModelSpanExtractor()
            relevant_spans = extractor.extract_spans(QUESTION, search_results())

        assert_recorded("model_span_extractor_highlighter", spans_by_chunk(relevant_spans))

    async def test_highlighter_extract_spans_async(self):
        with pytest.warns(VerbatimDeprecationWarning):
            extractor = ModelSpanExtractor()
            relevant_spans = await extractor.extract_spans_async(QUESTION, search_results())

        assert_recorded("model_span_extractor_highlighter", spans_by_chunk(relevant_spans))

    @pytest.mark.parametrize("output_mode", ["sentences", "spans"])
    def test_semantic_highlight_extract_spans(self, output_mode):
        with pytest.warns(VerbatimDeprecationWarning):
            extractor = SemanticHighlightExtractor(output_mode=output_mode)
            relevant_spans = extractor.extract_spans(QUESTION, search_results())

        assert_recorded(
            f"semantic_highlight_extractor_{output_mode}", spans_by_chunk(relevant_spans)
        )

    @pytest.mark.parametrize("output_mode", ["sentences", "spans"])
    async def test_semantic_highlight_extract_spans_async(self, output_mode):
        with pytest.warns(VerbatimDeprecationWarning):
            extractor = SemanticHighlightExtractor(output_mode=output_mode)
            relevant_spans = await extractor.extract_spans_async(QUESTION, search_results())

        assert_recorded(
            f"semantic_highlight_extractor_{output_mode}", spans_by_chunk(relevant_spans)
        )


# Both copies of VerbatimTransform and verbatim_query


class ChunkProvider(RAGProvider):
    def retrieve(self, question, k=5, filter=None):
        return [
            {"content": chunk["text"], "title": chunk["title"], "source": chunk["source"]}
            for chunk in CHUNKS[:k]
        ]


@pytest.fixture(params=["verbatim_core", "verbatim_rag"])
def transform_module(request):
    return pytest.importorskip(f"{request.param}.transform")


class TestVerbatimTransform:
    @pytest.mark.parametrize("template_mode", ["static", "contextual"])
    def test_transform(self, transform_module, template_mode):
        with pytest.warns(VerbatimDeprecationWarning):
            transform = transform_module.VerbatimTransform(
                llm_client=LLMClient(), max_display_spans=3, template_mode=template_mode
            )

        response = transform.transform(QUESTION, ChunkProvider().retrieve(QUESTION))

        assert_recorded(f"verbatim_transform_{template_mode}", response.model_dump(mode="json"))

    @pytest.mark.parametrize("template_mode", ["static", "contextual"])
    async def test_transform_async(self, transform_module, template_mode):
        with pytest.warns(VerbatimDeprecationWarning):
            transform = transform_module.VerbatimTransform(
                llm_client=LLMClient(), max_display_spans=3, template_mode=template_mode
            )

        response = await transform.transform_async(QUESTION, ChunkProvider().retrieve(QUESTION))

        assert_recorded(f"verbatim_transform_{template_mode}", response.model_dump(mode="json"))

    def test_verbatim_query(self, transform_module):
        with pytest.warns(VerbatimDeprecationWarning):
            response = transform_module.verbatim_query(ChunkProvider(), QUESTION, k=4)

        assert_recorded("verbatim_query", response.model_dump(mode="json"))

    async def test_verbatim_query_async(self, transform_module):
        with pytest.warns(VerbatimDeprecationWarning):
            response = await transform_module.verbatim_query_async(ChunkProvider(), QUESTION, k=4)

        assert_recorded("verbatim_query", response.model_dump(mode="json"))


# The enhance decorator and the verbatim-enhance CLI


class TestEnhance:
    def test_decorator(self):
        from verbatim_core.enhance import verbatim_enhance

        @verbatim_enhance(max_display_spans=3)
        def answer(question):
            return {"answer": "ignored", "context": ChunkProvider().retrieve(question)}

        response = answer(QUESTION)

        assert_recorded("verbatim_transform_contextual", response.model_dump(mode="json"))

    def test_cli(self, tmp_path, capsys):
        from verbatim_core.cli import main

        record = {"question": QUESTION, "context": ChunkProvider().retrieve(QUESTION)}
        (tmp_path / "records.jsonl").write_text(json.dumps(record) + "\n")

        assert main(["--input", str(tmp_path / "records.jsonl"), "--max-spans", "3"]) == 0

        [line] = capsys.readouterr().out.splitlines()
        assert_recorded("verbatim_transform_contextual", json.loads(line))


# The API server's transform route, which needs the full verbatim-rag package and fastapi


def test_api_transform_route(caplog):
    pytest.importorskip("verbatim_rag")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from api.app import app

    caplog.set_level(logging.DEBUG)
    request = {"question": QUESTION, "context": ChunkProvider().retrieve(QUESTION)}

    with TestClient(app) as client:
        response = client.post("/api/transform/verbatim", json=request)

    assert response.status_code == 200
    assert_recorded("verbatim_query", response.json())
    customer_texts = [QUESTION, *(chunk["text"] for chunk in CHUNKS)]
    for record in caplog.records:
        if not record.name.endswith(".content"):
            assert not [text for text in customer_texts if text in record.getMessage()]


class TestAPITransformRouteErrors:
    @pytest.fixture
    def client(self):
        pytest.importorskip("verbatim_rag")
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from api.app import app

        with TestClient(app) as client:
            yield client
        app.dependency_overrides.clear()

    def test_an_item_with_a_null_title_is_a_bad_request(self, client):
        context = [{"content": CHUNKS[0]["text"], "title": None}]

        response = client.post(
            "/api/transform/verbatim", json={"question": QUESTION, "context": context}
        )

        assert response.status_code == 400

    def test_a_failed_llm_call_is_a_server_error_logged_by_its_code(self, client, caplog):
        from api.app import app
        from api.dependencies import get_transformer

        class FailingTransformer:
            async def atransform(self, question, chunks, **settings):
                raise UpstreamError(f"Failed on {question}", code="timeout", service="llm")

        app.dependency_overrides[get_transformer] = FailingTransformer
        request = {"question": QUESTION, "context": ChunkProvider().retrieve(QUESTION)}

        response = client.post("/api/transform/verbatim", json=request)

        assert response.status_code == 500
        assert [
            record.getMessage() for record in caplog.records if record.levelname == "ERROR"
        ] == ["Verbatim transform failed: timeout"]


# VerbatimRAG and StreamingRAG, which need the full verbatim-rag package


class InMemoryIndex:
    def query(self, text=None, k=5, filter=None, hybrid_weights=None, rrf_k=60, search_params=None):
        from verbatim_rag.vector_stores.base import SearchResult

        results = [
            SearchResult(
                id=f"row-{i}",
                score=1.0 - i / 10,
                metadata={"title": chunk["title"], "source": chunk["source"]},
                text=chunk["text"],
            )
            for i, chunk in enumerate(CHUNKS)
        ]
        return results[:k]


class RecordingIndex(InMemoryIndex):
    def __init__(self):
        self.queries = []

    def query(self, text=None, **retrieval):
        self.queries.append(retrieval)
        return super().query(text, **retrieval)


class GreetingDetector:
    def detect(self, question):
        return {"route": "predefined", "answer": "Hello! Ask me about the Harbour Bridge."}


@pytest.fixture
def make_rag():
    verbatim_rag = pytest.importorskip("verbatim_rag")

    def _make(template_mode, intent_detector=None):
        with pytest.warns(VerbatimDeprecationWarning):
            return verbatim_rag.VerbatimRAG(
                InMemoryIndex(),
                k=4,
                max_display_spans=3,
                template_mode=template_mode,
                llm_client=LLMClient(),
                intent_detector=intent_detector,
            )

    return _make


class TestVerbatimRAG:
    @pytest.mark.parametrize("template_mode", ["static", "contextual"])
    def test_query(self, make_rag, template_mode):
        response = make_rag(template_mode).query(QUESTION)

        assert_recorded(f"verbatim_rag_query_{template_mode}", response.model_dump(mode="json"))

    @pytest.mark.parametrize("template_mode", ["static", "contextual"])
    async def test_query_async(self, make_rag, template_mode):
        response = await make_rag(template_mode).query_async(QUESTION)

        assert_recorded(f"verbatim_rag_query_{template_mode}", response.model_dump(mode="json"))

    def test_query_answered_by_the_intent_detector(self, make_rag):
        response = make_rag("contextual", GreetingDetector()).query(QUESTION)

        assert_recorded("verbatim_rag_query_intent", response.model_dump(mode="json"))

    async def test_query_async_answered_by_the_intent_detector(self, make_rag):
        response = await make_rag("contextual", GreetingDetector()).query_async(QUESTION)

        assert_recorded("verbatim_rag_query_intent", response.model_dump(mode="json"))


class TestStreamingRAG:
    @pytest.mark.parametrize("template_mode", ["static", "contextual"])
    def test_stream_query_sync(self, make_rag, template_mode):
        from verbatim_rag import StreamingRAG

        rag = make_rag(template_mode)

        with pytest.warns(VerbatimDeprecationWarning):
            events = StreamingRAG(rag).stream_query_sync(QUESTION)

        assert_recorded(f"streaming_rag_{template_mode}", without_timing(events))

    @pytest.mark.parametrize("template_mode", ["static", "contextual"])
    async def test_stream_query(self, make_rag, template_mode):
        from verbatim_rag import StreamingRAG

        rag = make_rag(template_mode)

        with pytest.warns(VerbatimDeprecationWarning):
            stream = StreamingRAG(rag).stream_query(QUESTION)
        events = [event async for event in stream]

        assert_recorded(f"streaming_rag_{template_mode}", without_timing(events))

    def test_stream_answered_by_the_intent_detector(self, make_rag):
        from verbatim_rag import StreamingRAG

        rag = make_rag("contextual", GreetingDetector())

        with pytest.warns(VerbatimDeprecationWarning):
            events = StreamingRAG(rag).stream_query_sync(QUESTION)

        assert_recorded("streaming_rag_intent", without_timing(events))


# The API server's streaming route, which needs fastapi as well


class FailingExtractor(SpanExtractor):
    def extract(self, question, chunks, *, config=None):
        cause = TimeoutError(f"No reply about {question}")
        raise UpstreamError("The LLM timed out.", code="timeout", service="llm") from cause


class TestAPIStreamRoute:
    @pytest.fixture
    def stream(self):
        pytest.importorskip("verbatim_rag")
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from api.app import app
        from api.dependencies import get_rag_instance

        def _stream(rag, **request):
            app.dependency_overrides[get_rag_instance] = lambda: rag
            try:
                with TestClient(app) as client:
                    response = client.post(
                        "/api/query/stream", json={"question": QUESTION, **request}
                    )
            finally:
                app.dependency_overrides.clear()
            assert response.status_code == 200
            return [json.loads(line) for line in response.text.splitlines()]

        return _stream

    @pytest.mark.parametrize("template_mode", ["static", "contextual"])
    def test_the_route_sends_the_recorded_stream(self, stream, make_rag, template_mode):
        events = stream(make_rag(template_mode), num_docs=4)

        assert_recorded(f"streaming_rag_{template_mode}", without_timing(events))

    def test_the_request_sets_the_retrieval_of_the_call(self, stream, make_rag):
        rag = make_rag("static")
        rag.index = RecordingIndex()
        retrieval = {
            "filter": "year > 1900",
            "hybrid_weights": {"dense": 1.0},
            "rrf_k": 10,
            "search_params": {"ef": 64},
        }

        events = stream(rag, num_docs=2, **retrieval)

        assert rag.index.queries == [{"k": 2, **retrieval}]
        assert len(events[0]["data"]) == 2

    def test_the_api_rag_streams_with_contextual_templates(self, stream):
        import verbatim_rag
        from api.dependencies import build_transformer

        llm_client = LLMClient()
        rag = verbatim_rag.VerbatimRAG(
            InMemoryIndex(), transformer=build_transformer(llm_client), llm_client=llm_client
        )

        events = stream(rag)

        assert events[-1]["data"]["answer"].startswith("The sources say:")

    def test_a_failure_is_the_last_event_and_logged_once(self, stream, caplog):
        import verbatim_rag

        rag = verbatim_rag.VerbatimRAG(
            InMemoryIndex(), transformer=Transformer(FailingExtractor(), TemplateManager())
        )

        events = stream(rag)

        assert [event["type"] for event in events] == ["documents", "error"]
        assert events[-1] == {"type": "error", "error": "timeout", "code": "timeout", "done": True}
        [record] = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert record.getMessage() == "Streamed query failed: timeout"
        assert "UpstreamError: The LLM timed out." in caplog.text
        customer_texts = [QUESTION, *(chunk["text"] for chunk in CHUNKS)]
        for record in caplog.records:
            if not record.name.endswith(".content"):
                logged = caplog.handler.format(record)
                assert not [text for text in customer_texts if text in logged]


# The verbatim-rag CLI, whose embedding model and Milvus store stand in for the in-memory index


class ChunkStore:
    def __init__(self, **settings):
        pass

    def query(self, *args, top_k=5, **settings):
        return InMemoryIndex().query(k=top_k)


class OneDimensionEmbedder:
    def __init__(self, **settings):
        pass

    def embed_text(self, text):
        return [1.0]

    def get_dimension(self):
        return 1


class TestVerbatimRAGCLI:
    @pytest.fixture
    def run(self, monkeypatch):
        cli = pytest.importorskip("verbatim_rag.cli")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        # The CLI loads an embedding model and opens a Milvus file itself
        monkeypatch.setattr(
            "verbatim_rag.embedding_providers.SentenceTransformersProvider", OneDimensionEmbedder
        )
        monkeypatch.setattr("verbatim_rag.vector_stores.LocalMilvusStore", ChunkStore)

        def _run(*arguments):
            cli.main(list(arguments))

        return _run

    @pytest.mark.parametrize(
        "template_mode, options", [("static", []), ("contextual", ["--contextual-templates"])]
    )
    def test_query(self, run, tmp_path, template_mode, options):
        output = tmp_path / "response.json"

        run(
            "query",
            *("--index", str(tmp_path / "index.db"), "--question", QUESTION),
            *("--num-docs", "4", "--max-display", "3", "--output", str(output), *options),
        )

        assert_recorded(f"verbatim_rag_query_{template_mode}", json.loads(output.read_text()))

    def test_query_answers_with_the_templates_of_the_template_command(self, run, tmp_path, capsys):
        templates, output = tmp_path / "templates.json", tmp_path / "response.json"

        run("template", "--questions", QUESTION, "--output", str(templates))
        defaults = len(RandomTemplate.DEFAULT_TEMPLATES)
        assert f"Added 2 templates to the {defaults} default ones" in capsys.readouterr().out
        run(
            "query",
            *("--index", str(tmp_path / "index.db"), "--question", QUESTION),
            *("--templates", str(templates), "--output", str(output)),
        )

        saved = json.loads(templates.read_text())
        assert saved["current_mode"] == "random"
        pool = saved["strategies"]["random"]["templates"]
        assert set(GENERATED_TEMPLATES) <= set(pool)
        # Written with one of the saved templates, not the static one
        answer = json.loads(output.read_text())["answer"]
        assert answer.startswith(tuple(template.split("[DISPLAY_SPANS]")[0] for template in pool))

    def test_query_prints_the_citations_by_their_type(self, run, tmp_path, capsys):
        # A template that shows the first of the three displayed spans: the others are references
        templates, output = tmp_path / "templates.json", tmp_path / "response.json"
        one_span = TemplateManager()
        one_span.use_random_mode(["Only this: [SPAN_1]"])
        one_span.save(str(templates))

        run(
            "query",
            *("--index", str(tmp_path / "index.db"), "--question", QUESTION, "--verbose"),
            *("--max-display", "3", "--templates", str(templates), "--output", str(output)),
        )

        printed = capsys.readouterr().out
        types = [
            c["type"] for c in json.loads(output.read_text())["structured_answer"]["citations"]
        ]
        displayed, references = types.count("display"), types.count("reference")
        assert displayed == 1
        assert f"({displayed} displayed, {references} as references)" in printed
        assert printed.count("(displayed):") == displayed
        assert printed.count("(reference):") == references

    def test_a_negative_number_of_displayed_spans_is_an_error(self, run, tmp_path, capsys):
        output = tmp_path / "response.json"

        run(
            "query",
            *("--index", str(tmp_path / "index.db"), "--question", QUESTION),
            *("--max-display", "-1", "--output", str(output)),
        )

        assert "Error:" in capsys.readouterr().out
        assert not output.exists()

    def test_num_docs_is_the_k_of_the_query(self, run, tmp_path):
        output = tmp_path / "response.json"

        run(
            "query",
            *("--index", str(tmp_path / "index.db"), "--question", QUESTION),
            *("--num-docs", "2", "--output", str(output)),
        )

        assert len(json.loads(output.read_text())["documents"]) == 2

    @pytest.mark.parametrize("mode, calls", [("batch", 1), ("individual", 4)])
    def test_the_extraction_mode_applies(self, run, tmp_path, monkeypatch, mode, calls):
        extraction_prompts = []

        def recording(messages, response_format=None, **settings):
            if response_format:
                extraction_prompts.append(messages[-1]["content"])
            return complete(messages, response_format, **settings)

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=recording))
        )
        openai = SimpleNamespace(OpenAI=lambda **settings: client, AsyncOpenAI=FakeAsyncOpenAI)
        monkeypatch.setattr("verbatim_core.llm_client.openai", openai)

        run(
            "query",
            *("--index", str(tmp_path / "index.db"), "--question", QUESTION),
            *("--num-docs", "4", "--extraction-mode", mode),
        )

        assert len(extraction_prompts) == calls
