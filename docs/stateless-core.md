# One interface per capability: stateless core, stores passed in

This document expands the idea-box entry of the same name. It defines the stateless interface for each capability verbatim-rag exposes, the types those interfaces share, the store interfaces the stateful classes are built on, and how today's public API keeps working with a deprecation warning until it is removed. Everything here is a proposal until the entry becomes an issue. The questions the team has to decide are collected at the end.

The capabilities are `verbatim.extract` and `verbatim.transform` with its streaming variant.

## The rule

The library has two layers.

The stateless core has one interface per capability. A call takes a typed request and returns a typed result, and the result depends only on the input, the model version and the config. Code in the core:

- writes nothing to disk and keeps nothing between calls
- has no global state and reads no environment variables
- never builds a model or an LLM client from a name inside a constructor. The loaded model or the client is passed in
- takes and returns the shared types `Chunk`, `Span` and `Usage`.

The stateful classes (`VerbatimRAG`, `VerbatimIndex`) keep today's convenience for open-source users. They are built from the core and from stores passed to them. Their own code retrieves, stores and calls the core, and nothing else.

Today's public API stays. Every class, function, method and argument that the core replaces becomes an adapter over the core, warns when it is used, and is removed after a deprecation window. Each step has one implementation, so the old and the new API can't drift apart.

## Shared types

These types are the vocabulary of every interface below. They live in `verbatim_core`.

```python
class Chunk(BaseModel):
    """A piece of source text."""

    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    metadata: dict[str, Any] = {}
    source_start: int | None = None  # position of the chunk in its source document
    source_end: int | None = None


class Span(BaseModel):
    """A part of a text, as "One span format across the libraries" defines it."""

    model_config = ConfigDict(frozen=True)

    start: int  # Unicode code points, inclusive
    end: int  # exclusive
    text: str  # always equal to the slice it points into
    score: float


class Usage(BaseModel):
    """What one model or service did during a call."""

    model_config = ConfigDict(frozen=True)

    model_id: str | None  # None for work without a model
    model_version: str | None
    units: dict[str, int]  # for example {"chunks": 12, "input_tokens": 3410}
```

- A span's offsets are relative to the chunk it comes from. Fields specific to extraction go into a subclass: `ExtractedSpan` adds `chunk_id`.
- Every result carries `usage: list[Usage]`, one entry per model or service that took part. A transform has one entry for the extractor model and one for the LLM that wrote the template. The library reports every unit it can count (chunks, characters, tokens, LLM calls).
- `model_version` identifies the weights exactly. `from_pretrained` records the resolved Hugging Face commit, not the repository id. For weights mounted from a local path (see "Mount local model weights and configure their paths in pydantic-settings configs") the caller passes `model_version`, because a path doesn't say which weights it holds. For a third-party LLM it is the model name the provider returned.
- Every result also carries `warnings`, and batch results carry `failures`, as "Typed errors, and no fallbacks that hide a failure" defines.
- The types are pydantic models because they cross the API boundary unchanged.

### Results are objects, not lists

The idea-box entry writes `extract(question, chunks) -> list[Span]`. A list has no place for `usage` and `warnings`, so each capability returns a result object:

| Capability | Result | Fields |
|------------|--------|--------|
| `verbatim.extract` | `Extraction` | `spans: list[ExtractedSpan]`, `usage`, `warnings` |
| `verbatim.transform` | `Answer` | `text`, `citations`, `spans: list[ExtractedSpan]`, `usage`, `warnings` |

## The method set

Every stateless interface has the same methods. A capability adds streaming methods only where it needs them.

| Capability | Interface | Sync | Async | Streaming |
|------------|-----------|------|-------|-----------|
| `verbatim.extract` | `SpanExtractor` | `extract` | `aextract` | – |
| `verbatim.transform` | `Transformer` | `transform` | `atransform` | `stream`, `astream` |

- Long calls take an `on_progress` callback that reports their progress. This is the callback that open question 1 of the logging convention recommends.
- The methods follow "Write each step once, with thin sync and async wrappers". The logic lives in functions without I/O, and the sync and async methods only make the calls. Local encoder inference runs in a worker thread behind the async method.
- The constructor takes what the interface depends on: a loaded model, an LLM client and a default config. Everything a caller may choose per request is a call argument, and a `config` argument replaces the default config for one call. Nothing is set on the object between calls.

## The interfaces

### verbatim.extract

```python
class SpanExtractor(ABC):
    def extract(
        self, question: str, chunks: Sequence[Chunk], *, config: ExtractConfig | None = None
    ) -> Extraction: ...

    async def aextract(
        self, question: str, chunks: Sequence[Chunk], *, config: ExtractConfig | None = None
    ) -> Extraction: ...
```

`SpanExtractor` stays the base class. It gains the two methods, and `extract_spans` becomes the deprecated adapter (see "Subclasses written against the old interface").

| Extractor | Constructor | Where the offsets come from | `score` |
|-----------|-------------|-----------------------------|---------|
| `ModelSpanExtractor`, highlighter models | `ModelSpanExtractor(model, config=...)` and `ModelSpanExtractor.from_pretrained(model_id, device=None)` | `.process()` returns character spans. Today the extractor keeps only their text (`verbatim_core/extractors.py:222`) | the model's span probability |
| `ModelSpanExtractor`, v1 sentence model | the same | the sentence boundaries from the splitter | the sentence probability |
| `SemanticHighlightExtractor` | `from_pretrained` as above. Today the constructor loads the model (`:336`) | the token regions it computes already | the region probability |
| `LLMSpanExtractor` | `LLMSpanExtractor(llm_client, config=...)`, with the client required | the extractor locates each quote in the chunk it came from. The fuzzy matcher already works with offsets and returns only the text (`:838`) | open question 2 |

`ExtractConfig` holds today's constructor settings: the threshold, `min_span_chars`, `merge_gap_chars`, `max_length` and `doc_stride` for the model, and the extraction mode, batch size, match mode and prompts for the LLM.

When a quote occurs more than once in a chunk, the extractor picks one occurrence. Today the response builder searches for the span text afterwards and highlights every occurrence (`verbatim_core/response_builder.py:122`).

### verbatim.transform

```python
class Transformer:
    def __init__(
        self,
        extractor: SpanExtractor,
        templates: TemplateManager,
        *,
        config: TransformConfig | None = None,
    ) -> None: ...

    def transform(
        self, question: str, chunks: Sequence[Chunk], *, config: TransformConfig | None = None
    ) -> Answer: ...

    async def atransform(...) -> Answer: ...

    def stream(...) -> Iterator[TransformEvent]: ...

    async def astream(...) -> AsyncIterator[TransformEvent]: ...
```

- `transform` is today's `VerbatimTransform.transform` without the conversion of dicts into search results. The response builder builds citations and highlights from the span offsets.
- `TransformConfig` holds the number of displayed spans and the template mode.
- `TemplateManager` gets its LLM client passed in. The static template mode needs none.
- `stream` and `astream` yield a `spans` event after extraction and an `answer` event at the end. They need no index, so `verbatim.transform-stream` works on chunks sent with the request.
- The `answer` argument, which `VerbatimTransform.transform` accepts and ignores today, is not carried over.

## Stores

Stored state sits behind `typing.Protocol` interfaces that the caller passes into the stateful classes. The library ships local implementations, and callers can pass their own. No method takes an owner id: a caller passes in a store already limited to the data it may read, so library code can't read other data by mistake.

The one store holds vectors and documents. Its protocol is `VectorStore`, defined in "Make the vector store easy to swap", its implementations are Milvus Lite and Milvus Cloud, as today, and `VerbatimIndex` uses it.

## Models and clients are passed in

- A class that runs a model takes the loaded model in its constructor. A `from_pretrained(model_id, ...)` classmethod is the one place where a name becomes a model, so the caller decides when the download and the load happen: at startup in a server, in the first cell of a notebook.
- `from_pretrained` accepts a local path, which is how a container loads weights mounted into it.
- A class that calls an LLM takes the client. The client comes from "One LLM-call layer in each library". Until that layer exists, it is `LLMClient`, the client verbatim-rag takes today.
- A model served remotely is passed in as a client. An encoder served remotely becomes one more implementation of the same interface.

## The stateful layer

### VerbatimRAG and VerbatimIndex

```python
transformer = Transformer(
    extractor=ModelSpanExtractor.from_pretrained("KRLabsOrg/verbatim-rag-modern-bert-v2"),
    templates=TemplateManager(llm_client=LLMClient("gpt-4o-mini")),
)
rag = VerbatimRAG(index, transformer=transformer)
response = rag.query("What does the warranty cover?", k=5)
```

- `query` retrieves from the index, converts the search results to `Chunk`, calls `transformer.transform` and builds the `QueryResponse` it returns today. `query_async` does the same with `atransform`.
- `query_stream(question, k=...)` replaces `StreamingRAG`. It sends the documents event, then the events of `transformer.astream`. `k` is an argument. Today `StreamingRAG` sets `rag.k` on the shared object for one request (`verbatim_rag/streaming.py:53`).
- Intent detection and reranking stay in `VerbatimRAG`. They are pipeline stages around the core.
- `VerbatimIndex` already takes its vector store and embedding providers. Its `query` keeps returning `SearchResult` (`verbatim_rag/index.py:552`), and `SearchResult.to_chunk()` converts a result before it reaches the core. The store interface belongs to "Make the vector store easy to swap".

### Defaults

A stateful class may still create a local store when none is passed, because keeping state locally is what the class is for. It never builds a model or an LLM client from a name. When the caller passes none, the class builds today's default for the length of the deprecation window and warns, so the quickstarts in the READMEs keep running. Open question 3 asks what happens after the window.

## Deprecating today's API

Every public name that the core replaces stays available, warns when it is used, and is removed after the deprecation window (open question 1).

### Adapters only

An old entry point converts its arguments to the new types, calls the core and converts the result back to today's shape. It keeps no logic of its own:

```python
class SpanExtractor(ABC):
    @deprecated("SpanExtractor.extract_spans", since="0.3.0", removed_in="0.5.0", use="extract()")
    def extract_spans(self, question: str, search_results: list[Any]) -> dict[str, list[str]]:
        chunks = [_chunk_from_result(i, result) for i, result in enumerate(search_results)]
        extraction = self.extract(question, chunks)
        texts = {chunk.id: chunk.text for chunk in chunks}
        spans: dict[str, list[str]] = {chunk.text: [] for chunk in chunks}
        for span in extraction.spans:
            spans[texts[span.chunk_id]].append(span.text)
        return spans
```

The two copies of `VerbatimTransform` (`verbatim_core/transform.py:54`, `verbatim_rag/transform.py:48`) become one adapter in `verbatim_core`.

### The warning

`verbatim_core` has a `_deprecation.py`. It defines the library's warning class and builds every message in one format:

```python
"""Deprecation warnings for the public API of verbatim_core and verbatim_rag."""

from __future__ import annotations

import warnings

from typing_extensions import deprecated as _deprecated


class VerbatimDeprecationWarning(DeprecationWarning):
    """Emitted when code uses a part of the API that will be removed."""


def _message(name: str, since: str, removed_in: str, use: str) -> str:
    return f"{name} is deprecated since {since} and will be removed in {removed_in}. Use {use}."


def deprecated(name: str, *, since: str, removed_in: str, use: str):
    """Mark a class, function or method as deprecated."""
    return _deprecated(_message(name, since, removed_in, use), category=VerbatimDeprecationWarning)


def warn_deprecated_argument(
    name: str, *, since: str, removed_in: str, use: str, stacklevel: int = 3
) -> None:
    """Warn about a deprecated argument, from inside the function that takes it."""
    warnings.warn(
        _message(name, since, removed_in, use), VerbatimDeprecationWarning, stacklevel=stacklevel
    )
```

- The library has its own subclass of `DeprecationWarning`, `VerbatimDeprecationWarning`, exported from the package root. A user can silence ours without silencing others, and our tests can turn ours into errors.
- `DeprecationWarning` is the category for warnings aimed at developers. Python shows it when the call comes from `__main__`, which covers scripts and notebooks, and pytest shows it in every test run. It stays hidden when the call comes from inside another installed package, whose users couldn't fix it. Python's default filter shows it once per call site, so a loop doesn't repeat it.
- `typing_extensions.deprecated` (PEP 702) warns at runtime, makes type checkers and IDEs mark every use, and keeps the message in `__deprecated__` for the API docs. It needs `typing_extensions` 4.5 or later. The package arrives through `pydantic` and `openai` already, and the library lists it as a direct dependency.
- Warnings fire when the old name is used: constructed, called, or passed as an argument. Importing a module that defines a deprecated class never warns. A module that moved warns on attribute access through a module-level `__getattr__`. `verbatim_rag/transform.py` becomes such a module, re-exporting the names from `verbatim_core`.
- The decorator can't mark an argument. A deprecated argument defaults to a sentinel, so the function can tell whether the caller passed it, and calls `warn_deprecated_argument`.

### Subclasses written against the old interface

Users subclass `SpanExtractor` today. The base class stays and keeps both sets of methods.

- A subclass that implements only the old method keeps working. The base class's new method calls the old one, converts the result and warns once per class that the subclass should implement the new method. For such an extractor, offsets are recovered by searching the chunk, as today.
- A subclass that implements only the new method gets the old method from the base class, through the adapter.
- `__init_subclass__` raises `TypeError` when a subclass implements neither, since the two default methods would call each other.
- Our own extractors derive from the base class, so the base class itself is never marked with the decorator. The decorator would warn when our own subclasses are defined, at import time.

### Endpoints of the API server

The FastAPI server in `api/` keeps its routes and response shapes. A route replaced by a new one is marked with `deprecated=True`, which marks it in the OpenAPI document, and its responses carry `Deprecation` and `Sunset` headers (RFC 9745, RFC 8594). New routes return the core result types unchanged, so an API result has the same shape as a library result.

### Behaviour that changes through the old API

The adapters keep today's shapes, but one behaviour changes, and it goes into the changelog: VerbatimRAG highlights the occurrence of a span the extractor chose, instead of every occurrence of the same text.

## New and replaced names

New in `verbatim_core`: the shared types, `extract` and `aextract` on every extractor, `from_pretrained` on the model extractors, `Transformer`, and `_deprecation.py`. New in `verbatim_rag`: `VerbatimRAG(index, transformer=...)`, `query_stream` and `SearchResult.to_chunk()`.

| Old | Replacement | The adapter |
|-----|-------------|-------------|
| `SpanExtractor.extract_spans`, `extract_spans_async` | `extract`, `aextract` | groups the span texts by chunk text |
| `ModelSpanExtractor(model_path=...)`, `SemanticHighlightExtractor(model_path=...)` | `from_pretrained(model_id)` | calls `from_pretrained` |
| `LLMSpanExtractor()` without `llm_client` | `LLMSpanExtractor(llm_client)` | builds `LLMClient(model)` as today |
| `VerbatimTransform`, `transform`, `transform_async` | `Transformer`, `transform`, `atransform` | turns the context dicts into chunks and the `Answer` into a `QueryResponse` |
| the module `verbatim_rag.transform` | `verbatim_core` | re-exports the core names |
| `verbatim_query`, `verbatim_query_async` | `provider.retrieve` followed by `Transformer` | unchanged behaviour |
| `StreamingRAG`, `stream_query`, `stream_query_sync` | `VerbatimRAG.query_stream` | yields today's event dicts |
| `VerbatimRAG(model=...)`, or without `transformer` | `VerbatimRAG(index, transformer=...)` | builds today's extractor and templates |

Internal callers that move in the same pull request: `verbatim_core/enhance.py:70` and `verbatim_core/cli.py:54` (`VerbatimTransform`), `verbatim_rag/verbatim_doc.py:135` (`extract_spans_async`), `api/app.py:261` (the forked `VerbatimTransform`) and `api/app.py:350` (`StreamingRAG`).

## The places that change

This table lists the places that change, in the same form as "Today's fallbacks" in the typed-errors document.

| Where | Today | After the change |
|-------|-------|------------------|
| `verbatim_core/extractors.py:38` | `extract_spans` returns `dict[chunk_text, list[str]]`, so two chunks with the same text share one entry | `extract` returns spans with `chunk_id`, offsets and score |
| `verbatim_core/extractors.py:222` | the highlighter's character spans are reduced to their text | offsets and score kept |
| `verbatim_core/extractors.py:131`, `:336` | the constructors download and load the model from `model_path` | `from_pretrained`, the constructor takes the loaded model |
| `verbatim_core/extractors.py:502` | `LLMClient(model)` built when no client is passed | client required, the old form warns |
| `verbatim_core/extractors.py:778`, `:838` | the matchers locate each quote and return only its text | they return the span with offsets |
| `verbatim_core/response_builder.py:122` | highlights recovered with `str.find`, every occurrence of the text highlighted | highlights from the span offsets |
| `verbatim_core/transform.py:70`, `verbatim_rag/transform.py:64` | two copies of `VerbatimTransform`, each building `LLMClient()`, an extractor and templates when none are passed | one `Transformer`, `VerbatimTransform` a deprecated adapter |
| `verbatim_rag/streaming.py:53`, `:68` | streaming needs a `VerbatimRAG` with an index and sets `rag.k` per request | `Transformer.astream` without an index, `VerbatimRAG.query_stream(k=...)` |
| `verbatim_rag/core.py:92` to `:101` | `VerbatimRAG` builds its client, extractor and templates from names | takes a `Transformer`, the old form warns |
| `api/app.py:261` | the transform route builds the forked `VerbatimTransform()` per request | a `Transformer` built at startup |

## Enforcement

Tests check the behaviour, and review covers what tests don't see. No Ruff rule can tell whether code keeps state.

### Tests

- `standard-tests` has one suite per interface (extractor, transformer). Every implementation runs it (see "Introduce multiple workspaces into the project structure"). The suites check that:
  - every span's text equals the slice at its offsets
  - the sync and async methods return equal results for the same input
  - a call with the model already loaded, run in an empty temporary directory set as working directory and `HOME`, leaves the directory empty
  - a second call with the same input returns the same result, also when a call with other input ran in between
  - `usage` has an entry with the model id and version.
- Each package has a `tests/test_deprecated.py`. For each old entry point it checks the warning class and the message, and compares the result with today's output recorded on a few fixtures, so the adapter keeps today's shape.
- pytest turns the library's own deprecation warnings into errors, so library code that calls a deprecated name fails the test suite. `pytest.warns` in `test_deprecated.py` records the warnings instead:

```toml
[tool.pytest.ini_options]
filterwarnings = [
    "error::verbatim_core.VerbatimDeprecationWarning",
]
```

- Importing each package in a fresh interpreter emits none of the library's deprecation warnings.

### Review checklist

- Does an adapter do anything beyond converting arguments and results?
- Does core code write a file, read an environment variable or keep anything on `self` between calls?
- Does a constructor download, load or build a model or a client from a name?
- Does a new result type carry `usage` and `warnings`?
- Does every removed or renamed public name have an adapter, a warning with the library's class, and a changelog entry?

## Rollout

The steps are ordered so each one can ship on its own.

1. Add the shared types and `_deprecation.py` to `verbatim_core`. Nothing changes behaviour yet.
2. Add the `standard-tests` suites for the extractor and transformer interfaces.
3. Add `extract` on every extractor and `from_pretrained`, then `Transformer` with streaming, with the response builder working from offsets.
4. Turn the old entry points into adapters, move the internal callers, and turn on the pytest filter.
5. Rebuild `VerbatimRAG` on the core: `query` and `query_stream`.
6. Add the new routes to the API server and deprecate the old ones.
7. Release 0.3.0 with "Deprecated" and "Changed" sections in the changelog and a migration page in the documentation site.
8. Remove the adapters in the release that the deprecation window names.

## Open questions

| \#  | Question | Decision |
|-----|----------|----------|
| 1   | How long is the deprecation window? Recommendation: deprecate in 0.3.0, remove in 0.5.0, and not earlier than three months after 0.3.0. |          |
| 2   | What does `score` hold when a method produces no probability, as in LLM span extraction? Recommendation: 1.0, documented as "extracted without a score", so `score` is always a number. |          |
| 3   | After the window, does `VerbatimRAG(index)` still build a default LLM client, or is the client required? Recommendation: required, with the READMEs showing the explicit form from 0.3.0 on. |          |
