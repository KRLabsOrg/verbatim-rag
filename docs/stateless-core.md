# One interface per capability: stateless core, stores passed in

This document expands the idea-box entry of the same name. It defines the stateless interface for each capability the libraries expose, the types those interfaces share, the store interfaces the stateful classes are built on, and how today's public API keeps working with a deprecation warning until it is removed. Everything here is a proposal until the entry becomes an issue. The questions the team has to decide are collected at the end.

The capabilities are `verbatim.extract`, `verbatim.transform` with its streaming variant, `lettuce.detect` with its typed and prompt variants, `rulechef.learn` and `rulechef.evaluate`. `rulechef.execute` is not one of them, but `rulechef.evaluate` runs rules, so rule execution gets a stateless function as well.

## The rule

Each library has two layers.

The stateless core has one interface per capability. A call takes a typed request and returns a typed result, and the result depends only on the input, the model version and the config. Code in the core:

- writes nothing to disk and keeps nothing between calls
- has no global state and reads no environment variables
- never builds a model or an LLM client from a name inside a constructor. The loaded model or the client is passed in
- takes and returns the shared types `Chunk`, `Span` and `Usage`.

The stateful classes (`VerbatimRAG`, `VerbatimIndex`, `RuleChef`) keep today's convenience for open-source users. They are built from the core and from stores passed to them. Their own code retrieves, stores and calls the core, and nothing else.

Today's public API stays. Every class, function, method and argument that the core replaces becomes an adapter over the core, warns when it is used, and is removed after a deprecation window. Each step has one implementation, so the old and the new API can't drift apart.

## Shared types

These types are the vocabulary of every interface below. Where they live is open question 1.

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

    model_id: str | None  # None for work without a model, such as running rules
    model_version: str | None
    units: dict[str, int]  # for example {"chunks": 12, "input_tokens": 3410}
```

- A span's offsets are relative to the text its container names: the chunk for `verbatim.extract`, the answer for `lettuce.detect`, the input field for RuleChef. Library-specific fields go into subclasses. `ExtractedSpan` adds `chunk_id`, and `DetectedSpan` adds `category`, `subcategory` and `reasoning`. RuleChef's `Span` (`core.py:78`) already has the four shared fields.
- Every result carries `usage: list[Usage]`, one entry per model or service that took part. A transform has one entry for the extractor model and one for the LLM that wrote the template. The library reports every unit it can count (chunks, characters, tokens, LLM calls).
- `model_version` identifies the weights exactly. `from_pretrained` records the resolved Hugging Face commit, not the repository id. For weights mounted from a local path (see "Mount local model weights and configure their paths in pydantic-settings configs") the caller passes `model_version`, because a path doesn't say which weights it holds. For a third-party LLM it is the model name the provider returned.
- Every result also carries `warnings`, and batch results carry `failures`, as "Typed errors, and no fallbacks that hide a failure" defines.
- The types are pydantic models because they cross the API boundary unchanged. LettuceDetect gets pydantic through `openai` today and lists it as a direct dependency.

### Results are objects, not lists

The idea-box entry writes `extract(question, chunks) -> list[Span]` and `learn(task, examples, config) -> Ruleset`. A list or a ruleset has no place for `usage` and `warnings`, so each capability returns a result object:

| Capability | Result | Fields |
|------------|--------|--------|
| `verbatim.extract` | `Extraction` | `spans: list[ExtractedSpan]`, `usage`, `warnings` |
| `verbatim.transform` | `Answer` | `text`, `citations`, `spans: list[ExtractedSpan]`, `usage`, `warnings` |
| `lettuce.detect` | `Detection` | `spans: list[DetectedSpan]`, `tokens: list[TokenScore] \| None`, `usage`, `warnings` |
| `rulechef.learn` | `LearnResult` | `ruleset`, `evaluation: EvalResult \| None`, `usage`, `warnings`, `failures` |
| `rulechef.evaluate` | `EvalResult` | today's fields, plus `usage`, `warnings` and a failure per rule that raised |

## The method set

Every stateless interface has the same methods. A capability adds batch or streaming methods only where it needs them.

| Capability | Interface | Sync | Async | Batch | Streaming |
|------------|-----------|------|-------|-------|-----------|
| `verbatim.extract` | `SpanExtractor` | `extract` | `aextract` | – | – |
| `verbatim.transform` | `Transformer` | `transform` | `atransform` | – | `stream`, `astream` |
| `lettuce.detect` and its variants | `Detector` | `detect` | `adetect` | `detect_batch`, `adetect_batch` | – |
| `rulechef.learn` | function | `learn` | `alearn` | – | – |
| `rulechef.evaluate` | function | `evaluate` | `aevaluate` | – | – |

- Batch detection is a capability of its own (`lettuce.detect-batch`), so the detector has batch methods. `rulechef.learn` and `rulechef.evaluate` run long too, but each call already runs over a whole example set and needs no batch variant.
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
| `LLMSpanExtractor` | `LLMSpanExtractor(llm_client, config=...)`, with the client required | the extractor locates each quote in the chunk it came from. The fuzzy matcher already works with offsets and returns only the text (`:838`) | open question 3 |

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

### lettuce.detect

```python
class SourcesInput(BaseModel):
    kind: Literal["sources"] = "sources"
    sources: list[Chunk]
    answer: str
    question: str | None = None  # None for summarisation


class PromptInput(BaseModel):
    kind: Literal["prompt"] = "prompt"
    prompt: str
    answer: str


DetectRequest = Annotated[SourcesInput | PromptInput, Field(discriminator="kind")]


class Detector(Protocol):
    model_id: str

    def detect(
        self, request: DetectRequest, *, min_score: float = 0.0, include_tokens: bool = False
    ) -> Detection: ...

    async def adetect(...) -> Detection: ...

    def detect_batch(
        self, requests: Sequence[DetectRequest], *, min_score: float = 0.0, include_tokens: bool = False
    ) -> DetectionBatch: ...

    async def adetect_batch(...) -> DetectionBatch: ...
```

- One request type covers `lettuce.detect`, `lettuce.detect-typed` and `lettuce.detect-prompt`. Every detector returns `Detection`, whichever model runs.
- `min_score` and `include_tokens` replace `min_confidence` and `output_format`. `Detection` always has the spans, with `score` set in every path. `tokens` is filled when the caller asks for it.
- Only the encoder scores tokens. `include_tokens=True` on another detector raises `InvalidInputError`. Today the LLM detector reports a fixed 0.9 or 0.1 per token (`detectors/llm.py:56`).
- `DetectionBatch` holds a result per request and the `failures`, as "Batches and jobs" in the typed-errors document describes.

| Detector | Constructor | Today |
|----------|-------------|-------|
| `EncoderDetector` | `EncoderDetector(model, tokenizer, typer=None, config=...)` and `from_pretrained(model_id, device=None, taxonomy_head=None)` | `TransformerDetector`. With a typer the spans carry `category` and `subcategory` |
| `GenerativeDetector` | `GenerativeDetector(client, model_id)`. The client is OpenAI-compatible, so any OpenAI-compatible server can serve the model | the "native" mode of `LLMDetector`, chosen by a name heuristic (`detectors/llm.py:26`) |
| `LLMJudgeDetector` | `LLMJudgeDetector(client, model, config=...)` | the rest of `LLMDetector`: the zero-shot and few-shot judge, reasoning and the verify pass |
| `RAGFactCheckerDetector` | unchanged, plus the `detect` methods | its `"detailed"` output (triplets and fact-check results) has no field in `Detection` and stays available through the deprecated `predict` until the judge detectors are designed |

`load_detector(model_id, **dependencies)` replaces `HallucinationDetector(method=...)`. It looks the id up in a registry of the KR Labs models the library knows, each with its detector class, and builds that class. For any other model the caller constructs the class directly. The registry replaces the name heuristic.

### rulechef.learn and rulechef.evaluate

```python
@dataclass(frozen=True)
class Ruleset:
    id: str
    task: Task
    rules: tuple[Rule, ...]
    version: int | None = None  # assigned by the store
    parent_version: int | None = None  # the version learning started from


@dataclass(frozen=True)
class TrainingData:
    examples: tuple[Example, ...]
    corrections: tuple[Correction, ...] = ()
    feedback: tuple[Feedback, ...] = ()


def learn(
    task: Task,
    data: TrainingData,
    config: LearnConfig,
    *,
    llm: OpenAI,
    base: Ruleset | None = None,
    coordinator: CoordinatorProtocol | None = None,
    on_progress: Callable[[LearnProgress], None] | None = None,
) -> LearnResult: ...


def evaluate(
    ruleset: Ruleset,
    examples: Sequence[Example],
    *,
    config: EvalConfig | None = None,
    executor: RuleExecutor | None = None,
) -> EvalResult: ...


def execute(ruleset: Ruleset, input_data: dict, *, executor: RuleExecutor | None = None) -> dict: ...
```

RuleChef keeps its dataclass style for these types, like the rest of `core.py`.

- `Ruleset` replaces `Dataset.rules` (`core.py:530`), which holds the current rules next to the training data without a version.
- `LearnConfig` collects the settings that are spread over the `RuleChef` constructor, `RuleLearner` and `learn_rules` today: allowed formats, sampling and synthesis strategy, rule and sample limits, temperature, the LLM budget settings, the refinement iterations, the holdout fraction and the split seed.
- `base` is the ruleset learning starts from, which today is the rules already in the dataset. `incremental_only` stays a field of `LearnConfig`.
- `learn` builds its `RuleLearner` for the call from `config`. Today `learn_rules` writes the sampling strategy for one run onto the shared learner and restores it afterwards (`pipeline.py:77`, `:131`).
- `llm` is the OpenAI client RuleChef takes today, until "One LLM-call layer in each library" replaces it.
- `evaluate` is today's `evaluate_dataset` (`evaluation.py:346`) with the rules applied by an executor. Today evaluation goes through `learner._apply_rules` (`learner.py:91`), so it needs a learner and with it an LLM client to run rules that need no LLM.
- `executor` defaults to today's `RuleExecutor` (`executor.py:189`). A caller can pass one that runs code rules in a sandbox. The timeout for code rules works only on the main thread today (`executor.py:38`), so a sandboxing executor needs its own.

## Stores

Stored state sits behind `typing.Protocol` interfaces that the caller passes into the stateful classes. Each library ships local implementations, and callers can pass their own. No method takes an owner id: a caller passes in a store already limited to the data it may read, so library code can't read other data by mistake.

| Store | Protocol | Local implementation | Used by |
|-------|----------|----------------------|---------|
| Vectors and documents | `VectorStore`, defined in "Make the vector store easy to swap" | Milvus Lite and Milvus Cloud, as today | `VerbatimIndex` |
| Rulesets | `RulesetStore` | `LocalRulesetStore(path)`, JSON files | `RuleChef` |
| Recorded traffic | `TrafficStore` | `JsonlTrafficStore(path)` and `InMemoryTrafficStore` | `RuleChef` observation |

```python
class RulesetStore(Protocol):
    def get(self, ruleset_id: str, version: int | None = None) -> Ruleset: ...

    def put(self, ruleset: Ruleset) -> Ruleset: ...  # returns the ruleset with its new version

    def versions(self, ruleset_id: str) -> list[int]: ...


class TrafficStore(Protocol):
    def append(self, records: Sequence[TrafficRecord]) -> None: ...

    def read(self, *, since: datetime | None = None, limit: int | None = None) -> list[TrafficRecord]: ...

    def delete(self, record_ids: Sequence[str]) -> None: ...
```

- A version never changes once stored. `put` always adds a version, so learning, evaluation and observation can refer to the same ruleset over time (`rulechef.ruleset`).
- `TrafficRecord` holds what the observer records today (the input, the LLM output, the model, a timestamp) and an id. Deletion of single records is part of the protocol from the start, because the traffic can hold personal data.
- A local implementation creates its directory on the first write. Today `DatasetStore` creates `./rulechef_data` in its constructor (`storage.py:21`).
- Training examples stay in the stateful `RuleChef`'s local dataset file, as today. The core takes them as an argument, so a caller can read them from wherever it keeps them. Open question 6 asks whether they need a protocol of their own.

## Models and clients are passed in

- A class that runs a model takes the loaded model in its constructor. A `from_pretrained(model_id, ...)` classmethod is the one place where a name becomes a model, so the caller decides when the download and the load happen: at startup in a server, in the first cell of a notebook.
- `from_pretrained` accepts a local path, which is how a container loads weights mounted into it.
- A class that calls an LLM takes the client. The client comes from "One LLM-call layer in each library". Until that layer exists, it is the client each library takes today: `LLMClient` in VerbatimRAG and LettuceDetect, the OpenAI client in RuleChef.
- A model served remotely is passed in as a client. `GenerativeDetector` works this way from the start, and an encoder served remotely becomes one more implementation of the same interface.

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

### RuleChef

```python
chef = RuleChef(
    task,
    client=OpenAI(),
    ruleset_store=LocalRulesetStore("./rulechef_data"),
    traffic_store=JsonlTrafficStore("./rulechef_data/traffic.jsonl"),
)
chef.add_example({"text": "..."}, {"spans": [...]})
rules, evaluation = chef.learn_rules()
```

- `learn_rules` builds `TrainingData` from the dataset, calls `learn` with the current ruleset as `base`, puts the result into the store and makes it the current ruleset. Its arguments and return value stay. Today the learning pipeline saves the dataset as a side effect (`pipeline.py:105`, `:119`).
- `evaluate`, `get_rule_metrics` and `rank_rules` call `evaluate` with the current ruleset and the dataset's examples.
- `load_rules(source, persist=True)` puts the loaded rules into the store as a new version.
- Observation (`start_observing`, `add_observation`, `add_raw_observation`) appends to the traffic store, and `export_traffic` reads from it.
- The constructor's learning settings stay and become the default `LearnConfig`.

### Defaults

A stateful class may still create a local store when none is passed, because keeping state locally is what the class is for. It never builds a model or an LLM client from a name. When the caller passes none, the class builds today's default for the length of the deprecation window and warns, so the quickstarts in the READMEs keep running. Open questions 4 and 5 ask what happens after the window.

## Deprecating today's API

Every public name that the core replaces stays available, warns when it is used, and is removed after the deprecation window (open question 2).

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

Each library has a `_deprecation.py`. It defines the library's warning class and builds every message in one format:

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

- Each library has its own subclass of `DeprecationWarning` (`VerbatimDeprecationWarning`, `LettuceDetectDeprecationWarning`, `RuleChefDeprecationWarning`), exported from the package root. A user can silence ours without silencing others, and our tests can turn ours into errors.
- `DeprecationWarning` is the category for warnings aimed at developers. Python shows it when the call comes from `__main__`, which covers scripts and notebooks, and pytest shows it in every test run. It stays hidden when the call comes from inside another installed package, whose users couldn't fix it. Python's default filter shows it once per call site, so a loop doesn't repeat it.
- `typing_extensions.deprecated` (PEP 702) warns at runtime, makes type checkers and IDEs mark every use, and keeps the message in `__deprecated__` for the API docs. It needs `typing_extensions` 4.5 or later. The package arrives through `pydantic` and `openai` already, and each library lists it as a direct dependency.
- Warnings fire when the old name is used: constructed, called, or passed as an argument. Importing a module that defines a deprecated class never warns. A module that moved warns on attribute access through a module-level `__getattr__`. `verbatim_rag/transform.py` becomes such a module, re-exporting the names from `verbatim_core`.
- The decorator can't mark an argument. A deprecated argument defaults to a sentinel, so the function can tell whether the caller passed it, and calls `warn_deprecated_argument`.

### Subclasses written against the old interface

Users subclass `SpanExtractor` and `BaseDetector` today. Both base classes stay and keep both sets of methods.

- A subclass that implements only the old method keeps working. The base class's new method calls the old one, converts the result and warns once per class that the subclass should implement the new method. For such an extractor, offsets are recovered by searching the chunk, as today.
- A subclass that implements only the new method gets the old method from the base class, through the adapter.
- `__init_subclass__` raises `TypeError` when a subclass implements neither, since the two default methods would call each other.
- Our own detectors and extractors derive from the base classes, so the base classes themselves are never marked with the decorator. The decorator would warn when our own subclasses are defined, at import time.

### Endpoints of the API servers in the repositories

The FastAPI servers in the three repositories keep their routes and response shapes. A route replaced by a new one is marked with `deprecated=True`, which marks it in the OpenAPI document, and its responses carry `Deprecation` and `Sunset` headers (RFC 9745, RFC 8594). New routes return the core result types unchanged, so an API result has the same shape as a library result.

### Behaviour that changes through the old API

The adapters keep today's shapes, but four behaviours change, and each one goes into the changelog:

- VerbatimRAG highlights the occurrence of a span the extractor chose, instead of every occurrence of the same text
- LettuceDetect spans carry `confidence` in every path, including the two that lack it today
- `LLMDetector` writes a cache file only when the caller passes `cache_file`. Today it writes one into the installed package directory by default (`detectors/llm.py:184`)
- the `RuleChef` constructor no longer creates `./rulechef_data`. The directory appears on the first save.

## Per library

### VerbatimRAG

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

### LettuceDetect

New: the shared types, `Detector` with `detect`, `adetect`, `detect_batch` and `adetect_batch` on `BaseDetector`, `EncoderDetector`, `GenerativeDetector`, `LLMJudgeDetector`, `load_detector`, `_deprecation.py`, and the route `POST /v1/detect`.

| Old | Replacement | The adapter |
|-----|-------------|-------------|
| `HallucinationDetector(method=..., **kwargs)` | `load_detector(model_id, ...)` | builds the new detector from the old arguments |
| `predict`, `predict_prompt`, `predict_prompt_batch` on every detector | `detect`, `detect_batch` | converts `Detection` into today's dicts, with `confidence` and `prob` taken from `score` |
| `make_detector` | `load_detector` | |
| `TransformerDetector(model_path=...)` | `EncoderDetector.from_pretrained(model_id)` | a subclass of `EncoderDetector` |
| `LLMDetector` with a generative model id | `GenerativeDetector(client, model_id)` | chooses the class with today's heuristic |
| `LLMDetector` with any other model | `LLMJudgeDetector(client, model)` | |
| `LLMDetector` without `client` | the same class with the client passed in | builds the client as today |
| `min_confidence`, `output_format` | `min_score`, `include_tokens` | |
| `POST /v1/lettucedetect/token`, `POST /v1/lettucedetect/spans` | `POST /v1/detect` | keep `hallucination_score` and today's fields |
| `LettuceClient` methods in `lettucedetect_api/client.py` | `LettuceClient.detect` | |

Internal callers that move in the same pull request: `cli.py:176`, `models/evaluator.py:13` and its three evaluation functions, `integrations/claude_code/check_answer.py:147` and `:162`, `integrations/langchain/callbacks.py:38` and `:158`, and `lettucedetect_api/server.py:41`.

### RuleChef

New: `Ruleset`, `TrainingData`, `LearnConfig`, `LearnResult`, `learn`, `alearn`, `evaluate`, `aevaluate`, `execute`, the `RulesetStore` and `TrafficStore` protocols with their local implementations, and `_deprecation.py`.

| Old | Replacement | The adapter |
|-----|-------------|-------------|
| `RuleChef(storage_path=..., dataset_name=...)` | `RuleChef(ruleset_store=LocalRulesetStore(path))` | builds the local store |
| `RuleChef` without `client` | `RuleChef(client=OpenAI())` | builds `OpenAI()` as today |
| `RuleChef.evaluate(verbose=True)` and the other `verbose` arguments | `format()` on the result | already deprecated by the logging convention |

`RuleLearner`, `LearningPipeline`, `DatasetStore` and `ObservationManager` are not exported from the package root, so they change without a warning. `tests/test_failure_clustering.py`, `tests/test_patch_budget.py` and `benchmarks/eval_tab_official.py` use `RuleLearner` and move with it.

Internal callers that move in the same pull request: `api/deps.py:44`, `api/routes/project.py:84` and `:120`.

## Today, per library

This table lists the places that change, in the same form as "Today's fallbacks" in the typed-errors document.

| Library | Where | Today | After the change |
|---------|-------|-------|------------------|
| VerbatimRAG | `verbatim_core/extractors.py:38` | `extract_spans` returns `dict[chunk_text, list[str]]`, so two chunks with the same text share one entry | `extract` returns spans with `chunk_id`, offsets and score |
| VerbatimRAG | `verbatim_core/extractors.py:222` | the highlighter's character spans are reduced to their text | offsets and score kept |
| VerbatimRAG | `verbatim_core/extractors.py:131`, `:336` | the constructors download and load the model from `model_path` | `from_pretrained`, the constructor takes the loaded model |
| VerbatimRAG | `verbatim_core/extractors.py:502` | `LLMClient(model)` built when no client is passed | client required, the old form warns |
| VerbatimRAG | `verbatim_core/extractors.py:778`, `:838` | the matchers locate each quote and return only its text | they return the span with offsets |
| VerbatimRAG | `verbatim_core/response_builder.py:122` | highlights recovered with `str.find`, every occurrence of the text highlighted | highlights from the span offsets |
| VerbatimRAG | `verbatim_core/transform.py:70`, `verbatim_rag/transform.py:64` | two copies of `VerbatimTransform`, each building `LLMClient()`, an extractor and templates when none are passed | one `Transformer`, `VerbatimTransform` a deprecated adapter |
| VerbatimRAG | `verbatim_rag/streaming.py:53`, `:68` | streaming needs a `VerbatimRAG` with an index and sets `rag.k` per request | `Transformer.astream` without an index, `VerbatimRAG.query_stream(k=...)` |
| VerbatimRAG | `verbatim_rag/core.py:92` to `:101` | `VerbatimRAG` builds its client, extractor and templates from names | takes a `Transformer`, the old form warns |
| VerbatimRAG | `api/app.py:261` | the transform route builds the forked `VerbatimTransform()` per request | a `Transformer` built at startup |
| LettuceDetect | `models/inference.py:19`, `detectors/factory.py:10` | the detector chosen by a method string, unknown arguments passed through `**kwargs` | `load_detector(model_id)` |
| LettuceDetect | `detectors/base.py:12` | three `predict` methods whose result shape depends on `output_format` | `detect` and `detect_batch` returning `Detection` |
| LettuceDetect | `detectors/transformer.py:62` | tokenizer and model loaded in the constructor | `EncoderDetector.from_pretrained` |
| LettuceDetect | `detectors/llm.py:26` | generative models recognised by their name and run inside the LLM judge | `GenerativeDetector`, chosen through the registry |
| LettuceDetect | `detectors/llm.py:155` | a client built from `provider` when none is passed | client required, the old form warns |
| LettuceDetect | `detectors/llm.py:184` | a default cache file written into the installed package | no cache in the core |
| LettuceDetect | `detectors/llm.py:297` | spans without `confidence` unless `include_reasoning` is set | `score` in every span (open question 3) |
| LettuceDetect | `lettucedetect_api/server.py:77`, `:100` | `confidence` renamed to `hallucination_score`, `category` dropped | `/v1/detect` returns `Detection` unchanged, the old routes deprecated |
| RuleChef | `engine.py:121` | `OpenAI()` built when no client is passed | client required, the old form warns |
| RuleChef | `engine.py:252`, `storage.py:21` | the constructor creates `./rulechef_data` and reads the dataset file | the store passed in, the directory created on the first write |
| RuleChef | `pipeline.py:77`, `:131` | the sampling strategy for one run written onto the shared learner | a `LearnConfig` argument |
| RuleChef | `pipeline.py:105`, `:119` | learning saves the dataset as a side effect | `learn` returns a `LearnResult`, `learn_rules` stores it |
| RuleChef | `core.py:530` | training data and the current rules in one `Dataset`, without versions | `TrainingData` and a versioned `Ruleset` |
| RuleChef | `learner.py:91`, `evaluation.py:346` | evaluation needs the learner, and with it an LLM client | `evaluate(ruleset, examples, executor=...)` |
| RuleChef | `engine.py:750` | observed traffic held in memory and exported to a JSONL file | `TrafficStore`, `export_traffic` reads from it |
| RuleChef | `api/deps.py:44` | `RuleChef(storage_path=...)` per workspace | the server's `RulesetStore` |

## Enforcement

Tests check the behaviour, and review covers what tests don't see. No Ruff rule can tell whether code keeps state.

### Tests

- `standard-tests` has one suite per interface (extractor, transformer, detector) and per store (ruleset, traffic). Every implementation runs it (see "Introduce multiple workspaces into the project structure"). The interface suites check that:
  - every span's text equals the slice at its offsets
  - the sync and async methods return equal results for the same input
  - a call with the model already loaded, run in an empty temporary directory set as working directory and `HOME`, leaves the directory empty
  - a second call with the same input returns the same result, also when a call with other input ran in between
  - `usage` has an entry with the model id and version.
- The store suites check that a stored ruleset reads back equal, that `put` never changes an existing version, and that a deleted traffic record is gone from `read`.
- Each package has a `tests/test_deprecated.py`. For each old entry point it checks the warning class and the message, and compares the result with today's output recorded on a few fixtures, so the adapter keeps today's shape.
- pytest turns the library's own deprecation warnings into errors, so library code that calls a deprecated name fails the test suite. `pytest.warns` in `test_deprecated.py` records the warnings instead:

```toml
[tool.pytest.ini_options]
filterwarnings = [
    "error::lettucedetect.LettuceDetectDeprecationWarning",
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

The steps are ordered so each one can ship on its own. Steps 3 to 5 are independent of each other.

1. Add the shared types and `_deprecation.py` to each library. Nothing changes behaviour yet.
2. Add the `standard-tests` suites for the three interfaces and the two stores.
3. VerbatimRAG: `extract` on every extractor and `from_pretrained`, then `Transformer` with streaming, with the response builder working from offsets.
4. LettuceDetect: `detect` on every detector, the split of `LLMDetector` into `GenerativeDetector` and `LLMJudgeDetector`, and `load_detector`.
5. RuleChef: `Ruleset`, `TrainingData`, `learn`, `evaluate`, `execute`, the store protocols and the local stores.
6. Turn the old entry points into adapters, move the internal callers, and turn on the pytest filter.
7. Rebuild the stateful classes on the core: `VerbatimRAG.query` and `query_stream`, and `RuleChef.learn_rules` and observation.
8. Add the new routes to the API servers in the repositories and deprecate the old ones.
9. Release each library as 0.3.0 with "Deprecated" and "Changed" sections in the changelog and a migration page in its documentation site.
10. Remove the adapters in the release that the deprecation window names.

## Open questions

| \#  | Question | Decision |
|-----|----------|----------|
| 1   | Do `Chunk`, `Span` and `Usage` live in one small shared package, or are the same names defined in each library? A shared package lets results pass between libraries without conversion. Separate copies keep the releases independent. Decide together with question 1 of the typed-errors document and question 1 of "One LLM-call layer in each library". |          |
| 2   | How long is the deprecation window? Recommendation: deprecate in 0.3.0, remove in 0.5.0, and not earlier than three months after 0.3.0. |          |
| 3   | What does `score` hold when a method produces no probability: LLM span extraction, and the LLM judge without reasoning? Recommendation: 1.0, documented as "flagged without a score", so `score` is always a number and `min_score` keeps these spans. |          |
| 4   | After the window, do `VerbatimRAG(index)` and `RuleChef(task)` still build a default LLM client, or is the client required? Recommendation: required, with the READMEs showing the explicit form from 0.3.0 on. |          |
| 5   | Does `RuleChef` without `ruleset_store` keep a local store in `./rulechef_data`, created on the first save? Recommendation: yes, keeping state locally is what the class is for. |          |
| 6   | Do training examples get a store protocol of their own? Recommendation: not now. The core takes them as an argument. |          |
| 7   | Do the store protocols get async methods now? A store backed by a network service is called from async code. Recommendation: yes for `RulesetStore` and `TrafficStore`, in line with whatever "Make the vector store easy to swap" decides for `VectorStore`. |          |
