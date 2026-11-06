# Migrating to 0.3.0

0.3.0 adds a stateless core: extractors return spans with their chunk ids and offsets, and a `Transformer` turns a question and chunks into a cited answer. `VerbatimRAG` is built on a `Transformer` you pass in. Models and LLM clients are passed into constructors, and the settings of a request are arguments of the call.

The old entry points keep working through the 0.3.x and 0.4.x releases. Each one warns with `VerbatimDeprecationWarning`, a `DeprecationWarning`, whose message names its replacement:

```text
VerbatimTransform is deprecated since 0.3.0 and will be removed in 0.5.0. Use Transformer.
```

They are removed in 0.5.0.

## Finding the old calls

Python hides `DeprecationWarning`s outside `__main__`. Turn the library's warnings into errors in your tests to find every old call:

```toml
# pyproject.toml
[tool.pytest.ini_options]
filterwarnings = ["error::verbatim_core.VerbatimDeprecationWarning"]
```

Or show them while running a script: `python -W always::DeprecationWarning your_script.py`. Importing either package never warns; using an old name does, including importing a name from a moved module such as `verbatim_rag.transform`.

## Answering from chunks

`VerbatimTransform` took context dicts and returned a `QueryResponse`. Build a `Transformer` from an extractor and templates instead, pass it chunks, and turn its `Answer` into a `QueryResponse` when you need one:

```python
# Before
from verbatim_core import VerbatimTransform

vt = VerbatimTransform(template_mode="contextual", max_display_spans=5)
response = vt.transform(question, context)
```

```python
# After
from verbatim_core import Chunk, TransformConfig, Transformer
from verbatim_core.extractors import LLMSpanExtractor
from verbatim_core.llm_client import LLMClient
from verbatim_core.response_builder import ResponseBuilder
from verbatim_core.templates import TemplateManager

client = LLMClient("gpt-4o-mini")
transformer = Transformer(
    LLMSpanExtractor(client),
    TemplateManager(llm_client=client, default_mode="contextual"),
    config=TransformConfig(max_display_spans=5),
)

chunks = [
    Chunk(id=f"ctx_{i}", text=item["content"], metadata={"title": item.get("title", "")})
    for i, item in enumerate(context)
]
answer = transformer.transform(question, chunks)
response = ResponseBuilder().build_from_answer(question, answer, chunks)
```

A chunk's `id` is yours to choose, unique within a call; the answer's citations point at it with offsets into its `text`.

| Old | New |
|---|---|
| `VerbatimTransform(...)`, `transform`, `transform_async` | `Transformer(extractor, templates, config=TransformConfig(...))`, `transform`, `atransform`, then `ResponseBuilder().build_from_answer` |
| `verbatim_query(provider, question, k)` | `context = provider.retrieve(question, k=k)`, chunks from the context items as above, `Transformer.transform`, then `ResponseBuilder().build_from_answer` |
| `verbatim_query_async(provider, question, k)` | `context = await provider.retrieve_async(question, k=k)`, chunks from the context items as above, `Transformer.atransform`, then `ResponseBuilder().build_from_answer` |
| `verbatim_rag.transform.VerbatimTransform`, `verbatim_query`, `verbatim_query_async` | the same names from `verbatim_core`, then the rows above |
| `verbatim_rag.transform.SearchResult` | `verbatim_rag.vector_stores.SearchResult` |

## Extractors

`extract` takes chunks and returns an `Extraction`: spans with their chunk ids, offsets and scores, the usage of the call and its warnings. `extract_spans` returned span texts keyed by chunk text.

```python
# Before
spans_by_text = extractor.extract_spans(question, search_results)

# After
extraction = extractor.extract(question, chunks)
for span in extraction.spans:
    print(span.chunk_id, span.start, span.end, span.text)
```

| Old | New |
|---|---|
| `extractor.extract_spans(question, results)` | `extractor.extract(question, chunks)` |
| `extractor.extract_spans_async(question, results)` | `await extractor.aextract(question, chunks)` |
| A `SpanExtractor` subclass that implements only `extract_spans` | Implement `extract(self, question, chunks, *, config=None)`, returning an `Extraction`. Until then, `extract` locates the span texts in their chunks and warns. |
| A subclass of `LLMSpanExtractor` or another extractor that overrides `extract_spans`, for example to filter the spans | Override `extract` instead, and filter the `Extraction`'s spans. Until then, `extract` and `aextract` call your `extract_spans` and warn. |
| `LLMSpanExtractor()`, `LLMSpanExtractor(model="gpt-4o")` | `LLMSpanExtractor(LLMClient("gpt-4o"))` |
| `LLMSpanExtractor(client, model=...)` | The model of the client: `LLMClient(model)` |
| `LLMSpanExtractor(client, extraction_mode=..., batch_size=..., span_match_mode=..., fuzzy_threshold=..., extraction_prompt=..., system_prompt=...)` | `LLMSpanExtractor(client, config=LLMExtractConfig(...))` with the same names |
| `LLMSpanExtractor(..., max_display_spans=...)`, `ModelSpanExtractor(..., max_display_spans=...)` | `Transformer(..., config=TransformConfig(max_display_spans=...))` |
| `ModelSpanExtractor()`, `ModelSpanExtractor("org/model")`, `ModelSpanExtractor(model_path=..., device=...)` | `ModelSpanExtractor.from_pretrained("org/model", device=...)` |
| `ModelSpanExtractor(..., threshold=..., min_span_chars=..., merge_gap_chars=..., max_length=..., doc_stride=...)` | `config=ModelExtractConfig(...)` with the same names |
| `ModelSpanExtractor(..., extraction_mode=...)` | Nothing in its place: it had no effect |
| `SemanticHighlightExtractor()`, `SemanticHighlightExtractor("org/model")`, `SemanticHighlightExtractor(model_name=..., device=..., max_tokens=...)` | `SemanticHighlightExtractor.from_pretrained("org/model", device=..., max_tokens=...)` |
| `SemanticHighlightExtractor(..., threshold=..., output_mode=..., language=..., min_span_tokens=..., merge_gap=...)` | `config=SemanticHighlightConfig(...)` with the same names |

`from_pretrained` is the one place a model name becomes a loaded model, so you decide when the download happens. A loaded model can also be passed in: `ModelSpanExtractor(model)`.

## Templates

The settings of a `TemplateManager` are its `config`, the default of every `Transformer` built with it. Pass other settings to the `Transformer` or to one call instead of changing the templates:

```python
# Before
templates.set_mode("random")
templates.set_citation_mode("hidden")

# After: for every call of this transformer
transformer = Transformer(
    extractor, templates, config=TransformConfig(template_mode="random", citation_mode="hidden")
)
# or for one call
answer = transformer.transform(question, chunks, config=TransformConfig(template_mode="random"))
```

| Old | New |
|---|---|
| `TemplateManager.set_mode(mode)`, assigning `TemplateManager.current_mode` | `TransformConfig(template_mode=mode)`; for structured mode, `TemplateManager.use_structured_mode()` |
| `TemplateManager.set_citation_mode(mode)`, assigning `TemplateManager.citation_mode` | `TransformConfig(citation_mode=mode)` |
| `TemplateManager.set_citation_format(format)` | `TransformConfig(citation_format=format)` |
| Contextual template mode without an LLM client, which used the static template | `TemplateManager(llm_client=client)` |
| `TemplateManager.set_rag_system`, `StructuredTemplate.set_rag_system`, `StructuredTemplate.fill_async` | `VerbatimRAG.query_async`, which answers structured questions itself |
| `QuestionSpecificTemplate(model_name=..., device=...)`, question-specific templates without an embedder | `QuestionSpecificTemplate.from_pretrained(model_name, device, templates=...)`, or `embedder=` a loaded model, also in `TemplateManager.use_question_specific_mode(templates, embedder=...)` |

## VerbatimRAG

`VerbatimRAG` takes the `Transformer` that writes its answers, and `k` is an argument of each query:

```python
# Before
rag = VerbatimRAG(index, model="gpt-4o-mini", k=5, template_mode="contextual", max_display_spans=5)
response = rag.query(question)

# After
client = LLMClient("gpt-4o-mini")
transformer = Transformer(
    LLMSpanExtractor(client),
    TemplateManager(llm_client=client, default_mode="contextual"),
    config=TransformConfig(max_display_spans=5),
)
rag = VerbatimRAG(index, transformer=transformer, llm_client=client)
response = rag.query(question, k=5)
```

The `llm_client` of `VerbatimRAG` serves structured template mode, which answers without the `Transformer`. Nothing is set on `VerbatimRAG` by a call, so one instance can serve concurrent queries.

| Old | New |
|---|---|
| `VerbatimRAG(index)`, and the arguments `model`, `template_manager`, `extractor`, `max_display_spans`, `template_mode` and `extraction_mode` | `VerbatimRAG(index, transformer=Transformer(extractor, templates, config=...), llm_client=client)` |
| `VerbatimRAG(index, k=...)` | `rag.query(question, k=...)`, `query_async` and `query_stream` likewise, and `VerbatimDOC(rag, k=...)` |
| Assigning `rag.extractor` or `rag.template_manager` | A `VerbatimRAG` with a `Transformer` of that extractor and those templates |
| Assigning `rag.max_display_spans` | `Transformer(..., config=TransformConfig(max_display_spans=...))` |
| `StreamingRAG(rag).stream_query(question, num_docs, ...)`, `stream_query_sync` | `rag.query_stream(question, k=..., ...)` |
| `verbatim_rag.document.Chunk` | `verbatim_rag.document.DocumentChunk` |

`query_stream` yields typed events instead of dicts, and raises instead of sending an `error` event:

```python
# Before
async for event in StreamingRAG(rag).stream_query(question, num_docs=5):
    if event["type"] == "documents":
        show_sources(event["data"])
    elif event["type"] == "answer":
        show_answer(event["data"])

# After
async for event in rag.query_stream(question, k=5):
    if event.type == "documents":
        show_sources(event.chunks)
    elif event.type == "spans":
        highlight(event.spans)
    else:
        show_answer(event.response)
```

The last event carries the same `QueryResponse` that `query` returns for the same input.

## What changes without a warning

Some behaviour changes in 0.3.0 for the old entry points too. The [changelog](https://github.com/KRLabsOrg/verbatim-rag/blob/main/CHANGELOG.md) has the details:

- A failed LLM call during extraction raises an `UpstreamError` or a `ConfigurationError`, instead of answering from the remaining chunks ([ADR 0006](https://github.com/KRLabsOrg/verbatim-rag/blob/main/docs/adr/0006-failed-llm-call-fails-extraction.md)).
- One span gives one highlight and one citation, where the span is, instead of one at every occurrence of its text ([ADR 0001](https://github.com/KRLabsOrg/verbatim-rag/blob/main/docs/adr/0001-one-occurrence-per-span.md)).
- Random template mode gives the same question the same template.
- A context item of the wrong shape raises `InvalidInputError`, a `ValueError`, instead of `TypeError`: `except TypeError` no longer catches it.
- A `QueryResponse` lists in `warnings` how its answer is degraded but still correct, such as quotes that couldn't be located.
- The steps of a call log at DEBUG, and customer text is logged only on the `verbatim_core.content` logger. See [Errors, warnings and logs](guide/errors-and-logs.md).
