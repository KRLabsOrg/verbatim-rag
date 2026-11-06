# Verbatim Core (Extractor)

The `verbatim_core` package provides RAG-agnostic verbatim span extraction. It works with any retrieval system -- you provide the question and context, it returns grounded answers with citations.

## Transformer

Writes a cited answer from a question and chunks you pass in, without an index. Pass the extractor and the templates it uses:

```python
from verbatim_core import Chunk, TransformConfig, Transformer
from verbatim_core.extractors import LLMSpanExtractor
from verbatim_core.llm_client import LLMClient
from verbatim_core.templates import TemplateManager

client = LLMClient(model="gpt-4o-mini")
transformer = Transformer(
    LLMSpanExtractor(client),
    TemplateManager(llm_client=client),
    config=TransformConfig(template_mode="contextual", max_display_spans=5),
)

answer = transformer.transform(
    "What methodology was used?",
    [
        Chunk(id="methods", text="We used a mixed-methods approach combining surveys and interviews."),
        Chunk(id="sample", text="The sample size was 500 participants across 3 hospitals."),
    ],
)
print(answer.text)
for citation in answer.citations:
    print(citation.number, citation.type, citation.chunk_id, citation.start, citation.end)
```

Each span of the answer gets one citation, which points at the span by chunk id and offsets. The first `max_display_spans` spans fill the template, and the citations of the spans the answer shows are marked `display`; the others are cited for reference only. In random template mode the template is chosen by the question and the config's `seed`, so the same question gets the same template. `answer.usage` lists the extractor's work and, when a template LLM ran, the LLM's, and `answer.warnings` says what made the answer less complete, such as a quote that couldn't be located or a template that fell back to the fallback template.

A `config` passed to a call replaces the transformer's default config for that call only, so one transformer can serve concurrent requests. `atransform` is the async version. `stream` and `astream` yield a `spans` event once the spans are extracted, then an `answer` event with the same `Answer` that `transform` returns:

```python
for event in transformer.stream(question, chunks):
    if event.type == "spans":
        show_sources(event.spans)
    else:
        show_answer(event.answer)
```

`ResponseBuilder().build_from_answer(question, answer, chunks)` turns an `Answer` into the `QueryResponse` that `VerbatimTransform` returns, with each cited span highlighted in its chunk.

Structured template mode is not available in the `Transformer`; use `VerbatimRAG` for structured answers.

## VerbatimTransform

Deprecated since 0.3.0, and removed in 0.5.0: use the `Transformer`. [Migrating to 0.3.0](../migration.md) maps each old entry point to its replacement. `VerbatimTransform` turns the context items into chunks with positional ids (`ctx_0`, `ctx_1`, ...), answers with a `Transformer`, and returns the answer as a `QueryResponse`. The same with a `Transformer`:

```python
from verbatim_core import Chunk, TransformConfig, Transformer
from verbatim_core.extractors import LLMSpanExtractor
from verbatim_core.llm_client import LLMClient
from verbatim_core.response_builder import ResponseBuilder
from verbatim_core.templates import TemplateManager

client = LLMClient()
transformer = Transformer(
    LLMSpanExtractor(client),
    TemplateManager(llm_client=client),
    config=TransformConfig(template_mode="contextual"),
)

chunks = [
    Chunk(
        id=f"ctx_{i}",
        text=item["content"],
        metadata={"title": item.get("title", ""), "source": item.get("source", "")},
    )
    for i, item in enumerate(context)
]
answer = transformer.transform(question, chunks)
response = ResponseBuilder().build_from_answer(question, answer, chunks)
```

### Context Format

Context items can be:

- **Dicts** with `"content"` (or `"text"`), plus optional `"title"`, `"source"`, `"metadata"`
- **Objects** with a `.text` attribute (e.g., search results from any vector store)

An item of another shape raises `InvalidInputError` with `param` naming it, such as `context[3]`: the code `missing_field` for a dict without `"content"` or `"text"`, and `invalid_value` otherwise.

## LLMSpanExtractor

Extracts verbatim spans from chunks using an LLM. The LLM quotes the text of each chunk that answers the question, and each quote is located in its chunk. A span's text is always the chunk's own text at the span's offsets.

```python
from verbatim_core import Chunk
from verbatim_core.extractors import LLMExtractConfig, LLMSpanExtractor
from verbatim_core.llm_client import LLMClient

extractor = LLMSpanExtractor(
    LLMClient(model="gpt-4o-mini"),
    config=LLMExtractConfig(
        extraction_mode="auto",  # batch for small inputs, individual for large
        batch_size=5,
        span_match_mode="fuzzy",  # "exact" or "fuzzy"
        fuzzy_threshold=0.8,
    ),
)

extraction = extractor.extract(
    "What methodology was used?",
    [
        Chunk(id="methods", text="We used a mixed-methods approach combining surveys and interviews."),
        Chunk(id="sample", text="The sample size was 500 participants across 3 hospitals."),
    ],
)
for span in extraction.spans:
    print(span.chunk_id, span.start, span.end, span.text)
```

`extract` returns an `Extraction`: the spans with their chunk ids, offsets and scores, the usage of the call (the model, the chunks and characters, the LLM calls and, where the LLM service reports them, the tokens) and its warnings. `aextract` is the async version and makes its LLM calls concurrently. A `config` passed to `extract` replaces the extractor's default config for that call.

Every span has the score 1.0: the LLM extracts without a score.

A quote that can't be located in its chunk is dropped and reported in `extraction.warnings` with the code `quote_not_found`. A failed LLM call raises, and the whole extraction fails: an `UpstreamError` for a timeout, a connection error, a rate limit, a server error or a reply that can't be read, and a `ConfigurationError` when the LLM service rejects the API key, the model or the account's quota.

`extract_spans`, which returns the span texts keyed by chunk text, is deprecated. Unlike `extract`, it tolerates failed LLM calls: a chunk whose call fails yields no spans.

## Fuzzy Span Matching

By default, a quote must match its chunk exactly. For documents with OCR artifacts, encoding issues (e.g. corrupted umlauts), or minor formatting differences, enable fuzzy matching in the extractor's config:

```python
extractor = LLMSpanExtractor(
    client,
    config=LLMExtractConfig(
        span_match_mode="fuzzy",
        fuzzy_threshold=0.8,  # 0-1, higher = stricter matching
    ),
)
```

In fuzzy mode:

- The first exact occurrence of a quote is tried first
- If there is none, `rapidfuzz` finds the best fuzzy alignment in the chunk, the earliest on a tie
- The span is the **chunk's own text** at that alignment, not the LLM's version
- Quotes below the threshold are dropped and reported in the warnings

This is particularly useful for medical documents, scanned PDFs, or any text that may have been through lossy format conversions.

## Custom Extraction Prompts

You can provide a custom extraction prompt to control what the LLM extracts. Prompts use [Jinja2](https://jinja.palletsprojects.com/) syntax with `{{ question }}` and `{{ documents }}` variables:

```python
custom_prompt = """Extract key findings from the documents.

Question: {{ question }}

Documents:
{{ documents }}

Return a JSON object mapping doc IDs to span arrays."""

extractor = LLMSpanExtractor(client, config=LLMExtractConfig(extraction_prompt=custom_prompt))
```

You can also pass an optional `system_prompt` to set the LLM's persona. It is sent only with a custom extraction prompt:

```python
extractor = LLMSpanExtractor(
    client,
    config=LLMExtractConfig(
        extraction_prompt=custom_prompt,
        system_prompt="You are a medical expert extracting clinical information.",
    ),
)
```

## Prompt Bank

Verbatim-core ships with built-in prompt templates that can be loaded, inspected, and used as starting points for custom prompts:

```python
from verbatim_core.prompts import load_prompt, list_prompts

# List available prompts
list_prompts()
# ['extraction/default', 'extraction/structured', 'template/aggregate',
#  'template/fallback', 'template/per_fact']

# Load a prompt template (raw, unrendered)
template = load_prompt("extraction/default")

# Load and render with variables
rendered = load_prompt("extraction/default", question="What is X?", documents="...")
```

Prompts are Jinja2 templates supporting variables (`{{ var }}`), conditionals (`{% if %}...{% endif %}`), and any other Jinja2 features. Literal braces (e.g. in JSON examples) pass through without escaping.

All built-in prompts (`LLMClient` extraction, template generation, fallback) load from the prompt bank -- no prompts are hardcoded in Python code.

## Template System

Controls how extracted spans are formatted into responses.

A `TemplateManager` holds the templates of each mode. Its constructor arguments, such as `default_mode`, describe its `config`, a `TransformConfig`. A `Transformer` built without a `config` uses the template manager's as its default. To choose the template mode, citation mode or citation format, pass a `TransformConfig` to the `Transformer` or to one call. The setters `set_mode`, `set_citation_mode` and `set_citation_format` are deprecated: they change the manager's config, which every transformer using it without its own config shares.

### Static Templates

Fast, deterministic -- no LLM calls. Static mode is the recommended choice for pipeline and tool-chain use cases where output is consumed by downstream tools or agents.

The default template frames output transparently as an unordered list of verbatim excerpts with no synthesis or ranking implied, avoiding misleading machine consumers:

```python
from verbatim_core.templates import TemplateManager

tm = TemplateManager(default_mode="static")
tm.use_static_mode(template="## Findings\n\n[DISPLAY_SPANS]")
```

### Contextual Templates

LLM-generated templates tailored to each question:

```python
from verbatim_core.llm_client import LLMClient

client = LLMClient()
tm = TemplateManager(llm_client=client, default_mode="contextual")
```

### Random Templates

Chooses a template from a pool by the question, so the same question always gets the same template. A `seed` in the `TransformConfig` changes the choice:

```python
from verbatim_core import TransformConfig

transformer.transform(question, chunks, config=TransformConfig(template_mode="random", seed=7))
```

### Question-Specific Templates

Pairs each template with example questions and chooses the template whose example is closest to the question. An embedding model compares them: pass any loaded model with an `encode(texts)` method, such as a `SentenceTransformer`, as `embedder`:

```python
from sentence_transformers import SentenceTransformer

templates = [
    {"template": "## Methods\n\n[RELEVANT_SENTENCES]", "examples": ["What methods were used?"]},
    {"template": "## Results\n\n[RELEVANT_SENTENCES]", "examples": ["What did they find?"]},
]
tm = TemplateManager()
tm.use_question_specific_mode(templates, embedder=SentenceTransformer("all-MiniLM-L6-v2"))
```

`QuestionSpecificTemplate.from_pretrained(model_name, templates=templates)` loads the model and builds the templates in one step. The examples are embedded when the templates are built, and each question when it is answered, so no call loads a model.

### Template Placeholders

- `[DISPLAY_SPANS]` -- all display spans aggregated
- `[SPAN_1]`, `[SPAN_2]`, ... -- individual span placeholders
  (`[FACT_N]` is the pre-0.2.7 name and still works in stored templates)
- `[CITATION_REFS]` -- citation-only reference numbers

### Linked Citations

By default, citations are appended as a flat list at the end. With linked citations, each display span can reference the specific sources that back it:

```python
display_spans = [
    {
        "text": "Treatment X showed 30% improvement over baseline.",
        "citation_ids": ["v1", "v3"],  # links to specific citation spans
    }
]
citation_spans = [
    {"text": "Table 2: Treatment X results...", "citation_id": "v1"},
    {"text": "Secondary outcomes were...", "citation_id": "v2"},
    {"text": "The 30% figure was significant (p<0.01).", "citation_id": "v3"},
]
```

Output with inline citations:

```
[1] Treatment X showed 30% improvement over baseline. [2] [4]
```

Each claim maps directly to its supporting sources. When linked citations are present, the flat `[CITATION_REFS]` placeholder is automatically suppressed to avoid double-referencing.

## @verbatim_enhance Decorator

Wrap any existing RAG function to add verbatim grounding:

```python
from verbatim_core import verbatim_enhance

@verbatim_enhance(max_display_spans=5)
def my_rag_function(question: str) -> list[dict]:
    # Your existing retrieval logic
    return [{"content": "Retrieved text...", "title": "Source"}]

response = my_rag_function("What is X?")
```

## CLI

Process JSON/JSONL files from the command line:

```bash
verbatim-enhance --input queries.jsonl --output results.jsonl
```

## Custom LLM Endpoints

Use any OpenAI-compatible API:

```python
from verbatim_core.llm_client import LLMClient

client = LLMClient(
    model="my-model",
    api_base="http://localhost:8080/v1",
)
```

The `complete()` method also supports an optional `system_prompt`:

```python
response = client.complete(
    prompt="Extract spans from this text...",
    json_mode=True,
    system_prompt="You are a helpful assistant.",
)
```
