# Verbatim Core (Extractor)

The `verbatim_core` package provides RAG-agnostic verbatim span extraction. It works with any retrieval system -- you provide the question and context, it returns grounded answers with citations.

## VerbatimTransform

The main entry point. Takes a question and retrieved context, extracts verbatim spans, and produces a cited response.

```python
from verbatim_core import VerbatimTransform

vt = VerbatimTransform(
    template_mode="contextual",  # or "static"
    extraction_mode="auto",      # "auto", "batch", or "individual"
    max_display_spans=5,
    span_match_mode="exact",     # "exact" or "fuzzy"
    fuzzy_threshold=0.8,         # for fuzzy mode
)

response = vt.transform(
    question="What methodology was used?",
    context=[
        {"content": "We used a mixed-methods approach combining surveys and interviews."},
        {"content": "The sample size was 500 participants across 3 hospitals."},
    ],
)
```

### Context Format

Context items can be:

- **Dicts** with `"content"` (or `"text"`), plus optional `"title"`, `"source"`, `"metadata"`
- **Objects** with a `.text` attribute (e.g., search results from any vector store)

## LLMSpanExtractor

Extracts verbatim spans from documents using an LLM. All extracted spans are verified to exist in the original text.

```python
from verbatim_core.extractors import LLMSpanExtractor
from verbatim_core.llm_client import LLMClient

client = LLMClient(model="gpt-4o-mini")
extractor = LLMSpanExtractor(
    llm_client=client,
    extraction_mode="auto",  # batch for small inputs, individual for large
    batch_size=5,
    span_match_mode="fuzzy",     # "exact" or "fuzzy"
    fuzzy_threshold=0.8,
)

spans = extractor.extract_spans("What is X?", search_results)
# Returns: {"doc text": ["verbatim span 1", "verbatim span 2"], ...}
```

## Fuzzy Span Matching

By default, extracted spans must match the source document exactly (substring match). For documents with OCR artifacts, encoding issues (e.g. corrupted umlauts), or minor formatting differences, enable fuzzy matching:

```python
vt = VerbatimTransform(
    span_match_mode="fuzzy",
    fuzzy_threshold=0.8,  # 0-1, higher = stricter matching
)
```

Or directly on the extractor:

```python
extractor = LLMSpanExtractor(
    llm_client=client,
    span_match_mode="fuzzy",
    fuzzy_threshold=0.8,
)
```

In fuzzy mode:

- Exact matches are tried first (fast path)
- If no exact match, `rapidfuzz` finds the best fuzzy alignment in the document
- The returned span is the **actual text from the document** (not the LLM's version), ensuring correct character offsets for highlighting
- Spans below the threshold are rejected with a warning

This is particularly useful for medical documents, scanned PDFs, or any text that may have been through lossy format conversions.

## Custom Extraction Prompts

You can provide a custom extraction prompt to control what the LLM extracts. Prompts use [Jinja2](https://jinja.palletsprojects.com/) syntax with `{{ question }}` and `{{ documents }}` variables:

```python
custom_prompt = """Extract key findings from the documents.

Question: {{ question }}

Documents:
{{ documents }}

Return a JSON object mapping doc IDs to span arrays."""

extractor = LLMSpanExtractor(
    llm_client=client,
    extraction_prompt=custom_prompt,
)
```

You can also pass an optional `system_prompt` to set the LLM's persona:

```python
extractor = LLMSpanExtractor(
    llm_client=client,
    extraction_prompt=custom_prompt,
    system_prompt="You are a medical expert extracting clinical information.",
)
```

These parameters are also available on `VerbatimTransform`:

```python
vt = VerbatimTransform(
    extraction_prompt=custom_prompt,
    system_prompt="You are a medical expert.",
    span_match_mode="fuzzy",
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

### Static Templates

Fast, deterministic -- no LLM calls. Static mode is the recommended choice for pipeline and tool-chain use cases where output is consumed by downstream tools or agents. The default template frames output transparently as an unordered list of verbatim excerpts with no synthesis or ranking implied, avoiding misleading machine consumers:

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

### Template Placeholders

- `[DISPLAY_SPANS]` -- all display spans aggregated
- `[FACT_1]`, `[FACT_2]`, ... -- individual span placeholders
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
