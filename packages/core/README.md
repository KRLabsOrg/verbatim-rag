# verbatim-core

Lightweight verbatim span extraction -- the RAG-agnostic core of [verbatim-rag](https://github.com/KRLabsOrg/verbatim-rag).

Extract source text spans from documents that answer a question. No vector
databases, embeddings, or heavy ML dependencies by default—runtime dependencies
are `openai`, `pydantic`, `rapidfuzz`, `jinja2` and `typing_extensions`.

## Installation

```bash
pip install verbatim-core
```

## Quick Start

```python
from verbatim_core import Chunk, Transformer
from verbatim_core.extractors import LLMSpanExtractor
from verbatim_core.llm_client import LLMClient
from verbatim_core.response_builder import ResponseBuilder
from verbatim_core.templates import TemplateManager

client = LLMClient("gpt-4o-mini")
transformer = Transformer(
    LLMSpanExtractor(client),
    TemplateManager(llm_client=client, default_mode="contextual"),
)

question = "What is the main finding?"
chunks = [
    Chunk(id="paper-a", text="The study found that X leads to Y.", metadata={"title": "Paper A"}),
    Chunk(
        id="paper-b",
        text="Results show Z is statistically significant.",
        metadata={"title": "Paper B"},
    ),
]

answer = transformer.transform(question, chunks)
print(answer.text)

# Each citation points at a span of a chunk, by id and offsets
for citation in answer.citations:
    print(citation.number, citation.chunk_id, citation.start, citation.end)

# A QueryResponse, with the highlights of each document
response = ResponseBuilder().build_from_answer(question, answer, chunks)
for doc in response.documents:
    for highlight in doc.highlights:
        print(f"  [{highlight.start}:{highlight.end}] {highlight.text}")
```

## What This Package Includes

- **Transformer** -- question + chunks -> answer cited from the chunks' own text (sync, async, streaming)
- **LLMSpanExtractor**, **ModelSpanExtractor**, **SemanticHighlightExtractor** -- extract verbatim spans, with their chunk ids and offsets
- **LLMClient** -- unified OpenAI API wrapper (sync + async)
- **TemplateManager** -- response formatting with multiple template strategies
- **@verbatim_enhance** -- decorator to enhance existing RAG functions
- **CLI** (`verbatim-enhance`) -- batch processing from the command line

## Model-Based Extraction

For ModernBERT or Zilliz semantic highlight extractors (adds torch, transformers):

```bash
pip install verbatim-core[model]
```

```python
from verbatim_core import Transformer
from verbatim_core.extractors import ModelSpanExtractor
from verbatim_core.templates import TemplateManager

extractor = ModelSpanExtractor.from_pretrained("KRLabsOrg/verbatim-rag-modern-bert-v2")
transformer = Transformer(extractor, TemplateManager())  # static templates: no LLM call
```

## Upgrading from 0.2.x

`VerbatimTransform`, `verbatim_query` and the other 0.2 entry points still work and warn until
0.5.0. The [migration guide](https://krlabsorg.github.io/verbatim-rag/migration/) maps each one to
its replacement.

## Environment

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Full RAG System

For the complete RAG pipeline with vector indexing, embeddings, and document processing, install the full package:

```bash
pip install verbatim-rag
```
