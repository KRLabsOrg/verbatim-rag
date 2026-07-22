# verbatim-core

Lightweight verbatim span extraction -- the RAG-agnostic core of [verbatim-rag](https://github.com/KRLabsOrg/verbatim-rag).

Extract source text spans from documents that answer a question. No vector
databases, embeddings, or heavy ML dependencies by default—runtime dependencies
are `openai`, `pydantic`, `rapidfuzz`, and `jinja2`.

## Installation

```bash
pip install verbatim-core
```

## Quick Start

```python
from verbatim_core import VerbatimTransform

vt = VerbatimTransform()
response = vt.transform(
    question="What is the main finding?",
    context=[
        {"content": "The study found that X leads to Y.", "title": "Paper A"},
        {"content": "Results show Z is statistically significant.", "title": "Paper B"},
    ],
)

print(response.answer)

# Access individual highlights and citations
for doc in response.documents:
    for highlight in doc.highlights:
        print(f"  [{highlight.start}:{highlight.end}] {highlight.text}")
```

## What This Package Includes

- **VerbatimTransform** -- question + context -> cited, grounded answer
- **LLMSpanExtractor** -- extract verbatim spans using an LLM
- **LLMClient** -- unified OpenAI API wrapper (sync + async)
- **TemplateManager** -- response formatting with multiple template strategies
- **@verbatim_enhance** -- decorator to enhance existing RAG functions
- **CLI** (`verbatim-enhance`) -- batch processing from the command line

## Model-Based Extraction

For ModernBERT or Zilliz semantic highlight extractors (adds torch, transformers):

```bash
pip install verbatim-core[model]
```

## Environment

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Full RAG System

For the complete RAG pipeline with vector indexing, embeddings, and document processing, install the full package:

```bash
pip install verbatim-rag
```
