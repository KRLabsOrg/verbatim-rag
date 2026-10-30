# verbatim-core

Lightweight verbatim span extraction -- the RAG-agnostic core of [verbatim-rag](https://github.com/KRLabsOrg/verbatim-rag).

Extract source text spans from documents that answer a question. No vector
databases, embeddings, or heavy ML dependencies by default—runtime dependencies
are `openai`, `pydantic`, `rapidfuzz`, and `jinja2`.

## Project Structure

```
.
├── README.md                           # Package documentation
├── pyproject.toml                      # Package metadata, dependencies and extras
├── ruff.toml                           # Package level ruff lint/format configuration
└── verbatim_core
    ├── __init__.py                     # Public API exports
    ├── cli.py                          # `verbatim-enhance` batch CLI (JSON/JSONL in, JSONL out)
    ├── enhance.py                      # @verbatim_enhance decorator for existing RAG functions
    ├── extractor_models
    │   ├── dataset.py                  # Data classes and torch Dataset for span-extractor training
    │   ├── model.py                    # QAModel: transformer-based sentence/span classifier
    │   ├── preprocess_ragbench.py      # Converts RAGBench into extractor training data
    │   ├── train.py                    # Training entry point (CLI)
    │   └── trainer.py                  # Training/evaluation loop and metrics
    ├── extractors.py                   # Span extractors (LLMSpanExtractor and model-based)
    ├── llm_client.py                   # Unified OpenAI client wrapper (sync + async)
    ├── models.py                       # Pydantic models for responses, documents and highlights
    ├── prompts
    │   ├── __init__.py                 # Prompt loader
    │   ├── extraction
    │   │   ├── default.txt             # Default span-extraction prompt
    │   │   └── structured.txt          # Structured (template-driven) extraction prompt
    │   └── template
    │       ├── aggregate.txt           # Prompt for one aggregate answer template
    │       ├── fallback.txt            # Prompt for fallback when no spans are found
    │       └── per_fact.txt            # Prompt for per-fact templates
    ├── providers.py                    # RAGProvider abstract interface (retrieve/generate)
    ├── response_builder.py             # Builds responses with highlights and citations
    ├── templates
    │   ├── __init__.py                 # Template exports
    │   ├── base.py                     # Base class for template strategies
    │   ├── contextual.py               # LLM-generated templates using retrieved context
    │   ├── filler.py                   # Fills templates with extracted spans
    │   ├── manager.py                  # TemplateManager: selects and applies strategies
    │   ├── question_specific.py        # LLM-generated templates tailored to the question
    │   ├── random.py                   # Randomly chosen predefined templates
    │   ├── static.py                   # Fixed predefined template
    │   └── structured.py               # Template-driven structured extraction
    ├── transform.py                    # VerbatimTransform: question + context -> cited answer
    ├── types.py                        # Shared typing protocols
    └── universal_document.py           # UniversalDocument: RAG-agnostic document dataclass
```

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
