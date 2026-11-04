# Installation

## Full RAG System

For the complete pipeline with document processing, vector indexing, and extraction:

```bash
pip install verbatim-rag
```

This installs all dependencies including torch, transformers, pymilvus, sentence-transformers, docling, and chonkie.

## Lightweight Core

If you only need verbatim span extraction without the full RAG pipeline:

```bash
pip install verbatim-core
```

This installs only `openai`, `pydantic`, `rapidfuzz`, `jinja2` and `typing_extensions` -- no heavy ML dependencies.

The `verbatim-core` package provides the same `verbatim_core` module with:

- `Transformer` -- question + chunks -> answer cited from the chunks' own text
- `LLMSpanExtractor` -- extract verbatim spans using an LLM
- `LLMClient` -- unified OpenAI API wrapper
- Template system for response formatting

## Model-Based Extraction

If you want to use the ModernBERT or Zilliz semantic highlight extractors without the full RAG pipeline:

```bash
pip install verbatim-core[model]
```

This adds `torch`, `transformers`, and `scikit-learn` on top of the lean install.

## Environment Setup

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

!!! note
    The OpenAI API key is only required for LLM-based extraction. The ModernBERT-based extractor (available in `verbatim-rag`) does not require an API key.

## Development Install

```bash
git clone https://github.com/KRLabsOrg/verbatim-rag.git
cd verbatim-rag
pip install -e packages/core/
pip install -e ".[dev]"
```
