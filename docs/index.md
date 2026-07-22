# Verbatim RAG

<p align="center">
  <img src="https://github.com/KRLabsOrg/verbatim-rag/blob/main/assets/chiliground.png?raw=true" alt="ChiliGround Logo" width="400"/>
  <br><em>Chill, I Ground!</em>
</p>

Provenance-first extractive RAG: select answer-relevant source passages and
return them with citations instead of freely rewriting the evidence.

[![PyPI](https://img.shields.io/pypi/v/verbatim-rag)](https://pypi.org/project/verbatim-rag/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![ACL 2025](https://img.shields.io/badge/ACL%20Anthology-2025.bionlp--share.8-blue)](https://aclanthology.org/2025.bionlp-share.8/)

## Why Verbatim RAG?

Traditional RAG systems retrieve relevant documents and then allow an LLM to freely generate responses based on that context. This can lead to hallucinations where the model invents facts not present in the source material.

Verbatim RAG reduces unsupported generation by **extracting source passages**.
Built-in verified paths return source text, but that contract does not guarantee
source truth, retrieval recall, extraction relevance/completeness, or generated
contextual framing. Use static templates for fixed deterministic framing.

## Two Packages

| Package | Install | Dependencies | Use case |
|---------|---------|-------------|----------|
| **verbatim-rag** | `pip install verbatim-rag` | Full (torch, milvus, etc.) | Complete RAG pipeline with indexing, search, and extraction |
| **verbatim-core** | `pip install verbatim-core` | Lean (openai, pydantic, rapidfuzz, jinja2) | Reusable evidence transform for integration into existing systems |

## Key Features

- **Reduced generative surface** -- Return evidence instead of freely paraphrasing it
- **Verbatim Extraction** -- Built-in verified paths return text from supplied sources
- **Citation Tracking** -- Responses include source citations and highlights
- **Multiple Extractors** -- LLM-based, fine-tuned ModernBERT, or Zilliz semantic highlighting
- **Local static operation** -- SPLADE + ModernBERT + static templates can run without generative LLM API calls
- **Template System** -- Flexible response formatting with multiple strategies

## How It Works

1. **Document Processing** -- Documents are processed using docling and chunked with chonkie
2. **Document Indexing** -- Indexed using vector embeddings (dense and/or sparse)
3. **Query Processing** -- Relevant documents are retrieved
4. **Span Extraction** -- Key passages are extracted verbatim
5. **Response Generation** -- Templates structure the answer with citations

## Citation

If you use Verbatim RAG in your research, please cite our paper:

```bibtex
@inproceedings{kovacs-etal-2025-kr,
    title = "{KR} Labs at {A}rch{EHR}-{QA} 2025: A Verbatim Approach for Evidence-Based Question Answering",
    author = "Kovacs, Adam and Schmitt, Paul and Recski, Gabor",
    booktitle = "Proceedings of the 24th Workshop on Biomedical Language Processing (Shared Tasks)",
    year = "2025",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.bionlp-share.8/",
}
```
