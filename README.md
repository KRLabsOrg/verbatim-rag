# Verbatim RAG

<p align="center">
  <img src="https://github.com/KRLabsOrg/verbatim-rag/blob/main/assets/chiliground.png?raw=true" alt="ChiliGround Logo" width="400"/>
  <br><em>Chill, I Ground! 🌶 ️</em>
</p>

Provenance-first extractive RAG: retrieve documents, select answer-relevant
passages, and return source excerpts with citations instead of freely rewriting
the evidence.

[![PyPI](https://img.shields.io/pypi/v/verbatim-rag)](https://pypi.org/project/verbatim-rag/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IACXwo3ezgA1yXarxVOC4yXjdUPmOI1H?usp=sharing)
[![ACL 2025](https://img.shields.io/badge/ACL%20Anthology-2025.bionlp--share.8-blue)](https://aclanthology.org/2025.bionlp-share.8/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.21102-b31b1b.svg)](https://arxiv.org/abs/2605.21102)
[![HF Collection](https://img.shields.io/badge/🤗%20HuggingFace-verbatim--rag--v1-yellow)](https://huggingface.co/collections/KRLabsOrg/verbatim-rag-v1)

## Concept

Traditional RAG systems retrieve relevant documents and then allow an LLM to
freely generate a response. Verbatim RAG reduces that generative surface by
selecting and displaying passages from retrieved context.

Built-in verified extraction paths return evidence text from the supplied
source. This is a provenance guarantee, not a truth guarantee: retrieval may be
incomplete, a source may be wrong, and an extractor may choose an irrelevant or
incomplete passage. The default contextual template may also generate
presentation text around cited excerpts; use `template_mode="static"` when the
framing must be fixed and deterministic.

On the paper's 100-row ACL-Verbatim benchmark, the 150M-parameter
ACL-specialized model achieved **53.6 micro Word-F1**, compared with **48.7** for
the strongest evaluated LLM extractor. In the generic v2
[model-card evaluation](https://huggingface.co/KRLabsOrg/verbatim-rag-modern-bert-v2),
v2 achieved higher micro Word-F1 than the evaluated Zilliz Semantic Highlight
and Provence baselines on ACL, RAGBench, Squeez, and QASPER slices. See the
[paper](https://arxiv.org/abs/2605.21102) for the benchmark design and
limitations.

The pipeline can also use local encoder models for retrieval and extraction plus
static rendering, without generative LLM API calls. With SPLADE and
`ModelSpanExtractor`, that configuration supports CPU execution after model
weights are available.

## What "verbatim" means

| Property | Built-in exact/static path | Outside the guarantee |
|---|---|---|
| Evidence text | Returned from retrieved source text | Custom/structured extractors must enforce their own contract |
| Rendering | Exact excerpts plus fixed transparent framing | Contextual mode can generate introductions, labels, and connective text |
| Citations | Source citations and highlights are returned | Repeated identical text can still make source-offset mapping ambiguous |
| Correctness | Provenance can be inspected | Source truth, retrieval recall, relevance, completeness, and entailment |

## Installation

```bash
# Install the package
pip install verbatim-rag
```

For local development:

```bash
pip install -e packages/core/
pip install -e .
```

## Lightweight Core

If you only need the reusable verbatim core without the full RAG pipeline (no torch, transformers, or Milvus):

```bash
pip install verbatim-core
```

```python
from verbatim_core import VerbatimTransform

vt = VerbatimTransform()
response = vt.transform(
    question="What is the main finding?",
    context=[
        {"content": "The study found that X leads to Y.", "title": "Paper A"},
        {"content": "Results show Z is significant.", "title": "Paper B"},
    ],
)
print(response.answer)
```

Dependencies: only `openai`, `pydantic`, `rapidfuzz`, and `jinja2`.

## Repository map

| Surface | Location | Responsibility |
|---|---|---|
| `verbatim-core` | This repository, `packages/core/` | Reusable question + context → evidence transform, validation, templates, citations |
| `verbatim-rag` | This repository, `verbatim_rag/` | Reference ingestion, indexing, retrieval, and orchestration pipeline |
| Research/training | [`KRLabsOrg/acl-verbatim`](https://github.com/KRLabsOrg/acl-verbatim) | Paper reproduction, v2 training, datasets, and canonical evaluation |
| Hosted client | [`KRLabsOrg/verbatim-client`](https://github.com/KRLabsOrg/verbatim-client) | SDK and CLI for hosted Verbatim services |
| Agent adapters | [`KRLabsOrg/verbatim-mcp`](https://github.com/KRLabsOrg/verbatim-mcp), [`KRLabsOrg/verbatim-skill`](https://github.com/KRLabsOrg/verbatim-skill) | Thin MCP and agent integrations |

## Quick Start

```python
from verbatim_rag import VerbatimIndex, VerbatimRAG
from verbatim_rag.ingestion import DocumentProcessor
from verbatim_rag.vector_stores import LocalMilvusStore
from verbatim_rag.embedding_providers import SpladeProvider

# Process documents with intelligent chunking
processor = DocumentProcessor()

# Process PDFs from URLs
document = processor.process_url(
    url="https://aclanthology.org/2025.bionlp-share.8.pdf",
    title="KR Labs at ArchEHR-QA 2025: A Verbatim Approach for Evidence-Based Question Answering",
    metadata={"authors": ["Adam Kovacs", "Paul Schmitt", "Gabor Recski"]}
)

# Create embedding provider and vector store
sparse_provider = SpladeProvider(
    model_name="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
    device="cpu"
)
vector_store = LocalMilvusStore(
    db_path="./index.db",
    collection_name="verbatim_rag",
    enable_dense=False,
    enable_sparse=True,
)

# Create index with providers
index = VerbatimIndex(
    vector_store=vector_store,
    sparse_provider=sparse_provider
)
index.add_documents([document])

# Then query the index
rag = VerbatimRAG(index)

response = rag.query("What is the main contribution of the paper?")
print(response.answer)
```


### Environment Setup

Set your OpenAI API key before using the system:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## How It Works

1. **Document Processing**: Documents are processed using docling for format conversion and chonkie for chunking
2. **Document Indexing**: Documents are indexed using vector embeddings (both dense and sparse)
3. **Template Management**: Response templates are created and stored for common question types
4. **Query Processing**: 
   - Relevant documents are retrieved
   - Key passages are extracted verbatim using either LLM-based or fine-tuned span extractors
   - Responses are structured using templates
   - Citations link back to source documents

The evidence excerpts remain inspectable source text. Retrieval, extraction
quality, and any generated contextual framing remain separate concerns.

## Architecture

### Core Components

- **VerbatimRAG** (`verbatim_rag/core.py`): Main orchestrator that coordinates document retrieval, span extraction, and response generation
- **VerbatimIndex** (`verbatim_rag/index.py`): Vector-based document indexing and retrieval
- **SpanExtractor** (`verbatim_rag/extractors.py`): Abstract interface for extracting relevant text spans from documents
  - **LLMSpanExtractor**: Uses OpenAI models to identify relevant spans
  - **ModelSpanExtractor**: Uses fine-tuned BERT-based models for span classification
- **DocumentProcessor** (`verbatim_rag/ingestion/`): Docling + Chonkie integration for intelligent document processing
- **Document** (`verbatim_rag/document.py`): Core document representation with metadata

### Data Flow
1. Documents are processed and chunked using docling and chonkie
2. Documents are indexed using vector embeddings
3. User queries retrieve relevant documents
4. Span extractors identify verbatim passages that answer the question
5. Response templates structure the final answer with citations
6. Responses expose selected source text with document references; guarantee
   details depend on the extractor and template mode described above

## API and web prototype

The repository contains a FastAPI API and Vite/React development UI. They are
not included in the PyPI wheel and are not yet part of the same compatibility
gate as `verbatim-core`. The Compose stack below runs both locally
([#27](https://github.com/KRLabsOrg/verbatim-rag/issues/27)); the document
lifecycle contract is tracked in
[#31](https://github.com/KRLabsOrg/verbatim-rag/issues/31).

## Local development stack (Docker Compose)

Docker Compose runs the API and the web UI together as a **development and demo
stack**. It is meant for local work and evaluation; it is not a production
deployment (no TLS, authentication, tenancy, or scaling).

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with Docker Compose v2.24+
  (the stack uses the optional `env_file` syntax)
- An OpenAI-compatible API key (Groq by default, see `api/dependencies.py`)

### Build and start

```bash
# 1. Create configuration from the example
cp .env.example .env

# 2. Edit .env and set your API key
#    OPENAI_API_KEY=your_api_key_here

# 3. Build the images and start the stack
docker compose up --build

# 4. Open http://localhost in your browser
```

Add `-d` to run detached. The first start downloads the embedding model from
HuggingFace, so it needs network access and can take several minutes; the API
health check covers that with a startup grace period (`start_period` in
`docker-compose.yml` — raise it on a slow connection), and the frontend waits
for the API to report healthy.

The API key is not validated at startup: a missing or empty `OPENAI_API_KEY`
still brings the stack up healthy and surfaces as an error on the first query.

### Logs and status

```bash
docker compose ps                  # service and health-check state
docker compose logs -f             # follow all services
docker compose logs -f api         # follow one service
```

The API health check polls `/api/status`, which fails visibly when a required
dependency or configuration is unavailable — a stuck `starting` state in
`docker compose ps` means the API could not initialise, and `docker compose
logs api` shows why.

### Stop and persistence

The Milvus Lite database lives in a Docker volume named `milvus_data`, so the
index survives restarts:

```bash
docker compose down          # stop containers, keep data
docker compose down -v       # also remove the database volume
```

### Secret handling

No key is baked into the image: `OPENAI_API_KEY` is passed at run time only.
Keep it in `.env`, which is git-ignored and excluded from the build context via
`.dockerignore`, or pass it inline for a single run:

```bash
OPENAI_API_KEY=your_key_here docker compose up --build
```

### Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | (required) | OpenAI-compatible API key |
| `INDEX_PATH` | `/data/index.db` | Milvus Lite database path in the container |
| `FRONTEND_PORT` | `80` | Host port the frontend is published on |

All of these can be set in `.env` (copy `.env.example`) or exported in the
shell. CORS configuration is not needed: the browser only talks to nginx, which
proxies `/api/` to the backend same-origin (the Vite dev server does the same
via its own proxy).

### Architecture

```mermaid
graph LR
    A[Browser] -->|http://localhost| B[nginx :80]
    B -->|/api/*| C[FastAPI :8000]
    C --> D[(Milvus Lite<br>/data/index.db)]
```

- **nginx** serves the React SPA and proxies `/api/` requests to the backend
- **FastAPI** handles all API requests using the Verbatim RAG pipeline
- **Milvus Lite** stores the vector index in a persistent Docker volume

The frontend is served as a static Vite build and calls the API through relative
`/api/` paths, which nginx proxies to the `api` service over the Compose
network. No host address is baked into the bundle, so changing `FRONTEND_PORT`
or serving the stack from another host needs no rebuild.

### Container dependency lock

The image installs from `docker/constraints.txt`, a generated lock of one
known-working environment. `pyproject.toml` keeps the library's broad supported
ranges; the lock narrows them for the container only, so a resolution that
breaks the stack cannot silently reach the image.

`docker/overrides.txt` is the hand-maintained input listing resolutions known to
break the container, and documents the regeneration command:

```bash
uv pip compile pyproject.toml --universal --python-version 3.11 \
    --override docker/overrides.txt -o docker/constraints.txt
```

`--universal` resolves with environment markers rather than for the host, so the
same lock builds on both arm64 and x86_64 machines. Regenerate after changing
dependencies in `pyproject.toml` or bumping `packages/core`, then verify with
`docker compose up --build`.

## ModernBERT Span Extractor

[KRLabsOrg/verbatim-rag-modern-bert-v2](https://huggingface.co/KRLabsOrg/verbatim-rag-modern-bert-v2) is a 150M-parameter query-conditioned token classifier built on `gte-reranker-modernbert-base`. It supports up to 8,192 tokens and is trained on scientific papers, Wikipedia QA, financial tables, medical literature, legal contracts, product manuals, and code/tool output.

The linked model card reports higher micro Word-F1 than the evaluated Zilliz
Semantic Highlight and Provence baselines on ACL, RAGBench, Squeez, and QASPER
slices. These are extractor evaluations, not end-to-end hallucination rates.

`ModelSpanExtractor` defaults to this model:

```python
from verbatim_rag.core import VerbatimRAG
from verbatim_rag.index import VerbatimIndex
from verbatim_rag.extractors import ModelSpanExtractor
from verbatim_rag.vector_stores import LocalMilvusStore
from verbatim_rag.embedding_providers import SpladeProvider

extractor = ModelSpanExtractor(
    model_path="KRLabsOrg/verbatim-rag-modern-bert-v2",  # default
    threshold=0.2,
    min_span_chars=30,
    merge_gap_chars=20,
    device=None,  # auto-detects cuda, mps, cpu
)

sparse_provider = SpladeProvider(
    model_name="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
    device="cpu"
)
vector_store = LocalMilvusStore(
    db_path="./index.db",
    collection_name="verbatim_rag",
    enable_dense=False,
    enable_sparse=True,
)
index = VerbatimIndex(vector_store=vector_store, sparse_provider=sparse_provider)

rag_system = VerbatimRAG(
    index=index,
    extractor=extractor,
    template_mode="static",  # no generated contextual framing
    k=5,
)
response = rag_system.query("Main findings of the paper?")
print(response.answer)
```

### Datasets

| Resource | Link |
|---|---|
| 114K ACL Anthology papers in structured Markdown | [KRLabsOrg/acl-anthology-md](https://huggingface.co/datasets/KRLabsOrg/acl-anthology-md) |
| Approximately 195K silver-labelled canonical query-chunk rows | [KRLabsOrg/verbatim-spans](https://huggingface.co/datasets/KRLabsOrg/verbatim-spans) |
| Human-annotated ACL extraction benchmark | [KRLabsOrg/acl-verbatim-spans](https://huggingface.co/datasets/KRLabsOrg/acl-verbatim-spans) |
| Training and evaluation pipeline | [KRLabsOrg/acl-verbatim](https://github.com/KRLabsOrg/acl-verbatim) |

## Citation

If you use Verbatim RAG or the extractive models in your research, please cite our papers:

```bibtex
@misc{Recski:2026,
    title={ACL-Verbatim: hallucination-free question answering for research},
    author={Gábor Recski and Szilveszter Tóth and Nadia Verdha and István Boros and Ádám Kovács},
    year={2026},
    eprint={2605.21102},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2605.21102},
}

@inproceedings{kovacs-etal-2025-kr,
    title = "{KR} Labs at {A}rch{EHR}-{QA} 2025: A Verbatim Approach for Evidence-Based Question Answering",
    author = "Kovacs, Adam  and
      Schmitt, Paul  and
      Recski, Gabor",
    editor = "Soni, Sarvesh  and
      Demner-Fushman, Dina",
    booktitle = "Proceedings of the 24th Workshop on Biomedical Language Processing (Shared Tasks)",
    month = aug,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.bionlp-share.8/",
    pages = "69--74",
    ISBN = "979-8-89176-276-3",
    abstract = "We present a lightweight, domain{-}agnostic verbatim pipeline for evidence{-}grounded question answering. Our pipeline operates in two steps: first, a sentence-level extractor flags relevant note sentences using either zero-shot LLM prompts or supervised ModernBERT classifiers. Next, an LLM drafts a question-specific template, which is filled verbatim with sentences from the extraction step. This prevents hallucinations and ensures traceability. In the ArchEHR{-}QA 2025 shared task, our system scored 42.01{\%}, ranking top{-}10 in core metrics and outperforming the organiser{'}s 70B{-}parameter Llama{-}3.3 baseline. We publicly release our code and inference scripts under an MIT license."
}
```
