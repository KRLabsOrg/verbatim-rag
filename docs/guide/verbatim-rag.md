# Full RAG System

The `verbatim_rag` package provides the complete pipeline: document processing, vector indexing, retrieval, and verbatim extraction.

## Architecture

```
Documents → DocumentProcessor → VerbatimIndex → VerbatimRAG → Response
                (docling/chonkie)   (Milvus)       (extractor + templates)
```

### Core Components

- **VerbatimRAG** -- Main orchestrator
- **VerbatimIndex** -- Vector-based document indexing and retrieval
- **DocumentProcessor** -- Docling + Chonkie document processing
- **SpanExtractor** -- LLM-based or model-based span extraction

## Document Processing

Process PDFs, URLs, and text files with intelligent chunking:

```python
from verbatim_rag.ingestion import DocumentProcessor

processor = DocumentProcessor()

# From URL
doc = processor.process_url(
    url="https://example.com/paper.pdf",
    title="Paper Title",
    metadata={"authors": ["Author A"]},
)

# From file
doc = processor.process_file("path/to/document.pdf", title="My Doc")
```

## Vector Stores

### Local Milvus (SQLite-backed)

```python
from verbatim_rag.vector_stores import LocalMilvusStore

store = LocalMilvusStore(
    db_path="./index.db",
    collection_name="my_collection",
    enable_dense=True,
    enable_sparse=True,
)
```

### Cloud Milvus

```python
from verbatim_rag.vector_stores import CloudMilvusStore

store = CloudMilvusStore(
    uri="https://your-cluster.zillizcloud.com",
    token="your_token",
    collection_name="my_collection",
)
```

## Embedding Providers

### Dense Embeddings

```python
from verbatim_rag.embedding_providers import SentenceTransformersProvider

dense = SentenceTransformersProvider(model_name="all-MiniLM-L6-v2")
```

### Sparse Embeddings (CPU-only)

```python
from verbatim_rag.embedding_providers import SpladeProvider

sparse = SpladeProvider(
    model_name="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
    device="cpu",
)
```

## Indexing and Querying

```python
from verbatim_core import Transformer
from verbatim_core.extractors import LLMSpanExtractor
from verbatim_core.llm_client import LLMClient
from verbatim_core.templates import TemplateManager
from verbatim_rag import VerbatimIndex, VerbatimRAG

index = VerbatimIndex(
    vector_store=store,
    dense_provider=dense,
    sparse_provider=sparse,
)

# Add documents
index.add_documents([doc1, doc2])

# Query
client = LLMClient()
transformer = Transformer(
    LLMSpanExtractor(client), TemplateManager(llm_client=client, default_mode="contextual")
)
rag = VerbatimRAG(index, transformer=transformer)
response = rag.query("What is the main contribution?", k=5)

print(response.answer)
for doc in response.documents:
    for h in doc.highlights:
        print(f"  [{h.start}:{h.end}] {h.text}")
```

`query` and `query_async` retrieve with `k` and the other retrieval arguments of the call, rerank when a reranker is given, and answer with the `Transformer`. Each retrieved row becomes a chunk whose id is the row's id: it stays the same while its document stays indexed, and changes when the document is indexed again. Nothing is set on `VerbatimRAG` by a call, so one instance can serve concurrent queries.

When the reranker's service fails, the results keep the retrieval order, and the response's `warnings` include `reranker_fallback`. `response.warnings` lists every way an answer is degraded but still correct, such as quotes that couldn't be located in their chunk (`quote_not_found`). Structured template mode answers without the `Transformer`, with the `llm_client` given to `VerbatimRAG`.

`VerbatimRAG(index)` and the arguments `model`, `k`, `template_manager`, `extractor`, `max_display_spans`, `template_mode` and `extraction_mode` still work and warn: without a transformer, `VerbatimRAG` builds the one it built before 0.3.0.

## Span Extractors

### LLM-based (default)

Uses OpenAI models for extraction. Requires `OPENAI_API_KEY`.

### ModernBERT-based (no API key)

```python
from verbatim_rag.extractors import ModelSpanExtractor

extractor = ModelSpanExtractor.from_pretrained("KRLabsOrg/verbatim-rag-modern-bert-v2")
rag = VerbatimRAG(index, transformer=Transformer(extractor, TemplateManager()))
```

`from_pretrained` downloads and loads the model, or loads it from a local path. It records the
model's Hugging Face commit as the version that the usage of each extraction reports. For a local
path, pass the version yourself: `from_pretrained("/models/v2", model_version="v2.1")`. The spans
keep the character offsets and probabilities the model computed.

### Semantic Highlighting

```python
from verbatim_rag.extractors import SemanticHighlightConfig, SemanticHighlightExtractor

extractor = SemanticHighlightExtractor.from_pretrained(
    "zilliz/semantic-highlight-bilingual-v1",
    config=SemanticHighlightConfig(threshold=0.5, output_mode="sentences"),
)
```

## API and web prototype

The source tree includes a FastAPI API and Vite/React UI prototype. They are not
part of the PyPI wheel or the current core compatibility gate. Follow
[#27](https://github.com/KRLabsOrg/verbatim-rag/issues/27) for the reproducible
local-stack work rather than treating these components as a supported
production deployment.
