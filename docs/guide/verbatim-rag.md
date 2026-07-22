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
from verbatim_rag import VerbatimIndex, VerbatimRAG

index = VerbatimIndex(
    vector_store=store,
    dense_provider=dense,
    sparse_provider=sparse,
)

# Add documents
index.add_documents([doc1, doc2])

# Query
rag = VerbatimRAG(index, k=5)
response = rag.query("What is the main contribution?")

print(response.answer)
for doc in response.documents:
    for h in doc.highlights:
        print(f"  [{h.start}:{h.end}] {h.text}")
```

## Span Extractors

### LLM-based (default)

Uses OpenAI models for extraction. Requires `OPENAI_API_KEY`.

### ModernBERT-based (no API key)

```python
from verbatim_rag.extractors import ModelSpanExtractor

extractor = ModelSpanExtractor("KRLabsOrg/verbatim-rag-modern-bert-v2")
rag = VerbatimRAG(index, extractor=extractor, template_mode="static")
```

### Semantic Highlighting

```python
from verbatim_rag.extractors import SemanticHighlightExtractor

extractor = SemanticHighlightExtractor(
    threshold=0.5,
    output_mode="sentences",
)
```

## API and web prototype

The source tree includes a FastAPI API and Vite/React UI prototype. They are not
part of the PyPI wheel or the current core compatibility gate. Follow
[#27](https://github.com/KRLabsOrg/verbatim-rag/issues/27) for the reproducible
local-stack work rather than treating these components as a supported
production deployment.
