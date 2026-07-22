# Quickstart

## Lightweight: VerbatimTransform

The fastest way to add verbatim grounding to your existing RAG system. Works with `verbatim-core` (lean) or `verbatim-rag` (full).

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

# Access highlights and citations
for doc in response.documents:
    for highlight in doc.highlights:
        print(f"  [{highlight.start}:{highlight.end}] {highlight.text}")
```

## Full Pipeline: VerbatimRAG

For the complete RAG system with document processing, indexing, and retrieval:

```python
from verbatim_rag import VerbatimIndex, VerbatimRAG
from verbatim_rag.ingestion import DocumentProcessor
from verbatim_rag.vector_stores import LocalMilvusStore
from verbatim_rag.embedding_providers import SpladeProvider

# Process documents
processor = DocumentProcessor()
document = processor.process_url(
    url="https://aclanthology.org/2025.bionlp-share.8.pdf",
    title="KR Labs at ArchEHR-QA 2025",
)

# Create sparse embedding provider (CPU-only)
sparse_provider = SpladeProvider(
    model_name="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
    device="cpu",
)

# Create vector store and index
vector_store = LocalMilvusStore(
    db_path="./index.db",
    collection_name="verbatim_rag",
    enable_dense=False,
    enable_sparse=True,
)

index = VerbatimIndex(
    vector_store=vector_store,
    sparse_provider=sparse_provider,
)
index.add_documents([document])

# Query
rag = VerbatimRAG(index)
response = rag.query("What is the main contribution?")
print(response.answer)
```

## Using the ModernBERT Extractor

Replace the LLM-based extractor with the current v2 encoder model and static
rendering—no generative LLM API call is made:

```python
from verbatim_rag.extractors import ModelSpanExtractor

extractor = ModelSpanExtractor("KRLabsOrg/verbatim-rag-modern-bert-v2")
rag = VerbatimRAG(index, extractor=extractor, template_mode="static")
response = rag.query("Main findings?")
```
