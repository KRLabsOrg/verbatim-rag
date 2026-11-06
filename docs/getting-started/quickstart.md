# Quickstart

## Lightweight: the Transformer

The fastest way to add verbatim grounding to your existing RAG system: pass a question and the chunks you retrieved, and get an answer whose citations point at the chunks' own text. Works with `verbatim-core` (lean) or `verbatim-rag` (full).

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

## Full Pipeline: VerbatimRAG

For the complete RAG system with document processing, indexing, and retrieval:

```python
from verbatim_core import Transformer
from verbatim_core.extractors import LLMSpanExtractor
from verbatim_core.llm_client import LLMClient
from verbatim_core.templates import TemplateManager
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

# Query: the Transformer writes the answers from the retrieved chunks
client = LLMClient("gpt-4o-mini")
transformer = Transformer(
    LLMSpanExtractor(client),
    TemplateManager(llm_client=client, default_mode="contextual"),
)
rag = VerbatimRAG(index, transformer=transformer)
response = rag.query("What is the main contribution?", k=5)
print(response.answer)
```

`k` and the other retrieval settings are arguments of each query, so one `VerbatimRAG` can serve concurrent queries.

## Using the ModernBERT Extractor

Replace the LLM-based extractor with the current v2 encoder model and static
rendering—no generative LLM API call is made:

```python
from verbatim_rag.extractors import ModelSpanExtractor

extractor = ModelSpanExtractor.from_pretrained("KRLabsOrg/verbatim-rag-modern-bert-v2")
rag = VerbatimRAG(index, transformer=Transformer(extractor, TemplateManager()))
response = rag.query("Main findings?", k=5)
```

`from_pretrained` downloads and loads the model when you call it. `TemplateManager()` writes the answers with the static template.

Upgrading from 0.2.x? See [Migrating to 0.3.0](../migration.md).
