# Verbatim RAG: Stop Hallucinating, Start Quoting

<p align="center">
  <img src="https://github.com/KRLabsOrg/verbatim-rag/blob/main/assets/chiliground.png?raw=true" alt="ChiliGround Logo" width="400"/>
  <br><em>Chill, I Ground!</em>
</p>

**TL;DR:**
- **Extract, Don't Generate**: LLMs select exact text spans instead of generating -> zero hallucinations
- **Wrap Existing RAG**: Integrate with LangChain/LlamaIndex in 15 lines of code
- **CPU-Only Pipeline**: No GPU, no API costs, works offline (SPLADE + ModernBERT) -> lightweight and efficient
- **Template Control**: Static, dynamic, or question-specific response formatting
- **Smart Chunking**: Structure-aware with raw/enhanced dual-chunk system
- **Production-Ready**:
  - Full RAG framework to get you started on your own project
  - Scalable (optional) cloud based compute or local storage
  - Included API and frontend for easy deployment

**Quick Links:**
- [⭐ GitHub](https://github.com/KRLabsOrg/verbatim-rag)
- [PyPI](https://pypi.org/project/verbatim-rag/)
- [Paper](https://aclanthology.org/2025.bionlp-share.8/) - ACL BioNLP 2025
- [Models](https://huggingface.co/KRLabsOrg) - Fine-tuned extractors

---

## Extract, Don't Generate

The fundamental issue with RAG systems is that LLMs generate text probabilistically. Even when provided with perfect context, the model samples from a probability distribution over tokens, which inevitably introduces approximations, paraphrasing, and factual drift.

This problem compounds in modern agentic RAG systems. When a single query triggers 7-8 sequential LLM calls (planning, retrieval, synthesis, verification, etc.), each with its own hallucination probability, the cumulative error rate becomes significant. If each call has a 10% chance of introducing an error, an 8-step agentic workflow has roughly a ~60% probability of containing at least one hallucination.

**Verbatim RAG takes a different approach:** Instead of asking the LLM to generate an answer based on retrieved documents, we constrain it to **extract exact text spans** that answer the question. These spans are then composed into a response without any generative rewriting.

<p align="center">
  <img src="https://github.com/KRLabsOrg/verbatim-rag/blob/main/assets/verbatim_architecture.png?raw=true" alt="Verbatim RAG Architecture" width="800"/>
</p>

**Why this works:** Extraction is a classification task (does this span answer the question?), not a generation task. The model never produces new tokens that approximate source content, it only identifies which existing tokens to include. This eliminates the issue of the probabilistic generation that causes hallucinations.

The result: every number, every fact, every claim in the response is directly traceable to the source text.

---

## Quickstart: Zero Hallucinations in 15 Lines

Install the required dependencies:
```bash
!pip install verbatim-rag
```

Then create a Verbatim RAG system:

```python
from verbatim_rag import VerbatimRAG, VerbatimIndex
from verbatim_rag.vector_stores import LocalMilvusStore
from verbatim_rag.embedding_providers import SpladeProvider
from verbatim_rag.schema import DocumentSchema

# 1. Create index (storage layer) - works on CPU!
store = LocalMilvusStore("./demo.db", enable_sparse=True, enable_dense=False)
embedder = SpladeProvider("opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill", device="cpu")
index = VerbatimIndex(vector_store=store, sparse_provider=embedder)

# 2. Add a document
doc = DocumentSchema(
    content="""
# Methods
We used two approaches: zero-shot LLM extraction and a fine-tuned ModernBERT classifier on 58k synthetic examples.

# Results
The system achieved 42.01% accuracy on ArchEHR-QA.
""",
    title="Research Paper"
)
index.add_documents([doc])

# 3. Create RAG with hallucination prevention
rag = VerbatimRAG(index, model="gpt-4o-mini")

# 4. Get exact quotes, not hallucinations
response = rag.query("What approaches were used?")
print(response.answer)
# [1] We used two approaches: zero-shot LLM extraction and a fine-tuned ModernBERT classifier on 58k synthetic examples.
```

**Note:** Use dense embeddings (SentenceTransformers, OpenAI) or hybrid search by swapping the embedding provider. See [docs](https://github.com/KRLabsOrg/verbatim-rag) for details.

---

## Already Have a RAG System? Wrap It in 15 Lines

A key integration pattern: wrap your existing LangChain/LlamaIndex retrieval with Verbatim's extraction layer. Your indexing pipeline stays unchanged, you're only adding the verbatim extraction step on top of your existing retrieval.

Install the required dependencies:
```bash
!pip install langchain langchain-openai langchain-community openai faiss-cpu langchain-text-splitters
```

Then wrap your existing LangChain RAG system:
```python
from verbatim_rag.providers import RAGProvider
from verbatim_rag import verbatim_query

# Your existing LangChain setup
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from typing import List, Dict, Any, Optional

# 1. Create sample documents
docs = [
    Document(
        page_content="""
# Methods
We used two approaches: zero-shot LLM extraction and a fine-tuned ModernBERT classifier on 58k synthetic examples.

# Results
The system achieved 42.01% accuracy on ArchEHR-QA.
        """,
        metadata={"title": "Research Paper 1", "source": "paper1.pdf", "year": 2025},
    )
]

# 2. Index with LangChain (your existing setup)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# 3. Wrap your LangChain retriever
class LangChainRAGProvider(RAGProvider):
    def __init__(self, langchain_retriever):
        self.retriever = langchain_retriever

    def retrieve(
        self, question: str, k: int = 5, filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        # Use LangChain's retrieval
        docs = self.retriever.invoke(question)

        # Convert to Verbatim format
        context = []
        for doc in docs[:k]:
            context.append(
                {
                    "content": doc.page_content,
                    "title": doc.metadata.get("title", ""),
                    "source": doc.metadata.get("source", ""),
                    "metadata": doc.metadata,
                }
            )
        return context


# 4. Use Verbatim with your existing LangChain RAG
provider = LangChainRAGProvider(retriever)
response = verbatim_query(provider, "What methods were used?", k=5)

print(response.answer)
# Output: [1] We used two approaches: zero-shot LLM extraction and a fine-tuned ModernBERT classifier on 58k synthetic examples.
```

This approach preserves your existing retrieval system (embeddings, vector store, chunking strategy) while adding verbatim extraction. The only change is at the final answer generation step.

---

## Why This Matters

Even with perfect retrieval, the generation step introduces hallucinations. Consider this example:

| | Traditional RAG | Verbatim RAG |
|---|---|---|
| **Question** | "How much synthetic data?" | "How much synthetic data?" |
| **Retrieved** | "We created 60k examples" | "We created 60k examples" |
| **Generated** | "Around 58,000 examples" | "[1] We created 60k examples" |

The LLM approximated "60k" to "58,000" during token generation—not because retrieval failed, but because the model is sampling from a learned distribution over approximate phrasings.

This happens because:
1. **Token-level generation** doesn't preserve exact numerals from context
2. **Training data** contains many approximate phrasings (\"around X\", \"approximately Y\")
3. **Autoregressive sampling** can drift from source text even with high context attention

Traditional RAG systems try to mitigate this through better prompting, retrieval, or reranking. But these don't address the root cause: the model is generating new tokens rather than selecting existing ones.

```python
# Traditional RAG: Model generates tokens (probabilistic)
answer = "They generated approximately 58,000 synthetic training samples"

# Verbatim RAG: Model selects spans (deterministic given the extraction)
answer = "[1] We created 60k examples."
```

---
## Architecture: Two Independent Layers

Verbatim RAG decouples **retrieval** from **extraction** through a clean interface boundary:

### Layer 1: VerbatimIndex (Storage)
Handles document ingestion, chunking, embedding, and retrieval:
- **Input**: Raw documents (PDFs, text, markdown)
- **Output**: Top-k relevant chunks for a query
- **Components**: Pluggable chunkers (markdown-aware, Chonkie), embedders (sparse/dense), vector stores (local/cloud Milvus)

### Layer 2: Verbatim Core (Extraction)
Consumes retrieved chunks and produces verbatim answers:
- **Input**: Query + retrieved chunks
- **Output**: Answer composed entirely from extracted spans
- **Components**: Span extractors (LLM or fine-tuned models), template managers, verification

### The Interface

```python
# Layer 1: Retrieval
chunks = index.query(question, k=5)  # Returns List[SearchResult]

# Layer 2: Extraction
spans = extractor.extract_spans(question, chunks)  # Returns Dict[str, List[str]]
answer = template_manager.compose(spans)  # Returns formatted response
```

This separation means you can:
- Swap Layer 1 for your existing retrieval (LangChain, custom)
- Use just Layer 2 to harden an existing RAG system
- Replace components independently (e.g., switch from LLM to ModernBERT extractor without touching retrieval)

---

## Index Real Research Papers

```python
from verbatim_rag.schema import DocumentSchema

# Add multiple papers with metadata
papers = [
    DocumentSchema.from_url(
        url="https://aclanthology.org/L16-1417.pdf",
        title="Building Concept Graphs from Monolingual Dictionary Entries",
        doc_type="academic_paper",
        authors=["Gabor Recski"],
        conference="LREC",
        year=2016,
    ),
    DocumentSchema.from_url(
        url="https://aclanthology.org/2025.bionlp-share.8.pdf",
        title="KR Labs at ArchEHR-QA 2025: A Verbatim Approach for Evidence-Based Question Answering",
        doc_type="academic_paper",
        authors=["Adam Kovacs", "Paul Schmitt", "Gabor Recski"],
        conference="BioNLP",
        year=2025,
    ),
]

index.add_documents(papers)
```

**Search with metadata filtering:**

```python
# Find papers from BioNLP
results = index.query(
    filter='metadata["conference"] == "BioNLP"',
    k=10
)

for result in results:
    print(f"{result.metadata['title']} ({result.metadata['year']})")
    print(f"Score: {result.score:.3f}")
```

**For production, use cloud storage:**

```python
from verbatim_rag.vector_stores import CloudMilvusStore

store = CloudMilvusStore(
    uri="https://your-milvus-instance.com",
    token="your-token",
    collection_name="production"
)
```

---

## CPU-Only Pipeline: No LLM API Calls

The entire pipeline can run on CPU without GPU or LLM API calls:

```python
from verbatim_rag.extractors import ModelSpanExtractor
from verbatim_rag.embedding_providers import SpladeProvider

# SPLADE for embeddings (sparse representations are CPU-efficient)
embedder = SpladeProvider("opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill", device="cpu")

# ModernBERT for extraction (fine-tuned on span classification)
extractor = ModelSpanExtractor(
    model_path="KRLabsOrg/verbatim-rag-modern-bert-v1",
    device="cpu"
)

# Complete CPU-only RAG
rag = VerbatimRAG(index, extractor=extractor)
```

**Tradeoffs**: The fine-tuned extractor has lower recall than LLM-based extractors on complex queries, but is deterministic, free, and works offline.

---

## Template Management

Template management controls how extracted spans are composed into final responses. Three modes are available:

### 1. **Static Mode** - Fixed Format

Use when response formatting must be identical across all queries:

```python
template = """
Based on our research documents:
[RELEVANT_SENTENCES]

Need more information? Contact support@yourcompany.com
"""

rag.template_manager.use_static_mode(template)

response = rag.query("What methods were used?")
# Output:
# Based on our research documents:
# [1] We used zero-shot LLM extraction and fine-tuned ModernBERT.
#
# Need more information? Contact support@yourcompany.com
```

The `[RELEVANT_SENTENCES]` placeholder is automatically replaced with numbered citations.

**Use cases:** Compliance-required formatting, API responses with fixed schemas, systems where output format must be predictable.

---

### 2. **Dynamic Mode** - Adaptive Formatting (Default)

The LLM generates a context-appropriate template for each query:

```python
rag.template_manager.use_contextual_mode()  # Default

response = rag.query("Who are the authors?")
# Gets: "The authors are: [1] ..."

response = rag.query("What were the results?")
# Gets: "The system achieved: [1] ..."
```

**Use cases:** Conversational interfaces, chatbots, systems where natural language flow matters more than format consistency.

### 3. Question-Specific Mode - Template Matching

Sometimes you want different response formats for different question types, without the randomness. Define templates paired with example questions—the system automatically selects the best-fitting template using semantic similarity:

```python
templates = [
    {
        "template": "**Methodology:**\n\n[RELEVANT_SENTENCES]\n\n*Methods extracted from source*",
        "examples": [
            "What methods were used?",
            "How was this done?",
            "Describe the methodology"
        ]
    },
    {
        "template": "**Key Findings:**\n\n[RELEVANT_SENTENCES]\n\n*Results from source*",
        "examples": [
            "What were the results?",
            "What did they find?",
            "What are the key findings?"
        ]
    }
]

rag.template_manager.use_question_specific_mode(templates)

# Questions automatically matched to appropriate templates
response = rag.query("How did they approach this?")
# Uses Methodology template

response = rag.query("What were the main findings?")
# Uses Key Findings template
```

The system embeds all example questions once and matches incoming queries using cosine similarity—no LLM call needed.

---

## Developer Tools: Inspection & Debugging

Verbatim RAG includes built-in tools for understanding and debugging your index:

### Inspect Your Index

```python
# Get overview statistics
stats = index.inspect()

print(f"Total documents: {stats['total_documents']}")
print(f"Total chunks: {stats['total_chunks']}")
print(f"Document types: {stats['doc_types']}")
```

### Examine Chunks Before Deployment


```python
# Get first 5 chunks to see what was indexed
chunks = index.get_all_chunks(limit=5)

for chunk in chunks:
    print(f"Raw text: {chunk.text[:100]}...")
    print(f"Enhanced (embedded): {chunk.enhanced_text[:100]}...")
    print(f"Metadata: {chunk.metadata}")
```


### Powerful Metadata Filtering

Add ANY custom fields to documents and filter at query time:

```python
# Add documents with custom metadata
doc = DocumentSchema.from_url(
    url="https://aclanthology.org/2025.bionlp-share.8.pdf",
    year=2025,
    conference="BioNLP",
    category="medical-nlp",
    user_id="user123",  # Any custom field!
    project_id="proj_A"  # Works with any field
)

# Combine semantic search + complex filters
results = index.query(
    text="clinical question answering",
    filter='metadata["year"] >= 2024 && metadata["category"] == "medical-nlp"',
    k=10
)
```

---

## Additional Features

### Bring Your Own LLM

Works with any OpenAI-compatible endpoint:

```python
from verbatim_rag.core import LLMClient

# Use Claude, local Ollama, Azure OpenAI, etc.
llm_client = LLMClient(
    model="moonshotai/kimi-k2-instruct-0905",
    api_base="https://api.groq.com/openai/v1/"
)
rag = VerbatimRAG(index, llm_client=llm_client)
```

### PDF Processing with Docling

Automatically convert PDFs to structured markdown:

```python
doc = DocumentSchema.from_url(
    url="https://aclanthology.org/2025.bionlp-share.8.pdf",
    title="Research Paper"
)
# Docling handles: PDF → Markdown with preserved structure
```

---

## Technical Tradeoffs

**When to use Verbatim RAG:**

Verbatim RAG trades generation flexibility for factual precision. Use it when:
- **Exactness matters**: Medical dosages, legal citations, financial figures must be character-perfect
- **Auditability required**: Every claim needs to trace back to source text
- **Liability concerns**: Approximate or paraphrased answers create legal/compliance risk
- **Existing RAG needs hardening**: You have a working system but need to eliminate hallucinations

**When traditional RAG may be preferable:**

- **Synthesis across sources**: Answering requires combining information from multiple passages in ways that don't exist verbatim in any single source
- **Natural language fluency**: Responses need to be conversational and flowing, not citation-heavy
- **Abstract/analytical queries**: Questions like "What are the main themes?" require interpretation, not extraction

**Hybrid approach**: Use Verbatim RAG for factual queries and traditional RAG for analytical/synthesis queries, routing based on query type.

---

### Resources
- [Tutorial Notebook](https://github.com/KRLabsOrg/verbatim-rag/blob/main/docs/build_verbatim.ipynb) - Complete guide with examples
- [Documentation](https://github.com/KRLabsOrg/verbatim-rag) - API reference and advanced usage
- [HuggingFace Models](https://huggingface.co/KRLabsOrg) - Fine-tuned extractors
- [GitHub Discussions](https://github.com/KRLabsOrg/verbatim-rag/discussions) - Questions and use cases
- [Report Issues](https://github.com/KRLabsOrg/verbatim-rag/issues)
- [KR Labs](https://krlabs.eu) - Enterprise support

---

## Research & Citation

Verbatim RAG is based on research published at ACL BioNLP 2025.

**Read the paper:** [ACL Anthology 2025](https://aclanthology.org/2025.bionlp-share.8/)

**If you use Verbatim RAG in your research:**
```bibtex
@inproceedings{kovacs-etal-2025-kr,
    title = "{KR} Labs at {A}rch{EHR}-{QA} 2025: A Verbatim Approach for Evidence-Based Question Answering",
    author = "Kovacs, Adam and Schmitt, Paul and Recski, Gabor",
    booktitle = "Proceedings of the 24th Workshop on Biomedical Language Processing",
    year = "2025",
    url = "https://aclanthology.org/2025.bionlp-share.8/",
    pages = "69--74"
}
```
