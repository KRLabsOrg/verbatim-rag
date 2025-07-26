# Verbatim RAG

A minimalistic approach to Retrieval-Augmented Generation (RAG) that prevents hallucination by ensuring all generated content is explicitly derived from source documents.

## Concept

Traditional RAG systems retrieve relevant documents and then allow an LLM to freely generate responses based on that context. This can lead to hallucinations where the model invents facts not present in the source material.

Verbatim RAG solves this by extracting verbatim text spans from documents and composing responses entirely from these exact passages, with direct citations linking back to sources.

For extraction, we can use LLM-based span extractors or fine-tuned encoder-based models like ModernBERT. We've trained our own ModernBERT model for this purpose, which is available on [HuggingFace](https://huggingface.co/KRLabsOrg/verbatim-rag-modern-bert-v1) (we've trained it on the [RAGBench](https://huggingface.co/datasets/galileo-ai/ragbench) dataset).

With this approach, **the whole RAG pipeline can be run without any usage of LLMs**, and with using SPLADE embeddings, the pipeline can be run entirely on CPU, making it lightweight and efficient.

## Installation

```bash
# Install the package
pip install verbatim-rag
```

## Quick Start

```python
from verbatim_rag import VerbatimIndex, VerbatimRAG
from verbatim_rag.ingestion import DocumentProcessor

# Process documents with intelligent chunking
processor = DocumentProcessor()

# Process PDFs from URLs
document = processor.process_url(
    url="https://aclanthology.org/2025.bionlp-share.8.pdf",
    title="KR Labs at ArchEHR-QA 2025: A Verbatim Approach for Evidence-Based Question Answering",
    metadata={"authors": ["Adam Kovacs", "Paul Schmitt", "Gabor Recski"]}
)

# Define SPLADE index with a sparse model
index = VerbatimIndex(
    sparse_model="naver/splade-v3", 
    db_path="./index.db"
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

This ensures all responses are grounded in the source material, preventing hallucinations.

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
6. All responses include exact text spans and document references

## Web Interface

The package includes a full web interface with React frontend and FastAPI backend:

```bash
# Start API server
python api/app.py

# Start React frontend (in another terminal)
cd frontend/
npm install
npm start
```

## ModernBERT Based Span Extractor

We've trained our own encoder model based on ModernBERT for sentence classification. This model is designed to classify text spans as relevant or not, providing a robust alternative to LLM-based extractors.

You can find our model on HuggingFace: [KRLabsOrg/verbatim-rag-modern-bert-v1](https://huggingface.co/KRLabsOrg/verbatim-rag-modern-bert-v1).

You can use it with the defined index as follows:

```python
from verbatim_rag.core import VerbatimRAG
from verbatim_rag.index import VerbatimIndex
from verbatim_rag.extractors import ModelSpanExtractor

# Load your trained extractor
extractor = ModelSpanExtractor("path/to/your/model")

# Create VerbatimRAG system with custom extractor
index = VerbatimIndex(
    sparse_model="naver/splade-v3", 
    db_path="./index.db"
)

rag_system = VerbatimRAG(
    index=index,
    extractor=extractor,
    k=5
)

# Query the system
response = rag_system.query("Main findings of the paper?")
print(response.answer)
```


## Citation

If you use Verbatim RAG in your research, please cite our paper:

```bibtex
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