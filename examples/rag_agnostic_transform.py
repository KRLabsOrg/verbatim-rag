"""
Examples of the Transformer over the context of any provider, and of VerbatimRAG.
"""

from verbatim_core import Chunk, Transformer
from verbatim_core.extractors import LLMSpanExtractor
from verbatim_core.llm_client import LLMClient
from verbatim_core.response_builder import ResponseBuilder
from verbatim_core.templates import TemplateManager

from verbatim_rag import (
    IndexProvider,
    VerbatimIndex,
    VerbatimRAG,
    VerbatimRAGProvider,
)
from verbatim_rag.embedding_providers import SpladeProvider
from verbatim_rag.schema import DocumentSchema
from verbatim_rag.vector_stores import LocalMilvusStore


def cited_response(transformer, question, context):
    """The QueryResponse of a provider's context items, which become chunks with positional ids."""
    chunks = [
        Chunk(
            id=f"ctx_{i}",
            text=item["content"],
            metadata={
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                **(item.get("metadata") or {}),
            },
        )
        for i, item in enumerate(context)
    ]
    return ResponseBuilder().build_from_answer(
        question, transformer.transform(question, chunks), chunks
    )


def main():
    # Build an index with sparse search (SPLADE)
    sparse_provider = SpladeProvider(model_name="naver/splade-v3", device="cpu")
    vector_store = LocalMilvusStore(
        db_path="./index.db",
        enable_dense=False,
        enable_sparse=True,
    )
    index = VerbatimIndex(vector_store=vector_store, sparse_provider=sparse_provider)

    # The Transformer writes the answers: LLM extraction and contextual templates
    llm_client = LLMClient()
    transformer = Transformer(
        LLMSpanExtractor(llm_client),
        TemplateManager(llm_client=llm_client, default_mode="contextual"),
    )

    # Add a raw-text document via DocumentSchema, through the RAG facade
    rag = VerbatimRAG(index, transformer=transformer, llm_client=llm_client)
    rag.add_document(
        DocumentSchema(
            content="Transformers improve attention. SPLADE enables hybrid sparse retrieval.",
            title="Note",
            doc_type="note",
            chunker_type="sentence",
            chunk_size=3,
        )
    )
    question = "What improves attention?"

    # 1) Any provider's context, then the Transformer
    resp = cited_response(transformer, question, IndexProvider(index).retrieve(question, k=5))
    print("Provider and Transformer:", resp.answer)

    # 2) VerbatimRAG retrieves and answers in one call
    print("VerbatimRAG:", rag.query(question, k=5).answer)

    # 3) VerbatimRAG as a provider
    resp3 = cited_response(transformer, question, VerbatimRAGProvider(rag).retrieve(question, k=5))
    print("RAG provider:", resp3.answer)


if __name__ == "__main__":
    main()
