"""
Examples for using the RAG-agnostic VerbatimTransform and providers.
"""

from verbatim_rag import VerbatimIndex, VerbatimRAG
from verbatim_rag import (
    VerbatimTransform,
    IndexProvider,
    VerbatimRAGProvider,
    verbatim_query,
)
from verbatim_rag.schema import DocumentSchema
from verbatim_rag.vector_stores import LocalMilvusStore
from verbatim_rag.embedding_providers import SpladeProvider


def main():
    # Build an index with sparse search (SPLADE)
    sparse_provider = SpladeProvider(model_name="naver/splade-v3", device="cpu")
    vector_store = LocalMilvusStore(
        db_path="./index.db",
        enable_dense=False,
        enable_sparse=True,
    )
    index = VerbatimIndex(vector_store=vector_store, sparse_provider=sparse_provider)

    # Add a raw-text document via DocumentSchema
    doc = DocumentSchema(
        content="Transformers improve attention. SPLADE enables hybrid sparse retrieval.",
        title="Note",
        doc_type="note",
        chunker_type="sentence",
        chunk_size=3,
    )

    # Use the RAG facade to ingest
    rag = VerbatimRAG(index)
    rag.add_document(doc)

    # 1) Provider + transform (two-step)
    provider = IndexProvider(index)
    context = provider.retrieve("What improves attention?", k=5)
    vt = VerbatimTransform()
    resp = vt.transform("What improves attention?", context)
    print("Two-step:", resp.answer)

    # 2) One-liner helper
    resp2 = verbatim_query(provider, "What improves attention?", k=5)
    print("Helper:", resp2.answer)

    # 3) VerbatimRAG as a provider
    ragp = VerbatimRAGProvider(rag)
    resp3 = verbatim_query(ragp, "What improves attention?", k=5)
    print("RAG provider:", resp3.answer)


if __name__ == "__main__":
    main()
