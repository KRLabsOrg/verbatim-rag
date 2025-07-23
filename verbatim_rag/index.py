"""
Unified index class for the Verbatim RAG system.
"""

from typing import List, Optional, Dict, Any

from verbatim_rag.document import Document
from verbatim_rag.embedding_providers import (
    DenseEmbeddingProvider,
    SparseEmbeddingProvider,
    SentenceTransformersProvider,
    SpladeProvider,
    OpenAIProvider,
)
from verbatim_rag.vector_stores import (
    VectorStore,
    LocalMilvusStore,
    CloudMilvusStore,
    SearchResult,
)
from verbatim_rag.config import (
    VerbatimRAGConfig,
    DenseEmbeddingModel,
    SparseEmbeddingModel,
    VectorDBType,
)


class VerbatimIndex:
    """
    A unified index for document retrieval supporting multiple embedding providers and vector stores.
    """

    def __init__(self, config: VerbatimRAGConfig):
        """
        Initialize the VerbatimIndex.

        :param config: Configuration object (required)
        """
        self.config = config
        self.dense_provider = self._create_dense_provider(config)
        self.sparse_provider = self._create_sparse_provider(config)
        self.vector_store = self._create_vector_store(config)

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the index.

        :param documents: List of Document objects to add
        """
        if not documents:
            return

        # Extract all processed chunks from documents
        all_chunks = []
        for doc in documents:
            for chunk in doc.chunks:
                for processed_chunk in chunk.processed_chunks:
                    all_chunks.append(
                        {
                            "document": doc,
                            "chunk": chunk,
                            "processed_chunk": processed_chunk,
                        }
                    )

        if not all_chunks:
            return

        # Prepare data for vector store
        ids = []
        texts = []
        metadatas = []
        dense_embeddings = []
        sparse_embeddings = []

        for item in all_chunks:
            doc = item["document"]
            chunk = item["chunk"]
            processed_chunk = item["processed_chunk"]

            # Extract text
            text = processed_chunk.enhanced_content
            texts.append(text)
            ids.append(processed_chunk.id)

            # Generate dense embedding
            dense_emb = self.dense_provider.embed_text(text)
            dense_embeddings.append(dense_emb)

            # Generate sparse embedding if provider available
            if self.sparse_provider:
                sparse_emb = self.sparse_provider.embed_text(text)
            else:
                sparse_emb = {}
            sparse_embeddings.append(sparse_emb)

            # Prepare metadata (minimal, essential fields only)
            metadata = {
                "document_id": doc.id,
                "title": doc.title,  # Keep for search display
                "source": doc.source,  # Keep for search display
                "chunk_type": chunk.chunk_type.value,
                "chunk_number": chunk.chunk_number,
                "headers": processed_chunk.headers,
                "page_number": chunk.metadata.get("page_number", 0),
            }
            metadatas.append(metadata)

        # Store in vector store
        self.vector_store.add_vectors(
            ids=ids,
            dense_vectors=dense_embeddings,
            sparse_vectors=sparse_embeddings,
            texts=texts,
            metadatas=metadatas,
        )

        # Store document metadata
        document_data = []
        for doc in documents:
            doc_dict = {
                "id": doc.id,
                "title": doc.title,
                "source": doc.source,
                "content_type": doc.content_type.value,
                "raw_content": doc.raw_content,
                "metadata": doc.metadata,
            }
            document_data.append(doc_dict)

        # Store documents if vector store supports it
        if hasattr(self.vector_store, "add_documents"):
            self.vector_store.add_documents(document_data)

    def search(
        self, query: str, k: int = 5, search_type: str = "auto"
    ) -> List[SearchResult]:
        """
        Search for documents similar to the query.

        :param query: The search query
        :param k: Number of documents to retrieve
        :param search_type: Type of search ("dense", "sparse", "hybrid", "auto")
        :return: List of SearchResult objects
        """
        # Auto-detect search type based on available providers
        if search_type == "auto":
            if self.sparse_provider:
                search_type = "hybrid"
            else:
                search_type = "dense"

        # Generate query embeddings
        query_dense = None
        query_sparse = None

        if search_type in ["dense", "hybrid"]:
            query_dense = self.dense_provider.embed_text(query)

        if search_type in ["sparse", "hybrid"] and self.sparse_provider:
            query_sparse = self.sparse_provider.embed_text(query)

        # Search using vector store
        return self.vector_store.search(
            dense_query=query_dense,
            sparse_query=query_sparse,
            text_query=query,
            top_k=k,
            search_type=search_type,
        )

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID.

        :param document_id: The document ID
        :return: Document metadata dict or None if not found
        """
        if hasattr(self.vector_store, "get_document"):
            return self.vector_store.get_document(document_id)
        return None

    def _create_dense_provider(
        self, config: VerbatimRAGConfig
    ) -> DenseEmbeddingProvider:
        """Create dense embedding provider from config."""
        if config.dense_embedding.model == DenseEmbeddingModel.SENTENCE_TRANSFORMERS:
            return SentenceTransformersProvider(
                model_name=config.dense_embedding.model_name or "all-MiniLM-L6-v2",
                device=config.dense_embedding.device,
            )
        elif config.dense_embedding.model == DenseEmbeddingModel.OPENAI:
            return OpenAIProvider(
                model_name=config.dense_embedding.model_name or "text-embedding-ada-002"
            )
        else:
            raise ValueError(
                f"Unsupported dense embedding model: {config.dense_embedding.model}"
            )

    def _create_sparse_provider(
        self, config: VerbatimRAGConfig
    ) -> Optional[SparseEmbeddingProvider]:
        """Create sparse embedding provider from config."""
        if not config.sparse_embedding.enabled:
            return None

        if config.sparse_embedding.model == SparseEmbeddingModel.SPLADE:
            return SpladeProvider(
                model_name=config.sparse_embedding.model_name or "naver/splade-v3",
                device=config.sparse_embedding.device,
            )
        else:
            raise ValueError(
                f"Unsupported sparse embedding model: {config.sparse_embedding.model}"
            )

    def _create_vector_store(self, config: VerbatimRAGConfig) -> VectorStore:
        """Create vector store from config."""
        if config.vector_db.type == VectorDBType.MILVUS_LOCAL:
            return LocalMilvusStore(
                db_path=config.vector_db.db_path,
                collection_name=config.vector_db.collection_name,
                dense_dim=self.dense_provider.get_dimension(),
            )
        elif config.vector_db.type == VectorDBType.MILVUS_CLOUD:
            return CloudMilvusStore(
                collection_name=config.vector_db.collection_name,
                dense_dim=self.dense_provider.get_dimension(),
                host=config.vector_db.host,
                port=str(config.vector_db.port or 19530),
                username=config.vector_db.api_key.split(":")[0]
                if config.vector_db.api_key
                else "",
                password=config.vector_db.api_key.split(":")[1]
                if config.vector_db.api_key
                else "",
            )
        else:
            raise ValueError(f"Unsupported vector store type: {config.vector_db.type}")

    @classmethod
    def from_config(cls, config: VerbatimRAGConfig) -> "VerbatimIndex":
        """Create VerbatimIndex from configuration."""
        return cls(config=config)
