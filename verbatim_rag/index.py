"""
Unified index class for the Verbatim RAG system.
"""

from typing import List, Optional, Dict, Any, Union

from verbatim_rag.document import Document
from verbatim_rag.schema import DocumentSchema
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
from verbatim_rag.document import DocumentType
from verbatim_rag.chunking import ChunkingService


class VerbatimIndex:
    """
    A unified index for document retrieval supporting multiple embedding providers and vector stores.
    """

    def __init__(
        self,
        db_path: str = "./milvus_verbatim.db",
        collection_name: str = "verbatim_rag",
        dense_model: Optional[str] = None,
        sparse_model: Optional[str] = None,
        config: Optional[VerbatimRAGConfig] = None,
    ):
        """
        Initialize the VerbatimIndex with simple parameters or config.

        Args:
            db_path: Path to Milvus database file
            collection_name: Name of the collection
            dense_model: Dense embedding model name (None to disable dense)
            sparse_model: Sparse embedding model name (None to disable sparse)
            config: Optional configuration object (overrides other params if provided)
        """
        if config is not None:
            self.config = config
        else:
            # Create simple config from parameters
            from verbatim_rag.config import create_default_config

            self.config = create_default_config()
            self.config.vector_db.db_path = db_path
            self.config.vector_db.collection_name = collection_name

            if dense_model:
                self.config.dense_embedding.model_name = dense_model
            # Track if dense is enabled for provider creation
            self._dense_enabled = dense_model is not None

            if sparse_model:
                self.config.sparse_embedding.enabled = True
                self.config.sparse_embedding.model_name = sparse_model
            else:
                self.config.sparse_embedding.enabled = False

            # Ensure at least one embedding type is enabled
            if not dense_model and not sparse_model:
                raise ValueError(
                    "At least one of dense_model or sparse_model must be provided"
                )

        # No longer needed - all metadata goes to single JSON field

        # Initialize chunking service
        self.chunking_service = ChunkingService()

        self.dense_provider = self._create_dense_provider(self.config)
        self.sparse_provider = self._create_sparse_provider(self.config)
        self.vector_store = self._create_vector_store(self.config)

    def add_documents(
        self,
        documents: List[Union[DocumentSchema, Document]],
        document_type: DocumentType = DocumentType.MARKDOWN,
    ) -> None:
        """
        Add documents to the index.

        :param documents: List of DocumentSchema or Document objects to add
        """
        if not documents:
            return

        # Handle DocumentSchema (new primary API) and legacy Document objects
        for doc in documents:
            if isinstance(doc, DocumentSchema):
                self._add_schema_document(doc)
            else:
                self._add_document_internal(doc, document_type)

    def _add_schema_document(self, doc: DocumentSchema) -> None:
        """Add a DocumentSchema to the index using chonkie for text chunking.

        Docling is only required for file/URL parsing via the ingestion pipeline.
        For raw text content provided in the schema, only chonkie is required here.
        """
        from verbatim_rag.document import (
            Document,
            Chunk,
            ChunkType,
            ProcessedChunk,
        )

        # Convert DocumentSchema to Document for processing
        # Properly flatten metadata to make custom fields available for filtering
        from datetime import datetime

        base_metadata = doc.model_dump(
            exclude={"id", "title", "source", "content", "metadata"}
        )
        custom_metadata = doc.metadata or {}
        flattened_metadata = {**base_metadata, **custom_metadata}

        # Handle datetime objects to prevent JSON serialization issues
        for key, value in flattened_metadata.items():
            if isinstance(value, datetime):
                flattened_metadata[key] = value.isoformat()

        document = Document(
            id=doc.id,
            title=doc.title or "",
            source=doc.source or "",
            content_type=doc.content_type,  # Use content_type from DocumentSchema
            raw_content=doc.content,
            metadata=flattened_metadata,  # Flattened metadata with custom fields at top level
        )

        # Use chunking service to handle text chunking with enhancement
        enhanced_chunks = self.chunking_service.chunk_document_enhanced(document)

        # Create Document chunks with proper structure
        for i, (chunk_text, enhanced_text) in enumerate(enhanced_chunks):
            # Create basic Chunk with inherited metadata
            doc_chunk = Chunk(
                document_id=document.id,
                content=chunk_text,  # Original text for extraction
                chunk_number=i,
                chunk_type=ChunkType.PARAGRAPH,
                metadata={},  # Keep chunk-level metadata minimal; doc.metadata added later
            )

            # Create ProcessedChunk
            processed_chunk = ProcessedChunk(
                chunk_id=doc_chunk.id,
                enhanced_content=enhanced_text,  # Enhanced text with headings/metadata
            )

            # Add to document structure
            doc_chunk.add_processed_chunk(processed_chunk)
            document.add_chunk(doc_chunk)

        # Use existing document addition logic
        self._add_document_internal(document)

    def _add_document_internal(
        self,
        doc: Document,
        document_type: DocumentType = DocumentType.MARKDOWN,
    ) -> None:
        """Add a Document object with chunks to the index."""
        # Extract all processed chunks from documents
        all_chunks = []
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
        enhanced_texts = []
        metadatas = []
        dense_embeddings = []
        sparse_embeddings = []

        for item in all_chunks:
            doc = item["document"]
            chunk = item["chunk"]
            processed_chunk = item["processed_chunk"]

            # Extract both texts
            original_text = chunk.content
            enhanced_text = processed_chunk.enhanced_content
            texts.append(original_text)
            enhanced_texts.append(enhanced_text)
            ids.append(processed_chunk.id)

            # Generate dense embedding if provider available (use enhanced text)
            if self.dense_provider:
                dense_emb = self.dense_provider.embed_text(enhanced_text)
            else:
                dense_emb = []
            dense_embeddings.append(dense_emb)

            # Generate sparse embedding if provider available (use enhanced text)
            if self.sparse_provider:
                sparse_emb = self.sparse_provider.embed_text(enhanced_text)
            else:
                sparse_emb = {}
            sparse_embeddings.append(sparse_emb)

            # Prepare metadata - everything goes into single JSON field
            metadata = {
                "document_id": doc.id,
                "title": doc.title,
                "source": doc.source,
                "doc_type": doc.metadata.get("doc_type"),
                "content_type": doc.content_type.value if doc.content_type else None,
                "chunk_type": chunk.chunk_type.value,
                "chunk_number": chunk.chunk_number,
                "page_number": chunk.metadata.get("page_number", 0),
                **(doc.metadata or {}),  # All document metadata
                **chunk.metadata,  # Chunk-specific metadata
            }
            metadatas.append(metadata)

        # Store in vector store - pass None for disabled embedding types
        dense_vectors_to_store = dense_embeddings if self.dense_provider else None
        sparse_vectors_to_store = sparse_embeddings if self.sparse_provider else None

        self.vector_store.add_vectors(
            ids=ids,
            dense_vectors=dense_vectors_to_store,
            sparse_vectors=sparse_vectors_to_store,
            texts=texts,
            enhanced_texts=enhanced_texts,
            metadatas=metadatas,
        )

        # Store document metadata
        document_data = []
        for doc in [item["document"] for item in all_chunks]:
            # Avoid duplicates
            if doc.id not in [d.get("id") for d in document_data]:
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

    def add_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        doc_id: str,
        document_type: DocumentType = DocumentType.MARKDOWN,
    ) -> str:
        """
        Add a single document using the new schema system.

        :param content: Document text content to be chunked and indexed
        :param metadata: Document metadata dict from DocumentSchema.to_storage_dict()
        :param doc_id: Document ID
        :return: Document ID
        """
        # Use chunking service to chunk the content
        chunks = self.chunking_service.chunk_with_metadata(
            content, metadata, document_type
        )

        # Create embeddings and store
        ids = []
        texts = []
        enhanced_texts = []
        metadatas = []
        dense_embeddings = []
        sparse_embeddings = []

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            ids.append(chunk_id)
            texts.append(chunk_text)  # Original text
            enhanced_texts.append(
                chunk_text
            )  # For now, same as original (no enhancement in simple path)

            # Generate embeddings
            if self.dense_provider:
                dense_emb = self.dense_provider.embed_text(chunk_text)
            else:
                dense_emb = []
            dense_embeddings.append(dense_emb)

            if self.sparse_provider:
                sparse_emb = self.sparse_provider.embed_text(chunk_text)
            else:
                sparse_emb = {}
            sparse_embeddings.append(sparse_emb)

            # Create chunk metadata - everything in single metadata field
            chunk_metadata = {
                "document_id": doc_id,
                "title": metadata.get("title", ""),
                "source": metadata.get("source", ""),
                "doc_type": metadata.get("doc_type", ""),
                "content_type": metadata.get("content_type"),
                "chunk_type": "paragraph",
                "chunk_number": i,
                "page_number": 0,
                **{
                    k: v
                    for k, v in metadata.items()
                    if k not in ["title", "source", "doc_type", "content_type"]
                },
            }
            metadatas.append(chunk_metadata)

        # Store in vector store
        dense_vectors_to_store = dense_embeddings if self.dense_provider else None
        sparse_vectors_to_store = sparse_embeddings if self.sparse_provider else None

        self.vector_store.add_vectors(
            ids=ids,
            dense_vectors=dense_vectors_to_store,
            sparse_vectors=sparse_vectors_to_store,
            texts=texts,
            enhanced_texts=enhanced_texts,
            metadatas=metadatas,
        )

        # Store document metadata once in documents collection
        if hasattr(self.vector_store, "add_documents"):
            doc_record = {
                "id": doc_id,
                "title": metadata.get("title") or "",
                "source": metadata.get("source") or "",
                "content_type": metadata.get("content_type") or "",
                "raw_content": "",  # do not store full content for schema-based docs
                "metadata": metadata,
            }
            self.vector_store.add_documents([doc_record])

        return doc_id

    def query(
        self,
        text: Optional[str] = None,
        k: int = 5,
        search_type: str = "auto",
        filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Query for documents using vector search or filtering.

        :param text: Optional text query for vector search
        :param k: Number of documents to retrieve
        :param search_type: Type of search ("dense", "sparse", "hybrid", "auto")
        :param filter: Optional Milvus filter expression for metadata filtering
        :return: List of SearchResult objects
        """
        # If no text provided, do filter-only query
        if not text:
            return self.vector_store.query(
                dense_query=None,
                sparse_query=None,
                text_query=None,
                top_k=k,
                filter=filter,
            )

        # Auto-detect search type based on available providers
        if search_type == "auto":
            if self.dense_provider and self.sparse_provider:
                search_type = "hybrid"
            elif self.sparse_provider:
                search_type = "sparse"
            elif self.dense_provider:
                search_type = "dense"
            else:
                raise ValueError("No embedding providers available")

        # Generate query embeddings
        query_dense = None
        query_sparse = None

        if search_type in ["dense", "hybrid"] and self.dense_provider:
            query_dense = self.dense_provider.embed_text(text)

        if search_type in ["sparse", "hybrid"] and self.sparse_provider:
            query_sparse = self.sparse_provider.embed_text(text)

        # Search using vector store
        return self.vector_store.query(
            dense_query=query_dense,
            sparse_query=query_sparse,
            text_query=text,
            top_k=k,
            search_type=search_type,
            filter=filter,
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

    def get_all_chunks(self, limit: int = 100) -> List[SearchResult]:
        """
        Get all chunks in the index (up to limit).

        :param limit: Maximum number of chunks to return
        :return: List of SearchResult objects
        """
        return self.query(k=limit)

    def get_all_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all unique documents from chunks.

        :param limit: Maximum number of document chunks to scan
        :return: List of unique documents with basic info
        """
        chunks = self.query(k=limit * 10)  # Get more chunks to find unique docs

        # Group by document_id to get unique documents
        docs_seen = set()
        unique_docs = []

        for chunk in chunks:
            metadata = chunk.metadata
            doc_id = metadata.get("document_id")
            if doc_id and doc_id not in docs_seen:
                docs_seen.add(doc_id)
                unique_docs.append(
                    {
                        "id": doc_id,
                        "title": metadata.get("title", ""),
                        "source": metadata.get("source", ""),
                        "doc_type": metadata.get("doc_type", ""),
                        "content_type": metadata.get("content_type", ""),
                    }
                )

                if len(unique_docs) >= limit:
                    break

        return unique_docs

    def get_chunks_by_document(
        self, document_id: str, limit: int = 100
    ) -> List[SearchResult]:
        """
        Get all chunks for a specific document.

        :param document_id: The document ID
        :param limit: Maximum number of chunks to return
        :return: List of SearchResult objects
        """
        filter_expr = f'metadata["document_id"] == "{document_id}"'
        return self.query(filter=filter_expr, k=limit)

    def inspect(self) -> Dict[str, Any]:
        """
        Get overview statistics about the index.

        :return: Dictionary with index statistics and sample data
        """
        # Get sample of chunks
        sample_chunks = self.get_all_chunks(limit=100)
        total_chunks = len(sample_chunks)

        # Get unique documents
        unique_docs = self.get_all_documents(limit=50)
        total_docs = len(unique_docs)

        # Analyze document types
        doc_types = {}
        content_types = {}

        for doc in unique_docs:
            doc_type = doc.get("doc_type", "unknown")
            content_type = doc.get("content_type", "unknown")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            content_types[content_type] = content_types.get(content_type, 0) + 1

        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "doc_types": doc_types,
            "content_types": content_types,
            "sample_documents": unique_docs[:5],
            "sample_chunks": [
                {
                    "id": chunk.id,
                    "text_preview": chunk.text[:100] + "..."
                    if len(chunk.text) > 100
                    else chunk.text,
                    "document_id": chunk.metadata.get("document_id"),
                    "chunk_number": chunk.metadata.get("chunk_number"),
                }
                for chunk in sample_chunks[:5]
            ],
        }

    def _create_dense_provider(
        self, config: VerbatimRAGConfig
    ) -> Optional[DenseEmbeddingProvider]:
        """Create dense embedding provider from config."""
        # Check if dense is disabled
        if hasattr(self, "_dense_enabled") and not self._dense_enabled:
            return None

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
            # Get dense dimension if dense provider exists, otherwise use default
            dense_dim = (
                self.dense_provider.get_dimension()
                if self.dense_provider
                else config.vector_db.dense_dim
            )
            return LocalMilvusStore(
                db_path=config.vector_db.db_path,
                collection_name=config.vector_db.collection_name,
                dense_dim=dense_dim,
                enable_dense=self.dense_provider is not None,
                enable_sparse=self.sparse_provider is not None,
            )
        elif config.vector_db.type == VectorDBType.MILVUS_CLOUD:
            # Get dense dimension if dense provider exists, otherwise use default
            dense_dim = (
                self.dense_provider.get_dimension()
                if self.dense_provider
                else config.vector_db.dense_dim
            )
            return CloudMilvusStore(
                collection_name=config.vector_db.collection_name,
                dense_dim=dense_dim,
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
