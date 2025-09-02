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
        filter_fields: Optional[List[str]] = None,
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

        # Which document-level metadata keys are duplicated onto chunks for filtering
        self.filter_fields = filter_fields or ["doc_type"]

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
                self._add_schema_document(doc, document_type)
            else:
                self._add_document_internal(doc, document_type)

    def _add_schema_document(
        self, doc: DocumentSchema, document_type: DocumentType = DocumentType.MARKDOWN
    ) -> None:
        """Add a DocumentSchema to the index using chonkie for text chunking.

        Docling is only required for file/URL parsing via the ingestion pipeline.
        For raw text content provided in the schema, only chonkie is required here.
        """
        from verbatim_rag.document import (
            Document,
            DocumentType,
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
            content_type=document_type,  # Default for direct creation
            raw_content=doc.content,
            metadata=flattened_metadata,  # Flattened metadata with custom fields at top level
        )

        # Chunk raw text with chonkie (no docling dependency in this path)
        try:
            import chonkie
        except ImportError as e:
            raise ImportError(
                "Text chunking requires chonkie. Install with: pip install chonkie"
            ) from e

        # Allow per-document overrides via metadata
        meta = document.metadata or {}
        chunker_type = str(meta.get("chunker_type", "recursive")).lower()
        chunker_recipe = str(
            meta.get(
                "chunker_recipe",
                "markdown" if document_type == DocumentType.MARKDOWN else "default",
            )
        )
        lang = str(meta.get("lang", "en"))
        chunk_size = int(meta.get("chunk_size", 512))
        chunk_overlap = int(meta.get("chunk_overlap", 50))
        merge_threshold = float(meta.get("merge_threshold", 0.7))
        split_threshold = float(meta.get("split_threshold", 0.3))

        if chunker_type == "recursive":
            chunker = chonkie.RecursiveChunker.from_recipe(chunker_recipe, lang=lang)
        elif chunker_type == "token":
            chunker = chonkie.TokenChunker(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        elif chunker_type == "sentence":
            chunker = chonkie.SentenceChunker(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        elif chunker_type == "word":
            chunker = chonkie.WordChunker(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        elif chunker_type == "sdpm":
            chunker = chonkie.SDPMChunker(
                chunk_size=chunk_size,
                merge_threshold=merge_threshold,
                split_threshold=split_threshold,
            )
        else:
            chunker = chonkie.RecursiveChunker.from_recipe(chunker_recipe, lang=lang)

        chunks = chunker(doc.content)

        # Create Document chunks with proper structure
        for i, chunk in enumerate(chunks):
            # Create basic Chunk with inherited metadata
            doc_chunk = Chunk(
                document_id=document.id,
                content=chunk.text,
                chunk_number=i,
                chunk_type=ChunkType.PARAGRAPH,
                metadata={},  # Keep chunk-level metadata minimal; doc.metadata added later
            )

            # Create ProcessedChunk
            processed_chunk = ProcessedChunk(
                chunk_id=doc_chunk.id,
                enhanced_content=chunk.text,
            )

            # Add to document structure
            doc_chunk.add_processed_chunk(processed_chunk)
            document.add_chunk(doc_chunk)

        # Use existing document addition logic
        self._add_document_internal(document)

    def _add_document_internal(
        self,
        doc: Document,
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

            # Generate dense embedding if provider available
            if self.dense_provider:
                dense_emb = self.dense_provider.embed_text(text)
            else:
                dense_emb = []
            dense_embeddings.append(dense_emb)

            # Generate sparse embedding if provider available
            if self.sparse_provider:
                sparse_emb = self.sparse_provider.embed_text(text)
            else:
                sparse_emb = {}
            sparse_embeddings.append(sparse_emb)

            # Prepare metadata
            # - Duplicate only allowlisted doc fields at top-level for direct filters (e.g., doc_type)
            allowed_doc_meta = {
                k: v for k, v in (doc.metadata or {}).items() if k in self.filter_fields
            }
            # - Include ALL document metadata so it ends up in custom_metadata JSON for flexible filters
            all_doc_meta = dict(doc.metadata or {})
            metadata = {
                "document_id": doc.id,
                "title": doc.title,  # Keep for search display
                "source": doc.source,  # Keep for search display
                "chunk_type": chunk.chunk_type.value,
                "chunk_number": chunk.chunk_number,
                "page_number": chunk.metadata.get("page_number", 0),
                **allowed_doc_meta,
                **all_doc_meta,  # Becomes custom_metadata in the store
                **chunk.metadata,  # Include chunk-specific metadata
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
        # Chunk text with chonkie (mandatory dependency)
        try:
            import chonkie
        except ImportError as e:
            raise ImportError(
                "Text chunking requires chonkie. Install with: pip install chonkie"
            ) from e

        CHUNKER_RECIPE = (
            "markdown" if document_type == DocumentType.MARKDOWN else "default"
        )
        chunker = chonkie.RecursiveChunker.from_recipe(CHUNKER_RECIPE, lang="en")
        chunks_objs = chunker(content)
        chunks = [c.text for c in chunks_objs if getattr(c, "text", "").strip()]
        if not chunks:
            chunks = [content]

        # Create embeddings and store
        ids = []
        texts = []
        metadatas = []
        dense_embeddings = []
        sparse_embeddings = []

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            ids.append(chunk_id)
            texts.append(chunk_text)

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

            # Create chunk metadata
            chunk_metadata = {
                "document_id": doc_id,
                "title": metadata.get("title", ""),
                "source": metadata.get("source", ""),
                "doc_type": metadata.get("doc_type", ""),  # Add this field
                "chunk_type": "paragraph",
                "chunk_number": i,
                "page_number": 0,
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
            metadatas=metadatas,
        )

        # Store document metadata once in documents collection
        if hasattr(self.vector_store, "add_documents"):
            doc_record = {
                "id": doc_id,
                "title": metadata.get("title") or "",
                "source": metadata.get("source") or "",
                "doc_type": metadata.get("doc_type") or "",
                "raw_content": "",  # do not store full content for schema-based docs
                "metadata": metadata,
            }
            self.vector_store.add_documents([doc_record])

        return doc_id

    def search(
        self,
        query: str,
        k: int = 5,
        search_type: str = "auto",
        filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for documents similar to the query.

        :param query: The search query
        :param k: Number of documents to retrieve
        :param search_type: Type of search ("dense", "sparse", "hybrid", "auto")
        :param filter: Optional Milvus filter expression for metadata filtering
        :return: List of SearchResult objects
        """
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
