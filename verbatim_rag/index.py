"""
Unified index class for the Verbatim RAG system.
"""

from typing import List, Optional, Dict, Any, Union, Tuple
from tqdm import tqdm
from verbatim_rag.document import Document
from verbatim_rag.schema import DocumentSchema
from verbatim_rag.embedding_providers import (
    DenseEmbeddingProvider,
    SparseEmbeddingProvider,
)
from verbatim_rag.vector_stores import (
    VectorStore,
    CloudMilvusStore,
    SearchResult,
)
from verbatim_rag.document import DocumentType
from verbatim_rag.chunker_providers import ChunkerProvider, MarkdownChunkerProvider


class VerbatimIndex:
    """
    A unified index for document retrieval supporting multiple embedding providers and vector stores.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        dense_provider: Optional[DenseEmbeddingProvider] = None,
        sparse_provider: Optional[SparseEmbeddingProvider] = None,
        chunker_provider: Optional[ChunkerProvider] = None,
    ):
        """
        Initialize VerbatimIndex with provider instances.

        Args:
            vector_store: Vector database instance (LocalMilvusStore, CloudMilvusStore, etc.)
            dense_provider: Optional dense embedding provider (SentenceTransformers, OpenAI, etc.)
            sparse_provider: Optional sparse embedding provider (SPLADE, etc.)
            chunker_provider: Optional chunker provider (MarkdownChunkerProvider, etc.)

        Example:
            store = LocalMilvusStore(db_path="./index.db")
            sparse = SpladeProvider("naver/splade-v3")
            chunker = MarkdownChunkerProvider()
            index = VerbatimIndex(
                vector_store=store,
                sparse_provider=sparse,
                chunker_provider=chunker
            )

        Raises:
            ValueError: If both embedding providers are None
        """
        if dense_provider is None and sparse_provider is None:
            raise ValueError(
                "At least one embedding provider (dense or sparse) must be provided"
            )

        self.vector_store = vector_store
        self.dense_provider = dense_provider
        self.sparse_provider = sparse_provider
        self.chunker_provider = chunker_provider or MarkdownChunkerProvider()

    # Helper methods for document processing

    def _flatten_schema_metadata(self, doc: DocumentSchema) -> Dict[str, Any]:
        """
        Extract and flatten all metadata from DocumentSchema.

        Converts DocumentSchema fields to a flat dict suitable for storage and filtering.
        Handles datetime serialization and merges custom metadata fields.

        Args:
            doc: DocumentSchema instance

        Returns:
            Flattened metadata dict with all fields at top level
        """
        from datetime import datetime

        # Extract known schema fields (exclude core document fields)
        base_metadata = doc.model_dump(
            exclude={"id", "title", "source", "content", "metadata"}
        )

        # Merge with custom metadata from metadata field
        custom_metadata = doc.metadata or {}
        flattened_metadata = {**base_metadata, **custom_metadata}

        # Handle datetime objects to prevent JSON serialization issues
        for key, value in flattened_metadata.items():
            if isinstance(value, datetime):
                flattened_metadata[key] = value.isoformat()

        return flattened_metadata

    def _convert_schema_to_document(self, doc: DocumentSchema) -> Document:
        """
        Convert DocumentSchema to Document with properly flattened metadata.

        This ensures all custom fields from DocumentSchema are preserved
        in the Document.metadata dict for filtering and retrieval.

        Args:
            doc: DocumentSchema instance

        Returns:
            Document instance with flattened metadata
        """
        from verbatim_rag.document import Document

        flattened_metadata = self._flatten_schema_metadata(doc)

        return Document(
            id=doc.id,
            title=doc.title or "",
            source=doc.source or "",
            content_type=doc.content_type,
            raw_content=doc.content,
            metadata=flattened_metadata,
        )

    def _chunk_document(self, doc: Document) -> List[Tuple["Chunk", "ProcessedChunk"]]:
        """
        Chunk a Document into Chunk and ProcessedChunk objects.

        Uses chunker provider for structural chunking, then adds document metadata.
        Does not add chunks to document or generate embeddings.

        Args:
            doc: Document to chunk

        Returns:
            List of (Chunk, ProcessedChunk) tuples
        """
        from verbatim_rag.document import Chunk, ChunkType, ProcessedChunk

        # Use chunker provider to get structural chunks (with heading hierarchy)
        chunk_tuples = self.chunker_provider.chunk(doc.raw_content)

        result = []
        for i, (raw_text, struct_enhanced) in enumerate(chunk_tuples):
            # Add document metadata to the structurally enhanced text
            final_enhanced = self._add_document_metadata(struct_enhanced, doc)

            # Create basic Chunk with inherited metadata
            doc_chunk = Chunk(
                document_id=doc.id,
                content=raw_text,  # Original text for extraction
                chunk_number=i,
                chunk_type=ChunkType.PARAGRAPH,
                metadata={},  # Keep chunk-level metadata minimal
            )

            # Create ProcessedChunk
            processed_chunk = ProcessedChunk(
                chunk_id=doc_chunk.id,
                enhanced_content=final_enhanced,  # Enhanced text with structure + metadata
            )

            result.append((doc_chunk, processed_chunk))

        return result

    def _add_document_metadata(self, text: str, doc: Document) -> str:
        """
        Add document metadata footer to enhanced text.

        Args:
            text: Text with structural enhancements (headings)
            doc: Document containing metadata

        Returns:
            Text with document metadata appended
        """
        parts = [text, "", "---"]
        parts.append(f"Document: {doc.title or 'Unknown'}")
        parts.append(f"Source: {doc.source or 'Unknown'}")

        # Add custom metadata (skip sensitive fields)
        if doc.metadata:
            skip_keys = {"user_id", "dataset_id", "userId"}
            for key, value in doc.metadata.items():
                if key not in skip_keys:
                    formatted_key = key.replace("_", " ").title()
                    parts.append(f"{formatted_key}: {value}")

        return "\n".join(parts)

    def _generate_embeddings(
        self, texts: List[str]
    ) -> Tuple[Optional[List], Optional[List]]:
        """
        Generate embeddings for a list of texts using batch processing.

        Creates both dense and sparse embeddings if respective providers are enabled.
        Uses batch methods for efficiency (single model call instead of N calls).

        Args:
            texts: List of text strings to embed

        Returns:
            Tuple of (dense_embeddings, sparse_embeddings)
            Returns None for disabled embedding types
        """
        dense_embeddings = None
        sparse_embeddings = None

        if self.dense_provider:
            dense_embeddings = self.dense_provider.embed_batch(texts)

        if self.sparse_provider:
            sparse_embeddings = self.sparse_provider.embed_batch(texts)

        return dense_embeddings, sparse_embeddings

    def _prepare_chunk_metadata(self, doc: Document, chunk: "Chunk") -> Dict[str, Any]:
        """
        Prepare metadata for a chunk to be stored in vector store.

        Assembles all metadata: document fields, custom metadata from doc.metadata,
        and chunk-specific fields. This ensures all fields are available for filtering.

        Args:
            doc: Document the chunk belongs to
            chunk: Chunk object

        Returns:
            Complete metadata dict for vector store
        """

        metadata = {
            # Standard document fields
            "document_id": doc.id,
            "title": doc.title,
            "source": doc.source,
            "doc_type": doc.metadata.get("doc_type"),
            "content_type": doc.content_type.value if doc.content_type else None,
            # Chunk fields
            "chunk_type": chunk.chunk_type.value,
            "chunk_number": chunk.chunk_number,
            "page_number": chunk.metadata.get("page_number", 0),
            # All custom metadata from document (authors, year, conference, etc.)
            **(doc.metadata or {}),
            # Chunk-specific metadata (can override document metadata)
            **chunk.metadata,
        }

        return metadata

    def _store_chunks(
        self,
        ids: List[str],
        texts: List[str],
        enhanced_texts: List[str],
        dense_embeddings: Optional[List],
        sparse_embeddings: Optional[List],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Store chunk vectors in vector store.

        Handles both dense and sparse embeddings, passing None for disabled types.

        Args:
            ids: List of chunk IDs
            texts: List of original chunk texts
            enhanced_texts: List of enhanced chunk texts (with context)
            dense_embeddings: Dense embeddings or None if disabled
            sparse_embeddings: Sparse embeddings or None if disabled
            metadatas: List of metadata dicts for each chunk
        """
        self.vector_store.add_vectors(
            ids=ids,
            dense_vectors=dense_embeddings,
            sparse_vectors=sparse_embeddings,
            texts=texts,
            enhanced_texts=enhanced_texts,
            metadatas=metadatas,
        )

    def _store_document_metadata(self, documents: List[Document]) -> None:
        """
        Store document metadata records in vector store.

        Deduplicates documents by ID and stores them if vector store supports it.

        Args:
            documents: List of Document objects to store
        """
        if not hasattr(self.vector_store, "add_documents"):
            return

        # Deduplicate documents by ID
        doc_dict = {}
        for doc in documents:
            if doc.id not in doc_dict:
                doc_dict[doc.id] = {
                    "id": doc.id,
                    "title": doc.title,
                    "source": doc.source,
                    "content_type": doc.content_type.value,
                    "raw_content": "",  # TODO: add raw content
                    "metadata": doc.metadata,
                }

        if doc_dict:
            self.vector_store.add_documents(list(doc_dict.values()))

    def add_documents(
        self,
        documents: List[Union[DocumentSchema, Document]],
        document_type: DocumentType = DocumentType.MARKDOWN,
    ) -> None:
        """
        Add documents to the index.

        Args:
            documents: List of DocumentSchema or Document objects to add
            document_type: Type of document (for legacy Document objects)
        """
        if not documents:
            return

        # Handle DocumentSchema (new primary API) and legacy Document objects
        for doc in tqdm(documents, desc="Adding documents"):
            if isinstance(doc, DocumentSchema):
                self._add_schema_document(doc)
            else:
                self._add_document_internal(doc, document_type)

    def _add_schema_document(self, doc: DocumentSchema) -> None:
        """
        Add a DocumentSchema to the index.

        Converts DocumentSchema to Document with properly flattened metadata,
        then delegates to _add_document_internal() for chunking and storage.

        Args:
            doc: DocumentSchema instance to add
        """
        # Convert DocumentSchema to Document with flattened metadata
        document = self._convert_schema_to_document(doc)

        # Delegate to unified internal method
        self._add_document_internal(document)

    def _add_document_internal(
        self,
        doc: Document,
        document_type: DocumentType = DocumentType.MARKDOWN,
    ) -> None:
        """
        Add a Document to the index with chunking, embedding, and storage.

        This is the unified internal method that handles all document addition.
        It chunks the document if needed, generates embeddings, and stores everything.

        Args:
            doc: Document to add
            document_type: Type of document (used for legacy path, ignored for schema path)
        """
        # Step 1: Chunk the document if it doesn't have chunks yet
        if not doc.chunks:
            chunks = self._chunk_document(doc)
        else:
            # Extract existing chunks (legacy path where Document has pre-populated chunks)
            chunks = [
                (chunk, processed_chunk)
                for chunk in doc.chunks
                for processed_chunk in chunk.processed_chunks
            ]

        if not chunks:
            return

        # Step 2: Extract texts for embedding
        ids = []
        texts = []
        enhanced_texts = []
        chunks_list = []  # Keep track of chunks for metadata prep

        for chunk, processed_chunk in chunks:
            ids.append(processed_chunk.id)
            texts.append(chunk.content)
            enhanced_texts.append(processed_chunk.enhanced_content)
            chunks_list.append(chunk)

        # Step 3: Generate embeddings for all enhanced texts
        dense_embeddings, sparse_embeddings = self._generate_embeddings(enhanced_texts)

        # Step 4: Prepare metadata for each chunk
        metadatas = [self._prepare_chunk_metadata(doc, chunk) for chunk in chunks_list]

        # Step 5: Store chunks in vector store
        self._store_chunks(
            ids=ids,
            texts=texts,
            enhanced_texts=enhanced_texts,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
            metadatas=metadatas,
        )

        # Step 6: Store document metadata
        self._store_document_metadata([doc])

    def add_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        doc_id: str,
        document_type: DocumentType = DocumentType.MARKDOWN,
    ) -> str:
        """
        Add a single document (legacy API - prefer DocumentSchema).

        This method converts simple parameters to DocumentSchema and delegates
        to the unified document addition pipeline.

        Args:
            content: Document text content to be chunked and indexed
            metadata: Document metadata dict
            doc_id: Document ID
            document_type: Type of document content

        Returns:
            Document ID
        """
        # Extract standard fields from metadata
        title = metadata.get("title", "")
        source = metadata.get("source", "")
        doc_type = metadata.get("doc_type")

        # Determine content_type
        if "content_type" in metadata:
            content_type_value = metadata.get("content_type")
            # Handle both string and DocumentType
            if isinstance(content_type_value, str):
                content_type = DocumentType(content_type_value)
            else:
                content_type = content_type_value
        else:
            content_type = document_type

        # Extract custom fields (everything except standard fields)
        custom_fields = {
            k: v
            for k, v in metadata.items()
            if k not in ["title", "source", "doc_type", "content_type"]
        }

        # Create DocumentSchema and delegate to unified path
        doc_schema = DocumentSchema(
            id=doc_id,
            content=content,
            title=title,
            source=source,
            doc_type=doc_type,
            content_type=content_type,
            **custom_fields,  # Pass all custom fields
        )

        # Use unified document addition pipeline (with batch embedding)
        self._add_schema_document(doc_schema)

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

        Args:
            text: Optional text query for vector search
            k: Number of documents to retrieve
            search_type: Type of search ("dense", "sparse", "hybrid", "auto")
            filter: Optional Milvus filter expression for metadata filtering

        Returns:
            List of SearchResult objects
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

        Args:
            document_id: The document ID

        Returns:
            Document metadata dict or None if not found
        """
        if hasattr(self.vector_store, "get_document"):
            return self.vector_store.get_document(document_id)
        return None

    def get_all_chunks(self, limit: int = 100) -> List[SearchResult]:
        """
        Get chunks from the index (up to limit).

        Note: This returns up to 'limit' chunks, not necessarily all chunks.

        Args:
            limit: Maximum number of chunks to return

        Returns:
            List of SearchResult objects
        """
        return self.query(k=limit)

    def get_all_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get unique documents from the index (up to limit).

        Note: This returns up to 'limit' unique documents, not necessarily all documents.

        Args:
            limit: Maximum number of unique documents to return

        Returns:
            List of unique documents with basic info
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

        Args:
            document_id: The document ID
            limit: Maximum number of chunks to return

        Returns:
            List of SearchResult objects
        """
        # Build backend-appropriate filter: Local uses JSON-path, Cloud uses promoted dynamic field
        if isinstance(self.vector_store, CloudMilvusStore):
            filter_expr = f'document_id == "{document_id}"'
        else:
            filter_expr = f'metadata["document_id"] == "{document_id}"'
        return self.query(filter=filter_expr, k=limit)

    def inspect(self) -> Dict[str, Any]:
        """
        Get overview statistics about the index.

        Returns:
            Dictionary with index statistics and sample data
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
