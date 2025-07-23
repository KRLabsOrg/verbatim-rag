"""
Vector storage for the Verbatim RAG system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from vector search."""

    id: str
    score: float
    metadata: Dict[str, Any]
    text: str


class VectorStore(ABC):
    """Base vector store interface."""

    @abstractmethod
    def add_vectors(
        self,
        ids: List[str],
        dense_vectors: Optional[List[List[float]]],
        sparse_vectors: Optional[List[Dict[int, float]]],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """Add vectors with metadata and text."""
        pass

    @abstractmethod
    def search(
        self,
        dense_query: Optional[List[float]] = None,
        sparse_query: Optional[Dict[int, float]] = None,
        text_query: Optional[str] = None,
        top_k: int = 5,
        search_type: str = "hybrid",
    ) -> List[SearchResult]:
        """Search for similar vectors using hybrid search."""
        pass

    @abstractmethod
    def delete(self, ids: List[str]):
        """Delete vectors by IDs."""
        pass


class LocalMilvusStore(VectorStore):
    """Local Milvus Lite storage with hybrid search support."""

    def __init__(
        self,
        db_path: str = "./milvus_verbatim.db",
        collection_name: str = "verbatim_rag",
        dense_dim: int = 384,
        enable_dense: bool = True,
        enable_sparse: bool = True,
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.documents_collection_name = f"{collection_name}_documents"
        self.dense_dim = dense_dim
        self.enable_dense = enable_dense
        self.enable_sparse = enable_sparse

        # Validate at least one embedding type is enabled
        if not enable_dense and not enable_sparse:
            raise ValueError(
                "At least one of enable_dense or enable_sparse must be True"
            )

        self._setup_client()

    def _setup_client(self):
        try:
            from pymilvus import MilvusClient, DataType

            # Create Milvus Lite client
            self.client = MilvusClient(self.db_path)

            # Create collection with sparse vector support
            if not self.client.has_collection(collection_name=self.collection_name):
                schema = self.client.create_schema(
                    auto_id=False,
                    enable_dynamic_fields=True,
                )

                # Add fields
                schema.add_field(
                    field_name="id",
                    datatype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=100,
                )
                # Add only the vector fields that are enabled
                if self.enable_dense:
                    schema.add_field(
                        field_name="dense_vector",
                        datatype=DataType.FLOAT_VECTOR,
                        dim=self.dense_dim,
                    )

                if self.enable_sparse:
                    schema.add_field(
                        field_name="sparse_vector",
                        datatype=DataType.SPARSE_FLOAT_VECTOR,
                    )
                schema.add_field(
                    field_name="text",
                    datatype=DataType.VARCHAR,
                    max_length=65535,
                    enable_analyzer=True,
                )
                schema.add_field(
                    field_name="document_id", datatype=DataType.VARCHAR, max_length=100
                )
                schema.add_field(
                    field_name="title", datatype=DataType.VARCHAR, max_length=512
                )
                schema.add_field(
                    field_name="source", datatype=DataType.VARCHAR, max_length=512
                )
                schema.add_field(
                    field_name="chunk_type", datatype=DataType.VARCHAR, max_length=50
                )
                schema.add_field(field_name="page_number", datatype=DataType.INT64)

                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name, schema=schema
                )

                # Create indexes for enabled vector fields
                index_params = self.client.prepare_index_params()

                if self.enable_dense:
                    # Dense vector index
                    index_params.add_index(
                        field_name="dense_vector",
                        index_type="IVF_FLAT",
                        metric_type="COSINE",
                        params={"nlist": 1024},
                    )

                if self.enable_sparse:
                    # Sparse vector index
                    index_params.add_index(
                        field_name="sparse_vector",
                        index_type="SPARSE_INVERTED_INDEX",
                        metric_type="IP",
                        params={"inverted_index_algo": "DAAT_MAXSCORE"},
                    )

                self.client.create_index(
                    collection_name=self.collection_name, index_params=index_params
                )

                logger.info(f"Created indexes for collection: {self.collection_name}")

            # Create documents collection (no vectors, just metadata)
            if not self.client.has_collection(
                collection_name=self.documents_collection_name
            ):
                doc_schema = self.client.create_schema(
                    auto_id=False,
                    enable_dynamic_fields=True,
                )

                # Add fields for documents (no chunks, just document metadata)
                doc_schema.add_field(
                    field_name="id",
                    datatype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=100,
                )
                doc_schema.add_field(
                    field_name="title", datatype=DataType.VARCHAR, max_length=512
                )
                doc_schema.add_field(
                    field_name="source", datatype=DataType.VARCHAR, max_length=512
                )
                doc_schema.add_field(
                    field_name="content_type", datatype=DataType.VARCHAR, max_length=50
                )
                doc_schema.add_field(
                    field_name="raw_content",
                    datatype=DataType.VARCHAR,
                    max_length=65535,
                )
                doc_schema.add_field(field_name="metadata", datatype=DataType.JSON)

                # Create documents collection
                self.client.create_collection(
                    collection_name=self.documents_collection_name, schema=doc_schema
                )

                logger.info(
                    f"Created documents collection: {self.documents_collection_name}"
                )

            logger.info(f"Connected to Milvus Lite: {self.db_path}")

        except ImportError:
            raise ImportError("pip install pymilvus")

    def add_vectors(
        self,
        ids: List[str],
        dense_vectors: Optional[List[List[float]]],
        sparse_vectors: Optional[List[Dict[int, float]]],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """Add vectors with metadata and text."""

        # Validate inputs based on enabled embedding types
        if self.enable_dense and (dense_vectors is None or len(dense_vectors) == 0):
            raise ValueError("Dense vectors required but not provided")
        if self.enable_sparse and (sparse_vectors is None or len(sparse_vectors) == 0):
            raise ValueError("Sparse vectors required but not provided")

        # Prepare data for Milvus with conditional vectors
        data = []
        for i in range(len(ids)):
            item = {
                "id": ids[i],
                "text": texts[i],
                "document_id": metadatas[i].get("document_id", ""),
                "title": metadatas[i].get("title", ""),  # Keep for search display
                "source": metadatas[i].get("source", ""),  # Keep for search display
                "chunk_type": metadatas[i].get("chunk_type", ""),
                "page_number": metadatas[i].get("page_number", 0),
            }

            # Add vectors only for fields that exist in the schema
            if self.enable_dense and dense_vectors:
                item["dense_vector"] = dense_vectors[i]
            if self.enable_sparse and sparse_vectors:
                item["sparse_vector"] = sparse_vectors[i]

            data.append(item)

        # Insert into Milvus
        self.client.insert(collection_name=self.collection_name, data=data)

        logger.info(f"Added {len(data)} vectors to Milvus")

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add document metadata to the documents collection."""

        # Prepare document data (no chunks, just metadata)
        doc_data = []
        for doc in documents:
            item = {
                "id": doc["id"],
                "title": doc["title"],
                "source": doc["source"],
                "content_type": doc["content_type"],
                "raw_content": doc["raw_content"],
                "metadata": doc["metadata"],
            }
            doc_data.append(item)

        # Insert into documents collection
        self.client.insert(
            collection_name=self.documents_collection_name, data=doc_data
        )

        logger.info(f"Added {len(doc_data)} documents to Milvus")

    def get_document(self, document_id: str) -> Dict[str, Any]:
        """Retrieve document metadata by ID."""

        results = self.client.query(
            collection_name=self.documents_collection_name,
            filter=f"id == '{document_id}'",
            output_fields=[
                "id",
                "title",
                "source",
                "content_type",
                "raw_content",
                "metadata",
            ],
        )

        if results:
            return results[0]
        else:
            return None

    def search(
        self,
        dense_query: Optional[List[float]] = None,
        sparse_query: Optional[Dict[int, float]] = None,
        text_query: Optional[str] = None,
        top_k: int = 5,
        search_type: str = "dense",
    ) -> List[SearchResult]:
        """Search using dense, sparse, or hybrid vectors."""

        output_fields = [
            "text",
            "document_id",
            "title",
            "source",
            "chunk_type",
            "page_number",
        ]

        if search_type == "dense" and dense_query:
            # Dense vector search
            results = self.client.search(
                collection_name=self.collection_name,
                data=[dense_query],
                anns_field="dense_vector",
                limit=top_k,
                output_fields=output_fields,
            )

        elif search_type == "sparse" and sparse_query:
            # Sparse vector search
            results = self.client.search(
                collection_name=self.collection_name,
                data=[sparse_query],
                anns_field="sparse_vector",
                limit=top_k,
                output_fields=output_fields,
            )

        elif search_type == "hybrid" and dense_query and sparse_query:
            # For now, fall back to dense search as hybrid search has API issues
            # TODO: Fix hybrid search implementation
            results = self.client.search(
                collection_name=self.collection_name,
                data=[dense_query],
                anns_field="dense_vector",
                limit=top_k,
                output_fields=output_fields,
            )

        else:
            raise ValueError(
                f"Invalid search configuration: type={search_type}, dense={dense_query is not None}, sparse={sparse_query is not None}"
            )

        # Convert results
        search_results = []
        for hit in results[0]:
            search_results.append(
                SearchResult(
                    id=hit.get("id"),
                    score=hit.get("distance", 0.0),
                    text=hit.get("entity", {}).get("text", ""),
                    metadata={
                        "document_id": hit.get("entity", {}).get("document_id", ""),
                        "title": hit.get("entity", {}).get("title", ""),
                        "source": hit.get("entity", {}).get("source", ""),
                        "chunk_type": hit.get("entity", {}).get("chunk_type", ""),
                        "page_number": hit.get("entity", {}).get("page_number", 0),
                    },
                )
            )

        return search_results

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents stored in the documents collection"""
        try:
            if self.client.has_collection(
                collection_name=self.documents_collection_name
            ):
                # Query all documents from the documents collection with limit
                results = self.client.query(
                    collection_name=self.documents_collection_name,
                    filter="",  # No filter to get all
                    output_fields=[
                        "id",
                        "title",
                        "source",
                        "content_type",
                        "raw_content",
                        "metadata",
                    ],
                    limit=1000,  # Set a reasonable limit
                )
                return results
            return []
        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
            return []

    def delete(self, ids: List[str]):
        # Delete by filter
        filter_expr = f"id in {ids}"
        self.client.delete(collection_name=self.collection_name, filter=filter_expr)


class CloudMilvusStore(VectorStore):
    """Cloud Milvus storage with hybrid search support."""

    def __init__(
        self,
        collection_name: str = "verbatim_rag",
        dense_dim: int = 384,
        sparse_dim: int = 30000,
        host: str = "localhost",
        port: str = "19530",
        username: str = "",
        password: str = "",
    ):
        self.collection_name = collection_name
        self.dense_dim = dense_dim
        self.sparse_dim = sparse_dim
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self._setup_milvus()

    def _setup_milvus(self):
        try:
            from pymilvus import (
                connections,
                Collection,
                CollectionSchema,
                FieldSchema,
                DataType,
            )

            # Connect to cloud Milvus
            connections.connect(
                "default",
                host=self.host,
                port=self.port,
                user=self.username,
                password=self.password,
            )

            # Create collection if it doesn't exist
            if not self._collection_exists():
                self._create_collection()

            self.collection = Collection(self.collection_name)
            logger.info(f"Connected to cloud Milvus collection: {self.collection_name}")

        except ImportError:
            raise ImportError("pip install pymilvus")

    def _collection_exists(self) -> bool:
        from pymilvus import utility

        return utility.has_collection(self.collection_name)

    def _create_collection(self):
        from pymilvus import Collection, CollectionSchema, FieldSchema, DataType

        fields = [
            FieldSchema(
                name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100
            ),
            FieldSchema(
                name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dense_dim
            ),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="page_number", dtype=DataType.INT64),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]

        schema = CollectionSchema(fields)
        collection = Collection(self.collection_name, schema)

        # Create indexes
        # Dense vector index
        dense_index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        collection.create_index("dense_vector", dense_index_params)

        # Sparse vector index
        sparse_index_params = {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
        }
        collection.create_index("sparse_vector", sparse_index_params)

        logger.info(
            f"Created cloud Milvus collection with hybrid search: {self.collection_name}"
        )

    def add_vectors(
        self,
        ids: List[str],
        dense_vectors: List[List[float]],
        sparse_vectors: List[Dict[int, float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """Add vectors with metadata and text."""

        # Extract metadata fields
        titles = [meta.get("title", "") for meta in metadatas]
        sources = [meta.get("source", "") for meta in metadatas]
        chunk_types = [meta.get("chunk_type", "") for meta in metadatas]
        page_numbers = [meta.get("page_number", 0) for meta in metadatas]

        data = [
            ids,
            dense_vectors,
            sparse_vectors,
            texts,
            titles,
            sources,
            chunk_types,
            page_numbers,
            metadatas,
        ]

        self.collection.insert(data)
        self.collection.flush()

    def search(
        self,
        dense_query: Optional[List[float]] = None,
        sparse_query: Optional[Dict[int, float]] = None,
        text_query: Optional[str] = None,
        top_k: int = 5,
        search_type: str = "hybrid",
    ) -> List[SearchResult]:
        """Search using hybrid dense + sparse vectors."""

        self.collection.load()

        output_fields = [
            "text",
            "title",
            "source",
            "chunk_type",
            "page_number",
            "metadata",
        ]

        if search_type == "dense" and dense_query:
            # Dense vector search only
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[dense_query],
                anns_field="dense_vector",
                param=search_params,
                limit=top_k,
                output_fields=output_fields,
            )

        elif search_type == "sparse" and sparse_query:
            # Sparse vector search only
            search_params = {"metric_type": "IP"}
            results = self.collection.search(
                data=[sparse_query],
                anns_field="sparse_vector",
                param=search_params,
                limit=top_k,
                output_fields=output_fields,
            )

        elif search_type == "hybrid" and dense_query and sparse_query:
            # Hybrid search - combine dense and sparse
            from pymilvus import AnnSearchRequest, WeightedRanker

            # Dense search request
            dense_search_param = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            dense_req = AnnSearchRequest(
                data=[dense_query],
                anns_field="dense_vector",
                param=dense_search_param,
                limit=top_k,
            )

            # Sparse search request
            sparse_search_param = {"metric_type": "IP"}
            sparse_req = AnnSearchRequest(
                data=[sparse_query],
                anns_field="sparse_vector",
                param=sparse_search_param,
                limit=top_k,
            )

            # Hybrid search with weighted ranking
            ranker = WeightedRanker(0.7, 0.3)  # 70% dense, 30% sparse
            results = self.collection.hybrid_search(
                reqs=[dense_req, sparse_req],
                ranker=ranker,
                limit=top_k,
                output_fields=output_fields,
            )

        else:
            raise ValueError(f"Invalid search configuration: type={search_type}")

        # Convert results
        search_results = []
        for hit in results[0]:
            search_results.append(
                SearchResult(
                    id=hit.id,
                    score=hit.score,
                    text=hit.entity.get("text", ""),
                    metadata={
                        "title": hit.entity.get("title", ""),
                        "source": hit.entity.get("source", ""),
                        "chunk_type": hit.entity.get("chunk_type", ""),
                        "page_number": hit.entity.get("page_number", 0),
                        **hit.entity.get("metadata", {}),
                    },
                )
            )

        return search_results

    def delete(self, ids: List[str]):
        self.collection.delete(f"id in {ids}")
        self.collection.flush()
