"""
Vector storage for the Verbatim RAG system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from vector search."""

    id: str
    score: float
    metadata: Dict[str, Any]
    text: str  # Original clean text for display
    enhanced_text: str = ""  # Enhanced text used for vectorization


class VectorStore(ABC):
    """Base vector store interface."""

    @abstractmethod
    def add_vectors(
        self,
        ids: List[str],
        dense_vectors: Optional[List[List[float]]],
        sparse_vectors: Optional[List[Dict[int, float]]],
        texts: List[str],
        enhanced_texts: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """Add vectors with metadata, original text, and enhanced text."""
        pass

    @abstractmethod
    def query(
        self,
        dense_query: Optional[List[float]] = None,
        sparse_query: Optional[Dict[int, float]] = None,
        text_query: Optional[str] = None,
        top_k: int = 5,
        search_type: str = "hybrid",
        filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Query for similar vectors using hybrid search."""
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
                    enable_dynamic_field=True,
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
                    field_name="enhanced_text",
                    datatype=DataType.VARCHAR,
                    max_length=65535,
                    enable_analyzer=True,
                )
                # Single JSON field for all metadata
                schema.add_field(field_name="metadata", datatype=DataType.JSON)

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
                    enable_dynamic_field=True,
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
        enhanced_texts: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """Add vectors with metadata and text."""

        # Validate inputs based on enabled embedding types
        if self.enable_dense and (dense_vectors is None or len(dense_vectors) == 0):
            raise ValueError("Dense vectors required but not provided")
        if self.enable_sparse and (sparse_vectors is None or len(sparse_vectors) == 0):
            raise ValueError("Sparse vectors required but not provided")

        # Prepare data for Milvus with conditional vectors
        from datetime import datetime
        from enum import Enum

        def json_serialize_safe(obj):
            """Safely serialize objects to JSON, handling datetime and enum objects."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Enum):
                return obj.value  # Convert enum to its string value
            elif isinstance(obj, dict):
                return {k: json_serialize_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [json_serialize_safe(item) for item in obj]
            else:
                return obj

        data = []
        for i in range(len(ids)):
            # Serialize all metadata - ensure all values are JSON-serializable
            safe_metadata = json_serialize_safe(metadatas[i])

            item = {
                "id": ids[i],
                "text": texts[i],  # Original clean text
                "enhanced_text": enhanced_texts[i],  # Enhanced text for vectorization
                "metadata": safe_metadata,
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
        import json
        from datetime import datetime
        from enum import Enum

        def json_serialize_safe(obj):
            """Safely serialize objects to JSON, handling datetime and enum objects."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Enum):
                return obj.value  # Convert enum to its string value
            elif isinstance(obj, dict):
                return {k: json_serialize_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [json_serialize_safe(item) for item in obj]
            else:
                return obj

        doc_data = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            if isinstance(metadata, dict):
                # Handle datetime objects in metadata, but keep as dict (not JSON string)
                safe_metadata = json_serialize_safe(metadata)
            elif isinstance(metadata, str):
                # If it's already a string, try to parse it back to dict for consistency
                try:
                    safe_metadata = json.loads(metadata)
                except:
                    safe_metadata = {
                        "raw": metadata
                    }  # Fallback for invalid JSON strings
            else:
                # Convert other types to dict format
                safe_metadata = json_serialize_safe(metadata) if metadata else {}

            item = {
                "id": doc.get("id", ""),
                "title": doc.get("title") or "",  # Convert None to empty string
                "source": doc.get("source") or "",  # Convert None to empty string
                "content_type": doc.get("doc_type")
                or doc.get("content_type")
                or "",  # Handle None values
                "raw_content": "",
                "metadata": safe_metadata,  # Store as dict, not JSON string
            }
            doc_data.append(item)

        # Insert into documents collection
        self.client.insert(
            collection_name=self.documents_collection_name, data=doc_data
        )

        logger.info(f"Added {len(doc_data)} documents to Milvus")

    def add_document_schema(self, document_dict: Dict[str, Any], doc_id: str = None):
        """
        Add a single document using the new schema system.

        :param document_dict: Document dict from DocumentSchema.to_storage_dict()
        :param doc_id: Optional document ID override
        """
        if doc_id:
            document_dict["id"] = doc_id

        self.add_documents([document_dict])

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

    def query(
        self,
        dense_query: Optional[List[float]] = None,
        sparse_query: Optional[Dict[int, float]] = None,
        text_query: Optional[str] = None,
        top_k: int = 5,
        search_type: str = "dense",
        filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Query using vector search or filter-only browsing."""

        # If no vectors provided, do filter-only query
        if not dense_query and not sparse_query:
            return self._filter_only_query(filter, top_k)

        output_fields = [
            "text",
            "enhanced_text",
            "metadata",
        ]

        if search_type == "dense" and dense_query:
            # Dense vector search
            results = self.client.search(
                collection_name=self.collection_name,
                data=[dense_query],
                anns_field="dense_vector",
                limit=top_k,
                output_fields=output_fields,
                filter=filter,
            )

        elif search_type == "sparse" and sparse_query:
            # Sparse vector search
            results = self.client.search(
                collection_name=self.collection_name,
                data=[sparse_query],
                anns_field="sparse_vector",
                limit=top_k,
                output_fields=output_fields,
                filter=filter,
            )

        elif search_type == "hybrid" and dense_query and sparse_query:
            # Hybrid search: combine dense and sparse results
            # Note: Milvus doesn't have native hybrid search, so we implement it
            # by combining results from both dense and sparse searches
            try:
                # Perform dense search
                dense_results = self.client.search(
                    collection_name=self.collection_name,
                    data=[dense_query],
                    anns_field="dense_vector",
                    limit=top_k * 2,  # Get more results for merging
                    output_fields=output_fields,
                    filter=filter,
                )

                # Perform sparse search
                sparse_results = self.client.search(
                    collection_name=self.collection_name,
                    data=[sparse_query],
                    anns_field="sparse_vector",
                    limit=top_k * 2,  # Get more results for merging
                    output_fields=output_fields,
                    filter=filter,
                )

                # Combine results using reciprocal rank fusion (RRF)
                results = self._merge_hybrid_results(
                    dense_results[0], sparse_results[0], top_k, alpha=0.5
                )
                results = [results]  # Wrap in list to match expected format

            except Exception as e:
                logger.warning(
                    f"Hybrid search failed: {e}, falling back to dense search"
                )
                # Fallback to dense search
                results = self.client.search(
                    collection_name=self.collection_name,
                    data=[dense_query],
                    anns_field="dense_vector",
                    limit=top_k,
                    output_fields=output_fields,
                    filter=filter,
                )

        else:
            raise ValueError(
                f"Invalid search configuration: type={search_type}, dense={dense_query is not None}, sparse={sparse_query is not None}"
            )

        # Convert results
        search_results = []
        for hit in results[0]:
            entity = hit.get("entity", {})

            # Get metadata from single JSON field
            metadata = entity.get("metadata", {})
            # Handle both dict and JSON string cases
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = {"raw": metadata}

            search_results.append(
                SearchResult(
                    id=hit.get("id"),
                    score=hit.get("distance", 0.0),
                    text=entity.get("text", ""),
                    enhanced_text=entity.get("enhanced_text", ""),
                    metadata=metadata,
                )
            )

        return search_results

    def _filter_only_query(
        self, filter: Optional[str], limit: int
    ) -> List[SearchResult]:
        """Query without vector search - just filtering/browsing."""
        try:
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter or "",  # Empty filter gets all
                output_fields=["id", "text", "enhanced_text", "metadata"],
                limit=limit,
            )

            # Convert to SearchResult format
            search_results = []
            for result in results:
                metadata = result.get("metadata", {})
                # Handle both dict and JSON string cases
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except Exception:
                        metadata = {"raw": metadata}

                search_results.append(
                    SearchResult(
                        id=result.get("id", ""),
                        score=1.0,  # No relevance score for filter-only queries
                        text=result.get("text", ""),
                        enhanced_text=result.get("enhanced_text", ""),
                        metadata=metadata,
                    )
                )

            return search_results

        except Exception as e:
            logger.error(f"Failed to query chunks: {e}")
            return []

    def _merge_hybrid_results(
        self, dense_hits, sparse_hits, top_k: int, alpha: float = 0.5
    ):
        """
        Merge dense and sparse search results using reciprocal rank fusion (RRF).

        :param dense_hits: Results from dense vector search
        :param sparse_hits: Results from sparse vector search
        :param top_k: Number of final results to return
        :param alpha: Weight for combining scores (0.5 = equal weight)
        :return: Merged results list
        """
        # Create score maps based on ranking
        dense_scores = {}
        sparse_scores = {}
        all_ids = set()

        # Process dense results
        for rank, hit in enumerate(dense_hits):
            hit_id = hit.get("id")
            if hit_id:
                # RRF score: 1 / (rank + 1)
                dense_scores[hit_id] = 1.0 / (rank + 1)
                all_ids.add(hit_id)

        # Process sparse results
        for rank, hit in enumerate(sparse_hits):
            hit_id = hit.get("id")
            if hit_id:
                sparse_scores[hit_id] = 1.0 / (rank + 1)
                all_ids.add(hit_id)

        # Combine scores and create merged results
        merged_results = []
        hit_map = {}

        # Create hit map for easy lookup
        for hit in dense_hits:
            hit_id = hit.get("id")
            if hit_id:
                hit_map[hit_id] = hit

        for hit in sparse_hits:
            hit_id = hit.get("id")
            if hit_id and hit_id not in hit_map:
                hit_map[hit_id] = hit

        # Calculate combined scores and create results
        for hit_id in all_ids:
            dense_score = dense_scores.get(hit_id, 0.0)
            sparse_score = sparse_scores.get(hit_id, 0.0)

            # Weighted combination of RRF scores
            combined_score = alpha * dense_score + (1 - alpha) * sparse_score

            if hit_id in hit_map:
                hit = hit_map[hit_id].copy()
                hit["distance"] = (
                    1.0 - combined_score
                )  # Convert back to distance (lower is better)
                merged_results.append(hit)

        # Sort by combined score (higher is better) and return top_k
        merged_results.sort(key=lambda x: 1.0 - x.get("distance", 1.0), reverse=True)
        return merged_results[:top_k]

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
        uri: Optional[str] = None,
        token: Optional[str] = None,
        enable_dense: bool = True,
        enable_sparse: bool = True,
    ):
        self.collection_name = collection_name
        self.documents_collection_name = f"{collection_name}_documents"
        self.dense_dim = dense_dim
        self.sparse_dim = sparse_dim
        self.uri = uri
        self.token = token
        self.enable_dense = enable_dense
        self.enable_sparse = enable_sparse

        # Validate at least one embedding type is enabled
        if not enable_dense and not enable_sparse:
            raise ValueError(
                "At least one of enable_dense or enable_sparse must be True"
            )

        self._setup_milvus()

    def _setup_milvus(self):
        try:
            from pymilvus import MilvusClient, DataType

            if not self.uri:
                raise ValueError(
                    "CloudMilvusStore requires 'uri' for cloud connections"
                )

            # Connect via MilvusClient (Cloud)
            if self.token:
                self.client = MilvusClient(uri=self.uri, token=self.token)
            else:
                self.client = MilvusClient(uri=self.uri)

            # Create main chunks collection if missing
            if not self.client.has_collection(collection_name=self.collection_name):
                schema = self.client.create_schema(
                    auto_id=False, enable_dynamic_field=True
                )
                schema.add_field(
                    field_name="id",
                    datatype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=100,
                )
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
                    field_name="enhanced_text",
                    datatype=DataType.VARCHAR,
                    max_length=65535,
                    enable_analyzer=True,
                )
                schema.add_field(field_name="metadata", datatype=DataType.JSON)

                self.client.create_collection(
                    collection_name=self.collection_name, schema=schema
                )

                index_params = self.client.prepare_index_params()
                if self.enable_dense:
                    index_params.add_index(
                        field_name="dense_vector",
                        index_type="IVF_FLAT",
                        metric_type="COSINE",
                        params={"nlist": 1024},
                    )
                if self.enable_sparse:
                    index_params.add_index(
                        field_name="sparse_vector",
                        index_type="SPARSE_INVERTED_INDEX",
                        metric_type="IP",
                        params={"inverted_index_algo": "DAAT_MAXSCORE"},
                    )
                self.client.create_index(
                    collection_name=self.collection_name, index_params=index_params
                )
                logger.info(f"Created cloud Milvus collection: {self.collection_name}")

            # Create documents collection (include dummy vector to satisfy cloud constraint)
            if not self.client.has_collection(
                collection_name=self.documents_collection_name
            ):
                doc_schema = self.client.create_schema(
                    auto_id=False, enable_dynamic_field=True
                )
                doc_schema.add_field(
                    field_name="id",
                    datatype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=100,
                )
                doc_schema.add_field(
                    field_name="title", datatype=DataType.VARCHAR, max_length=4096
                )
                doc_schema.add_field(
                    field_name="source", datatype=DataType.VARCHAR, max_length=4096
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
                # Add required dummy vector (dim=2)
                doc_schema.add_field(
                    field_name="dummy_vector", datatype=DataType.FLOAT_VECTOR, dim=2
                )

                self.client.create_collection(
                    collection_name=self.documents_collection_name, schema=doc_schema
                )
                # Create flat index on dummy_vector so collection can be loaded
                doc_index_params = self.client.prepare_index_params()
                doc_index_params.add_index(
                    field_name="dummy_vector",
                    index_type="FLAT",
                    metric_type="L2",
                    params={},
                )
                self.client.create_index(
                    collection_name=self.documents_collection_name,
                    index_params=doc_index_params,
                )
                logger.info(
                    f"Created cloud Milvus documents collection: {self.documents_collection_name}"
                )

            logger.info("Connected to Milvus Cloud via MilvusClient")

        except ImportError:
            raise ImportError("pip install pymilvus")

    def _collection_exists(self) -> bool:
        return self.client.has_collection(self.collection_name)

    def _create_collection(self):
        from pymilvus import DataType

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(
            field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100
        )
        if self.enable_dense:
            schema.add_field(
                field_name="dense_vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=self.dense_dim,
            )
        if self.enable_sparse:
            schema.add_field(
                field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
            )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
        )
        schema.add_field(
            field_name="enhanced_text",
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
        )
        schema.add_field(field_name="metadata", datatype=DataType.JSON)

        self.client.create_collection(
            collection_name=self.collection_name, schema=schema
        )

        index_params = self.client.prepare_index_params()
        if self.enable_dense:
            index_params.add_index(
                field_name="dense_vector",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 1024},
            )
        if self.enable_sparse:
            index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"inverted_index_algo": "DAAT_MAXSCORE"},
            )
        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params
        )
        logger.info(
            f"Created cloud Milvus collection with dynamic fields: {self.collection_name}"
        )

    def _create_documents_collection(self):
        from pymilvus import DataType

        doc_schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        doc_schema.add_field(
            field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100
        )
        doc_schema.add_field(
            field_name="title", datatype=DataType.VARCHAR, max_length=4096
        )
        doc_schema.add_field(
            field_name="source", datatype=DataType.VARCHAR, max_length=4096
        )
        doc_schema.add_field(
            field_name="content_type", datatype=DataType.VARCHAR, max_length=50
        )
        doc_schema.add_field(
            field_name="raw_content", datatype=DataType.VARCHAR, max_length=65535
        )
        doc_schema.add_field(field_name="metadata", datatype=DataType.JSON)
        # Add required dummy vector (dim=2)
        doc_schema.add_field(
            field_name="dummy_vector", datatype=DataType.FLOAT_VECTOR, dim=2
        )

        self.client.create_collection(
            collection_name=self.documents_collection_name, schema=doc_schema
        )
        logger.info(
            f"Created cloud Milvus documents collection: {self.documents_collection_name}"
        )

    def add_vectors(
        self,
        ids: List[str],
        dense_vectors: Optional[List[List[float]]],
        sparse_vectors: Optional[List[Dict[int, float]]],
        texts: List[str],
        enhanced_texts: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """Add vectors with metadata and text using MilvusClient."""

        def promote_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
            # Copy to avoid mutating original
            md = dict(metadata or {})

            # Keys to promote to top-level dynamic fields if present
            # Simplicity: only promote keys we need to filter on
            promotable_keys = {"user_id", "document_id", "dataset_id"}

            promoted: Dict[str, Any] = {}
            for key in list(md.keys()):
                if key in promotable_keys:
                    promoted[key] = md.pop(key)

            return promoted, md

        from datetime import datetime
        from enum import Enum

        def json_serialize_safe(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return getattr(obj, "value", str(obj))
            if isinstance(obj, dict):
                return {str(k): json_serialize_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [json_serialize_safe(item) for item in obj]
            return obj

        data: List[Dict[str, Any]] = []
        for i in range(len(ids)):
            promoted, cleaned_metadata = promote_metadata(metadatas[i])
            safe_metadata = json_serialize_safe(cleaned_metadata)

            item: Dict[str, Any] = {
                "id": ids[i],
                "text": texts[i],
                "enhanced_text": enhanced_texts[i],
                "metadata": safe_metadata,
                **promoted,
            }

            if self.enable_dense and dense_vectors:
                item["dense_vector"] = dense_vectors[i]
            if self.enable_sparse and sparse_vectors:
                item["sparse_vector"] = sparse_vectors[i]

            data.append(item)

        self.client.insert(collection_name=self.collection_name, data=data)
        logger.info(f"Added {len(data)} vectors to Milvus (cloud)")

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add document metadata to documents collection."""
        if not documents:
            return

        from datetime import datetime
        from enum import Enum

        def json_serialize_safe(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return getattr(obj, "value", str(obj))
            if isinstance(obj, dict):
                return {str(k): json_serialize_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [json_serialize_safe(item) for item in obj]
            return obj

        rows: List[Dict[str, Any]] = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            safe_metadata = (
                json_serialize_safe(metadata)
                if isinstance(metadata, dict)
                else metadata
            )
            # Promote common filter fields to dynamic fields
            promoted: Dict[str, Any] = {}
            if isinstance(metadata, dict):
                for key in ["user_id", "dataset_id", "document_id"]:
                    if key in metadata:
                        promoted[key] = metadata.get(key)
            row = {
                "id": doc.get("id", ""),
                "title": doc.get("title") or "",
                "source": doc.get("source") or "",
                "content_type": doc.get("doc_type") or doc.get("content_type") or "",
                "raw_content": doc.get("raw_content", ""),
                "metadata": safe_metadata,
                "dummy_vector": [0.0, 0.0],
                **promoted,
            }
            rows.append(row)

        self.client.insert(collection_name=self.documents_collection_name, data=rows)
        logger.info(f"Added {len(rows)} documents to Milvus (cloud)")

    def add_document_schema(self, document_dict: Dict[str, Any], doc_id: str = None):
        if doc_id:
            document_dict["id"] = doc_id
        self.add_documents([document_dict])

    def query(
        self,
        dense_query: Optional[List[float]] = None,
        sparse_query: Optional[Dict[int, float]] = None,
        text_query: Optional[str] = None,
        top_k: int = 5,
        search_type: str = "hybrid",
        filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Query using hybrid dense + sparse vectors or filter-only browsing."""

        # Request fields; include promoted dynamic fields
        dynamic_fields = ["user_id", "document_id", "dataset_id"]
        output_fields = ["text", "enhanced_text", "metadata", *dynamic_fields]

        if not dense_query and not sparse_query:
            try:
                results = self.client.query(
                    collection_name=self.collection_name,
                    filter=filter or "",
                    output_fields=["id", *output_fields],
                    limit=top_k,
                )
                search_results: List[SearchResult] = []
                for r in results:
                    md = r.get("metadata", {}) or {}
                    if isinstance(md, str):
                        try:
                            md = json.loads(md)
                        except Exception:
                            md = {"raw": md}
                    for f in dynamic_fields:
                        if f in r and r[f] is not None:
                            md[f] = r[f]
                    search_results.append(
                        SearchResult(
                            id=r.get("id", ""),
                            score=1.0,
                            text=r.get("text", ""),
                            enhanced_text=r.get("enhanced_text", ""),
                            metadata=md,
                        )
                    )
                return search_results
            except Exception as e:
                logger.error(f"Filter-only query failed: {e}")
                return []

        if search_type == "dense" and dense_query:
            # Dense vector search only
            results = self.client.search(
                collection_name=self.collection_name,
                data=[dense_query],
                anns_field="dense_vector",
                limit=top_k,
                output_fields=output_fields,
                filter=filter,
            )

        elif search_type == "sparse" and sparse_query:
            # Sparse vector search only
            results = self.client.search(
                collection_name=self.collection_name,
                data=[sparse_query],
                anns_field="sparse_vector",
                limit=top_k,
                output_fields=output_fields,
                filter=filter,
            )

        elif search_type == "hybrid" and dense_query and sparse_query:
            # Hybrid search - combine dense and sparse
            # Fallback hybrid: combine results from two searches
            try:
                dense_results = self.client.search(
                    collection_name=self.collection_name,
                    data=[dense_query],
                    anns_field="dense_vector",
                    limit=top_k * 2,
                    output_fields=output_fields,
                    filter=filter,
                )
                sparse_results = self.client.search(
                    collection_name=self.collection_name,
                    data=[sparse_query],
                    anns_field="sparse_vector",
                    limit=top_k * 2,
                    output_fields=output_fields,
                    filter=filter,
                )
                merged = self._merge_hybrid_results(
                    dense_results[0], sparse_results[0], top_k, alpha=0.5
                )
                results = [merged]
            except Exception as e:
                logger.warning(f"Hybrid search failed: {e}, falling back to dense-only")
                results = self.client.search(
                    collection_name=self.collection_name,
                    data=[dense_query],
                    anns_field="dense_vector",
                    limit=top_k,
                    output_fields=output_fields,
                    filter=filter,
                )

        else:
            raise ValueError(f"Invalid search configuration: type={search_type}")

        # Convert results
        search_results = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            md = entity.get("metadata", {}) or {}
            if isinstance(md, str):
                try:
                    md = json.loads(md)
                except Exception:
                    md = {"raw": md}
            for f in dynamic_fields:
                val = entity.get(f)
                if val is not None:
                    md[f] = val
            search_results.append(
                SearchResult(
                    id=hit.get("id"),
                    score=hit.get("distance", 0.0),
                    text=entity.get("text", ""),
                    enhanced_text=entity.get("enhanced_text", ""),
                    metadata=md,
                )
            )

        return search_results

    def delete(self, ids: List[str]):
        if not ids:
            return
        quoted = ",".join([f'"{_id}"' for _id in ids])
        self.client.delete(
            collection_name=self.collection_name, filter=f"id in [{quoted}]"
        )

    def delete_document(self, document_id: str):
        """Delete document row and all chunks for a document by document_id."""
        self.client.delete(
            collection_name=self.collection_name,
            filter=f'document_id == "{document_id}"',
        )
        try:
            self.client.delete(
                collection_name=self.documents_collection_name,
                filter=f'id == "{document_id}"',
            )
        except Exception as e:
            logger.warning(f"Failed to delete document row for {document_id}: {e}")
