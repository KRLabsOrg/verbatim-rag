"""
Base Milvus store with shared implementation for Local and Cloud variants.
"""

import json
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from .base import SearchResult, VectorStore
from .hybrid_search import (
    convert_hits_to_results,
    merge_hybrid_results,
    sanitize_hybrid_weights,
)
from .utils import json_serialize_safe, promote_metadata

logger = logging.getLogger(__name__)

# Milvus VARCHAR field limit (slightly under 65535 to be safe)
MAX_TEXT_LENGTH = 60000


class BaseMilvusStore(VectorStore):
    """
    Base class for Milvus stores with shared implementation.

    Subclasses must implement:
    - _setup_client(): Initialize Milvus client and create collections
    - _get_dynamic_fields(): Return list of dynamic field names for queries
    """

    def __init__(
        self,
        collection_name: str = "verbatim_rag",
        dense_dim: int = 384,
        enable_dense: bool = True,
        enable_sparse: bool = True,
        enable_full_text: bool = False,
        index_type: str = "IVF_FLAT",
        nlist: int = 8192,
    ):
        self.collection_name = collection_name
        self.documents_collection_name = f"{collection_name}_documents"
        self.dense_dim = dense_dim
        self.enable_dense = enable_dense
        self.enable_sparse = enable_sparse
        self.enable_full_text = enable_full_text
        self.index_type = index_type
        self.nlist = nlist
        self.client = None

        # Validate at least one embedding type is enabled
        if not enable_dense and not enable_sparse and not enable_full_text:
            raise ValueError(
                "At least one of enable_dense, enable_sparse, or enable_full_text must be True"
            )

    @abstractmethod
    def _setup_client(self):
        """Initialize Milvus client and create collections."""
        pass

    def _get_dynamic_fields(self) -> List[str]:
        """
        Return list of dynamic field names to include in query output.

        Override in subclasses if needed.
        """
        return []

    def _get_output_fields(self) -> List[str]:
        """Return standard output fields plus dynamic fields."""
        base_fields = ["text", "enhanced_text", "metadata"]
        return base_fields + self._get_dynamic_fields()

    def _truncate_text(self, text: str, field_name: str, chunk_id: str) -> str:
        """Truncate text to MAX_TEXT_LENGTH bytes if needed, with warning."""
        encoded = text.encode("utf-8")
        if len(encoded) <= MAX_TEXT_LENGTH:
            return text
        # Truncate by bytes, then decode back (may be slightly under limit)
        truncated = encoded[:MAX_TEXT_LENGTH].decode("utf-8", errors="ignore")
        logger.warning(
            f"Truncating {field_name} for chunk {chunk_id}: "
            f"{len(encoded)} bytes -> {len(truncated.encode('utf-8'))} bytes"
        )
        return truncated

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

        data = []
        for i in range(len(ids)):
            promoted, cleaned_metadata = promote_metadata(metadatas[i])
            safe_metadata = json_serialize_safe(cleaned_metadata)

            item = {
                "id": ids[i],
                "text": self._truncate_text(texts[i], "text", ids[i]),
                "enhanced_text": self._truncate_text(enhanced_texts[i], "enhanced_text", ids[i]),
                "metadata": safe_metadata,
                **promoted,
            }

            if self.enable_dense and dense_vectors:
                item["dense_vector"] = dense_vectors[i]
            if self.enable_sparse and sparse_vectors:
                item["sparse_vector"] = sparse_vectors[i]

            data.append(item)

        self.client.insert(collection_name=self.collection_name, data=data)
        logger.info(f"Added {len(data)} vectors to Milvus")

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add document metadata to the documents collection."""
        if not documents:
            return

        rows = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            safe_metadata = (
                json_serialize_safe(metadata) if isinstance(metadata, dict) else metadata
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
                **promoted,
            }

            # Add dummy vector if needed (for cloud Milvus)
            if hasattr(self, "_needs_dummy_vector") and self._needs_dummy_vector:
                row["dummy_vector"] = [0.0, 0.0]

            rows.append(row)

        self.client.insert(collection_name=self.documents_collection_name, data=rows)
        logger.info(f"Added {len(rows)} documents to Milvus")

    def add_document_schema(self, document_dict: Dict[str, Any], doc_id: str = None):
        """Add a single document using the schema system."""
        if doc_id:
            document_dict["id"] = doc_id
        self.add_documents([document_dict])

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
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
        return results[0] if results else None

    def query(
        self,
        dense_query: Optional[List[float]] = None,
        sparse_query: Optional[Dict[int, float]] = None,
        text_query: Optional[str] = None,
        top_k: int = 5,
        search_type: str = "hybrid",
        filter: Optional[str] = None,
        search_params: Optional[Dict[str, Any]] = None,
        hybrid_weights: Optional[Dict[str, float]] = None,
        rrf_k: int = 60,
    ) -> List[SearchResult]:
        """
        Query using vector search, full text search, or filter-only browsing.

        Args:
            dense_query: Optional dense vector query
            sparse_query: Optional sparse vector query
            text_query: Optional text query for full text search (BM25)
            top_k: Number of results to return
            search_type: Type of search ("dense", "sparse", "hybrid", "full_text", "auto")
            filter: Optional filter expression
            search_params: Optional search parameters
            hybrid_weights: Optional dict of weights for hybrid search
            rrf_k: RRF constant for hybrid search (default 60)
        """
        # If hybrid_weights provided, use N-way hybrid search
        if hybrid_weights is not None:
            return self._hybrid_search_with_weights(
                dense_query,
                sparse_query,
                text_query,
                top_k,
                filter,
                search_params,
                hybrid_weights,
                rrf_k,
            )

        # Full text search using BM25
        if search_type == "full_text" and text_query and self.enable_full_text:
            return self._full_text_search(text_query, top_k, filter)

        # If no vectors provided, do filter-only query
        if not dense_query and not sparse_query:
            return self._filter_only_query(filter, top_k)

        output_fields = self._get_output_fields()
        dynamic_fields = self._get_dynamic_fields()

        if search_type == "dense" and dense_query:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[dense_query],
                anns_field="dense_vector",
                limit=top_k,
                output_fields=output_fields,
                filter=filter,
                search_params=search_params,
            )

        elif search_type == "sparse" and sparse_query:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[sparse_query],
                anns_field="sparse_vector",
                limit=top_k,
                output_fields=output_fields,
                filter=filter,
                search_params=search_params,
            )

        elif search_type == "hybrid" and dense_query and sparse_query:
            try:
                dense_results = self.client.search(
                    collection_name=self.collection_name,
                    data=[dense_query],
                    anns_field="dense_vector",
                    limit=top_k * 2,
                    output_fields=output_fields,
                    filter=filter,
                    search_params=search_params,
                )

                sparse_results = self.client.search(
                    collection_name=self.collection_name,
                    data=[sparse_query],
                    anns_field="sparse_vector",
                    limit=top_k * 2,
                    output_fields=output_fields,
                    filter=filter,
                    search_params=search_params,
                )

                results_by_method = {
                    "dense": dense_results[0],
                    "sparse": sparse_results[0],
                }
                merged = merge_hybrid_results(
                    results_by_method,
                    top_k,
                    {"dense": 0.5, "sparse": 0.5},
                    rrf_k=rrf_k,
                    log_label=self.__class__.__name__,
                )
                results = [merged]

            except Exception as e:
                logger.warning(f"Hybrid search failed: {e}, falling back to dense search")
                results = self.client.search(
                    collection_name=self.collection_name,
                    data=[dense_query],
                    anns_field="dense_vector",
                    limit=top_k,
                    output_fields=output_fields,
                    filter=filter,
                    search_params=search_params,
                )
        else:
            raise ValueError(
                f"Invalid search configuration: type={search_type}, "
                f"dense={dense_query is not None}, sparse={sparse_query is not None}"
            )

        return convert_hits_to_results(results[0], dynamic_fields)

    def _filter_only_query(self, filter: Optional[str], limit: int) -> List[SearchResult]:
        """Query without vector search - just filtering/browsing."""
        try:
            dynamic_fields = self._get_dynamic_fields()
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter or "",
                output_fields=["id", "text", "enhanced_text", "metadata"] + dynamic_fields,
                limit=limit,
            )

            search_results = []
            for result in results:
                metadata = result.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except Exception:
                        metadata = {"raw": metadata}

                for f in dynamic_fields:
                    if f in result and result[f] is not None:
                        metadata[f] = result[f]

                search_results.append(
                    SearchResult(
                        id=result.get("id", ""),
                        score=1.0,
                        text=result.get("text", ""),
                        enhanced_text=result.get("enhanced_text", ""),
                        metadata=metadata,
                    )
                )

            return search_results

        except Exception as e:
            logger.error(f"Failed to query chunks: {e}")
            return []

    def _full_text_search(
        self, text_query: str, top_k: int, filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Perform full text search using BM25.

        Note: Only supported in Cloud Milvus. LocalMilvusStore overrides this
        to raise an error.
        """
        raise NotImplementedError("Full text search not supported in this store type")

    def _hybrid_search_with_weights(
        self,
        dense_query: Optional[List[float]],
        sparse_query: Optional[Dict[int, float]],
        text_query: Optional[str],
        top_k: int,
        filter: Optional[str],
        search_params: Optional[Dict[str, Any]],
        hybrid_weights: Dict[str, float],
        rrf_k: int,
    ) -> List[SearchResult]:
        """Execute N-way hybrid search with custom weights."""
        hybrid_weights = sanitize_hybrid_weights(hybrid_weights)

        # Warn if full_text requested but not available
        if "full_text" in hybrid_weights and not self.enable_full_text:
            logger.warning(
                f"full_text not available on {self.__class__.__name__}, "
                "removing from hybrid_weights"
            )
            hybrid_weights = {k: v for k, v in hybrid_weights.items() if k != "full_text"}

        if not hybrid_weights:
            raise ValueError("No valid search methods in hybrid_weights")

        output_fields = self._get_output_fields()
        dynamic_fields = self._get_dynamic_fields()
        results_by_method = {}

        # Execute each search method
        if "dense" in hybrid_weights and dense_query is not None:
            dense_results = self.client.search(
                collection_name=self.collection_name,
                data=[dense_query],
                anns_field="dense_vector",
                limit=top_k * 2,
                output_fields=output_fields,
                filter=filter,
                search_params=search_params,
            )
            results_by_method["dense"] = dense_results[0]

        if "sparse" in hybrid_weights and sparse_query is not None:
            sparse_results = self.client.search(
                collection_name=self.collection_name,
                data=[sparse_query],
                anns_field="sparse_vector",
                limit=top_k * 2,
                output_fields=output_fields,
                filter=filter,
                search_params=search_params,
            )
            results_by_method["sparse"] = sparse_results[0]

        if "full_text" in hybrid_weights and text_query is not None and self.enable_full_text:
            try:
                full_text_results = self.client.search(
                    collection_name=self.collection_name,
                    data=[text_query],
                    anns_field="bm25_vector",
                    limit=top_k * 2,
                    output_fields=output_fields,
                    filter=filter,
                )
                results_by_method["full_text"] = full_text_results[0]
            except Exception as e:
                logger.warning(f"Full text search failed: {e}, excluding from hybrid")

        # Handle results
        if len(results_by_method) == 0:
            logger.warning("Hybrid search: no valid methods executed after validation")
            return []
        elif len(results_by_method) == 1:
            method_results = list(results_by_method.values())[0]
            logger.debug(
                "Hybrid search (%s): methods=%s weights=%s rrf_k=%s top_k=%s",
                self.__class__.__name__,
                list(results_by_method.keys()),
                hybrid_weights,
                rrf_k,
                top_k,
            )
            return convert_hits_to_results(method_results[:top_k], dynamic_fields)

        # Multiple methods: merge with RRF
        logger.debug("Merging %d search methods using RRF", len(results_by_method))
        merged_hits = merge_hybrid_results(
            results_by_method,
            top_k,
            hybrid_weights,
            rrf_k,
            log_label=self.__class__.__name__,
        )
        return convert_hits_to_results(merged_hits, dynamic_fields)

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents stored in the documents collection."""
        try:
            if self.client.has_collection(collection_name=self.documents_collection_name):
                results = self.client.query(
                    collection_name=self.documents_collection_name,
                    filter="",
                    output_fields=[
                        "id",
                        "title",
                        "source",
                        "content_type",
                        "raw_content",
                        "metadata",
                    ],
                    limit=1000,
                )
                return results
            return []
        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
            return []

    def delete(self, ids: List[str]):
        """Delete vectors by IDs."""
        if not ids:
            return
        quoted = ",".join([f'"{_id}"' for _id in ids])
        self.client.delete(collection_name=self.collection_name, filter=f"id in [{quoted}]")

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
