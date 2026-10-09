"""
Local Milvus Lite storage implementation.
"""

import logging

from .milvus_base import BaseMilvusStore

logger = logging.getLogger(__name__)


class LocalMilvusStore(BaseMilvusStore):
    """
    Local Milvus Lite storage with hybrid search support.

    Note: Full text search (BM25) is NOT supported in Milvus Lite.
    This feature requires Milvus Standalone, Distributed, or Zilliz Cloud.
    """

    def __init__(
        self,
        db_path: str = "./milvus_verbatim.db",
        collection_name: str = "verbatim_rag",
        dense_dim: int = 384,
        enable_dense: bool = True,
        enable_sparse: bool = True,
        enable_full_text: bool = False,
        index_type: str = "IVF_FLAT",
        nlist: int = 8192,
    ):
        # Milvus Lite does not support BM25 full text search
        if enable_full_text:
            logger.warning(
                "Full text search (BM25) is not supported in Milvus Lite. "
                "This feature requires Milvus Standalone, Distributed, or Zilliz Cloud. "
                "Full text search will be disabled."
            )
            enable_full_text = False

        # Validate at least one embedding type is enabled (before calling super)
        if not enable_dense and not enable_sparse:
            raise ValueError("At least one of enable_dense or enable_sparse must be True")

        self.db_path = db_path
        # Milvus Lite 3 rejects a collection without a vector field, so the documents
        # collection carries a placeholder vector, as in CloudMilvusStore.
        self._needs_dummy_vector = True

        super().__init__(
            collection_name=collection_name,
            dense_dim=dense_dim,
            enable_dense=enable_dense,
            enable_sparse=enable_sparse,
            enable_full_text=enable_full_text,
            index_type=index_type,
            nlist=nlist,
        )

        self._setup_client()

    def _setup_client(self):
        """Initialize Milvus Lite client and create collections."""
        try:
            from pymilvus import DataType, MilvusClient

            self.client = MilvusClient(self.db_path)

            # Create main chunks collection if missing
            if not self.client.has_collection(collection_name=self.collection_name):
                schema = self.client.create_schema(
                    auto_id=False,
                    enable_dynamic_field=True,
                )

                schema.add_field(
                    field_name="id",
                    datatype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=512,
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

                self.client.create_collection(collection_name=self.collection_name, schema=schema)

                # Create indexes
                index_params = self.client.prepare_index_params()

                if self.enable_dense:
                    index_params.add_index(
                        field_name="dense_vector",
                        index_type=self.index_type,
                        metric_type="COSINE",
                        params={"nlist": self.nlist},
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

                logger.info(f"Created indexes for collection: {self.collection_name}")

            # Create documents collection (metadata plus a placeholder vector)
            if not self.client.has_collection(collection_name=self.documents_collection_name):
                doc_schema = self.client.create_schema(
                    auto_id=False,
                    enable_dynamic_field=True,
                )

                doc_schema.add_field(
                    field_name="id",
                    datatype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=100,
                )
                doc_schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
                doc_schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=512)
                doc_schema.add_field(
                    field_name="content_type", datatype=DataType.VARCHAR, max_length=50
                )
                doc_schema.add_field(
                    field_name="raw_content",
                    datatype=DataType.VARCHAR,
                    max_length=65535,
                )
                doc_schema.add_field(field_name="metadata", datatype=DataType.JSON)
                doc_schema.add_field(
                    field_name="dummy_vector", datatype=DataType.FLOAT_VECTOR, dim=2
                )

                self.client.create_collection(
                    collection_name=self.documents_collection_name, schema=doc_schema
                )

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

                logger.info(f"Created documents collection: {self.documents_collection_name}")

            logger.info(f"Connected to Milvus Lite: {self.db_path}")

        except ImportError:
            raise ImportError("pip install pymilvus")
