"""
Configuration system for VerbatimRAG ecosystem.
Supports both local and cloud deployment modes with unified configuration.
"""

import os
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator


class DeploymentMode(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    HYBRID = "hybrid"


class VectorDBType(str, Enum):
    MILVUS_LOCAL = "milvus_local"
    MILVUS_CLOUD = "milvus_cloud"


class RerankerType(str, Enum):
    NONE = "none"
    LOCAL = "local"
    JINA = "jina"
    COHERE = "cohere"
    ANTHROPIC = "anthropic"


class DenseEmbeddingModel(str, Enum):
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    CUSTOM = "custom"


class SparseEmbeddingModel(str, Enum):
    SPLADE = "splade"
    CUSTOM = "custom"


class DocumentProcessor(str, Enum):
    BASIC = "basic"
    DOCLING = "docling"


class ChunkingStrategy(str, Enum):
    BASIC = "basic"
    CHONKIE = "chonkie"


class SearchType(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class DenseEmbeddingConfig(BaseModel):
    """Configuration for dense embedding models"""

    model: DenseEmbeddingModel = DenseEmbeddingModel.SENTENCE_TRANSFORMERS
    model_name: Optional[str] = "all-MiniLM-L6-v2"
    api_key: Optional[str] = None
    device: str = "cpu"
    batch_size: int = 100

    @validator("api_key", pre=True, always=True)
    def get_api_key(cls, v, values):
        if v is None and values.get("model") == DenseEmbeddingModel.OPENAI:
            return os.getenv("OPENAI_API_KEY")
        return v


class SparseEmbeddingConfig(BaseModel):
    """Configuration for sparse embedding models"""

    model: SparseEmbeddingModel = SparseEmbeddingModel.SPLADE
    model_name: Optional[str] = "naver/splade-v3"
    device: str = "cpu"
    batch_size: int = 100
    enabled: bool = False


class VectorDBConfig(BaseModel):
    """Configuration for vector databases"""

    type: VectorDBType = VectorDBType.MILVUS_LOCAL
    host: Optional[str] = None
    port: Optional[int] = None
    collection_name: str = "verbatim_rag"
    api_key: Optional[str] = None

    # Milvus Local settings
    db_path: str = "./milvus_verbatim.db"
    dense_dim: int = 384

    # Milvus index settings
    dense_metric_type: str = "COSINE"
    dense_index_type: str = "IVF_FLAT"
    sparse_metric_type: str = "IP"
    sparse_index_type: str = "SPARSE_INVERTED_INDEX"
    nlist: int = 1024


class RerankerConfig(BaseModel):
    """Configuration for rerankers"""

    type: RerankerType = RerankerType.NONE
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    top_k: int = 10

    # Local reranker settings
    device: str = "cpu"
    batch_size: int = 16

    @validator("api_key", pre=True, always=True)
    def get_api_key(cls, v, values):
        reranker_type = values.get("type")
        if v is None:
            if reranker_type == RerankerType.JINA:
                return os.getenv("JINA_API_KEY")
            elif reranker_type == RerankerType.COHERE:
                return os.getenv("COHERE_API_KEY")
            elif reranker_type == RerankerType.ANTHROPIC:
                return os.getenv("ANTHROPIC_API_KEY")
        return v


class DocumentProcessorConfig(BaseModel):
    """Configuration for document processing"""

    type: DocumentProcessor = DocumentProcessor.BASIC

    # Docling-specific settings
    extract_tables: bool = True
    extract_images: bool = True
    preserve_layout: bool = True
    output_format: str = "markdown"


class ChunkingConfig(BaseModel):
    """Configuration for text chunking"""

    strategy: ChunkingStrategy = ChunkingStrategy.BASIC
    chunk_size: int = 512
    overlap: int = 50

    # Chonkie-specific settings
    semantic_chunking: bool = True
    context_aware: bool = True
    min_chunk_size: int = 100
    max_chunk_size: int = 1000


class SearchConfig(BaseModel):
    """Configuration for search and retrieval"""

    type: SearchType = SearchType.DENSE
    top_k: int = 10

    # Hybrid search settings
    dense_weight: float = 0.7
    sparse_weight: float = 0.3

    # SPLADE settings
    splade_model: str = "naver/splade-cocondenser-ensembledistil"
    splade_device: str = "cpu"


class TemplateConfig(BaseModel):
    """Configuration for template generation"""

    realtime_generation: bool = False
    cache_templates: bool = True
    default_template: str = "default"
    custom_templates_path: Optional[str] = None


class VerbatimDOCConfig(BaseModel):
    """Configuration for VerbatimDOC system"""

    enabled: bool = True
    query_syntax: str = "[!query={}]"
    max_query_length: int = 500
    output_formats: List[str] = ["markdown", "docx", "pdf"]
    preserve_formatting: bool = True

    # Query processing settings
    parallel_processing: bool = True
    max_concurrent_queries: int = 10
    query_timeout: int = 30  # seconds


class CacheConfig(BaseModel):
    """Configuration for caching system"""

    enabled: bool = True
    cache_embeddings: bool = True
    cache_results: bool = True
    cache_ttl: int = 3600  # seconds
    max_cache_size: int = 1000  # number of items


class VerbatimRAGConfig(BaseModel):
    """Main configuration for VerbatimRAG system"""

    deployment_mode: DeploymentMode = DeploymentMode.LOCAL

    # Component configurations
    dense_embedding: DenseEmbeddingConfig = Field(default_factory=DenseEmbeddingConfig)
    sparse_embedding: SparseEmbeddingConfig = Field(
        default_factory=SparseEmbeddingConfig
    )
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    document_processor: DocumentProcessorConfig = Field(
        default_factory=DocumentProcessorConfig
    )
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    template: TemplateConfig = Field(default_factory=TemplateConfig)
    verbatim_doc: VerbatimDOCConfig = Field(default_factory=VerbatimDOCConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    # General settings
    debug: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "VerbatimRAGConfig":
        """Load configuration from YAML file"""
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

    @classmethod
    def from_env(cls) -> "VerbatimRAGConfig":
        """Load configuration from environment variables"""
        config = cls()

        # Override with environment variables
        if os.getenv("VERBATIM_DEPLOYMENT_MODE"):
            config.deployment_mode = DeploymentMode(
                os.getenv("VERBATIM_DEPLOYMENT_MODE")
            )

        if os.getenv("VERBATIM_VECTOR_DB_TYPE"):
            config.vector_db.type = VectorDBType(os.getenv("VERBATIM_VECTOR_DB_TYPE"))

        if os.getenv("VERBATIM_RERANKER_TYPE"):
            config.reranker.type = RerankerType(os.getenv("VERBATIM_RERANKER_TYPE"))

        if os.getenv("VERBATIM_DENSE_EMBEDDING_MODEL"):
            config.dense_embedding.model = DenseEmbeddingModel(
                os.getenv("VERBATIM_DENSE_EMBEDDING_MODEL")
            )

        if os.getenv("VERBATIM_SPARSE_EMBEDDING_ENABLED"):
            config.sparse_embedding.enabled = (
                os.getenv("VERBATIM_SPARSE_EMBEDDING_ENABLED").lower() == "true"
            )

        if os.getenv("VERBATIM_SEARCH_TYPE"):
            config.search.type = SearchType(os.getenv("VERBATIM_SEARCH_TYPE"))

        if os.getenv("VERBATIM_DEBUG"):
            config.debug = os.getenv("VERBATIM_DEBUG").lower() == "true"

        return config

    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file"""
        with open(output_path, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False)

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []

        # Validate API keys based on selected services
        if (
            self.dense_embedding.model == DenseEmbeddingModel.OPENAI
            and not self.dense_embedding.api_key
        ):
            errors.append("OpenAI API key is required for OpenAI embeddings")

        if self.reranker.type == RerankerType.JINA and not self.reranker.api_key:
            errors.append("Jina API key is required for Jina reranker")

        if self.reranker.type == RerankerType.COHERE and not self.reranker.api_key:
            errors.append("Cohere API key is required for Cohere reranker")

        if (
            self.vector_db.type == VectorDBType.MILVUS_CLOUD
            and not self.vector_db.api_key
        ):
            errors.append("Milvus cloud API key is required for Milvus cloud")

        if self.vector_db.type == VectorDBType.MILVUS_CLOUD and not self.vector_db.host:
            errors.append("Milvus cloud host is required for Milvus cloud")

        # Validate hybrid search weights
        if self.search.type == SearchType.HYBRID:
            total_weight = self.search.dense_weight + self.search.sparse_weight
            if abs(total_weight - 1.0) > 0.001:
                errors.append(
                    f"Dense and sparse weights must sum to 1.0, got {total_weight}"
                )

        # Validate chunking settings
        if self.chunking.overlap >= self.chunking.chunk_size:
            errors.append("Chunk overlap must be less than chunk size")

        if self.chunking.strategy == ChunkingStrategy.CHONKIE:
            if self.chunking.min_chunk_size > self.chunking.max_chunk_size:
                errors.append("Min chunk size must be less than max chunk size")

        return errors


def load_config(config_path: Optional[Union[str, Path]] = None) -> VerbatimRAGConfig:
    """
    Load configuration from file or environment variables.

    Args:
        config_path: Path to YAML configuration file. If None, loads from environment.

    Returns:
        VerbatimRAGConfig: Loaded configuration

    Raises:
        ValueError: If configuration validation fails
    """
    if config_path:
        config = VerbatimRAGConfig.from_yaml(config_path)
    else:
        config = VerbatimRAGConfig.from_env()

    # Validate configuration
    errors = config.validate_config()
    if errors:
        raise ValueError(
            "Configuration validation failed:\n"
            + "\n".join(f"- {error}" for error in errors)
        )

    return config


def create_default_config() -> VerbatimRAGConfig:
    """Create a default configuration for local development"""
    return VerbatimRAGConfig(
        deployment_mode=DeploymentMode.LOCAL,
        dense_embedding=DenseEmbeddingConfig(
            model=DenseEmbeddingModel.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
        ),
        sparse_embedding=SparseEmbeddingConfig(
            model=SparseEmbeddingModel.SPLADE,
            model_name="naver/splade-v3",
            enabled=True,
        ),
        vector_db=VectorDBConfig(
            type=VectorDBType.MILVUS_LOCAL, db_path="./milvus_verbatim.db"
        ),
        reranker=RerankerConfig(type=RerankerType.NONE),
        document_processor=DocumentProcessorConfig(type=DocumentProcessor.DOCLING),
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.CHONKIE, chunk_size=512, overlap=50
        ),
        search=SearchConfig(type=SearchType.HYBRID, top_k=10),
        template=TemplateConfig(realtime_generation=False, cache_templates=True),
        verbatim_doc=VerbatimDOCConfig(enabled=True, parallel_processing=True),
        cache=CacheConfig(enabled=True, cache_embeddings=True, cache_results=True),
    )
