"""
Simplified configuration system for VerbatimRAG.
Only includes configurations that are actually used in the codebase.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel, Field, validator


class DenseEmbeddingModel(str, Enum):
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class SparseEmbeddingModel(str, Enum):
    SPLADE = "splade"


class VectorDBType(str, Enum):
    MILVUS_LOCAL = "milvus_local"
    MILVUS_CLOUD = "milvus_cloud"


class DenseEmbeddingConfig(BaseModel):
    """Configuration for dense embedding models"""

    model: DenseEmbeddingModel = DenseEmbeddingModel.SENTENCE_TRANSFORMERS
    model_name: Optional[str] = "all-MiniLM-L6-v2"
    api_key: Optional[str] = None
    device: str = "cpu"

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


class VerbatimRAGConfig(BaseModel):
    """Main configuration for VerbatimRAG system"""

    # Component configurations
    dense_embedding: DenseEmbeddingConfig = Field(default_factory=DenseEmbeddingConfig)
    sparse_embedding: SparseEmbeddingConfig = Field(
        default_factory=SparseEmbeddingConfig
    )
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "VerbatimRAGConfig":
        """Load configuration from YAML file"""
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file"""
        with open(output_path, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False)


def create_default_config() -> VerbatimRAGConfig:
    """Create a default configuration for local development"""
    return VerbatimRAGConfig(
        dense_embedding=DenseEmbeddingConfig(
            model=DenseEmbeddingModel.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
        ),
        sparse_embedding=SparseEmbeddingConfig(
            model=SparseEmbeddingModel.SPLADE,
            model_name="naver/splade-v3",
            enabled=False,
        ),
        vector_db=VectorDBConfig(
            type=VectorDBType.MILVUS_LOCAL, db_path="./milvus_verbatim.db"
        ),
    )


def load_config(config_path: Optional[Union[str, Path]] = None) -> VerbatimRAGConfig:
    """
    Load configuration from file or create default.

    Args:
        config_path: Path to YAML configuration file. If None, creates default.

    Returns:
        VerbatimRAGConfig: Loaded configuration
    """
    if config_path:
        return VerbatimRAGConfig.from_yaml(config_path)
    else:
        return create_default_config()
