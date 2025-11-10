"""
Simple embedding providers for the Verbatim RAG system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict
import logging

import numpy as np

logger = logging.getLogger(__name__)


class DenseEmbeddingProvider(ABC):
    """Base dense embedding provider interface."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass


class SparseEmbeddingProvider(ABC):
    """Base sparse embedding provider interface."""

    @abstractmethod
    def embed_text(self, text: str) -> Dict[int, float]:
        """Embed a single text string to sparse vector."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[Dict[int, float]]:
        """Embed a batch of text strings to sparse vectors."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get vocabulary size."""
        pass


class SentenceTransformersProvider(DenseEmbeddingProvider):
    """Local SentenceTransformers provider."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Loaded SentenceTransformers model: {self.model_name}")
        except ImportError:
            raise ImportError("pip install sentence-transformers")

    def embed_text(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


class OpenAIProvider(DenseEmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self, model_name: str = "text-embedding-ada-002"):
        self.model_name = model_name
        self._setup_client()

    def _setup_client(self):
        try:
            import openai
            import os

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")

            self.client = openai.OpenAI(api_key=api_key)
            logger.info(f"Initialized OpenAI provider: {self.model_name}")
        except ImportError:
            raise ImportError("pip install openai")

    def embed_text(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]

    def get_dimension(self) -> int:
        return 1536 if "ada-002" in self.model_name else 1536


class SpladeProvider(SparseEmbeddingProvider):
    """SPLADE sparse embedding provider."""

    def __init__(self, model_name: str = "naver/splade-v3", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SparseEncoder

            self.model = SparseEncoder(self.model_name, device=self.device)
            logger.info(f"Loaded SPLADE model: {self.model_name}")
        except ImportError:
            raise ImportError("pip install sentence-transformers")

    def embed_text(self, text: str) -> Dict[int, float]:
        """Get sparse embedding as dict {token_id: weight}."""
        sparse_embedding = self.model.encode([text])[0]
        # Convert to sparse dict format
        sparse_dict = {}
        for idx, weight in enumerate(sparse_embedding):
            if abs(weight) > 1e-6:  # Filter out very small values
                sparse_dict[idx] = float(weight)
        return sparse_dict

    def embed_batch(self, texts: List[str]) -> List[Dict[int, float]]:
        """Get batch of sparse embeddings."""
        logging.debug("calling model.encode...")
        embeddings = self.model.encode(texts)
        logging.debug(f"{type(embeddings)=}")
        logging.debug(f"{embeddings.shape=}")
        logging.debug("done, running cutoff...")
        result = []

        if self.device == "cpu":
            embeddings = embeddings.to_dense().numpy()
        else:
            embeddings = embeddings.to_dense().cpu().numpy()
        for embedding in embeddings:
            indices = np.nonzero(embedding)[0]
            result.append({int(idx): float(embedding[idx]) for idx in indices})

        logging.debug("done, returning result.")
        return result

    def get_dimension(self) -> int:
        return 30522  # BERT vocab size
