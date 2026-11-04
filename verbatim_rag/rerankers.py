"""
Reranker interfaces and provider adapters for VerbatimRAG.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import List

from verbatim_core.errors import ConfigurationError, UpstreamError, VerbatimError

from verbatim_rag.vector_stores.base import SearchResult


class Reranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(self, question: str, results: List[SearchResult]) -> List[SearchResult]:
        raise NotImplementedError

    async def rerank_async(self, question: str, results: List[SearchResult]) -> List[SearchResult]:
        return await asyncio.to_thread(self.rerank, question, results)


class BaseReranker(Reranker):
    """Base helper for rerankers with shared options."""

    def __init__(self, rerank_k: int = 50, text_field: str = "text"):
        self.rerank_k = rerank_k
        self.text_field = text_field

    def _split_results(self, results: List[SearchResult]):
        head = results[: self.rerank_k]
        tail = results[self.rerank_k :]
        return head, tail

    def _get_texts(self, results: List[SearchResult]) -> List[str]:
        if self.text_field == "enhanced_text":
            return [r.enhanced_text or r.text for r in results]
        return [r.text for r in results]


class CohereReranker(BaseReranker):
    """Cohere reranker adapter."""

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-english-v3.0",
        rerank_k: int = 50,
        text_field: str = "text",
    ):
        super().__init__(rerank_k=rerank_k, text_field=text_field)
        try:
            import cohere
        except ImportError as exc:
            raise ImportError("pip install cohere") from exc
        self.client = cohere.Client(api_key)
        self.model = model

    def rerank(self, question: str, results: List[SearchResult]) -> List[SearchResult]:
        head, tail = self._split_results(results)
        if not head:
            return results
        texts = self._get_texts(head)
        # The Cohere SDK raises its errors with the HTTP status, and httpx's when no reply came.
        # Any other error is a bug, which propagates.
        try:
            response = self.client.rerank(model=self.model, query=question, documents=texts)
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status is None and type(e).__module__.split(".")[0] != "httpx":
                raise
            raise _service_error("Cohere", status, timed_out="Timeout" in type(e).__name__) from e
        try:
            ranked = [head[item.index] for item in response.results]
        except (AttributeError, IndexError, TypeError) as e:
            raise _invalid_reply("Cohere") from e
        return ranked + tail


class JinaReranker(BaseReranker):
    """Jina reranker adapter (HTTP API)."""

    def __init__(
        self,
        api_key: str,
        model: str = "jina-reranker-v1-base-en",
        base_url: str = "https://api.jina.ai/v1/rerank",
        rerank_k: int = 50,
        text_field: str = "text",
    ):
        super().__init__(rerank_k=rerank_k, text_field=text_field)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def rerank(self, question: str, results: List[SearchResult]) -> List[SearchResult]:
        head, tail = self._split_results(results)
        if not head:
            return results
        try:
            import requests
        except ImportError as exc:
            raise ImportError("pip install requests") from exc

        payload = {
            "model": self.model,
            "query": question,
            "documents": self._get_texts(head),
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = requests.post(self.base_url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as e:
            status = e.response.status_code if e.response is not None else None
            timed_out = isinstance(e, requests.Timeout)
            raise _service_error("Jina", status, timed_out=timed_out) from e
        try:
            ranked = [head[item["index"]] for item in resp.json().get("results", [])]
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as e:
            raise _invalid_reply("Jina") from e
        return ranked + tail


class SentenceTransformersReranker(BaseReranker):
    """Local cross-encoder reranker using sentence-transformers."""

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        rerank_k: int = 50,
        text_field: str = "text",
    ):
        super().__init__(rerank_k=rerank_k, text_field=text_field)
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("pip install sentence-transformers") from exc
        self.model = CrossEncoder(model, device=device, trust_remote_code=True)

    def rerank(self, question: str, results: List[SearchResult]) -> List[SearchResult]:
        head, tail = self._split_results(results)
        if not head:
            return results
        texts = self._get_texts(head)
        pairs = [(question, text) for text in texts]
        scores = self.model.predict(pairs)
        ranked = [r for _, r in sorted(zip(scores, head), reverse=True)]
        return ranked + tail


class JinaV3Reranker(BaseReranker):
    """Jina V3 reranker using transformers."""

    def __init__(
        self,
        model: str = "jinaai/jina-reranker-v3",
        rerank_k: int = 50,
        text_field: str = "text",
    ):
        super().__init__(rerank_k=rerank_k, text_field=text_field)
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("pip install transformers") from exc
        self.model = AutoModel.from_pretrained(
            model,
            dtype="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def rerank(self, question: str, results: List[SearchResult]) -> List[SearchResult]:
        head, tail = self._split_results(results)
        if not head:
            return results
        texts = self._get_texts(head)
        results = self.model.rerank(question, texts, top_n=self.rerank_k)
        return [head[item["index"]] for item in results]


def _service_error(service: str, status: int | None, *, timed_out: bool = False) -> VerbatimError:
    """The error of a failed rerank call, by the HTTP status of the reply when there is one.

    A rejected key is a ConfigurationError. Every other failure is an UpstreamError, which
    VerbatimRAG answers by keeping the retrieval order.
    """
    if status == 401:
        return ConfigurationError(
            f"The {service} reranker rejected the API key. Check the key.",
            code="authentication_failed",
        )
    if status == 403:
        return ConfigurationError(
            f"The {service} reranker refused the request. Check the key's permissions.",
            code="permission_denied",
        )
    if status == 404:
        return ConfigurationError(
            f"The {service} reranker doesn't know the model, or the address is wrong. Check both.",
            code="model_not_found",
        )
    if status == 429:
        return UpstreamError(
            f"The {service} reranker limited the rate of requests. Retry later.",
            code="rate_limited",
            service="reranker",
            retryable=True,
        )
    if status is not None and status < 500:
        return UpstreamError(
            f"The {service} reranker rejected the request (HTTP {status}). Check the model and "
            "the number of results reranked.",
            code="request_rejected",
            service="reranker",
        )
    if status is not None:
        return UpstreamError(
            f"The {service} reranker failed (HTTP {status}). Retry later.",
            code="server_error",
            service="reranker",
            retryable=True,
        )
    return UpstreamError(
        f"The {service} reranker did not answer. Check the connection, and retry.",
        code="timeout" if timed_out else "connection_failed",
        service="reranker",
        retryable=True,
    )


def _invalid_reply(service: str) -> UpstreamError:
    """The error of a reply that can't be read, which a retry may fix, as for the LLM."""
    return UpstreamError(
        f"The {service} reranker sent a reply that can't be read. Retry, or check the model and "
        "the service's address.",
        code="invalid_reply",
        service="reranker",
        retryable=True,
    )
