"""Span extraction with decision models, such as Vela 2.0 (ADR 0009).

A decision model answers typed questions about a request and its context in one call,
``system_one(state, questions)``. A span question returns spans of the context with offsets in
Unicode code points and a probability each. `DecisionModelExtractor` asks one span question per
chunk: which spans of the chunk answer the question.

The model runs in memory (`LocalDecisionModel`, with transformers) or on a server
(`RemoteDecisionModel`, the ``POST /v1/systemone`` protocol of Vela's ``vela2_serve.py``). Any other
backend is an object with a ``system_one`` method of the same shape, and optionally an async
``asystem_one``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import httpx
from pydantic import Field
from typing_extensions import override

from verbatim_core._checks import _call_config, _check_unique_ids
from verbatim_core._loading import _load_errors_translated, _log_model_loaded
from verbatim_core.errors import ConfigurationError, InvalidInputError, UpstreamError
from verbatim_core.extractors import ExtractConfig, SpanExtractor, _best_device, _commit, _model_span, _model_usage
from verbatim_core.types import Chunk, ExtractedSpan, Extraction, ResultWarning

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "DecisionModel",
    "DecisionModelConfig",
    "DecisionModelExtractor",
    "LocalDecisionModel",
    "RemoteDecisionModel",
]

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "vllm-sr/Vela-2.0-4B"
_QUESTION_ID = "evidence"


class DecisionModelConfig(ExtractConfig):
    """Settings of `DecisionModelExtractor`."""

    # The span question asked over each chunk, with the user's question as the request
    instructions: str = Field(default="Which spans of the context answer the request?", min_length=1)
    label: str = Field(default="answer", min_length=1)
    description: str = Field(default="text that answers the request", min_length=1)
    # The lowest probability of a span. None: the model's calibrated threshold (Vela 2.0's broad
    # span head: 0.55).
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


@runtime_checkable
class DecisionModel(Protocol):
    """A decision model, in memory or behind a server: what `DecisionModelExtractor` calls."""

    model_id: str | None
    model_version: str | None

    def system_one(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        """Answer the questions about the state, in the response shape of Vela's ``system_one``."""
        ...


class LocalDecisionModel:
    """A decision model loaded with transformers, called in this process.

    Calls take turns: one model runs one call at a time. For calls in parallel, load one model per
    worker or serve the model (`RemoteDecisionModel`).
    """

    def __init__(self, model: Any, *, model_id: str | None = None, model_version: str | None = None) -> None:  # noqa: ANN401  # remote code of the model
        """Wrap a loaded model that has ``system_one``, such as ``AutoModel.from_pretrained("vllm-sr/Vela-2.0-4B", trust_remote_code=True)``."""
        if not callable(getattr(model, "system_one", None)):
            msg = "The model has no system_one method. Load a decision model, such as vllm-sr/Vela-2.0-4B."
            raise ConfigurationError(msg, code="invalid_value")
        self.model = model
        self.model_id: str | None = model_id
        self.model_version: str | None = model_version
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls, model_id: str = DEFAULT_MODEL, device: str | None = None, *, revision: str | None = None
    ) -> LocalDecisionModel:
        """Download and load a decision model, or load it from a local path.

        The model's own code runs (``trust_remote_code``): load only models you trust, and pin
        ``revision`` to a commit for a reproducible deployment.

        :param model_id: A Hugging Face model id or the path of a local model
        :param device: ``"cuda"``, ``"mps"`` or ``"cpu"``, or None to choose the best available
        :param revision: A commit, branch or tag of the model on Hugging Face
        :raises ConfigurationError: If torch or transformers is missing, or the model can't be loaded
        """
        try:
            import torch  # noqa: PLC0415  # optional, heavy dependency
            import transformers  # noqa: PLC0415  # optional, heavy dependency
        except ImportError as e:
            msg = (
                "A local decision model needs torch and transformers, in the versions its model card "
                "names. Install them, or serve the model and use RemoteDecisionModel."
            )
            raise ConfigurationError(msg, code="missing_dependency") from e
        device = device or _best_device(torch)
        started = time.perf_counter()
        with _load_errors_translated(model_id):
            model = transformers.AutoModel.from_pretrained(model_id, trust_remote_code=True, revision=revision)
        model.to(device)
        model.eval()
        _log_model_loaded(logger, model_id, device, started)
        return cls(model, model_id=model_id, model_version=_commit(model) or revision)

    def system_one(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        """Answer the questions about the state."""
        with self._lock:
            return self.model.system_one(state, questions)

    async def asystem_one(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        """Answer the questions about the state, in a worker thread."""
        return await asyncio.to_thread(self.system_one, state, questions)


class RemoteDecisionModel:
    """A decision model behind a server that speaks ``POST /v1/systemone``.

    The request is ``{"model", "state", "questions"}`` and the response has the shape of
    ``system_one``, as Vela's ``vela2_serve.py`` serves it. A server with another protocol needs an
    object of its own with ``system_one``.
    """

    def __init__(  # noqa: PLR0913  # the settings of one HTTP client
        self,
        base_url: str,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        model_version: str | None = None,
        timeout: float = 60.0,
        path: str = "/v1/systemone",
        http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Set up the client. Nothing is sent before the first call.

        :param base_url: The server, for example ``"http://127.0.0.1:8001"``
        :param model: The name the server serves the model under
        :param api_key: Sent as a bearer token, when the server asks for one
        :param model_version: The commit of the served weights, for the usage of a call
        :param timeout: Seconds to wait for a response
        :param path: The path of the system_one endpoint
        :param http_client: The httpx client of `system_one`, for proxies, certificates or tests
        :param async_http_client: The httpx client of `asystem_one`
        """
        self.url = base_url.rstrip("/") + path
        self.model_id: str | None = model
        self.model_version: str | None = model_version
        self._headers = {} if api_key is None else {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout
        self._client = http_client or httpx.Client()
        self._async_client = async_http_client or httpx.AsyncClient()

    def system_one(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        """Answer the questions about the state, on the server."""
        try:
            response = self._client.post(self.url, **self._request(state, questions))
        except httpx.HTTPError as e:
            raise _translate_error(e) from e
        return _read(response)

    async def asystem_one(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        """Answer the questions about the state, on the server."""
        try:
            response = await self._async_client.post(self.url, **self._request(state, questions))
        except httpx.HTTPError as e:
            raise _translate_error(e) from e
        return _read(response)

    def _request(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        body = {"model": self.model_id, "state": state, "questions": questions}
        return {"json": body, "headers": self._headers, "timeout": self._timeout}


_HTTP_TOO_LARGE = 413
_HTTP_UNAUTHORIZED = {401, 403}
_HTTP_RETRY = {408, 429}
_HTTP_SERVER_ERROR = 500


def _translate_error(error: httpx.HTTPError) -> UpstreamError:
    if isinstance(error, httpx.TimeoutException):
        return UpstreamError(
            "The decision model server did not answer in time. Retry later, or give it a longer timeout.",
            code="timeout",
            service="decision_model",
            retryable=True,
        )
    return UpstreamError(
        "Could not connect to the decision model server. Check its base URL.",
        code="connection_failed",
        service="decision_model",
        retryable=True,
    )


def _read(response: httpx.Response) -> dict[str, Any]:
    """Return the body of a response, or raise the library's error for its status."""
    status = response.status_code
    if status in _HTTP_UNAUTHORIZED:
        msg = "The decision model server rejected the API key. Give RemoteDecisionModel a valid key."
        raise ConfigurationError(msg, code="authentication_failed")
    if status == _HTTP_TOO_LARGE:
        msg = "A chunk is too long for the decision model. Split the documents into shorter chunks."
        raise InvalidInputError(msg, code="input_too_long", param="chunks")
    if status in _HTTP_RETRY or status >= _HTTP_SERVER_ERROR:
        msg = f"The decision model server failed with status {status}. Retry later."
        raise UpstreamError(msg, code="server_error", service="decision_model", retryable=True)
    if response.is_error:
        # 422: the model name or the request doesn't match what the server serves
        msg = (
            f"The decision model server rejected the request with status {status}. Check the model "
            "name and that the server speaks the system_one protocol."
        )
        raise UpstreamError(msg, code="request_rejected", service="decision_model")
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        msg = "The decision model server sent a reply that can't be read. Check that it speaks the system_one protocol."
        raise UpstreamError(msg, code="invalid_reply", service="decision_model", retryable=True)
    return body


class DecisionModelExtractor(SpanExtractor[DecisionModelConfig]):
    """Extract spans with a decision model: one span question per chunk.

    The user's question is the request and the chunk the context. A span is kept when the model
    returns it above the threshold and its offsets hold the text the model returned; one that
    doesn't is dropped with the warning ``span_offsets_mismatch``. Its probability is the span's
    score.

    ``aextract`` asks about all chunks at once when the model has ``asystem_one``.
    """

    def __init__(self, model: DecisionModel, *, config: DecisionModelConfig | None = None) -> None:
        """Build an extractor on a decision model: a `LocalDecisionModel`, a `RemoteDecisionModel` or your own."""
        if not isinstance(model, DecisionModel):
            msg = "The model needs model_id, model_version and system_one(state, questions)."
            raise ConfigurationError(msg, code="invalid_value")
        self.model = model
        self.config = config or DecisionModelConfig()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        *,
        revision: str | None = None,
        config: DecisionModelConfig | None = None,
    ) -> DecisionModelExtractor:
        """Load a decision model in memory (`LocalDecisionModel.load`) and extract with it."""
        return cls(LocalDecisionModel.load(model_id, device, revision=revision), config=config)

    @classmethod
    def from_url(  # noqa: PLR0913  # the settings of one HTTP client
        cls,
        base_url: str,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        model_version: str | None = None,
        timeout: float = 60.0,
        config: DecisionModelConfig | None = None,
    ) -> DecisionModelExtractor:
        """Extract with a decision model on a server (`RemoteDecisionModel`)."""
        remote = RemoteDecisionModel(base_url, model, api_key=api_key, model_version=model_version, timeout=timeout)
        return cls(remote, config=config)

    @override
    def extract(
        self, question: str, chunks: Sequence[Chunk], *, config: DecisionModelConfig | None = None
    ) -> Extraction:
        """Extract the spans of the chunks that answer the question.

        :param question: The query or question
        :param chunks: Chunks with unique ids
        :param config: Replaces the extractor's default config for this call
        :return: The spans, with the usage and warnings of the call
        :raises InvalidInputError: If two chunks share an id, or a chunk is too long for the model
        :raises UpstreamError: If the model's server fails
        """
        config = _call_config(config, self.config, DecisionModelConfig)
        _check_unique_ids(chunks)
        asked = [chunk for chunk in chunks if chunk.text.strip()]
        replies = [self.model.system_one(*_request(question, chunk, config)) for chunk in asked]
        return self._extraction(asked, replies)

    @override
    async def aextract(
        self, question: str, chunks: Sequence[Chunk], *, config: DecisionModelConfig | None = None
    ) -> Extraction:
        """Async version of `extract`."""
        config = _call_config(config, self.config, DecisionModelConfig)
        _check_unique_ids(chunks)
        asked = [chunk for chunk in chunks if chunk.text.strip()]
        call = getattr(self.model, "asystem_one", None)
        if call is None:
            replies = [
                await asyncio.to_thread(self.model.system_one, *_request(question, chunk, config)) for chunk in asked
            ]
        else:
            replies = await asyncio.gather(*(call(*_request(question, chunk, config)) for chunk in asked))
        return self._extraction(asked, list(replies))

    def _extraction(self, asked: list[Chunk], replies: list[dict[str, Any]]) -> Extraction:
        spans: list[ExtractedSpan] = []
        mismatched = 0
        for chunk, reply in zip(asked, replies, strict=True):
            for found in _reply_spans(reply):
                offsets = _offsets(chunk.text, found)
                if offsets is None:
                    mismatched += 1
                    continue
                spans.append(_model_span(chunk, *offsets, found.get("probability", 0.0)))
        warnings = []
        if mismatched:
            message = f"Dropped {mismatched} spans whose offsets don't hold the text the decision model returned."
            warnings.append(ResultWarning(code="span_offsets_mismatch", message=message))
            logger.warning(
                "Dropped %d decision model spans with wrong offsets",
                mismatched,
                extra={"event": "fallback_used", "warning_code": "span_offsets_mismatch"},
            )
        usage = _model_usage(self.model.model_id, self.model.model_version, asked)
        return Extraction(spans=spans, usage=[usage], warnings=warnings)


def _request(question: str, chunk: Chunk, config: DecisionModelConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the state and the span question of one chunk."""
    span_question: dict[str, Any] = {
        "type": "span",
        "instructions": config.instructions,
        "over": "context",
        "criteria": {config.label: config.description},
    }
    if config.threshold is not None:
        span_question["threshold"] = config.threshold
    return {"request": question, "context": chunk.text}, {_QUESTION_ID: span_question}


def _reply_spans(reply: dict[str, Any]) -> list[dict[str, Any]]:
    spans = (reply.get("spans") or {}).get(_QUESTION_ID) or []
    if not isinstance(spans, list):
        msg = "The decision model's reply has no list of spans. Check that it is a system_one response."
        raise UpstreamError(msg, code="invalid_reply", service="decision_model")
    return [span for span in spans if isinstance(span, dict)]


def _offsets(text: str, found: dict[str, Any]) -> tuple[int, int] | None:
    """Return a span's offsets if they lie in the text, hold more than whitespace, and hold the returned text."""
    start, end, returned = found.get("start"), found.get("end"), found.get("text")
    if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text)):
        return None
    if not text[start:end].strip() or (returned is not None and text[start:end] != returned):
        return None
    return start, end
