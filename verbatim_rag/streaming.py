"""
StreamingRAG: a streamed query as dict events.

Deprecated: use VerbatimRAG.query_stream, which yields typed events and raises typed errors.
"""

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from verbatim_core import (
    AnswerCitation,
    Chunk,
    ExtractedSpan,
    InvalidInputError,
    SpansEvent,
    VerbatimError,
)
from verbatim_core._deprecation import deprecated
from verbatim_core.response_builder import _documents_and_citations

from .core import DocumentsEvent, QueryStreamEvent, VerbatimRAG

logger = logging.getLogger(__name__)

_USE_QUERY_STREAM = "VerbatimRAG.query_stream"


@deprecated("StreamingRAG", use=_USE_QUERY_STREAM)
class StreamingRAG:
    """`VerbatimRAG.query_stream` as dict events: documents, progress, highlights, then the
    answer, or an error."""

    def __init__(self, rag: VerbatimRAG):
        self.rag = rag

    @deprecated("StreamingRAG.stream_query", use=_USE_QUERY_STREAM)
    def stream_query(
        self,
        question: str,
        num_docs: Optional[int] = None,
        filter: Optional[str] = None,
        hybrid_weights: Optional[Dict[str, float]] = None,
        rrf_k: int = 60,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream a query as dict events.

        :param question: The user's question
        :param num_docs: The number of documents to retrieve
        :param filter: Optional vector-store filter expression
        :param hybrid_weights: Optional weights for hybrid search
        :param rrf_k: RRF constant for hybrid search
        :param search_params: Optional parameters for vector search (e.g. nprobe)
        :return: The events, as dicts with a "type"; a failure is the last event, of type "error"
        """
        return self._dict_events(
            question,
            num_docs,
            filter=filter,
            hybrid_weights=hybrid_weights,
            rrf_k=rrf_k,
            search_params=search_params,
        )

    @deprecated("StreamingRAG.stream_query_sync", use=_USE_QUERY_STREAM)
    def stream_query_sync(
        self, question: str, num_docs: Optional[int] = None, filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """The dict events of `stream_query`, as a list."""

        async def collect() -> List[Dict[str, Any]]:
            return [event async for event in self._dict_events(question, num_docs, filter=filter)]

        return asyncio.run(collect())

    def _dict_events(
        self, question: str, num_docs: Optional[int], **retrieval: Any
    ) -> AsyncIterator[Dict[str, Any]]:
        return _dict_events(self.rag.query_stream(question, k=num_docs, **retrieval))


async def _dict_events(events: AsyncIterator[QueryStreamEvent]) -> AsyncIterator[Dict[str, Any]]:
    """
    The dict events of a streamed query, which StreamingRAG and the API server's streaming route
    send: "documents", then "progress" with the time the extraction took, "highlights" and
    "answer", or "error" as the last event when the query fails.
    """
    chunks: Sequence[Chunk] = []
    extraction_started = time.perf_counter()
    try:
        async for event in events:
            if isinstance(event, DocumentsEvent):
                chunks = event.chunks
                yield {"type": "documents", "data": _documents(chunks)}
                extraction_started = time.perf_counter()
            elif isinstance(event, SpansEvent):
                elapsed_ms = int((time.perf_counter() - extraction_started) * 1000)
                yield {"type": "progress", "stage": "extraction_complete", "elapsed_ms": elapsed_ms}
                yield {"type": "highlights", "data": _documents(chunks, event.spans)}
            else:
                yield {"type": "answer", "data": event.response.model_dump(), "done": True}
    # The dict stream has no exceptions: the frontend and StreamingRAG's callers read a failure as
    # its last event.
    except Exception as error:
        yield _error_event(error)


def _documents(
    chunks: Sequence[Chunk], spans: Sequence[ExtractedSpan] = ()
) -> List[Dict[str, Any]]:
    """The documents of the response, before its answer is written: the chunks, with the spans
    highlighted as the response highlights them, since it cites every span once, in order."""
    citations = [
        AnswerCitation(
            number=number, type="reference", chunk_id=span.chunk_id, start=span.start, end=span.end
        )
        for number, span in enumerate(spans, start=1)
    ]
    documents, _ = _documents_and_citations(citations, chunks)
    return [document.model_dump() for document in documents]


def _error_event(error: Exception) -> Dict[str, Any]:
    """The failure as the last dict event, logged with its traceback, since no caller receives
    the exception. The event carries the error's code, and for an InvalidInputError its message,
    which tells the caller what to fix. The other failures are the server's to fix, from the
    log."""
    code = error.code if isinstance(error, VerbatimError) else "internal_error"
    logger.error("Streamed query failed: %s", code, exc_info=_as_logged(error))
    message = str(error) if isinstance(error, InvalidInputError) else code
    return {"type": "error", "error": message, "code": code, "done": True}


class _LoggedError(Exception):
    """An exception as the log shows it: its class and traceback, and only a VerbatimError's
    message, since another exception's message may quote the question or a chunk."""


def _as_logged(error: BaseException, seen: Optional[set] = None) -> _LoggedError:
    """The error and its causes as the log shows them."""
    seen = (seen or set()) | {id(error)}
    shown = type(error).__qualname__
    if isinstance(error, VerbatimError):
        shown = f"{shown}: {error}"
    logged = _LoggedError(shown).with_traceback(error.__traceback__)
    cause = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    if cause is not None and id(cause) not in seen:
        logged.__cause__ = _as_logged(cause, seen)
    return logged
