"""Decorator for enhancing existing RAG functions with verbatim transformation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from functools import wraps
from typing import TYPE_CHECKING, Any

from verbatim_core._context import _default_transformer, _query_response
from verbatim_core.llm_client import LLMClient

if TYPE_CHECKING:
    from verbatim_core.models import QueryResponse

_ANSWER_SOURCES_PAIR_LEN = 2  # a wrapped function may return an (answer, sources) pair


def _to_context_dicts(obj: object) -> list[dict[str, Any]]:
    """Best-effort conversion of sources/context to context dicts."""
    ctx: list[dict[str, Any]] = []
    if obj is None:
        return ctx
    if isinstance(obj, Mapping):
        # Single dict
        data = dict(obj)
        if "content" in data or "text" in data:
            ctx.append(
                {
                    "content": data.get("content") or data.get("text"),
                    "title": data.get("title", ""),
                    "source": data.get("source", ""),
                    "metadata": data.get("metadata") or {},
                }
            )
        return ctx
    if isinstance(obj, (list, tuple)):
        for item in obj:
            ctx.extend(_to_context_dicts(item))
        return ctx
    # Fallback: treat as raw text
    if isinstance(obj, str) and obj.strip():
        ctx.append({"content": obj})
    return ctx


def verbatim_enhance(
    max_display_spans: int = 5,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Enhance existing RAG functions with verbatim transformation.

    The wrapped function can return:
    - dict with keys: 'question'?, 'answer'?, 'context' or 'sources'
    - tuple: (answer, sources)
    - or just 'sources' (context list/dict) if no answer
    Currently, provided 'answer' is ignored; verbatim answer is derived from context.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> QueryResponse:  # noqa: ANN401  # passes the wrapped function's arguments through
            result = fn(*args, **kwargs)
            question = kwargs.get("question") or (args[0] if args else "")
            context: Iterable[dict[str, Any]] = []

            if isinstance(result, dict):
                context = result.get("context") or result.get("sources") or []
            elif isinstance(result, (list, tuple)):
                if len(result) == _ANSWER_SOURCES_PAIR_LEN:
                    _answer, context = result
                else:
                    context = result
            else:
                context = result

            context_dicts = _to_context_dicts(context)

            transformer = _default_transformer(LLMClient(), max_display_spans)
            return _query_response(transformer, question or "", context_dicts)

        return wrapper

    return decorator
