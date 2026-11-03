"""Decorator for enhancing existing RAG functions with verbatim transformation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from functools import wraps
from typing import Any

from verbatim_core.transform import VerbatimTransform

_ANSWER_SOURCES_PAIR_LEN = 2


def _to_context_dicts(obj: object) -> list[dict]:
    """Best-effort conversion of sources/context to context dicts."""
    ctx: list[dict] = []
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
                },
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
        def wrapper(*args: object, **kwargs: object) -> object:  # decorator passthrough
            result = fn(*args, **kwargs)
            question = kwargs.get("question") or (args[0] if args else "")
            answer = None
            context: Iterable[dict[str, Any]] = []

            if isinstance(result, dict):
                answer = result.get("answer")
                context = result.get("context") or result.get("sources") or []
            elif isinstance(result, (list, tuple)):
                if len(result) == _ANSWER_SOURCES_PAIR_LEN:
                    answer, context = result
                else:
                    context = result
            else:
                context = result

            context_dicts = _to_context_dicts(context)

            vt = VerbatimTransform(max_display_spans=max_display_spans)
            return vt.transform(question=question or "", context=context_dicts, answer=answer)

        return wrapper

    return decorator
