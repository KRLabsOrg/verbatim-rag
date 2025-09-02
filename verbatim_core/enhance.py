from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict, Iterable, Mapping

from .transform import VerbatimTransform


def _to_context_dicts(obj: Any) -> list[dict]:
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
    """Decorator to enhance existing RAG functions with verbatim transformation.

    The wrapped function can return:
    - dict with keys: 'question'?, 'answer'?, 'context' or 'sources'
    - tuple: (answer, sources)
    - or just 'sources' (context list/dict) if no answer
    Currently, provided 'answer' is ignored; verbatim answer is derived from context.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            question = kwargs.get("question") or (args[0] if args else "")
            answer = None
            context: Iterable[Dict[str, Any]] = []

            if isinstance(result, dict):
                answer = result.get("answer")
                context = result.get("context") or result.get("sources") or []
            elif isinstance(result, (list, tuple)):
                if len(result) == 2:
                    answer, context = result
                else:
                    context = result
            else:
                context = result

            context_dicts = _to_context_dicts(context)

            vt = VerbatimTransform(max_display_spans=max_display_spans)
            resp = vt.transform(
                question=question or "", context=context_dicts, answer=answer
            )
            return resp

        return wrapper

    return decorator
