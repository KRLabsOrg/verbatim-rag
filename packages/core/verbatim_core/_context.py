"""Answering context items, the input of VerbatimTransform, verbatim_enhance and the CLI, with a
Transformer, as a QueryResponse."""

from __future__ import annotations

from typing import Any, Iterable, List

from .errors import InvalidInputError
from .extractors import LLMSpanExtractor
from .llm_client import LLMClient
from .models import QueryResponse
from .response_builder import ResponseBuilder
from .templates import TemplateManager
from .transformer import TransformConfig, Transformer
from .types import Chunk


def _chunks_from_context(context: Iterable[Any]) -> List[Chunk]:
    """Chunks with positional ids (ctx_0, ctx_1, ...) from context items: dicts with "content" or
    "text", and optionally "title", "source" and "metadata", or objects with a text attribute.

    :raises InvalidInputError: If an item has another shape
    """
    chunks = []
    for position, item in enumerate(context):
        param = f"context[{position}]"
        if not isinstance(item, dict) and isinstance(getattr(item, "text", None), str):
            text, metadata = item.text, getattr(item, "metadata", None) or {}
        elif isinstance(item, dict):
            if "content" not in item and "text" not in item:
                raise InvalidInputError(
                    f"{param} has no 'content' or 'text'. Put the text of each context item in "
                    "one of them.",
                    code="missing_field",
                    param=param,
                )
            text = item.get("content") or item.get("text")
            _check_metadata(item.get("metadata") or {}, param)
            metadata = {
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                **(item.get("metadata") or {}),
            }
        else:
            raise InvalidInputError(
                f"{param} must be a dict with 'content' or 'text', or an object with a text "
                "attribute.",
                code="invalid_value",
                param=param,
            )
        if not isinstance(text, str) or not text:
            raise InvalidInputError(
                f"The text of {param} must be a non-empty string.",
                code="invalid_value",
                param=param,
            )
        _check_metadata(metadata, param)
        chunks.append(Chunk(id=f"ctx_{position}", text=text, metadata=metadata))
    return chunks


def _check_metadata(metadata: Any, param: str) -> None:
    """The metadata of a context item: a dict with string keys, whose title and source, which the
    response shows, are strings."""
    if not isinstance(metadata, dict) or not all(isinstance(key, str) for key in metadata):
        raise InvalidInputError(
            f"The metadata of {param} must be a dict with string keys.",
            code="invalid_value",
            param=f"{param}.metadata",
        )
    for field in ("title", "source"):
        if not isinstance(metadata.get(field, ""), str):
            raise InvalidInputError(
                f"The {field} of {param} must be a string.",
                code="invalid_value",
                param=f"{param}.{field}",
            )


def _default_transformer(llm_client: LLMClient, max_display_spans: int) -> Transformer:
    """The transformer VerbatimTransform builds without arguments: LLM extraction and contextual
    templates."""
    return Transformer(
        LLMSpanExtractor(llm_client),
        TemplateManager(llm_client=llm_client),
        config=TransformConfig(template_mode="contextual", max_display_spans=max_display_spans),
    )


def _query_response(
    transformer: Transformer,
    question: str,
    context: Iterable[Any],
    config: TransformConfig | None = None,
) -> QueryResponse:
    """Answer the question from the context items, as a QueryResponse."""
    chunks = _chunks_from_context(context)
    answer = transformer.transform(question, chunks, config=config)
    return ResponseBuilder().build_from_answer(question, answer, chunks)


async def _aquery_response(
    transformer: Transformer,
    question: str,
    context: Iterable[Any],
    config: TransformConfig | None = None,
) -> QueryResponse:
    """Async version of `_query_response`."""
    chunks = _chunks_from_context(context)
    answer = await transformer.atransform(question, chunks, config=config)
    return ResponseBuilder().build_from_answer(question, answer, chunks)
