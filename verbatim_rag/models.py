"""
Compatibility re-exports for models from verbatim_core, plus local QueryRequest.
"""

from pydantic import BaseModel, Field
from verbatim_core.models import (
    QueryResponse,
    Highlight,
    Citation,
    DocumentWithHighlights,
    StructuredAnswer,
    StreamingResponse,
    StreamingResponseType,
)


class QueryRequest(BaseModel):
    """Request model for the query endpoint (kept in rag for API compatibility)."""

    question: str
    num_docs: int = Field(default=5, ge=1)


__all__ = [
    "QueryRequest",
    "QueryResponse",
    "Highlight",
    "Citation",
    "DocumentWithHighlights",
    "StructuredAnswer",
    "StreamingResponse",
    "StreamingResponseType",
]
