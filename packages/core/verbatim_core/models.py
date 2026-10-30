"""Pydantic models for verbatim_core (RAG-agnostic)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Highlight(BaseModel):
    """A highlighted character span within a document."""

    text: str = Field(..., min_length=1)
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_end_after_start(self) -> Highlight:
        """Ensure the highlight ends after it starts."""
        if self.end <= self.start:
            msg = "end must be greater than start"
            raise ValueError(msg)
        return self


class DocumentWithHighlights(BaseModel):
    """A document together with its highlighted spans."""

    content: str = Field(..., min_length=1)
    highlights: list[Highlight] = Field(default_factory=list)
    title: str = Field(default="")
    source: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """A citation pointing to a highlight in a document."""

    text: str = Field(..., min_length=1)
    doc_index: int = Field(..., ge=0)
    highlight_index: int = Field(..., ge=0)
    number: int | None = Field(default=None, ge=1)
    type: str | None = Field(default=None)


class StructuredAnswer(BaseModel):
    """Answer text with its citations."""

    text: str = Field(..., min_length=1)
    citations: list[Citation] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """Full response to a query."""

    model_config = {"arbitrary_types_allowed": True}

    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    structured_answer: StructuredAnswer
    documents: list[DocumentWithHighlights] = Field(default_factory=list)


class StreamingResponseType(Enum):
    """Kinds of streaming response chunks."""

    DOCUMENTS = "documents"
    HIGHLIGHTS = "highlights"
    ANSWER = "answer"


class StreamingResponse(BaseModel):
    """A single chunk of a streaming response."""

    type: StreamingResponseType
    data: Any
    done: bool = False
