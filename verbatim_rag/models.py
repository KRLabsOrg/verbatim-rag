"""
Data models for structured responses from the Verbatim RAG system.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Highlight(BaseModel):
    """A highlighted span of text in a document."""

    text: str = Field(..., min_length=1)
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_end_after_start(self) -> "Highlight":
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class DocumentWithHighlights(BaseModel):
    """A document with highlighted spans."""

    content: str = Field(..., min_length=1)
    highlights: list[Highlight] = Field(default_factory=list)


class Citation(BaseModel):
    """A citation linking a text span to its source document."""

    text: str = Field(..., min_length=1)
    doc_index: int = Field(..., ge=0)
    highlight_index: int = Field(..., ge=0)


class StructuredAnswer(BaseModel):
    """A structured answer with citations."""

    text: str = Field(..., min_length=1)
    citations: list[Citation] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """The complete result of a query, including the answer and source documents."""

    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    structured_answer: StructuredAnswer
    documents: list[DocumentWithHighlights] = Field(..., min_items=0)

    class Config:
        arbitrary_types_allowed = True


class QueryRequest(BaseModel):
    """Request model for the query endpoint."""

    question: str
    num_docs: int = Field(default=5, ge=1)


class StreamingResponseType(Enum):
    DOCUMENTS = "documents"
    HIGHLIGHTS = "highlights"
    ANSWER = "answer"


class StreamingResponse(BaseModel):
    """Streaming response for progressive updates."""

    type: StreamingResponseType
    data: Any
    done: bool = False
