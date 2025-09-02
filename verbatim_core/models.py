"""
Pydantic models for verbatim_core (RAG-agnostic).
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, model_validator


class Highlight(BaseModel):
    text: str = Field(..., min_length=1)
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_end_after_start(self) -> "Highlight":
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class DocumentWithHighlights(BaseModel):
    content: str = Field(..., min_length=1)
    highlights: list[Highlight] = Field(default_factory=list)
    title: str = Field(default="")
    source: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    text: str = Field(..., min_length=1)
    doc_index: int = Field(..., ge=0)
    highlight_index: int = Field(..., ge=0)
    number: int | None = Field(default=None, ge=1)
    type: str | None = Field(default=None)


class StructuredAnswer(BaseModel):
    text: str = Field(..., min_length=1)
    citations: list[Citation] = Field(default_factory=list)


class QueryResponse(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    structured_answer: StructuredAnswer
    documents: list[DocumentWithHighlights] = Field(..., min_items=0)

    class Config:
        arbitrary_types_allowed = True


class StreamingResponseType(Enum):
    DOCUMENTS = "documents"
    HIGHLIGHTS = "highlights"
    ANSWER = "answer"


class StreamingResponse(BaseModel):
    type: StreamingResponseType
    data: Any
    done: bool = False
