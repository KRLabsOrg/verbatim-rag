"""
Types shared by the stateless interfaces of verbatim_core.

Where they live is provisional: they may move to a package shared with other libraries.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator


class HasText(Protocol):
    text: str


class Chunk(BaseModel):
    """A piece of source text with an id, the unit spans are extracted from."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    id: str = Field(min_length=1)
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_start: int | None = Field(default=None, ge=0)  # position in the source document
    source_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_source_offsets_in_order(self) -> Chunk:
        if (
            self.source_start is not None
            and self.source_end is not None
            and self.source_end < self.source_start
        ):
            raise ValueError("source_end must not precede source_start")
        return self


class Span(BaseModel):
    """A part of a text, given by offsets in Unicode code points.

    The type is shared with other libraries, whose spans point into other texts. In this library
    the text is always a chunk, and extractors return an `ExtractedSpan`.
    """

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    start: int = Field(ge=0)  # inclusive
    end: int  # exclusive
    text: str  # always equal to the slice the offsets point into
    score: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_text_fits_offsets(self) -> Span:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        if len(self.text) != self.end - self.start:
            raise ValueError("text must be exactly as long as the span from start to end")
        return self


class ExtractedSpan(Span):
    """A span of a chunk, as an extractor returns it. Its offsets are relative to the chunk."""

    chunk_id: str = Field(min_length=1)


class Usage(BaseModel):
    """What one model or service did during a call."""

    # Older pydantic 2 releases reserve the "model_" prefix for their own methods.
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True, protected_namespaces=())

    model_id: str | None  # None for work without a model, or with a model the code doesn't name
    model_version: str | None
    units: dict[str, NonNegativeInt]  # for example {"chunks": 12, "input_tokens": 3410}


class ResultWarning(BaseModel):
    """Something that made a result less complete without making it wrong."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    code: str = Field(min_length=1)  # stable snake_case identifier, for example "quote_not_found"
    message: str


class Extraction(BaseModel):
    """The spans an extractor found in a set of chunks."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    spans: list[ExtractedSpan] = Field(default_factory=list)
    usage: list[Usage] = Field(default_factory=list)
    warnings: list[ResultWarning] = Field(default_factory=list)


class AnswerCitation(BaseModel):
    """A numbered citation in an answer. It points at its span by chunk id and offsets."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    number: int = Field(ge=1)
    type: Literal["display", "reference"]  # shown in the answer, or cited for reference only
    chunk_id: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int


class Answer(BaseModel):
    """A cited answer, written from the spans extracted from a set of chunks."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    text: str
    citations: list[AnswerCitation] = Field(default_factory=list)
    spans: list[ExtractedSpan] = Field(default_factory=list)
    usage: list[Usage] = Field(default_factory=list)
    warnings: list[ResultWarning] = Field(default_factory=list)
