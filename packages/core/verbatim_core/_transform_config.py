"""The settings of a Transformer call, apart from the Transformer so the templates can hold one."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, NonNegativeInt, field_validator


class TransformConfig(BaseModel):
    """Settings of a `Transformer` call."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    # The first spans are shown in the answer, the rest are cited for reference only.
    max_display_spans: NonNegativeInt = 5
    # The Transformer rejects "structured": structured answers stay in VerbatimRAG (ADR 0004).
    template_mode: Literal["static", "contextual", "random", "question_specific", "structured"] = "static"
    citation_mode: Literal["inline", "hidden"] = "inline"  # citation markers in the text, or none
    citation_format: str = "[{number}]"  # the marker, with {number} and {span_id}
    seed: int | None = None  # random template mode chooses by the question and this seed

    @field_validator("citation_format")
    @classmethod
    def validate_citation_format(cls, citation_format: str) -> str:
        try:
            citation_format.format(number=1, span_id="1")
        except (KeyError, IndexError, ValueError) as e:
            msg = "citation_format may use only {number} and {span_id}"
            raise ValueError(msg) from e
        return citation_format
