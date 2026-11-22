"""The settings of a Transformer call, apart from the Transformer so the templates can hold one."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, NonNegativeInt, field_validator

from verbatim_core.plan import PlanConfig  # noqa: TC001  # pydantic reads the annotation at runtime


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
    # Leave out a span whose words an earlier span already has (the same sentence in two
    # textbooks) before choosing the shown spans. The answer's `spans` keep every span.
    drop_repeated_spans: bool = False
    # Contextual mode only: the LLM writes an answer plan (ADR 0007, ADR 0008) instead of
    # placeholder text, with these settings. None: placeholder templates.
    answer_plan: PlanConfig | None = None

    @field_validator("citation_format")
    @classmethod
    def validate_citation_format(cls, citation_format: str) -> str:
        try:
            citation_format.format(number=1, span_id="1")
        except (KeyError, IndexError, ValueError) as e:
            msg = "citation_format may use only {number} and {span_id}"
            raise ValueError(msg) from e
        return citation_format
