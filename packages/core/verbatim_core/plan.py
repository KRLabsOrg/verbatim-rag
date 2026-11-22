"""Answer plans: what an answer shows, validated in code (ADR 0007, ADR 0008).

An answer plan says which spans an answer shows, in which order and under which labels. It holds
meaning only: no span text and no layout. Items refer to spans by number, from 1. `verbatim_core.render`
lays a plan out as markdown and fills it with the spans.

A plan comes from an LLM (`plan_from_llm`, through structured output) or from the developer. Either
way `validate_plan` checks it against the spans of the call: unknown span numbers are dropped, a
span is used at most once, empty items and sections are dropped, and the number of shown items is
capped. `fallback_plan` is the deterministic plan used when nothing valid remains.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = [
    "AnswerPlan",
    "FieldItem",
    "LLMPlan",
    "PlanConfig",
    "PlanSection",
    "PlanText",
    "QuoteItem",
    "ReferencesItem",
    "RestItem",
    "distinct_spans",
    "fallback_plan",
    "field_items",
    "fill_fields",
    "plan_candidates",
    "plan_from_llm",
    "renumber_plan",
    "shows_spans",
    "validate_plan",
]


class _PlanModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class QuoteItem(_PlanModel):
    """Spans that say the same thing: the first is shown, the others are cited next to it."""

    kind: Literal["quote"] = "quote"
    spans: list[PositiveInt] = Field(min_length=1)


class RestItem(_PlanModel):
    """The shown spans no other item uses, in order, at most ``max``: the plan form of [DISPLAY_SPANS]."""

    kind: Literal["rest"] = "rest"
    max: PositiveInt | None = None


class ReferencesItem(_PlanModel):
    """Citation markers of spans the text doesn't show: the plan form of [CITATION_REFS].

    Without ``spans``, the spans the rest of the answer doesn't use.
    """

    kind: Literal["references"] = "references"
    spans: list[PositiveInt] | None = None


class FieldItem(_PlanModel):
    """A named question whose answer is extracted for this item, such as one row of a checklist.

    The Transformer extracts ``question`` over the call's chunks and fills ``spans`` with the
    field's spans, by their place in `Answer.spans` (ADR 0010). A plan you write leaves it out.
    """

    kind: Literal["field"] = "field"
    name: str = Field(min_length=1)
    question: str = Field(min_length=1)
    required: bool = False  # an empty required field shows PlanText.no_evidence
    max_spans: PositiveInt = 1
    spans: list[PositiveInt] | None = None  # filled by the Transformer; [] when nothing was found


PlanItem = Annotated[QuoteItem | RestItem | ReferencesItem | FieldItem, Field(discriminator="kind")]


class PlanSection(_PlanModel):
    """A group of items under an optional label."""

    label: str | None = None
    items: list[PlanItem] = Field(default_factory=list)


class AnswerPlan(_PlanModel):
    """Which spans an answer shows, in which order and under which labels."""

    intro: str | None = None
    sections: list[PlanSection] = Field(default_factory=list)


class PlanText(_PlanModel):
    """The language of plan answers: its name, for the LLM, and the fixed texts of the renderer."""

    # The bundled plan prompt asks for the intro and labels in this language; None: the quotes'
    language: str | None = None
    # Introduces the fallback plan, when the LLM couldn't write a plan
    fallback_intro: str = "The sources contain these passages:"
    # Before the numbers of cited spans the answer doesn't show, with PlanConfig(show_references=True)
    references_label: str = "Further sources:"
    # A rest item with no span to show
    no_spans: str = "No relevant information found in the provided documents."
    # A required field item with no span (field items are filled by a later release)
    no_evidence: str = "No supporting passage found."


class PlanConfig(_PlanModel):
    """The settings of answers rendered from plans: `TransformConfig(answer_plan=PlanConfig())`."""

    # The LLM chooses from at most this many spans, without repeats, chunks taking turns
    max_candidates: PositiveInt = 20
    # Items that show a span
    max_items: PositiveInt = 6
    # Supporting spans an item may cite next to its shown span
    max_support: NonNegativeInt = 2
    # Show quotes without the layout of their source: each line's spaces collapsed and its
    # indentation removed, paragraphs as hard line breaks. Citations keep the exact span text.
    compact_quotes: bool = True
    # List the cited spans the answer doesn't show after it, under PlanText.references_label
    show_references: bool = False
    # The plan of every answer, written by the developer (ADR 0010): the LLM writes none, in any
    # template mode. None: the LLM writes the plan, in contextual mode.
    plan: AnswerPlan | None = None
    # The fixed texts of the answer. None: the contextual strategy's plan_text, else English.
    text: PlanText | None = None


class _LLMItem(_PlanModel):
    spans: list[int]


class _LLMSection(_PlanModel):
    label: str | None
    items: list[_LLMItem]


class LLMPlan(_PlanModel):
    """The plan an LLM writes: the JSON schema of the structured output asks for this shape."""

    intro: str | None
    sections: list[_LLMSection]


def plan_from_llm(plan: LLMPlan) -> AnswerPlan:
    """Return the answer plan of an LLM's plan, not yet validated against the spans."""
    sections = [
        PlanSection(
            label=_text_or_none(section.label),
            items=[
                QuoteItem(spans=[n for n in item.spans if n > 0])
                for item in section.items
                if any(n > 0 for n in item.spans)
            ],
        )
        for section in plan.sections
    ]
    return AnswerPlan(intro=_text_or_none(plan.intro), sections=sections)


def _text_or_none(text: str | None) -> str | None:
    return (text.strip() or None) if text is not None else None


def validate_plan(
    plan: AnswerPlan, span_count: int, max_items: int | None = None, max_support: int | None = None
) -> AnswerPlan:
    """Return the plan with what can't be shown taken out.

    Span numbers above ``span_count`` are dropped, and a span is shown or supports at most once
    across the plan, the first time. Quote items left without spans, and sections left without
    items, are dropped. At most ``max_items`` items show spans: quote items count one each, and a
    rest item gets the places left. A quote item keeps at most ``max_support`` supporting spans
    after its shown one; the others stay unused. Field items stay.
    """
    used: set[int] = set()
    shown = 0
    sections = []
    for section in plan.sections:
        items: list[QuoteItem | RestItem | ReferencesItem | FieldItem] = []
        for item in section.items:
            if isinstance(item, QuoteItem):
                numbers = [n for n in dict.fromkeys(item.spans) if n <= span_count and n not in used]
                if max_support is not None:
                    numbers = numbers[: max_support + 1]
                if not numbers or (max_items is not None and shown >= max_items):
                    continue
                used.update(numbers)
                shown += 1
                items.append(QuoteItem(spans=numbers))
            elif isinstance(item, RestItem) and max_items is not None:
                left = max_items - shown
                if left <= 0:
                    continue
                items.append(RestItem(max=min(item.max or left, left)))
                shown = max_items
            elif isinstance(item, ReferencesItem) and item.spans is not None:
                numbers = [n for n in dict.fromkeys(item.spans) if n <= span_count]
                if numbers:
                    items.append(ReferencesItem(spans=numbers))
            else:
                items.append(item)
        if items:
            sections.append(PlanSection(label=section.label, items=items))
    return AnswerPlan(intro=plan.intro, sections=sections)


def shows_spans(plan: AnswerPlan) -> bool:
    """Return whether a plan has an item that shows spans."""
    return any(isinstance(item, (QuoteItem, RestItem)) for section in plan.sections for item in section.items)


def fallback_plan(span_count: int, max_items: int, text: PlanText | None = None) -> AnswerPlan:
    """Return the plan that shows the first spans in order, under the fallback intro."""
    text = text or PlanText()
    items = [QuoteItem(spans=[n]) for n in range(1, min(span_count, max_items) + 1)]
    return AnswerPlan(intro=text.fallback_intro, sections=[PlanSection(items=items)])


def renumber_plan(plan: AnswerPlan, numbers: Sequence[int]) -> AnswerPlan:
    """Return the plan with span number n replaced by ``numbers[n - 1]``.

    The Transformer uses it to point a plan written over the candidates at `Answer.spans`.
    """

    def renumber(spans: list[int]) -> list[int]:
        return [numbers[n - 1] for n in spans]

    sections = []
    for section in plan.sections:
        items: list[QuoteItem | RestItem | ReferencesItem | FieldItem] = []
        for item in section.items:
            if isinstance(item, QuoteItem):
                items.append(QuoteItem(spans=renumber(item.spans)))
            elif isinstance(item, ReferencesItem) and item.spans is not None:
                items.append(ReferencesItem(spans=renumber(item.spans)))
            elif isinstance(item, FieldItem) and item.spans is not None:
                items.append(item.model_copy(update={"spans": renumber(item.spans)}))
            else:
                items.append(item)
        sections.append(PlanSection(label=section.label, items=items))
    return AnswerPlan(intro=plan.intro, sections=sections)


def field_items(plan: AnswerPlan | None) -> list[FieldItem]:
    """Return the field items of a plan, in order."""
    if plan is None:
        return []
    return [item for section in plan.sections for item in section.items if isinstance(item, FieldItem)]


def fill_fields(plan: AnswerPlan, spans: Sequence[Sequence[int]]) -> AnswerPlan:
    """Return the plan with the i-th field item's spans set to ``spans[i]``."""
    filled = iter(spans)
    sections = [
        PlanSection(
            label=section.label,
            items=[
                item.model_copy(update={"spans": list(next(filled))}) if isinstance(item, FieldItem) else item
                for item in section.items
            ],
        )
        for section in plan.sections
    ]
    return AnswerPlan(intro=plan.intro, sections=sections)


_REPEAT_KEY = re.compile(r"\W+")


def distinct_spans(texts: Sequence[str]) -> list[int]:
    """Return the positions of the spans whose words no earlier kept span already has.

    Words are letters and digits, ignoring case, so the same sentence printed in two textbooks
    counts once. Spans without words are left out.
    """
    kept: list[int] = []
    seen: list[str] = []
    for position, text in enumerate(texts):
        words = _REPEAT_KEY.sub("", text).casefold()
        if words and not any(words in earlier for earlier in seen):
            kept.append(position)
            seen.append(words)
    return kept


def plan_candidates(texts: Sequence[str], chunk_ids: Sequence[str], max_candidates: int) -> list[int]:
    """Return the positions of the spans a plan chooses from.

    The `distinct_spans` are taken in turns between chunks, in the order the chunks first appear,
    so one chunk can't fill the list, up to ``max_candidates``.
    """
    by_chunk: dict[str, list[int]] = {}
    for position in distinct_spans(texts):
        by_chunk.setdefault(chunk_ids[position], []).append(position)
    rounds = max((len(positions) for positions in by_chunk.values()), default=0)
    turns = [positions[i] for i in range(rounds) for positions in by_chunk.values() if i < len(positions)]
    return turns[:max_candidates]
