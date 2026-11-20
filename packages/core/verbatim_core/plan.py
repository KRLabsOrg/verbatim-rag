"""Answer plans: the structure of an answer, validated in code and rendered without an LLM.

An answer plan says which spans an answer shows, in which order and under which labels. It never
holds span text: items refer to spans by number. The renderer turns a plan and the spans into
markdown and numbers the citations in the order they appear, so the markers in the text and the
citations always agree.

A plan comes from an LLM (`plan_from_llm`, through structured output) or from placeholder text
(`parse_template`). Either way `validate_plan` checks it against the spans of the call: unknown span
numbers are dropped, a span is used at most once, empty items and sections are dropped, and the
number of shown items is capped. `fallback_plan` is the deterministic plan used when nothing valid
remains. See ADR 0007.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from verbatim_core.errors import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = [
    "AnswerPlan",
    "FieldItem",
    "LLMPlan",
    "PlanCitation",
    "PlanSection",
    "PlanText",
    "QuoteItem",
    "ReferencesItem",
    "RenderedPlan",
    "RestItem",
    "TextItem",
    "compact_quote",
    "fallback_plan",
    "parse_template",
    "plan_candidates",
    "plan_from_llm",
    "render_plan",
    "shows_spans",
    "validate_plan",
]


class _PlanModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class QuoteItem(_PlanModel):
    """Spans that say the same thing: the first is shown, the others are cited next to it."""

    kind: Literal["quote"] = "quote"
    spans: list[PositiveInt] = Field(min_length=1)  # span numbers, from 1


class RestItem(_PlanModel):
    """Every shown span that no other item uses, in order: the plan form of [DISPLAY_SPANS]."""

    kind: Literal["rest"] = "rest"
    max: PositiveInt | None = None


class ReferencesItem(_PlanModel):
    """The numbers of the cited spans the text doesn't show: the plan form of [CITATION_REFS]."""

    kind: Literal["references"] = "references"


class FieldItem(_PlanModel):
    """A named question whose answer is extracted for this item, such as one row of a checklist.

    Designed in ADR 0007 and validated here, but not filled yet: a later release extracts each
    field's question over the chunks (with #74). `render_plan` raises for a plan that has one.
    """

    kind: Literal["field"] = "field"
    name: str = Field(min_length=1)
    question: str = Field(min_length=1)
    required: bool = False  # an empty required field shows PlanText.no_evidence
    max_spans: PositiveInt = 1


class TextItem(_PlanModel):
    """Text of the template around the spans, kept as it is (headings, connecting words)."""

    kind: Literal["text"] = "text"
    text: str


PlanItem = Annotated[QuoteItem | FieldItem | RestItem | ReferencesItem | TextItem, Field(discriminator="kind")]


class PlanSection(_PlanModel):
    """A group of items under an optional label."""

    label: str | None = None
    items: list[PlanItem] = Field(default_factory=list)


class AnswerPlan(_PlanModel):
    """Which spans an answer shows, in which order and under which labels.

    In ``list`` layout each quote is a list item and each label a bold line. In ``inline`` layout,
    the layout of a parsed placeholder template, the items are joined as they are, so the text
    items carry the layout.
    """

    intro: str | None = None
    sections: list[PlanSection] = Field(default_factory=list)
    layout: Literal["list", "inline"] = "list"


class PlanText(_PlanModel):
    """The fixed texts of rendered plans. Set them in the language of your answers."""

    # Introduces the fallback plan, when the LLM couldn't write a plan
    fallback_intro: str = "The sources contain these passages:"
    # Before the numbers of cited spans the answer doesn't show, when `show_references` is set
    references_label: str = "Further sources:"
    # A rest item with no span to show
    no_spans: str = "No relevant information found in the provided documents."
    # A required field item with no span (field items are filled by a later release)
    no_evidence: str = "No supporting passage found."
    show_references: bool = False
    # Show quotes without the layout of their source: each line's spaces collapsed and its
    # indentation removed, paragraphs as hard line breaks. Citations keep the exact span text.
    compact_quotes: bool = True


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


_PER_SPAN = re.compile(r"\[(?:SPAN|FACT)_(\d+)\]|\[CITATION_REFS\]")
_AGGREGATE = re.compile(r"\[DISPLAY_SPANS\]|\[RELEVANT_SENTENCES\]|\[CITATION_REFS\]")


def parse_template(template: str) -> AnswerPlan:
    """Parse placeholder template text into an inline plan, keeping every other character.

    A template with per-span placeholders ([SPAN_n], [FACT_n]) is read as per-span: there
    [DISPLAY_SPANS] stays text, as `TemplateFiller` leaves it. Otherwise [DISPLAY_SPANS] and
    [RELEVANT_SENTENCES] become rest items. [CITATION_REFS] becomes a references item.
    """
    per_span = re.search(r"\[(?:SPAN|FACT)_\d+\]", template) is not None
    pattern = _PER_SPAN if per_span else _AGGREGATE
    items: list[QuoteItem | RestItem | ReferencesItem | TextItem] = []
    position = 0
    for match in pattern.finditer(template):
        if match.start() > position:
            items.append(TextItem(text=template[position : match.start()]))
        placeholder = match.group(0)
        if placeholder == "[CITATION_REFS]":
            items.append(ReferencesItem())
        elif per_span:
            number = int(match.group(1))
            # [SPAN_0] fills nothing
            items.append(QuoteItem(spans=[number]) if number > 0 else TextItem(text=""))
        else:
            items.append(RestItem())
        position = match.end()
    if position < len(template):
        items.append(TextItem(text=template[position:]))
    return AnswerPlan(sections=[PlanSection(items=items)], layout="inline")


def validate_plan(
    plan: AnswerPlan, span_count: int, max_items: int | None = None, max_support: int | None = None
) -> AnswerPlan:
    """Return the plan with what can't be shown taken out.

    Span numbers outside 1..``span_count`` are dropped, and a span is used at most once across the
    plan, the first time. Quote items left without spans, and sections left without items, are
    dropped; in an inline plan, so is the list bullet that led to a dropped quote. At most
    ``max_items`` items show spans: quote items count one each, and a rest item gets the places
    left. A quote item keeps at most ``max_support`` supporting spans after its shown one; the
    others stay unused. Text and references items stay.
    """
    used: set[int] = set()
    shown = 0
    sections = []
    for section in plan.sections:
        items: list[QuoteItem | FieldItem | RestItem | ReferencesItem | TextItem] = []
        for item in section.items:
            if isinstance(item, QuoteItem):
                numbers = [n for n in dict.fromkeys(item.spans) if 1 <= n <= span_count and n not in used]
                if max_support is not None:
                    numbers = numbers[: max_support + 1]
                if not numbers or (max_items is not None and shown >= max_items):
                    _drop_bullet(items)
                    continue
                used.update(numbers)
                shown += 1
                items.append(QuoteItem(spans=numbers))
            elif isinstance(item, RestItem) and max_items is not None:
                left = max_items - shown
                if left <= 0:
                    _drop_bullet(items)
                    continue
                items.append(RestItem(max=min(item.max or left, left)))
                shown = max_items
            else:
                items.append(item)
        if any(not isinstance(item, TextItem) for item in items) or (plan.layout == "inline" and items):
            sections.append(PlanSection(label=section.label, items=items))
    return AnswerPlan(intro=plan.intro, sections=sections, layout=plan.layout)


_TRAILING_BULLET = re.compile(r"\n?[ \t]*(?:[-*+]|\d+[.)])[ \t]*$")


def _drop_bullet(items: list[QuoteItem | FieldItem | RestItem | ReferencesItem | TextItem]) -> None:
    """Take a list bullet that led to a dropped item off the end of the text before it."""
    if items and isinstance(items[-1], TextItem):
        items[-1] = TextItem(text=_TRAILING_BULLET.sub("", items[-1].text))


def shows_spans(plan: AnswerPlan) -> bool:
    """Return whether a plan has an item that shows spans."""
    return any(isinstance(item, (QuoteItem, RestItem)) for section in plan.sections for item in section.items)


def fallback_plan(span_count: int, max_items: int, text: PlanText | None = None) -> AnswerPlan:
    """Return the plan that shows the first spans in order, under the fallback intro."""
    text = text or PlanText()
    items = [QuoteItem(spans=[n]) for n in range(1, min(span_count, max_items) + 1)]
    return AnswerPlan(intro=text.fallback_intro, sections=[PlanSection(items=items)])


MIN_TABLE_LINES = 2

_LIST_ITEM_START = re.compile(r"[ \t]*(?:[-*+]|\d+[.)])[ \t]+")
_BLANK_LINES = re.compile(r"\n[ \t]*\n\s*")


def list_item_indent(template: str, position: int) -> int | None:
    """Return the content indent of the list item the position is in, or None outside lists."""
    line_start = template.rfind("\n", 0, position) + 1
    match = _LIST_ITEM_START.match(template, line_start, position)
    return None if match is None else match.end() - line_start


def fit_list_item(text: str, indent: int) -> str:
    """Keep a quote with blank lines inside its markdown list item.

    A blank line inside the quote, as between the entries of a list in a PDF, ended the list item,
    and the rest of the quote followed as loose paragraphs. Paragraphs become backslash hard line
    breaks, which `ResponseBuilder.clean_answer` leaves alone, and every line is indented to the
    item's content. A nested list line keeps its own indentation on top; other lines lose theirs,
    which in PDF text is layout, and four spaces would start a code block. Only whitespace changes.
    """
    pad = " " * indent
    paragraphs = [paragraph.rstrip() for paragraph in _BLANK_LINES.split(text.strip())]
    lines = "\\\n".join(paragraphs).split("\n")
    return lines[0] + "".join("\n" + pad + _nested_line(line) for line in lines[1:])


def _nested_line(line: str) -> str:
    return line if _LIST_ITEM_START.match(line) else line.lstrip(" \t")


def compact_quote(text: str, indent: int = 0) -> str:
    """Return a quote without the layout of its source page, for display.

    PDF text breaks lines inside sentences and indents them with long runs of spaces. Each line's
    spaces are collapsed and the lines of a paragraph are joined with a space; a line that starts a
    list starts a new line. Paragraphs are joined with backslash hard line breaks, indented by
    ``indent`` so the quote stays in its list item. Only whitespace changes.
    """
    pad = " " * indent
    paragraphs = []
    for paragraph in _BLANK_LINES.split(text.strip()):
        joined = ""
        for line in (" ".join(raw.split()) for raw in paragraph.split("\n")):
            if not line:
                continue
            if joined and _LIST_ITEM_START.match(line):
                joined += "\\\n" + pad + line
            else:
                joined = f"{joined} {line}" if joined else line
        if joined:
            paragraphs.append(joined)
    return ("\\\n" + pad).join(paragraphs)


def has_blank_lines(text: str) -> bool:
    """Return whether a span has a blank line inside, which would end a markdown list item."""
    return _BLANK_LINES.search(text.strip()) is not None


def is_table(text: str) -> bool:
    """Return whether a span is a markdown table: at least half of its lines have pipes."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if len(lines) < MIN_TABLE_LINES:
        return False
    pipe_lines = sum(1 for line in lines if "|" in line)
    if pipe_lines < MIN_TABLE_LINES:
        return False
    return pipe_lines >= len(lines) / 2


_REPEAT_KEY = re.compile(r"\W+")


def plan_candidates(texts: Sequence[str], chunk_ids: Sequence[str], max_candidates: int) -> list[int]:
    """Return the positions of the spans a plan chooses from.

    A span whose words (letters and digits, ignoring case) an earlier span already has is left
    out. The others are taken in turns between chunks, in the order the chunks first appear, so
    one chunk can't fill the list, up to ``max_candidates``.
    """
    by_chunk: dict[str, list[int]] = {}
    seen: list[str] = []
    for position, (text, chunk_id) in enumerate(zip(texts, chunk_ids, strict=True)):
        words = _REPEAT_KEY.sub("", text).casefold()
        if not words or any(words in earlier for earlier in seen):
            continue
        seen.append(words)
        by_chunk.setdefault(chunk_id, []).append(position)
    rounds = max((len(positions) for positions in by_chunk.values()), default=0)
    turns = [positions[i] for i in range(rounds) for positions in by_chunk.values() if i < len(positions)]
    return turns[:max_candidates]


@dataclass(frozen=True)
class PlanCitation:
    """A citation of a rendered plan: its number, the span's position, and whether it is shown."""

    number: int
    span: int  # position in the spans given to `render_plan`, from 0
    type: Literal["display", "reference"]


@dataclass(frozen=True)
class RenderedPlan:
    """The text of a rendered plan and its citations, numbered in the order they appear."""

    text: str
    citations: list[PlanCitation]


def render_plan(  # noqa: PLR0913  # keyword-only rendering settings
    plan: AnswerPlan,
    spans: Sequence[str],
    *,
    text: PlanText | None = None,
    citation_mode: str = "inline",
    citation_format: str = "[{number}]",
    shown_limit: int | None = None,
) -> RenderedPlan:
    """Render a validated plan with the texts of its spans, numbering citations as they appear.

    Span number n of the plan is ``spans[n - 1]``. A shown span is a ``display`` citation; a span
    cited next to a shown one, or not used by the plan, is a ``reference`` citation. Spans the plan
    doesn't use get the numbers after the others, and appear in the text only in a references item
    or, with ``PlanText.show_references``, after the answer. Span text is never changed; a span
    with blank lines inside a list item gets hard line breaks and indented lines, as in
    `TemplateFiller`. With ``citation_mode="hidden"`` the text has no markers. A rest item shows
    only spans before ``shown_limit`` (all by default): the candidates the plan was written for.
    """
    if any(isinstance(item, FieldItem) for section in plan.sections for item in section.items):
        msg = "Field items are not rendered yet: fill them in a later release (ADR 0007), or use quote items."
        raise InvalidInputError(msg, code="invalid_value", param="plan")
    limit = len(spans) if shown_limit is None else shown_limit
    return _Renderer(plan, spans, text or PlanText(), citation_mode, citation_format, limit).render()


class _Renderer:
    def __init__(  # noqa: PLR0913, PLR0917  # the settings of render_plan
        self,
        plan: AnswerPlan,
        spans: Sequence[str],
        text: PlanText,
        citation_mode: str,
        citation_format: str,
        shown_limit: int,
    ) -> None:
        self.plan = plan
        self.spans = spans
        self.text = text
        self.inline = citation_mode == "inline"
        self.citation_format = citation_format
        self.numbers: dict[int, int] = {}  # span position -> citation number
        self.types: dict[int, Literal["display", "reference"]] = {}
        items = [item for section in plan.sections for item in section.items]
        quoted = {n - 1 for item in items if isinstance(item, QuoteItem) for n in item.spans}
        pool = [i for i in range(min(shown_limit, len(spans))) if i not in quoted]
        rest_items = [item for item in items if isinstance(item, RestItem)]
        capacity = (
            len(pool) if any(item.max is None for item in rest_items) else sum(item.max or 0 for item in rest_items)
        )
        # the spans the rest items will show, in order: references leave them out
        self.rest_spans = pool[:capacity]

    def cite(self, position: int, kind: Literal["display", "reference"]) -> str:
        if position not in self.numbers:
            self.numbers[position] = len(self.numbers) + 1
            self.types[position] = kind
        if not self.inline:
            return ""
        number = self.numbers[position]
        return self.citation_format.format(number=number, span_id=str(number))

    def quote(self, position: int, before: str) -> str:
        marker = self.cite(position, "display")
        span_text = self.spans[position].strip()
        if is_table(span_text):
            return f"{marker}\n\n{span_text}" if marker else span_text
        indent = list_item_indent(before, len(before))
        if self.text.compact_quotes:
            compact = compact_quote(span_text, indent or 0)
            return f"{marker} {compact}" if marker else compact
        quoted = f"{marker} {span_text}" if marker else span_text
        return quoted if indent is None or not has_blank_lines(span_text) else fit_list_item(quoted, indent)

    def quote_item(self, item: QuoteItem, before: str) -> str:
        first, *support = (n - 1 for n in item.spans)
        shown = self.quote(first, before)
        markers = " ".join(self.cite(n, "reference") for n in support).strip()
        return f"{shown} {markers}" if markers else shown

    def rest_item(self, item: RestItem) -> str:
        positions = self.take_rest(item)
        blocks = [self.quote(position, "") for position in positions if self.spans[position].strip()]
        return "\n\n".join(blocks) if blocks else self.text.no_spans

    def take_rest(self, item: RestItem) -> list[int]:
        positions = self.rest_spans[: item.max] if item.max is not None else self.rest_spans
        self.rest_spans = self.rest_spans[len(positions) :]
        return positions

    def references(self) -> str:
        unused = [i for i in range(len(self.spans)) if i not in self.numbers and i not in self.rest_spans]
        return " ".join(self.cite(i, "reference") for i in unused).strip()

    def render(self) -> RenderedPlan:
        body = self.render_inline() if self.plan.layout == "inline" else self.render_list()
        if self.text.show_references and self.inline and (markers := self.references()):
            body = f"{body}\n\n{self.text.references_label} {markers}" if body else markers
        for position in range(len(self.spans)):
            self.cite(position, "reference")
        citations = [PlanCitation(number=self.numbers[i], span=i, type=self.types[i]) for i in self.numbers]
        return RenderedPlan(text=body.strip(), citations=sorted(citations, key=lambda c: c.number))

    def render_inline(self) -> str:
        out = ""
        for section in self.plan.sections:
            for item in section.items:
                if isinstance(item, TextItem):
                    out += item.text
                elif isinstance(item, QuoteItem):
                    out += self.quote_item(item, out)
                elif isinstance(item, RestItem):
                    out += self.rest_item(item)
                elif isinstance(item, ReferencesItem):
                    out += self.references()
        return out

    def render_list(self) -> str:
        blocks = [self.plan.intro] if bool(self.plan.intro) else []
        for section in self.plan.sections:
            labelled = bool(section.label)
            lines = [f"**{section.label}**"] if labelled else []
            for item in section.items:
                if isinstance(item, QuoteItem):
                    lines.append("- " + self.quote_item(item, "- "))
                elif isinstance(item, RestItem):
                    positions = self.take_rest(item)
                    lines.extend("- " + self.quote(p, "- ") for p in positions if self.spans[p].strip())
                elif isinstance(item, ReferencesItem):
                    if markers := self.references():
                        lines.append(markers)
                elif isinstance(item, TextItem) and item.text.strip():
                    lines.append(item.text.strip())
            if len(lines) > (1 if labelled else 0):
                blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
