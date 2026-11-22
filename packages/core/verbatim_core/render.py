"""The renderer of every answer: a layout of text and slots, filled with spans (ADR 0008).

A layout is a list of pieces: text, kept as it is, and slots that span text and citation markers
go into. `layout_plan` lays out an answer plan as markdown; `TemplateFiller` lays out a placeholder
template. `render` fills either, and is the only code that writes span text and markers, so the
markers in the text and the citations it returns always agree. Span text never changes; only its
whitespace may, to keep it inside a markdown list item.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from verbatim_core.errors import InvalidInputError
from verbatim_core.plan import AnswerPlan, FieldItem, QuoteItem, ReferencesItem, RestItem

if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = [
    "Layout",
    "Markers",
    "Passages",
    "Quote",
    "RenderOptions",
    "Rendered",
    "RenderedCitation",
    "compact_quote",
    "layout_plan",
    "render",
]


@dataclass(frozen=True)
class Quote:
    """A shown span, the first, with the others cited next to it. Numbers from 1."""

    spans: tuple[int, ...]


@dataclass(frozen=True)
class Markers:
    """Citation markers of spans, without their text."""

    spans: tuple[int, ...]


@dataclass(frozen=True)
class Passages:
    """Quotes one after another with a blank line between, or ``empty`` when none has text."""

    quotes: tuple[Quote, ...]
    empty: str


Layout = list["str | Quote | Markers | Passages"]


@dataclass(frozen=True)
class RenderOptions:
    """How `render` writes span text and citation markers."""

    citation_mode: Literal["inline", "hidden"] = "inline"  # markers in the text, or none
    citation_format: str = "[{number}]"  # with {number} and {span_id}
    # "appearance": citations are numbered in the order they appear, then the spans the text
    # doesn't cite. "position": span n is citation n, as placeholder templates number.
    numbering: Literal["appearance", "position"] = "appearance"
    compact_quotes: bool = False  # see `compact_quote`
    # With inline markers: after the answer, this label and the markers of the spans it doesn't cite
    references_label: str | None = None


@dataclass(frozen=True)
class RenderedCitation:
    """A citation of a rendered answer: its number, the span's position from 0, and whether it is shown."""

    number: int
    span: int
    type: Literal["display", "reference"]


@dataclass(frozen=True)
class Rendered:
    """The text of a rendered answer and a citation for every span, by number."""

    text: str
    citations: list[RenderedCitation]


def layout_plan(
    plan: AnswerPlan, spans: Sequence[str], shown_limit: int | None = None, no_evidence: str | None = None
) -> Layout:
    """Lay out a validated plan as markdown: the intro, a bold line per label, a list item per quote.

    A rest item shows, in order, the spans among the first ``shown_limit`` (all by default) that no
    quote or field item uses, each as a list item. A references item without spans cites the spans
    the plan doesn't show or support. A field item is a list item that starts with its name in
    bold: one span follows the name, several are a nested list. An empty field shows
    ``no_evidence`` when it is required, and is left out otherwise. Spans without text are left out.

    :raises InvalidInputError: If a field item has no spans filled in (the Transformer fills them)
    """
    items = [item for section in plan.sections for item in section.items]
    if any(isinstance(item, FieldItem) and item.spans is None for item in items):
        msg = "A field item has no spans filled in. Render plans with field items through the Transformer."
        raise InvalidInputError(msg, code="invalid_value", param="plan")
    has_text = [bool(text.strip()) for text in spans]
    quoted = {n for item in items if isinstance(item, (QuoteItem, FieldItem)) for n in item.spans or ()}
    limit = len(spans) if shown_limit is None else min(shown_limit, len(spans))
    pool = [n for n in range(1, limit + 1) if n not in quoted and has_text[n - 1]]
    rest = {}  # the spans of each rest item, by its place in the plan
    for place, item in enumerate(items):
        if isinstance(item, RestItem):
            rest[place] = pool if item.max is None else pool[: item.max]
            pool = pool[len(rest[place]) :]
    used = quoted | {n for taken in rest.values() for n in taken}
    unused = tuple(n for n in range(1, len(spans) + 1) if n not in used)

    layout: Layout = []
    place = 0

    def block(pieces: Layout) -> None:
        if layout:
            layout.append("\n\n")
        layout.extend(pieces)

    if bool(plan.intro):  # an empty intro shows nothing
        block([plan.intro])
    for section in plan.sections:
        lines: list[Layout] = []
        for item in section.items:
            if isinstance(item, QuoteItem) and has_text[item.spans[0] - 1]:
                lines.append(["- ", Quote(tuple(item.spans))])
            elif isinstance(item, RestItem):
                lines.extend(["- ", Quote((n,))] for n in rest[place])
            elif isinstance(item, ReferencesItem) and (refs := tuple(item.spans or unused)):
                lines.append([Markers(refs)])
            elif isinstance(item, FieldItem):
                lines.extend(_field_lines(item, has_text, no_evidence))
            place += 1
        if lines:
            if bool(section.label):
                lines.insert(0, [f"**{section.label}**"])
            block([piece for i, line in enumerate(lines) for piece in (["\n"] if i else []) + line])
    return layout


def _field_lines(item: FieldItem, has_text: list[bool], no_evidence: str | None) -> list[Layout]:
    """Return the lines of a field: its name and spans, or the no-evidence text, or nothing."""
    shown = [n for n in item.spans or () if has_text[n - 1]]
    name = f"- **{item.name}:**"
    if len(shown) == 1:
        return [[f"{name} ", Quote((shown[0],))]]
    if shown:
        nested: list[Layout] = [["  - ", Quote((n,))] for n in shown]
        return [[name], *nested]
    if item.required and bool(no_evidence):
        return [[f"{name} {no_evidence}"]]
    return []


def render(
    layout: Layout,
    spans: Sequence[str],
    options: RenderOptions | None = None,
    span_ids: Sequence[str] | None = None,
) -> Rendered:
    """Fill a layout with the spans' text and citation markers.

    Span number n is ``spans[n - 1]``. A shown span is a ``display`` citation, any other span a
    ``reference`` citation; every span gets a citation. ``span_ids`` fill ``{span_id}`` in the
    citation format, the citation number by default.
    """
    return _Renderer(spans, options or RenderOptions(), span_ids).render(layout)


class _Renderer:
    def __init__(self, spans: Sequence[str], options: RenderOptions, span_ids: Sequence[str] | None) -> None:
        self.spans = spans
        self.options = options
        self.span_ids = span_ids
        self.inline = options.citation_mode == "inline"
        self.numbers: dict[int, int] = {}  # span position -> citation number
        self.shown: set[int] = set()
        self.cited: set[int] = set()  # spans the text cites

    def number(self, position: int) -> int:
        if position not in self.numbers:
            by_position = self.options.numbering == "position"
            self.numbers[position] = position + 1 if by_position else len(self.numbers) + 1
        return self.numbers[position]

    def marker(self, position: int) -> str:
        number = self.number(position)
        self.cited.add(position)
        if not self.inline:
            return ""
        span_id = self.span_ids[position] if self.span_ids is not None else str(number)
        return self.options.citation_format.format(number=number, span_id=span_id)

    def markers(self, numbers: Sequence[int]) -> str:
        return " ".join(marker for n in numbers if (marker := self.marker(n - 1)))

    def quote(self, quote: Quote, before: str | None) -> str:
        """Render a quote; ``before`` is the layout up to it, for the list item it is in."""
        first, *support = quote.spans
        text = self.spans[first - 1].strip()
        if not text:
            return ""
        self.shown.add(first - 1)
        marker = self.marker(first - 1)
        refs = self.markers(support)
        if _is_table(text):
            markers = " ".join(m for m in (marker, refs) if m)
            return f"{markers}\n\n{text}" if markers else text
        indent = None if before is None else _list_item_indent(before)
        if self.options.compact_quotes:
            return " ".join(part for part in (marker, compact_quote(text, indent or 0), refs) if part)
        joined = " ".join(part for part in (marker, text, refs) if part)
        return joined if indent is None or not _BLANK_LINES.search(text) else _fit_list_item(joined, indent)

    def render(self, layout: Layout) -> Rendered:
        out = ""
        echo = ""  # the layout's text up to the current piece, each slot as one character
        for piece in layout:
            if isinstance(piece, str):
                out += piece
                echo += piece
                continue
            if isinstance(piece, Quote):
                out += self.quote(piece, echo)
            elif isinstance(piece, Markers):
                out += self.markers(piece.spans)
            else:
                blocks = [block for quote in piece.quotes if (block := self.quote(quote, None))]
                out += "\n\n".join(blocks) if blocks else piece.empty
            echo += "\0"
        text = out.strip()
        unused = [p + 1 for p in range(len(self.spans)) if p not in self.cited]
        if bool(self.options.references_label) and self.inline and unused:
            markers = self.markers(unused)
            text = f"{text}\n\n{self.options.references_label} {markers}" if text else markers
        for position in range(len(self.spans)):
            self.number(position)
        citations = [
            RenderedCitation(number, position, "display" if position in self.shown else "reference")
            for position, number in self.numbers.items()
        ]
        return Rendered(text=text, citations=sorted(citations, key=lambda c: c.number))


_MIN_TABLE_LINES = 2
_LIST_ITEM_START = re.compile(r"[ \t]*(?:[-*+]|\d+[.)])[ \t]+")
_BLANK_LINES = re.compile(r"\n[ \t]*\n\s*")


def _list_item_indent(text: str) -> int | None:
    """Return the content indent of the list item the end of the text is in, or None outside lists."""
    line_start = text.rfind("\n") + 1
    match = _LIST_ITEM_START.match(text, line_start)
    return None if match is None else match.end() - line_start


def _fit_list_item(text: str, indent: int) -> str:
    """Keep a quote with blank lines inside its markdown list item.

    A blank line inside the quote, as between the entries of a list in a PDF, ended the list item,
    and the rest of the quote followed as loose paragraphs. Paragraphs become backslash hard line
    breaks, and every line is indented to the item's content. A nested list line keeps its own
    indentation on top; other lines lose theirs, which in PDF text is layout, and four spaces would
    start a code block. Only whitespace changes.
    """
    pad = " " * indent
    paragraphs = [paragraph.rstrip() for paragraph in _BLANK_LINES.split(text.strip())]
    lines = "\\\n".join(paragraphs).split("\n")
    return lines[0] + "".join(
        "\n" + pad + (line if _LIST_ITEM_START.match(line) else line.lstrip(" \t")) for line in lines[1:]
    )


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


def _is_table(text: str) -> bool:
    """Return whether a span is a markdown table: at least half of its lines have pipes."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    pipe_lines = sum(1 for line in lines if "|" in line)
    return len(lines) >= _MIN_TABLE_LINES and pipe_lines >= _MIN_TABLE_LINES and pipe_lines >= len(lines) / 2
