"""Placeholder templates, laid out for the renderer of every answer (ADR 0008).

A placeholder template is markdown with per-span placeholders ([SPAN_1], [FACT_1], ...) or
aggregate ones ([DISPLAY_SPANS], [RELEVANT_SENTENCES]), and [CITATION_REFS]. `TemplateFiller`
reads them as the library always has: display spans first, then citation-only spans, span n is
citation n, and the source layout of a span is kept. The template's text stays as it is.
"""

import re
from typing import Any, ClassVar

from verbatim_core.render import Layout, Markers, Passages, Quote, Rendered, RenderOptions, render
from verbatim_core.templates.base import has_span_placeholder

NO_SPANS = "No relevant information found in the provided documents."

_PER_SPAN = re.compile(r"\[(?:SPAN|FACT)_(\d+)\]|\[CITATION_REFS\]")
_AGGREGATE = re.compile(r"\[DISPLAY_SPANS\]|\[RELEVANT_SENTENCES\]|\[CITATION_REFS\]")
_HAS_PER_SPAN = re.compile(r"\[(?:SPAN|FACT)_\d+\]")


def layout_template(template: str, display_spans: list[dict[str, Any]], citation_spans: list[dict[str, Any]]) -> Layout:
    """Lay out a placeholder template for the display and citation-only spans of a call.

    A template with per-span placeholders is read as per-span: there [DISPLAY_SPANS] stays text.
    [SPAN_n] shows display span n, cites a citation-only span by marker, or is left out when there
    is no span n. [DISPLAY_SPANS] and [RELEVANT_SENTENCES] show every display span, a blank line
    between them. [CITATION_REFS] cites the citation-only spans, unless display spans link their
    own (``citation_ids``), which are cited next to the display span that links them.
    """
    shown = len(display_spans)
    total = shown + len(citation_spans)
    number_of = {  # citation_id -> span number
        str(span["citation_id"]): shown + offset
        for offset, span in enumerate(citation_spans, start=1)
        if span.get("citation_id")
    }

    def quote(number: int) -> Quote:
        ids = map(str, display_spans[number - 1].get("citation_ids", []))
        return Quote((number, *(number_of[key] for key in ids if key in number_of)))

    linked_anywhere = any(span.get("citation_ids") for span in display_spans)
    per_span = _HAS_PER_SPAN.search(template) is not None
    layout: Layout = []
    position = 0
    for match in (_PER_SPAN if per_span else _AGGREGATE).finditer(template):
        layout.append(template[position : match.start()])
        position = match.end()
        # A placeholder that cites nothing stays in the layout as empty markers: the line it is on
        # is a list item only if the line starts as one, placeholders included.
        if match.group(0) == "[CITATION_REFS]":
            cited = not linked_anywhere
            layout.append(Markers(tuple(range(shown + 1, total + 1)) if cited else ()))
        elif not per_span:
            layout.append(Passages(tuple(quote(n) for n in range(1, shown + 1)), NO_SPANS))
        elif 1 <= (number := int(match.group(1))) <= shown:
            layout.append(quote(number))
        else:
            layout.append(Markers((number,) if shown < number <= total else ()))
    layout.append(template[position:])
    return layout


class TemplateFiller:
    """Fill placeholder templates with spans and citation markers."""

    ALLOWED_MODES: ClassVar[set[str]] = {"inline", "hidden"}

    def __init__(self, citation_mode: str = "inline", citation_format: str = "[{number}]") -> None:
        """Initialize the filler.

        :param citation_mode: "inline" to embed markers in text, "hidden" for plain text.
        :param citation_format: str.format template for citation markers.
            Available variables: {number} (sequential integer), {span_id} (span's own id,
            falls back to str(number) when absent from span data).
            Default "[{number}]" reproduces pre-0.2.8 output exactly.
            Example: "[{span_id}]" renders "[cite1]" when span_data contains
            {"span_id": "cite1"}.
        """
        self.set_citation_mode(citation_mode)
        self.citation_format = citation_format

    def set_citation_mode(self, citation_mode: str) -> None:
        """Set the citation mode ("inline" or "hidden")."""
        if citation_mode not in self.ALLOWED_MODES:
            msg = f"Unsupported citation mode: {citation_mode}. Allowed values: {sorted(self.ALLOWED_MODES)}"
            raise ValueError(msg)
        self.citation_mode = citation_mode

    def render(
        self,
        template: str,
        display_spans: list[dict[str, Any]],
        citation_spans: list[dict[str, Any]],
    ) -> Rendered:
        """Fill the template, and return the text with a citation for every span."""
        spans = display_spans + citation_spans
        options = RenderOptions(
            citation_mode="hidden" if self.citation_mode == "hidden" else "inline",
            citation_format=self.citation_format,
            numbering="position",
        )
        span_ids = [str(span.get("span_id", number)) for number, span in enumerate(spans, start=1)]
        layout = layout_template(template, display_spans, citation_spans)
        return render(layout, [span.get("text", "") for span in spans], options, span_ids)

    def fill(
        self,
        template: str,
        display_spans: list[dict[str, Any]],
        citation_spans: list[dict[str, Any]],
    ) -> str:
        """Fill the template with display and citation spans."""
        return self.render(template, display_spans, citation_spans).text

    @staticmethod
    def ensure_placeholder(template: str, placeholder: str = "[DISPLAY_SPANS]") -> str:
        """Append the placeholder to the template when it has no placeholder that shows spans.

        A per-span template that leaves out [SPAN_1] has one: appending [DISPLAY_SPANS] to it
        put the raw placeholder into the answer, because per-span filling never fills it.
        """
        if has_span_placeholder(template):
            return template
        return template + f"\n\n{placeholder}"
