"""Template filling utilities for the Verbatim RAG system.

Handles aggregate placeholders ([DISPLAY_SPANS], [RELEVANT_SENTENCES]) and
per-span placeholders ([SPAN_1], ...). Supports toggling inline citation
numbering on or off so callers can choose between numbered excerpts or
clean text with citations handled separately.
"""

from typing import Any, ClassVar

from verbatim_core.plan import (
    QuoteItem,
    RestItem,
    TextItem,
    fit_list_item,
    has_blank_lines,
    is_table,
    list_item_indent,
    parse_template,
)
from verbatim_core.templates.base import has_span_placeholder

MIN_TABLE_LINES = 2  # also in verbatim_core.plan


class TemplateFiller:
    """Fill template placeholders with spans and citation markers."""

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
            raise ValueError(
                msg,
            )
        self.citation_mode = citation_mode

    def _render_marker(self, number: int, span_data: dict[str, Any]) -> str:
        """Render a citation marker using citation_format.

        Provides {number} and {span_id} (falls back to str(number) if absent).
        """
        span_id = span_data.get("span_id", str(number))
        return self.citation_format.format(number=number, span_id=span_id)

    def fill(
        self,
        template: str,
        display_spans: list[dict[str, Any]],
        citation_spans: list[dict[str, Any]],
    ) -> str:
        """Fill the template with display and citation spans."""
        if not template:
            return ""

        citation_number_by_id = self._build_citation_number_map(display_spans, citation_spans)
        span_id_by_citation_id = self._build_span_id_map(citation_spans, len(display_spans) + 1)
        has_linked_citations = self._has_linked_citations(display_spans)
        citation_refs = ""
        if citation_spans and self.citation_mode == "inline" and not has_linked_citations:
            start_num = len(display_spans) + 1
            citation_refs = " ".join(self._render_marker(start_num + i, span) for i, span in enumerate(citation_spans))

        # The template is parsed into an answer plan and the plan is filled item by item, with the
        # numbering and layout the placeholder filler has always had (ADR 0007).
        display_content: str | None = None
        filled = ""
        echo = ""  # the template up to the current item, for the list item a span is in
        for item in parse_template(template).sections[0].items:
            if isinstance(item, TextItem):
                filled += item.text
                echo += item.text
            elif isinstance(item, QuoteItem):
                filled += self._fill_span(
                    item.spans[0], echo, display_spans, citation_spans, citation_number_by_id, span_id_by_citation_id
                )
                echo += f"[SPAN_{item.spans[0]}]"
            elif isinstance(item, RestItem):
                if display_content is None:
                    display_content = self._display_content(
                        display_spans, citation_number_by_id, span_id_by_citation_id
                    )
                filled += display_content
                echo += "[DISPLAY_SPANS]"
            else:  # a references item
                filled += citation_refs
                echo += "[CITATION_REFS]"

        return filled.strip()

    def _fill_span(  # the state of one fill
        self,
        idx: int,
        echo: str,
        display_spans: list[dict[str, Any]],
        citation_spans: list[dict[str, Any]],
        citation_number_by_id: dict[str, int],
        span_id_by_citation_id: dict[str, str],
    ) -> str:
        """Fill one per-span placeholder: a displayed span, a citation-only marker, or nothing."""
        total_spans = display_spans + citation_spans
        if not 1 <= idx <= len(total_spans):
            return ""
        if idx > len(display_spans):
            return self._render_marker(idx, total_spans[idx - 1]) if self.citation_mode == "inline" else ""
        span_data = display_spans[idx - 1]
        formatted = self._format_span(span_data, idx, citation_number_by_id, span_id_by_citation_id)
        indent = list_item_indent(echo, len(echo))
        text = span_data.get("text", "")
        # Only a span that would end its list item changes: other output stays as it was
        if indent is None or not has_blank_lines(text) or self._is_table(text):
            return formatted
        return fit_list_item(formatted, indent)

    def _display_content(
        self,
        display_spans: list[dict[str, Any]],
        citation_number_by_id: dict[str, int],
        span_id_by_citation_id: dict[str, str],
    ) -> str:
        """Return what [DISPLAY_SPANS] and [RELEVANT_SENTENCES] are filled with."""
        formatted = []
        for i, span_data in enumerate(display_spans, 1):
            block = self._format_span(span_data, i, citation_number_by_id, span_id_by_citation_id)
            if block:
                formatted.append(block)
        return "\n\n".join(formatted) if formatted else "No relevant information found in the provided documents."

    def _format_span(
        self,
        span_data: dict[str, Any],
        index: int,
        citation_number_by_id: dict[str, int],
        span_id_by_citation_id: dict[str, str],
    ) -> str:
        text = span_data.get("text", "")
        cleaned = text.strip()
        if not cleaned:
            return ""

        linked_refs = self._format_linked_citation_refs(
            span_data,
            citation_number_by_id,
            span_id_by_citation_id,
        )
        is_table = self._is_table(cleaned)
        marker = self._render_marker(index, span_data)

        if self.citation_mode == "inline":
            if is_table:
                if linked_refs:
                    return f"{marker} {linked_refs}\n\n{cleaned}"
                return f"{marker}\n\n{cleaned}"
            if linked_refs:
                return f"{marker} {cleaned} {linked_refs}"
            return f"{marker} {cleaned}"

        return cleaned

    @staticmethod
    def _has_linked_citations(display_spans: list[dict[str, Any]]) -> bool:
        return any(span.get("citation_ids") for span in display_spans)

    @staticmethod
    def _build_citation_number_map(
        display_spans: list[dict[str, Any]],
        citation_spans: list[dict[str, Any]],
    ) -> dict[str, int]:
        start_num = len(display_spans) + 1
        citation_number_by_id: dict[str, int] = {}
        for offset, citation_span in enumerate(citation_spans):
            citation_id = citation_span.get("citation_id")
            if citation_id:
                citation_number_by_id[str(citation_id)] = start_num + offset
        return citation_number_by_id

    @staticmethod
    def _build_span_id_map(
        citation_spans: list[dict[str, Any]],
        start_num: int,
    ) -> dict[str, str]:
        """Map citation_id → span_id for linked citation refs."""
        result: dict[str, str] = {}
        for offset, span in enumerate(citation_spans):
            cid = span.get("citation_id")
            if cid:
                result[str(cid)] = span.get("span_id", str(start_num + offset))
        return result

    def _format_linked_citation_refs(
        self,
        span_data: dict[str, Any],
        citation_number_by_id: dict[str, int],
        span_id_by_citation_id: dict[str, str] | None = None,
    ) -> str:
        if self.citation_mode != "inline":
            return ""

        citation_ids = span_data.get("citation_ids", [])
        parts = []
        for citation_id in citation_ids:
            key = str(citation_id)
            if key not in citation_number_by_id:
                continue
            number = citation_number_by_id[key]
            sid = (span_id_by_citation_id or {}).get(key, str(number))
            parts.append(self.citation_format.format(number=number, span_id=sid))
        return " ".join(parts)

    @staticmethod
    def _is_table(text: str) -> bool:
        return is_table(text)

    @staticmethod
    def _is_table_content(text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < MIN_TABLE_LINES:
            return False

        pipe_lines = sum(1 for line in lines if "|" in line)

        return pipe_lines >= len(lines) / 2

    @staticmethod
    def ensure_placeholder(template: str, placeholder: str = "[DISPLAY_SPANS]") -> str:
        """Append the placeholder to the template when it has no placeholder that shows spans.

        A per-span template that leaves out [SPAN_1] has one: appending [DISPLAY_SPANS] to it
        put the raw placeholder into the answer, because per-span filling never fills it.
        """
        if has_span_placeholder(template):
            return template

        if template.endswith(":"):
            return template + f"\n\n{placeholder}"
        return template + f"\n\n{placeholder}"
