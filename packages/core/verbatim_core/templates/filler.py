"""Template filling utilities for the Verbatim RAG system.

Handles aggregate placeholders ([DISPLAY_SPANS], [RELEVANT_SENTENCES]) and
per-span placeholders ([SPAN_1], ...). Supports toggling inline citation
numbering on or off so callers can choose between numbered excerpts or
clean text with citations handled separately.
"""

import re
from typing import Any


class TemplateFiller:
    ALLOWED_MODES = {"inline", "hidden"}

    def __init__(self, citation_mode: str = "inline", citation_format: str = "[{number}]") -> None:
        """:param citation_mode: "inline" to embed markers in text, "hidden" for plain text.
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
        if citation_mode not in self.ALLOWED_MODES:
            msg = (
                f"Unsupported citation mode: {citation_mode}. "
                f"Allowed values: {sorted(self.ALLOWED_MODES)}"
            )
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
        if not template:
            return ""

        citation_number_by_id = self._build_citation_number_map(display_spans, citation_spans)
        span_id_by_citation_id = self._build_span_id_map(citation_spans, len(display_spans) + 1)
        has_linked_citations = self._has_linked_citations(display_spans)
        citation_refs = ""
        if citation_spans and self.citation_mode == "inline" and not has_linked_citations:
            start_num = len(display_spans) + 1
            citation_refs = " ".join(
                self._render_marker(start_num + i, span) for i, span in enumerate(citation_spans)
            )

        fact_pattern = re.compile(r"\[(?:SPAN|FACT)_(\d+)\]")
        if fact_pattern.search(template):
            filled = self._fill_per_fact_placeholders(
                template,
                display_spans,
                citation_spans,
                citation_number_by_id,
                span_id_by_citation_id,
            )
            if "[CITATION_REFS]" in filled:
                filled = filled.replace("[CITATION_REFS]", citation_refs)
        else:
            filled = self._fill_aggregate_placeholders(
                template,
                display_spans,
                citation_refs,
                citation_number_by_id,
                span_id_by_citation_id,
            )

        return filled.strip()

    def _fill_per_fact_placeholders(
        self,
        template: str,
        display_spans: list[dict[str, Any]],
        citation_spans: list[dict[str, Any]],
        citation_number_by_id: dict[str, int],
        span_id_by_citation_id: dict[str, str],
    ) -> str:
        fact_pattern = re.compile(r"\[(?:SPAN|FACT)_(\d+)\]")
        total_spans = display_spans + citation_spans

        def replace_fact(match):
            idx = int(match.group(1))
            if 1 <= idx <= len(total_spans):
                if idx <= len(display_spans):
                    span_data = display_spans[idx - 1]
                    return self._format_span(
                        span_data, idx, citation_number_by_id, span_id_by_citation_id,
                    )
                if self.citation_mode == "inline":
                    return self._render_marker(idx, total_spans[idx - 1])
                return ""
            return ""

        return fact_pattern.sub(replace_fact, template)

    def _fill_aggregate_placeholders(
        self,
        template: str,
        display_spans: list[dict[str, Any]],
        citation_refs: str,
        citation_number_by_id: dict[str, int],
        span_id_by_citation_id: dict[str, str],
    ) -> str:
        if display_spans:
            formatted = []
            for i, span_data in enumerate(display_spans, 1):
                block = self._format_span(
                    span_data, i, citation_number_by_id, span_id_by_citation_id,
                )
                if block:
                    formatted.append(block)

            display_content = (
                "\n\n".join(formatted)
                if formatted
                else "No relevant information found in the provided documents."
            )
        else:
            display_content = "No relevant information found in the provided documents."

        filled = template.replace("[DISPLAY_SPANS]", display_content)
        filled = filled.replace("[RELEVANT_SENTENCES]", display_content)

        if "[CITATION_REFS]" in filled:
            filled = filled.replace("[CITATION_REFS]", citation_refs)

        return filled

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
            span_data, citation_number_by_id, span_id_by_citation_id,
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
        lines = [line for line in text.strip().splitlines() if line.strip()]
        if len(lines) < 2:
            return False

        pipe_lines = sum(1 for line in lines if "|" in line)
        if pipe_lines < 2:
            return False

        return pipe_lines >= len(lines) / 2

    @staticmethod
    def _is_table_content(text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return False

        pipe_lines = sum(1 for line in lines if "|" in line)

        return pipe_lines >= len(lines) / 2

    @staticmethod
    def ensure_placeholder(template: str, placeholder: str = "[DISPLAY_SPANS]") -> str:
        acceptable = ["[RELEVANT_SENTENCES]", "[DISPLAY_SPANS]", "[SPAN_1]", "[FACT_1]"]

        if any(p in template for p in acceptable):
            return template

        if template.endswith(":"):
            return template + f"\n\n{placeholder}"
        return template + f"\n\n{placeholder}"
