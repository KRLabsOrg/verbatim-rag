"""
Template filling utilities for the Verbatim RAG system.

Handles aggregate placeholders ([DISPLAY_SPANS], [RELEVANT_SENTENCES]) and
per-fact placeholders ([FACT_1], ...). Supports toggling inline citation
numbering on or off so callers can choose between numbered excerpts or
clean text with citations handled separately.
"""

import re
from typing import List, Dict, Any


class TemplateFiller:
    ALLOWED_MODES = {"inline", "hidden"}

    def __init__(self, citation_mode: str = "inline"):
        self.set_citation_mode(citation_mode)

    def set_citation_mode(self, citation_mode: str) -> None:
        if citation_mode not in self.ALLOWED_MODES:
            raise ValueError(
                f"Unsupported citation mode: {citation_mode}. "
                f"Allowed values: {sorted(self.ALLOWED_MODES)}"
            )
        self.citation_mode = citation_mode

    def fill(
        self,
        template: str,
        display_spans: List[Dict[str, Any]],
        citation_spans: List[Dict[str, Any]],
    ) -> str:
        if not template:
            return ""

        citation_refs = ""
        if citation_spans and self.citation_mode == "inline":
            start_num = len(display_spans) + 1
            end_num = len(display_spans) + len(citation_spans)
            citation_refs = " ".join(f"[{i}]" for i in range(start_num, end_num + 1))

        fact_pattern = re.compile(r"\[FACT_(\d+)\]")
        if fact_pattern.search(template):
            filled = self._fill_per_fact_placeholders(
                template, display_spans, citation_spans
            )
            if "[CITATION_REFS]" in filled:
                filled = filled.replace("[CITATION_REFS]", citation_refs)
        else:
            filled = self._fill_aggregate_placeholders(
                template, display_spans, citation_refs
            )

        return filled.strip()

    def _fill_per_fact_placeholders(
        self,
        template: str,
        display_spans: List[Dict[str, Any]],
        citation_spans: List[Dict[str, Any]],
    ) -> str:
        fact_pattern = re.compile(r"\[FACT_(\d+)\]")
        total_spans = display_spans + citation_spans

        def replace_fact(match):
            idx = int(match.group(1))
            if 1 <= idx <= len(total_spans):
                if idx <= len(display_spans):
                    span_text = display_spans[idx - 1].get("text", "")
                    return self._format_span(span_text, idx, self._is_table(span_text))
                else:
                    return f"[{idx}]" if self.citation_mode == "inline" else ""
            return ""

        return fact_pattern.sub(replace_fact, template)

    def _fill_aggregate_placeholders(
        self, template: str, display_spans: List[Dict[str, Any]], citation_refs: str
    ) -> str:
        if display_spans:
            formatted = []
            for i, span_data in enumerate(display_spans, 1):
                span_text = span_data.get("text", "")
                block = self._format_span(span_text, i, self._is_table(span_text))
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

    def _format_span(self, text: str, index: int, is_table: bool = False) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""

        if self.citation_mode == "inline":
            if is_table:
                return f"[{index}]\n\n{cleaned}"
            return f"[{index}] {cleaned}"

        return cleaned

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
        acceptable = ["[RELEVANT_SENTENCES]", "[DISPLAY_SPANS]", "[FACT_1]"]

        if any(p in template for p in acceptable):
            return template

        if template.endswith(":"):
            return template + f"\n\n{placeholder}"
        return template + f"\n\n{placeholder}"
