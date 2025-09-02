"""
Template filling utilities for the Verbatim RAG system.

This module provides the TemplateFiller class which handles the complex logic
of replacing placeholders in templates with actual content, including support
for per-fact placeholders, table detection, and proper markdown formatting.
"""

import re
from typing import List, Dict, Any


class TemplateFiller:
    """
    Handles template placeholder replacement with proper formatting.

    Supports both legacy aggregate placeholders ([DISPLAY_SPANS], [RELEVANT_SENTENCES])
    and modern per-fact placeholders ([FACT_1], [FACT_2], etc.).
    """

    @staticmethod
    def fill(
        template: str,
        display_spans: List[Dict[str, Any]],
        citation_spans: List[Dict[str, Any]],
    ) -> str:
        """
        Fill template with display spans and citation references.

        :param template: Template string with placeholders
        :param display_spans: Spans to display verbatim with metadata
        :param citation_spans: Spans for citation reference only
        :return: Filled template
        """
        if not template:
            return ""

        # Handle citation references
        citation_refs = ""
        if citation_spans:
            start_num = len(display_spans) + 1
            end_num = len(display_spans) + len(citation_spans)
            citation_refs = " ".join(f"[{i}]" for i in range(start_num, end_num + 1))

        # Check for per-fact placeholders
        fact_pattern = re.compile(r"\[FACT_(\d+)\]")
        total_spans = display_spans + citation_spans

        if fact_pattern.search(template):
            # Use per-fact placeholder replacement
            filled_template = TemplateFiller._fill_per_fact_placeholders(
                template, display_spans, citation_spans
            )

            # Still support [CITATION_REFS] if present
            if "[CITATION_REFS]" in filled_template and citation_spans:
                filled_template = filled_template.replace(
                    "[CITATION_REFS]", citation_refs
                )
            elif "[CITATION_REFS]" in filled_template:
                filled_template = filled_template.replace("[CITATION_REFS]", "")
        else:
            # Use aggregate placeholder replacement
            filled_template = TemplateFiller._fill_aggregate_placeholders(
                template, display_spans, citation_refs
            )

        return filled_template.strip()

    @staticmethod
    def _fill_per_fact_placeholders(
        template: str,
        display_spans: List[Dict[str, Any]],
        citation_spans: List[Dict[str, Any]],
    ) -> str:
        """Fill per-fact placeholders like [FACT_1], [FACT_2], etc."""
        fact_pattern = re.compile(r"\[FACT_(\d+)\]")
        total_spans = display_spans + citation_spans

        def replace_fact(match):
            idx = int(match.group(1))
            if 1 <= idx <= len(total_spans):
                if idx <= len(display_spans):
                    # Display span with verbatim text
                    span_text = display_spans[idx - 1]["text"]
                    if TemplateFiller._is_table(span_text):
                        # Place citation number on its own line, blank line, then table
                        return f"[{idx}]\n\n{span_text.strip()}"
                    else:
                        return f"[{idx}] {span_text}"
                else:
                    # Citation-only span - just the number
                    return f"[{idx}]"
            return ""  # Remove placeholder if out of range

        return fact_pattern.sub(replace_fact, template)

    @staticmethod
    def _fill_aggregate_placeholders(
        template: str, display_spans: List[Dict[str, Any]], citation_refs: str
    ) -> str:
        """Fill aggregate placeholders like [DISPLAY_SPANS], [RELEVANT_SENTENCES]."""
        # Format display spans
        if display_spans:
            formatted_content = []
            for i, span_data in enumerate(display_spans, 1):
                span_text = span_data["text"]
                if TemplateFiller._is_table(span_text):
                    # Tables need special formatting
                    block = f"[{i}]\n\n{span_text.strip()}"
                else:
                    block = f"[{i}] {span_text}"
                formatted_content.append(block)
            display_content = "\n\n".join(formatted_content)
        else:
            display_content = "No relevant information found in the provided documents."

        # Replace all aggregate placeholders
        filled = template.replace("[DISPLAY_SPANS]", display_content)
        filled = filled.replace("[RELEVANT_SENTENCES]", display_content)

        # Handle citation references
        if "[CITATION_REFS]" in filled:
            filled = filled.replace("[CITATION_REFS]", citation_refs)

        return filled

    @staticmethod
    def _format_display_spans(spans: List[Dict[str, Any]]) -> str:
        """
        Format display spans for presentation.

        :param spans: Display spans with text and metadata
        :return: Formatted string
        """
        if not spans:
            return "(No relevant information found)"

        # Check if content looks like structured data (tables, lists)
        formatted_lines = []

        for i, span in enumerate(spans, 1):
            content = span.get("text", "").strip()
            if not content:
                continue

            # Detect table-like content
            if TemplateFiller._is_table_content(content):
                formatted_lines.append(f"[{i}] {content}")
            else:
                # Regular text content with proper quotation
                formatted_lines.append(f'[{i}] "{content}"')

        if not formatted_lines:
            return "(No relevant information found)"

        return "\n\n".join(formatted_lines)

    @staticmethod
    def _is_table(text: str) -> bool:
        """
        Heuristic to detect markdown tables in a span.

        :param text: Text to check
        :return: True if text appears to be a markdown table
        """
        lines = [line for line in text.strip().splitlines() if line.strip()]
        if len(lines) < 2:
            return False

        # Count lines with pipe characters
        pipe_lines = sum(1 for line in lines if "|" in line)
        if pipe_lines < 2:
            return False

        # Basic table detection - more than half the lines have pipes
        return pipe_lines >= len(lines) / 2

    @staticmethod
    def ensure_placeholder(template: str, placeholder: str = "[DISPLAY_SPANS]") -> str:
        """
        Ensure template has at least one acceptable placeholder.

        :param template: Template to check
        :param placeholder: Placeholder to add if none found
        :return: Template with placeholder guaranteed
        """
        acceptable = ["[RELEVANT_SENTENCES]", "[DISPLAY_SPANS]", "[FACT_1]"]

        if any(p in template for p in acceptable):
            return template

        # Add placeholder naturally
        if template.endswith(":"):
            return template + f"\n\n{placeholder}"
        else:
            return template + f"\n\n{placeholder}"
