"""Static template strategy (copy)."""

from __future__ import annotations

from typing import List, Dict, Any
from .base import TemplateStrategy


class StaticTemplate(TemplateStrategy):
    def generate(self, question: str, spans: List[str], citation_count: int) -> str:
        # Simple header + placeholders for citations
        return "Answer:\n[DISPLAY_SPANS]\n[CITATION_REFS]"

    def fill(
        self,
        template: str,
        display_spans: List[Dict[str, Any]],
        citation_spans: List[Dict[str, Any]],
    ) -> str:
        # Render display spans with numbers
        rendered = []
        for i, span in enumerate(display_spans, start=1):
            rendered.append(f"[{i}] {span['text']}")

        display_block = "\n".join(rendered) if rendered else "(no direct spans found)"
        citations_block = " ".join(
            f"[{i}]"
            for i in range(
                len(display_spans) + 1, len(display_spans) + len(citation_spans) + 1
            )
        )

        return template.replace("[DISPLAY_SPANS]", display_block).replace(
            "[CITATION_REFS]", citations_block
        )
