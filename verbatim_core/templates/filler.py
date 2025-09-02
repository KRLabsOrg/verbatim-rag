"""Template filler utilities (copy)."""

from __future__ import annotations

from typing import List, Dict, Any


class TemplateFiller:
    @staticmethod
    def fill(
        template: str,
        display_spans: List[Dict[str, Any]],
        citation_spans: List[Dict[str, Any]],
    ) -> str:
        rendered = [f"[{i}] {s['text']}" for i, s in enumerate(display_spans, start=1)]
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
