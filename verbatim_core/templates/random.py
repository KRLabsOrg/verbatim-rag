"""Random template strategy (copy)."""

from __future__ import annotations

import random
from typing import List, Dict, Any
from .base import TemplateStrategy
from verbatim_core.llm_client import LLMClient


class RandomTemplate(TemplateStrategy):
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client
        self.templates = [
            "Answer:\n[DISPLAY_SPANS]\n[CITATION_REFS]",
            "Key facts:\n[DISPLAY_SPANS]\nReferences: [CITATION_REFS]",
        ]

    def generate(self, question: str, spans: List[str], citation_count: int) -> str:
        if not self.llm_client:
            return random.choice(self.templates)
        prompt = f"Generate a short answer template for question: {question}"
        t = self.llm_client.simple_complete(prompt)
        return t or random.choice(self.templates)

    def fill(
        self,
        template: str,
        display_spans: List[Dict[str, Any]],
        citation_spans: List[Dict[str, Any]],
    ) -> str:
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
