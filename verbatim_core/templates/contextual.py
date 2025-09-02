"""Contextual (LLM-generated) template strategy (copy)."""

from __future__ import annotations

from typing import List, Dict, Any
from .base import TemplateStrategy
from verbatim_core.llm_client import LLMClient


class ContextualTemplate(TemplateStrategy):
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client

    def generate(self, question: str, spans: List[str], citation_count: int) -> str:
        if not self.llm_client:
            # Fallback to simple template
            return "Answer:\n[DISPLAY_SPANS]\n[CITATION_REFS]"
        prompt = (
            "Create a concise answer template that will place numbered verbatim facts "
            "inline where appropriate. Use placeholders like [FACT_1], [FACT_2]. "
            f"Question: {question}. Facts: {spans}. Extra citations: {citation_count}."
        )
        t = self.llm_client.simple_complete(prompt)
        return t or "Answer:\n[DISPLAY_SPANS]\n[CITATION_REFS]"

    async def generate_async(
        self, question: str, spans: List[str], citation_count: int
    ) -> str:
        if not self.llm_client:
            return "Answer:\n[DISPLAY_SPANS]\n[CITATION_REFS]"
        prompt = (
            "Create a concise answer template that will place numbered verbatim facts "
            "inline where appropriate. Use placeholders like [FACT_1], [FACT_2]. "
            f"Question: {question}. Facts: {spans}. Extra citations: {citation_count}."
        )
        t = await self.llm_client.simple_complete_async(prompt)
        return t or "Answer:\n[DISPLAY_SPANS]\n[CITATION_REFS]"

    def fill(
        self,
        template: str,
        display_spans: List[Dict[str, Any]],
        citation_spans: List[Dict[str, Any]],
    ) -> str:
        # Replace FACT_n placeholders in order; remaining facts go into [CITATION_REFS]
        filled = template
        for i, span in enumerate(display_spans, start=1):
            filled = filled.replace(
                f"[FACT_{i}]", span["text"]
            )  # best-effort replacement

        # Backfill if template has no FACT placeholders
        if filled == template:
            rendered = [
                f"[{i}] {s['text']}" for i, s in enumerate(display_spans, start=1)
            ]
            facts_block = "\n".join(rendered) if rendered else "(no direct spans found)"
            filled = filled.replace("[DISPLAY_SPANS]", facts_block)

        # Append citation-only references
        citations_block = " ".join(
            f"[{i}]"
            for i in range(
                len(display_spans) + 1, len(display_spans) + len(citation_spans) + 1
            )
        )
        filled = filled.replace("[CITATION_REFS]", citations_block)
        return filled
