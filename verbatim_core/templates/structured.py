"""
Template-driven structured extraction for VerbatimRAG.

This strategy uses the template to guide extraction. The LLM receives the full
template structure and extracts spans organized by placeholder.
"""

import re
from typing import Dict, Any, List, Optional

from .base import TemplateStrategy


class StructuredTemplate(TemplateStrategy):
    """
    Structured extraction template strategy.

    The template controls what gets extracted - the LLM returns per-placeholder
    spans which are then used to fill the template.

    Example template:
        ## Methodology
        [METHODOLOGY]

        ## Results
        [RESULTS]
    """

    PLACEHOLDER_PATTERN = re.compile(r"\[([A-Z][A-Z0-9_]+)\]")
    SYSTEM_PLACEHOLDERS = {"DISPLAY_SPANS", "RELEVANT_SENTENCES", "CITATION_REFS"}

    # Standard mappings from placeholder names to extraction hints
    STANDARD_MAPPINGS: Dict[str, str] = {
        "METHODOLOGY": "methodology or methods used",
        "METHOD": "method used",
        "APPROACH": "approach taken",
        "RESULTS": "results or findings",
        "FINDINGS": "findings",
        "CONCLUSION": "conclusion",
        "CONTRIBUTIONS": "main contributions",
        "LIMITATIONS": "limitations",
        "FUTURE_WORK": "future work suggested",
        "BACKGROUND": "background information",
        "DATASET": "dataset used",
        "METRICS": "metrics used",
        "ACCURACY": "accuracy achieved",
        "PERFORMANCE": "performance results",
        "BASELINE": "baseline used",
        "RELATED_WORK": "related work discussed",
        "IMPLEMENTATION": "implementation details",
        "EVALUATION": "evaluation approach",
    }

    def __init__(
        self,
        rag_system=None,
        template: Optional[str] = None,
        placeholder_mappings: Optional[Dict[str, str]] = None,
        citation_mode: str = "inline",
    ):
        self.rag_system = rag_system
        self.template = template
        self.custom_mappings = placeholder_mappings or {}
        self.citation_mode = citation_mode

    # ------------------------------------------------------------------ helpers
    def set_rag_system(self, rag_system) -> None:
        self.rag_system = rag_system

    def set_template(self, template: str) -> None:
        self.validate_template(template)
        self.template = template

    def validate_template(self, template: str) -> None:
        if not template or not template.strip():
            raise ValueError("Template cannot be empty")

        has_semantic = bool(self.PLACEHOLDER_PATTERN.search(template))
        has_standard = any(
            p in template
            for p in ("[DISPLAY_SPANS]", "[RELEVANT_SENTENCES]", "[FACT_1]")
        )

        if not (has_semantic or has_standard):
            raise ValueError(
                "Structured templates must contain semantic placeholders like "
                "[METHODOLOGY] or standard placeholders such as [DISPLAY_SPANS]"
            )

    def add_placeholder_mapping(self, placeholder: str, hint: str) -> None:
        """Add custom mapping from placeholder name to extraction hint."""
        self.custom_mappings[placeholder] = hint

    def get_placeholder_mappings(self) -> Dict[str, str]:
        """Get all placeholder mappings (standard + custom)."""
        return {**self.STANDARD_MAPPINGS, **self.custom_mappings}

    def get_placeholder_hints(self) -> Dict[str, str]:
        """
        Get hints for all placeholders in the current template.

        Returns dict mapping placeholder names to their extraction hints.
        """
        if not self.template:
            return {}

        hints = {}
        all_mappings = self.get_placeholder_mappings()

        for match in self.PLACEHOLDER_PATTERN.finditer(self.template):
            name = match.group(1)
            if name.startswith("FACT_"):
                continue
            if name in self.SYSTEM_PLACEHOLDERS:
                continue

            # Get hint from mappings or generate from name
            hint = all_mappings.get(name, name.replace("_", " ").lower())
            hints[name] = hint

        return hints

    def set_citation_mode(self, citation_mode: str) -> None:
        self.citation_mode = citation_mode

    # ---------------------------------------------------------------- TemplateStrategy interface
    def generate(self, question: str, spans: List[str], citation_count: int = 0) -> str:
        if not self.template:
            raise ValueError("Structured template not set")
        return self.template

    def fill(
        self,
        template: str,
        display_spans: List[Dict[str, Any]],
        citation_spans: List[Dict[str, Any]],
    ) -> str:
        # Standard fill not used for structured mode
        return template

    def save_state(self) -> Dict[str, Any]:
        return {
            "type": "structured",
            "template": self.template,
            "placeholder_mappings": self.custom_mappings,
            "citation_mode": self.citation_mode,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.template = state.get("template", self.template)
        self.custom_mappings = state.get("placeholder_mappings", {})
        if "citation_mode" in state:
            self.citation_mode = state["citation_mode"]

    # ---------------------------------------------------------------- structured filling
    def fill_with_spans(self, span_map: Dict[str, List[str]]) -> str:
        """
        Fill the template with per-placeholder spans.

        :param span_map: Dict mapping placeholder names to lists of spans
        :return: Filled template
        """
        if not self.template:
            raise ValueError("Template not set")

        result = self.template

        # Find all placeholders and replace them
        for match in reversed(list(self.PLACEHOLDER_PATTERN.finditer(self.template))):
            name = match.group(1)
            if name.startswith("FACT_") or name in self.SYSTEM_PLACEHOLDERS:
                continue

            spans = span_map.get(name, [])
            replacement = self._format_spans(spans)
            result = result[: match.start()] + replacement + result[match.end() :]

        return result

    def _format_spans(self, spans: List[str]) -> str:
        """Format spans for display."""
        if not spans:
            return "(no relevant information found)"

        cleaned = [s.strip() for s in spans if s.strip()]
        if not cleaned:
            return "(no relevant information found)"

        if self.citation_mode == "inline":
            if len(cleaned) == 1:
                return cleaned[0]
            return "\n\n".join(f"[{i}] {span}" for i, span in enumerate(cleaned, 1))

        # hidden citation mode
        if len(cleaned) == 1:
            return cleaned[0]
        return "\n\n".join(cleaned)

    # ---------------------------------------------------------------- async fill (deprecated, use RAG.query)
    async def fill_async(
        self,
        question: str,
        template: Optional[str] = None,
        placeholder_mappings: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Fill template via structured extraction.

        Note: Prefer using rag.query_async() which handles this automatically
        when in structured mode. This method is kept for backwards compatibility.
        """
        if not self.rag_system:
            raise ValueError("RAG system not set")

        if template:
            self.set_template(template)
        if placeholder_mappings:
            for name, hint in placeholder_mappings.items():
                self.add_placeholder_mapping(name, hint)

        # Delegate to RAG query which handles structured mode
        response = await self.rag_system.query_async(question)
        return response.answer
