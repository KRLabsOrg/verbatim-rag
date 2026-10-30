"""Template-driven structured extraction for VerbatimRAG.

This strategy uses the template to guide extraction. The LLM receives the full
template structure and extracts spans organized by placeholder.
"""

import re
from typing import Any, ClassVar

from verbatim_core.templates.base import TemplateStrategy


class StructuredTemplate(TemplateStrategy):
    """Structured extraction template strategy.

    The template controls what gets extracted - the LLM returns per-placeholder
    spans which are then used to fill the template.

    Example template:
        ## Methodology
        [METHODOLOGY]

        ## Results
        [RESULTS]
    """

    PLACEHOLDER_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"\[([A-Z][A-Z0-9_]+)\]")
    SYSTEM_PLACEHOLDERS: ClassVar[set[str]] = {"DISPLAY_SPANS", "RELEVANT_SENTENCES", "CITATION_REFS"}

    # Standard mappings from placeholder names to extraction hints
    STANDARD_MAPPINGS: ClassVar[dict[str, str]] = {
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
        rag_system: Any = None,  # noqa: ANN401  # duck-typed RAG system
        template: str | None = None,
        placeholder_mappings: dict[str, str] | None = None,
        citation_mode: str = "inline",
    ) -> None:
        """Initialize the structured template strategy."""
        self.rag_system = rag_system
        self.template = template
        self.custom_mappings = placeholder_mappings or {}
        self.citation_mode = citation_mode

    # ------------------------------------------------------------------ helpers
    def set_rag_system(self, rag_system: Any) -> None:  # noqa: ANN401  # duck-typed RAG system
        """Set the RAG system used for structured extraction."""
        self.rag_system = rag_system

    def set_template(self, template: str) -> None:
        """Validate and set the template."""
        self.validate_template(template)
        self.template = template

    def validate_template(self, template: str) -> None:
        """Validate that the template has semantic or standard placeholders."""
        if not template or not template.strip():
            msg = "Template cannot be empty"
            raise ValueError(msg)

        has_semantic = bool(self.PLACEHOLDER_PATTERN.search(template))
        has_standard = any(p in template for p in ("[DISPLAY_SPANS]", "[RELEVANT_SENTENCES]", "[SPAN_1]", "[FACT_1]"))

        if not (has_semantic or has_standard):
            msg = (
                "Structured templates must contain semantic placeholders like "
                "[METHODOLOGY] or standard placeholders such as [DISPLAY_SPANS]"
            )
            raise ValueError(
                msg,
            )

    def add_placeholder_mapping(self, placeholder: str, hint: str) -> None:
        """Add custom mapping from placeholder name to extraction hint."""
        self.custom_mappings[placeholder] = hint

    def get_placeholder_mappings(self) -> dict[str, str]:
        """Get all placeholder mappings (standard + custom)."""
        return {**self.STANDARD_MAPPINGS, **self.custom_mappings}

    def get_placeholder_hints(self) -> dict[str, str]:
        """Get hints for all placeholders in the current template.

        Returns dict mapping placeholder names to their extraction hints.
        """
        if not self.template:
            return {}

        hints = {}
        all_mappings = self.get_placeholder_mappings()

        for match in self.PLACEHOLDER_PATTERN.finditer(self.template):
            name = match.group(1)
            if name.startswith(("FACT_", "SPAN_")):
                continue
            if name in self.SYSTEM_PLACEHOLDERS:
                continue

            # Get hint from mappings or generate from name
            hint = all_mappings.get(name, name.replace("_", " ").lower())
            hints[name] = hint

        return hints

    def set_citation_mode(self, citation_mode: str) -> None:
        """Set the citation mode."""
        self.citation_mode = citation_mode

    # ---------------------------------------------------------------- TemplateStrategy interface
    def generate(self, question: str, spans: list[str], citation_count: int = 0) -> str:
        """Return the configured template."""
        if not self.template:
            msg = "Structured template not set"
            raise ValueError(msg)
        return self.template

    def fill(
        self,
        template: str,
        display_spans: list[dict[str, Any]],
        citation_spans: list[dict[str, Any]],
    ) -> str:
        """Return the template unchanged; standard filling is unused here."""
        # Standard fill not used for structured mode
        return template

    def save_state(self) -> dict[str, Any]:
        """Return the serializable state of this strategy."""
        return {
            "type": "structured",
            "template": self.template,
            "placeholder_mappings": self.custom_mappings,
            "citation_mode": self.citation_mode,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore strategy state from a saved dictionary."""
        self.template = state.get("template", self.template)
        self.custom_mappings = state.get("placeholder_mappings", {})
        if "citation_mode" in state:
            self.citation_mode = state["citation_mode"]

    # ---------------------------------------------------------------- structured filling
    def fill_with_spans(self, span_map: dict[str, list]) -> str:
        """Fill the template with per-placeholder spans.

        :param span_map: Dict mapping placeholder names to lists of spans
                         Supports both old format (list of strings) and
                         new format (list of {text, doc} dicts)
        :return: Filled template
        """
        if not self.template:
            msg = "Template not set"
            raise ValueError(msg)

        result = self.template

        # Global citation counter for consistent numbering across all placeholders
        citation_counter = [1]  # Use list to allow mutation in nested function

        # Find all placeholders in order (not reversed) to maintain citation order
        matches = list(self.PLACEHOLDER_PATTERN.finditer(self.template))

        # Process in reverse order for string replacement, but track citation numbers in forward order
        # First pass: calculate citation numbers for each placeholder
        placeholder_citations = {}
        for match in matches:
            name = match.group(1)
            if name.startswith("FACT_") or name in self.SYSTEM_PLACEHOLDERS:
                continue

            items = span_map.get(name, [])
            texts = self._extract_texts(items)

            # Assign citation numbers
            if texts:
                start_num = citation_counter[0]
                placeholder_citations[name] = (texts, start_num)
                citation_counter[0] += len(texts)
            else:
                placeholder_citations[name] = ([], 0)

        # Second pass: replace placeholders with formatted spans
        for match in reversed(matches):
            name = match.group(1)
            if name.startswith("FACT_") or name in self.SYSTEM_PLACEHOLDERS:
                continue

            texts, start_num = placeholder_citations.get(name, ([], 0))
            replacement = self._format_spans_with_offset(texts, start_num)
            result = result[: match.start()] + replacement + result[match.end() :]

        return result

    def _extract_texts(self, items: list) -> list[str]:
        """Extract text strings from items (handles both string and dict formats)."""
        texts = []
        for item in items:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = item.get("text", "").strip()
            else:
                continue
            if text:
                texts.append(text)
        return texts

    def _format_spans_with_offset(self, texts: list[str], start_num: int) -> str:
        """Format spans for display with global citation numbering.

        :param texts: List of span texts
        :param start_num: Starting citation number for this placeholder
        """
        if not texts:
            return "(no relevant information found)"

        if self.citation_mode == "inline":
            if len(texts) == 1:
                return f"[{start_num}] {texts[0]}"
            # Global sequential numbering
            return "\n\n".join(f"[{start_num + i}] {text}" for i, text in enumerate(texts))

        # hidden citation mode - just text
        if len(texts) == 1:
            return texts[0]
        return "\n\n".join(texts)

    # ---------------------------------------------------------------- async fill (deprecated, use RAG.query)
    async def fill_async(
        self,
        question: str,
        template: str | None = None,
        placeholder_mappings: dict[str, str] | None = None,
    ) -> str:
        """Fill template via structured extraction.

        Note: Prefer using rag.query_async() which handles this automatically
        when in structured mode. This method is kept for backwards compatibility.
        """
        if not self.rag_system:
            msg = "RAG system not set"
            raise ValueError(msg)

        if template:
            self.set_template(template)
        if placeholder_mappings:
            for name, hint in placeholder_mappings.items():
                self.add_placeholder_mapping(name, hint)

        # Delegate to RAG query which handles structured mode
        response = await self.rag_system.query_async(question)
        return response.answer
