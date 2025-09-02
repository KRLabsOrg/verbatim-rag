"""
Random template strategy for the Verbatim RAG system.

Provides random template selection from a pool of diverse templates.
Templates can be generated using LLM or manually curated.
"""

import random
from typing import List, Dict, Any, Optional
from .base import TemplateStrategy
from .filler import TemplateFiller
from verbatim_core.llm_client import LLMClient


class RandomTemplate(TemplateStrategy):
    """
    Random template strategy that selects from a pool of templates.

    This strategy maintains a collection of templates and randomly selects
    one for each use. Templates can be manually added or generated using
    an LLM for diversity.
    """

    DEFAULT_TEMPLATES = [
        """## Key Findings

Based on the documents, here are the relevant facts:

[DISPLAY_SPANS]

*These excerpts are taken verbatim from the source materials.*""",
        """### Response

The following information addresses your question:

[DISPLAY_SPANS]

---
*Source: Verbatim extracts from provided documents*""",
        """## Answer

Here's what I found in the documents:

[DISPLAY_SPANS]

These excerpts are copied exactly from the source materials to ensure accuracy.""",
        """**Summary of Findings:**

[DISPLAY_SPANS]

*Note: All content above is extracted word-for-word from the source documents.*""",
        """### Document Analysis

The relevant information from the documents includes:

[DISPLAY_SPANS]

---
*These are verbatim excerpts to maintain accuracy.*""",
    ]

    def __init__(
        self, templates: List[str] = None, llm_client: Optional[LLMClient] = None
    ):
        """
        Initialize random template strategy.

        :param templates: Optional list of templates (uses defaults if None)
        :param llm_client: Optional LLM client for generating more templates
        """
        self.templates = templates or self.DEFAULT_TEMPLATES.copy()
        self.llm_client = llm_client
        self.filler = TemplateFiller()

        # Validate all templates
        for template in self.templates:
            self.validate_template(template)

    def generate(self, question: str, spans: List[str], citation_count: int = 0) -> str:
        """
        Randomly select a template from the pool.

        :param question: User's question (not used for selection)
        :param spans: List of spans (not used for selection)
        :param citation_count: Number of citation spans (not used for selection)
        :return: Randomly selected template
        """
        if not self.templates:
            return self._get_fallback_template()

        return random.choice(self.templates)

    def fill(
        self,
        template: str,
        display_spans: List[Dict[str, Any]],
        citation_spans: List[Dict[str, Any]],
    ) -> str:
        """
        Fill the template with actual span content.

        :param template: Template string with placeholders
        :param display_spans: Spans to display with full text
        :param citation_spans: Spans for citation reference only
        :return: Filled template
        """
        return self.filler.fill(template, display_spans, citation_spans)

    def save_state(self) -> Dict[str, Any]:
        """
        Save the current template pool.

        :return: Dictionary with template pool state
        """
        return {"type": "random", "templates": self.templates}

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load template pool from saved state.

        :param state: Dictionary with template pool configuration
        """
        if "templates" in state:
            self.templates = state["templates"]
            # Validate all loaded templates
            for template in self.templates:
                try:
                    self.validate_template(template)
                except ValueError as e:
                    print(f"Warning: Invalid template removed during load: {e}")
                    self.templates.remove(template)

    def add_template(self, template: str) -> None:
        """
        Add a new template to the pool.

        :param template: Template to add
        :raises ValueError: If template is invalid
        """
        self.validate_template(template)
        if template not in self.templates:
            self.templates.append(template)

    def remove_template(self, template: str) -> bool:
        """
        Remove a template from the pool.

        :param template: Template to remove
        :return: True if template was found and removed
        """
        if template in self.templates:
            self.templates.remove(template)
            return True
        return False

    def clear_templates(self) -> None:
        """Clear all templates from the pool."""
        self.templates.clear()

    def get_template_count(self) -> int:
        """
        Get the number of templates in the pool.

        :return: Number of templates
        """
        return len(self.templates)

    def generate_diverse_templates(self, count: int = 10) -> None:
        """
        Generate diverse templates using LLM and add to pool.

        :param count: Number of templates to generate
        :raises ValueError: If no LLM client is available
        """
        if not self.llm_client:
            raise ValueError("LLM client required for template generation")

        prompt = f"""Create {count} diverse answer templates for a Q&A system using **Markdown formatting**. Each should:

1. Include exactly one [DISPLAY_SPANS] placeholder
2. Have different styles (casual, formal, brief, detailed, etc.)
3. Work for any question type
4. Show variety in structure and tone
5. **Use markdown formatting** (headers, bold, italic, lists) for better presentation

Use markdown elements like:
- **Bold text** for emphasis
- *Italic text* for subtle emphasis  
- ### Headers for sections
- - Bullet points for lists
- > Blockquotes for highlighting

Return only the templates, one per line. No numbering or explanations."""

        try:
            response = self.llm_client.complete(prompt, temperature=0.8)
            templates = [t.strip() for t in response.split("\n") if t.strip()]

            # Filter and add valid templates
            added_count = 0
            for template in templates:
                try:
                    self.validate_template(template)
                    if template not in self.templates:
                        self.templates.append(template)
                        added_count += 1
                except ValueError:
                    continue  # Skip invalid templates

            print(f"Generated and added {added_count} new templates to the pool")

        except Exception as e:
            print(f"Template generation failed: {e}")
            # Add some fallback templates if generation fails
            fallback_templates = [
                "Here's what the documents reveal:\n\n[DISPLAY_SPANS]",
                "Key findings from the sources:\n\n[DISPLAY_SPANS]",
                "The documents indicate:\n\n[DISPLAY_SPANS]",
            ]

            for template in fallback_templates:
                if template not in self.templates:
                    self.templates.append(template)

    def _get_fallback_template(self) -> str:
        """Get a fallback template when pool is empty."""
        return """## Response

Based on the available documents:

[DISPLAY_SPANS]

---
*These excerpts are taken verbatim from the source documents.*"""
