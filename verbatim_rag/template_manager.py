"""
Clean and flexible template management for the Verbatim RAG system.
"""

import json
import os
import random
from typing import Optional, List, Dict, Any
from difflib import SequenceMatcher

try:  # Optional dependency; template generation fast-path works without it
    import openai  # type: ignore
except Exception:  # pragma: no cover
    openai = None  # fallback handled in code


class TemplateManager:
    """
    Clean template manager with three modes:
    - Single: One template for all questions
    - Random: Random selection from multiple templates
    - Question-specific: Templates matched to specific questions
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize template manager."""
        self.model = model

        # Template storage
        self._single_template: Optional[str] = None
        self._random_templates: List[str] = []
        self._question_templates: Dict[str, str] = {}

        # Current mode state
        self._mode = "single"  # single, random, question_specific

        # Set sensible default
        self._single_template = self._default_template()

    def _default_template(self) -> str:
        """Default template with markdown formatting."""
        return """## Response

Based on the available documents, here are the key findings:

[RELEVANT_SENTENCES]

---
*These excerpts are taken verbatim from the source documents to ensure accuracy.*"""

    def _validate_template(self, template: str) -> None:
        """Validate template has required placeholder.

        Accept either legacy aggregate placeholder [RELEVANT_SENTENCES] / [DISPLAY_SPANS]
        or at least one per-fact placeholder [FACT_1].
        """
        if any(p in template for p in ("[RELEVANT_SENTENCES]", "[DISPLAY_SPANS]")):
            return
        if "[FACT_1]" in template:
            return
        raise ValueError("Template must contain [RELEVANT_SENTENCES] or [DISPLAY_SPANS] or per-fact placeholders like [FACT_1]")

    # =============================================================================
    # Main interface - used by VerbatimRAG
    # =============================================================================

    def get_template(self, question: str = "") -> str:
        """
        Get template for a question based on current mode.
        This is the main method called by VerbatimRAG.
        """
        if self._mode == "single":
            return self._single_template or self._default_template()

        elif self._mode == "random":
            if self._random_templates:
                return random.choice(self._random_templates)
            return self._default_template()

        elif self._mode == "question_specific":
            # Try exact match
            if question in self._question_templates:
                return self._question_templates[question]

            # Try fuzzy match
            template = self._fuzzy_match_template(question)
            if template:
                return template

            # Fallback to default
            return self._default_template()

        return self._default_template()

    # =============================================================================
    # Mode configuration - used by users
    # =============================================================================

    def use_single_mode(self, template: str = None) -> None:
        """Use single template for all questions."""
        if template:
            self._validate_template(template)
            self._single_template = template
        elif not self._single_template:
            self._single_template = self._default_template()

        self._mode = "single"

    def use_random_mode(self, templates: List[str] = None) -> None:
        """Use random template selection."""
        if templates:
            for template in templates:
                self._validate_template(template)
            self._random_templates = templates
        elif not self._random_templates:
            # Generate some if none provided
            self.generate_random_templates(20)

        self._mode = "random"

    def use_question_specific_mode(self, templates: Dict[str, str] = None) -> None:
        """Use question-specific template matching."""
        if templates:
            for template in templates.values():
                self._validate_template(template)
            self._question_templates = templates

        self._mode = "question_specific"

    # =============================================================================
    # Template management
    # =============================================================================

    def set_single_template(self, template: str) -> None:
        """Set the single template."""
        self._validate_template(template)
        self._single_template = template
        if self._mode != "single":
            self._mode = "single"

    def add_random_template(self, template: str) -> None:
        """Add a template to random pool."""
        self._validate_template(template)
        self._random_templates.append(template)
        if self._mode != "random":
            self._mode = "random"

    def add_question_template(self, question: str, template: str) -> None:
        """Add question-specific template."""
        self._validate_template(template)
        self._question_templates[question] = template
        if self._mode != "question_specific":
            self._mode = "question_specific"

    def generate_random_templates(self, count: int = 20) -> None:
        """Generate diverse random templates using LLM."""
        templates = self._generate_templates_with_llm(count)
        self._random_templates = templates
        self._mode = "random"
        print(f"Generated {len(templates)} random templates")

    def generate_contextual_template(
        self, 
        question: str, 
        display_spans: List[str], 
        citation_count: int = 0
    ) -> str:
        """
        Generate a template based on the actual content that will fill it.
        
        :param question: The user's question
        :param display_spans: The spans that will be displayed verbatim
        :param citation_count: Number of additional citations (reference only)
        :return: A template with appropriate placeholders
        """
        if not display_spans:
            return self._default_template()
        
        # Build enumerated span preview (truncate each for token safety)
        span_lines = []
        for i, span in enumerate(display_spans, start=1):
            clean = span.replace('\n', ' ').strip()
            span_lines.append(f"{i}. {clean}")
        spans_block = "\n".join(span_lines)
        # If small number of spans we can deterministically build a simple template
        # without spending an LLM call (acts as a fast-path & safety fallback)
        # if len(display_spans) <= 8:
        #     bullets = []
        #     for i in range(1, len(display_spans) + 1):
        #         bullets.append(f"- **Fact {i}:** [FACT_{i}]")
        #     citation_line = "\n\n[CITATION_REFS]" if citation_count > 0 else ""
        #     return (
        #         "## Answer\n\n"
        #         "Below are the verbatim facts relevant to your question.\n\n"
        #         + "\n".join(bullets)
        #         + citation_line
        #     )

        # Fallback to aggregate placeholder generation via LLM if many spans
        prompt = f"""Generate a response template for this Q&A scenario (OUTPUT MUST BE VALID GITHUB-FLAVORED MARKDOWN):

Question: {question}

Content that will be inserted into the template:
- Total verbatim facts to show (display facts): {len(display_spans)}
- Full list of verbatim facts (truncated for reference only):
{spans_block}
- Additional citation-only facts (only numbers, no text shown): {citation_count}

Template strategy rules (Markdown correctness is critical):
- Use [DISPLAY_SPANS] exactly once for the aggregate of all verbatim spans.
- If citation-only facts exist, you MAY place [CITATION_REFS] exactly once where their numbers should appear, otherwise omit it.

Markdown formatting requirements:
- Use only GitHub-Flavored Markdown (GFM): headings (##, ###), paragraphs, bullet/numbered lists, bold/italic, blockquotes, and tables.
- Do NOT wrap the entire template in code fences.
- Every heading must be followed by a blank line unless immediately followed by a list.
- Placeholders must not be inside backticks, code blocks, or HTML tags.

Instructions:
- Intro: 1 concise sentence tying question to spans.
- Provide a section header then include the aggregate placeholder.
- Do NOT invent or paraphrase span content; placeholders stand in for verbatim content only.
- Avoid nested lists; keep structure shallow and clean.

Template requirements:
- Must contain [DISPLAY_SPANS].
- {"Include [CITATION_REFS] once" if citation_count > 0 else "Do NOT include [CITATION_REFS]"}.
- End without extra commentary like "Hope this helps".

Return ONLY the template text (no explanation)."""

        try:
            # Only attempt if openai client is available / configured
            if hasattr(openai, "chat"):
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                template = response.choices[0].message.content.strip()
            else:
                raise RuntimeError("OpenAI client not available")

            try:
                self._validate_template(template)
            except Exception:
                template = self._ensure_display_placeholder(template)

            if citation_count > 0 and "[CITATION_REFS]" not in template:
                template += "\n\n[CITATION_REFS]"
            elif citation_count == 0 and "[CITATION_REFS]" in template:
                template = template.replace("[CITATION_REFS]", "").strip()

            return template
        except Exception as e:
            print(f"Contextual template generation failed: {e}")
            return self._generate_fallback_template(citation_count > 0)


    def _ensure_display_placeholder(self, template: str) -> str:
        """Ensure template has the display placeholder."""
        if "[DISPLAY_SPANS]" not in template:
            # Try to add it naturally
            if template.endswith(":"):
                return template + "\n\n[DISPLAY_SPANS]"
            else:
                return template + "\n\n[DISPLAY_SPANS]"
        return template

    def _generate_fallback_template(self, has_citations: bool) -> str:
        """Generate a simple fallback template with markdown formatting."""
        template = """### Key Findings

Based on the available documents:

[DISPLAY_SPANS]"""
        
        if has_citations:
            template += "\n\n**Additional References:** [CITATION_REFS]"
        
        template += "\n\n---\n*All content above is extracted verbatim from source documents.*"
        
        return template

    # =============================================================================
    # Convenience template creators
    # =============================================================================

    @staticmethod
    def create_simple(intro: str = None, outro: str = None) -> str:
        """Create simple template."""
        intro = intro or "Thanks for your question! Based on the documents:"
        parts = [intro, "", "[RELEVANT_SENTENCES]"]

        if outro:
            parts.extend(["", outro])

        return "\n".join(parts)

    @staticmethod
    def create_academic() -> str:
        """Create academic-style template."""
        return "Based on the available literature:\n\n[RELEVANT_SENTENCES]\n\nThese findings provide evidence for the research question."

    @staticmethod
    def create_brief() -> str:
        """Create brief template."""
        return "Key points:\n\n[RELEVANT_SENTENCES]"

    # =============================================================================
    # Persistence and info
    # =============================================================================

    def save(self, filepath: str) -> None:
        """Save templates to file."""
        data = {
            "mode": self._mode,
            "single_template": self._single_template,
            "random_templates": self._random_templates,
            "question_templates": self._question_templates,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """Load templates from file."""
        if not os.path.exists(filepath):
            return

        with open(filepath, "r") as f:
            data = json.load(f)

        self._mode = data.get("mode", "single")
        self._single_template = data.get("single_template")
        self._random_templates = data.get("random_templates", [])
        self._question_templates = data.get("question_templates", {})

    def info(self) -> Dict[str, Any]:
        """Get current state info."""
        return {
            "mode": self._mode,
            "single_template_set": self._single_template is not None,
            "random_templates_count": len(self._random_templates),
            "question_templates_count": len(self._question_templates),
        }

    # =============================================================================
    # Internal helpers
    # =============================================================================

    def _fuzzy_match_template(self, question: str) -> Optional[str]:
        """Find best matching question template."""
        if not self._question_templates:
            return None

        best_match = None
        best_score = 0
        threshold = 0.6

        question_lower = question.lower()

        for q, template in self._question_templates.items():
            score = SequenceMatcher(None, question_lower, q.lower()).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = template

        return best_match

    def _generate_templates_with_llm(self, count: int) -> List[str]:
        """Generate diverse templates using LLM with markdown formatting."""
        prompt = f"""Create {count} diverse answer templates for a Q&A system using **Markdown formatting**. Each should:

1. Include exactly one [RELEVANT_SENTENCES] placeholder
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
            response = openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                # temperature=0.8,
            )

            text = response.choices[0].message.content
            templates = [t.strip() for t in text.split("\n") if t.strip()]

            # Filter valid templates
            valid = [t for t in templates if "[RELEVANT_SENTENCES]" in t]

            return valid[:count]  # Ensure we don't exceed requested count

        except Exception as e:
            print(f"LLM generation failed: {e}")
            # Fallback templates
            return [
                "Thanks for your question! Here's what I found:\n\n[RELEVANT_SENTENCES]",
                "Based on the documents:\n\n[RELEVANT_SENTENCES]",
                "Here are the key points:\n\n[RELEVANT_SENTENCES]",
                "According to the sources:\n\n[RELEVANT_SENTENCES]\n\nHope this helps!",
                "The documents reveal:\n\n[RELEVANT_SENTENCES]",
            ][:count]
