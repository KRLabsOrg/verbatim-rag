"""Question-specific template strategy for the Verbatim RAG system.

Provides template matching based on example questions using semantic similarity.
Users define templates paired with example questions, and the system automatically
selects the best-matching template for each query.
"""

import logging
from typing import Any

from .base import TemplateStrategy
from .filler import TemplateFiller

logger = logging.getLogger(__name__)


class QuestionSpecificTemplate(TemplateStrategy):
    """Question-specific template strategy using semantic similarity matching.

    Maintains a list of (template, example_questions) pairs. When a query arrives,
    computes similarity to all examples and selects the template from the
    best-matching pair.

    Example usage:
        templates = [
            {
                "template": "Methodology: [RELEVANT_SENTENCES]",
                "examples": ["What methods were used?", "How was this done?"]
            },
            {
                "template": "Key Findings: [RELEVANT_SENTENCES]",
                "examples": ["What were the results?", "What did they find?"]
            }
        ]
        strategy = QuestionSpecificTemplate(templates)
    """

    DEFAULT_TEMPLATE = """## Response

Based on the available documents:

[DISPLAY_SPANS]

---
*These excerpts are taken verbatim from the source documents to ensure accuracy.*"""

    def __init__(
        self,
        templates: list[dict[str, Any]] | None = None,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        citation_mode: str = "inline",
    ) -> None:
        """Initialize question-specific template strategy.

        :param templates: List of template definitions, each with 'template' and 'examples' keys
        :param model_name: SentenceTransformer model name for embedding
        :param device: Device to run model on ("cpu" or "cuda")
        """
        self.templates = templates or []
        self.model_name = model_name
        self.device = device
        self.citation_mode = citation_mode
        self.filler = TemplateFiller(citation_mode=citation_mode)
        self._embedder = None
        self._example_embeddings = None  # Will store stacked embeddings for efficient matching

        if self.templates:
            self._validate_and_embed_templates()

    def set_citation_mode(self, citation_mode: str) -> None:
        """Set the citation mode and propagate it to the filler."""
        self.citation_mode = citation_mode
        self.filler.set_citation_mode(citation_mode)

    def _get_embedder(self) -> Any:  # noqa: ANN401  # optional dependency type (SentenceTransformer)
        """Lazy load the embedding model."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(self.model_name, device=self.device)
                logger.info("Loaded embedding model: %s", self.model_name)
            except ImportError as err:
                msg = (
                    "sentence-transformers required for question-specific templates. "
                    "Install with: pip install sentence-transformers"
                )
                raise ImportError(
                    msg,
                ) from err
        return self._embedder

    def _validate_and_embed_templates(self) -> None:
        """Validate template format and precompute example embeddings."""
        if not self.templates:
            return

        embedder = self._get_embedder()

        # Collect all examples and their template indices
        all_examples = []
        template_indices = []  # Which template each example belongs to

        for idx, config in enumerate(self.templates):
            if "template" not in config:
                msg = f"Template config {idx} missing 'template' key"
                raise ValueError(msg)
            if "examples" not in config or not config["examples"]:
                msg = f"Template config {idx} missing or empty 'examples'"
                raise ValueError(msg)

            # Validate template has required placeholders
            self.validate_template(config["template"])

            # Track which template these examples belong to
            examples = config["examples"]
            all_examples.extend(examples)
            template_indices.extend([idx] * len(examples))

        # Embed all examples at once (more efficient)
        embeddings = embedder.encode(all_examples)
        self._example_embeddings = {
            "embeddings": embeddings,  # numpy array of shape (N, embedding_dim)
            "template_indices": template_indices,  # which template each embedding belongs to
        }

        logger.info("Validated and embedded %d example questions", len(all_examples))

    def set_question_templates(self, templates: list[dict[str, Any]]) -> None:
        """Set templates with example questions.

        :param templates: List of template configs, each with:
                          - "template": Template string with [RELEVANT_SENTENCES]
                          - "examples": List of example questions
        :raises ValueError: If template format is invalid
        """
        self.templates = templates
        self._example_embeddings = None
        self._validate_and_embed_templates()

    def _find_best_match(self, question: str) -> tuple[int, float]:
        """Find the best matching template for a question.

        :param question: User's question
        :return: Tuple of (template_index, best_similarity_score)
        """
        if not self.templates or self._example_embeddings is None:
            return -1, 0.0

        embedder = self._get_embedder()
        question_embedding = embedder.encode([question])[0]  # Shape: (embedding_dim,)

        import numpy as np

        # Compute similarities to all examples at once using numpy
        similarities = np.dot(self._example_embeddings["embeddings"], question_embedding) / (
            np.linalg.norm(self._example_embeddings["embeddings"], axis=1) * np.linalg.norm(question_embedding)
        )

        # Find the best similarity
        best_example_idx = np.argmax(similarities)
        best_similarity = float(similarities[best_example_idx])
        best_template_idx = self._example_embeddings["template_indices"][best_example_idx]

        logger.debug(
            "Question matched template %s (similarity: %.3f)",
            best_template_idx,
            best_similarity,
        )

        return best_template_idx, best_similarity

    def generate(self, question: str, spans: list[str], citation_count: int = 0) -> str:
        """Generate template by finding best match for the question.

        :param question: User's question
        :param spans: List of spans (not used for matching)
        :param citation_count: Number of citation spans (not used for matching)
        :return: Best matching template or default template
        """
        template_idx, _similarity = self._find_best_match(question)

        if template_idx >= 0 and template_idx < len(self.templates):
            return self.templates[template_idx]["template"]

        logger.info("Using default template (no templates configured)")
        return self.DEFAULT_TEMPLATE

    def fill(
        self,
        template: str,
        display_spans: list[dict[str, Any]],
        citation_spans: list[dict[str, Any]],
    ) -> str:
        """Fill the template with actual span content.

        :param template: Template string with placeholders
        :param display_spans: Spans to display with full text
        :param citation_spans: Spans for citation reference only
        :return: Filled template
        """
        return self.filler.fill(template, display_spans, citation_spans)

    def save_state(self) -> dict[str, Any]:
        """Save the current template configuration.

        :return: Dictionary with template state
        """
        return {
            "type": "question_specific",
            "templates": self.templates,
            "model_name": self.model_name,
            "device": self.device,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load template configuration from saved state.

        :param state: Dictionary with template configuration
        """
        if "templates" in state:
            self.templates = state["templates"]
        if "model_name" in state:
            self.model_name = state["model_name"]
        if "device" in state:
            self.device = state["device"]

        # Clear cached embeddings and revalidate
        self._example_embeddings = None
        self._embedder = None
        if self.templates:
            self._validate_and_embed_templates()

    def add_template(self, template: str, examples: list[str]) -> None:
        """Add a new template to the list.

        :param template: Template string with [RELEVANT_SENTENCES]
        :param examples: List of example questions for this template
        :raises ValueError: If template is invalid
        """
        self.validate_template(template)
        if not examples:
            msg = "At least one example question is required"
            raise ValueError(msg)

        self.templates.append({"template": template, "examples": examples})

        # Revalidate and re-embed all templates
        self._example_embeddings = None
        self._validate_and_embed_templates()
        logger.info("Added template (total: %d)", len(self.templates))

    def remove_template(self, index: int) -> None:
        """Remove a template by index.

        :param index: Index of template to remove
        :raises IndexError: If index is out of range
        """
        if index < 0 or index >= len(self.templates):
            msg = f"Template index {index} out of range [0, {len(self.templates) - 1}]"
            raise IndexError(msg)

        self.templates.pop(index)
        self._example_embeddings = None
        if self.templates:
            self._validate_and_embed_templates()
        logger.info("Removed template at index %d (total: %d)", index, len(self.templates))
