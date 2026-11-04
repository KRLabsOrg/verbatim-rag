"""Question-specific template strategy for the Verbatim RAG system.

Provides template matching based on example questions using semantic similarity.
Users define templates paired with example questions, and the system automatically
selects the best-matching template for each query.
"""

import logging
import math
import time
from collections.abc import Sequence
from typing import Any

from verbatim_core._deprecation import _UNSET, warn_deprecated
from verbatim_core._loading import _load_errors_translated, _log_model_loaded
from verbatim_core.errors import ConfigurationError
from verbatim_core.templates.base import TemplateStrategy
from verbatim_core.templates.filler import TemplateFiller

logger = logging.getLogger(__name__)

# The embedder loaded by name when the templates were given none, as before 0.3.0.
_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


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
        strategy = QuestionSpecificTemplate.from_pretrained("all-MiniLM-L6-v2", templates=templates)
    """

    DEFAULT_TEMPLATE = """## Response

Based on the available documents:

[DISPLAY_SPANS]

---
*These excerpts are taken verbatim from the source documents to ensure accuracy.*"""

    def __init__(
        self,
        templates: list[dict[str, Any]] | None = None,
        model_name: str = _UNSET,
        device: str = _UNSET,
        citation_mode: str = "inline",
        *,
        embedder: Any = None,
    ):
        """Initialize question-specific template strategy.

        :param templates: List of template definitions, each with 'template' and 'examples' keys
        :param model_name: Deprecated. Load the embedder with `from_pretrained(model_name)`.
        :param device: Deprecated, with `model_name`.
        :param embedder: A loaded embedding model with an `encode(texts)` method, such as a
            SentenceTransformer. It embeds the template examples here and each question.
        """
        # Loading the embedder by name warns once: here if the caller named the model, else when
        # it is loaded.
        self._warn_on_load = model_name is _UNSET and device is _UNSET
        if not self._warn_on_load:
            warn_deprecated(
                "QuestionSpecificTemplate(model_name=..., device=...)",
                use="QuestionSpecificTemplate.from_pretrained(model_name, device)",
            )

        self.templates = templates or []
        # The name of the model that embeds, saved and loaded by. None for an embedder given
        # without a name.
        if model_name is not _UNSET:
            self.model_name = model_name
        else:
            self.model_name = None if embedder is not None else _DEFAULT_MODEL_NAME
        self.device = "cpu" if device is _UNSET else device
        self.citation_mode = citation_mode
        self.filler = TemplateFiller(citation_mode=citation_mode)
        self._embedder = embedder
        self._example_embeddings = None  # Will store stacked embeddings for efficient matching

        if self.templates:
            self._validate_and_embed_templates()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str | None = None,
        *,
        templates: list[dict[str, Any]] | None = None,
        citation_mode: str = "inline",
    ) -> "QuestionSpecificTemplate":
        """Load a sentence-transformers embedding model and build the templates with it.

        :param model_name: A sentence-transformers model id, or the path of a local model
        :param device: The device to run the model on. None lets sentence-transformers choose.
        :param templates: List of template definitions, each with 'template' and 'examples' keys
        :param citation_mode: "inline" or "hidden"
        :raises ConfigurationError: If sentence-transformers is not installed, or the model can't
            be loaded
        """
        embedder = _load_embedder(model_name, device)
        strategy = cls(templates, citation_mode=citation_mode, embedder=embedder)
        strategy.model_name = model_name
        strategy.device = str(embedder.device)
        return strategy

    def set_citation_mode(self, citation_mode: str) -> None:
        self.citation_mode = citation_mode
        self.filler.set_citation_mode(citation_mode)

    def _get_embedder(self):
        """The embedder. Without one, it is loaded by name, as before 0.3.0."""
        if self._embedder is None:
            if self.model_name is None:
                msg = (
                    "The question-specific templates don't name the model that embedded their "
                    "examples. Pass that model as embedder, for example with "
                    "TemplateManager.use_question_specific_mode(embedder=...) before load."
                )
                raise ConfigurationError(
                    msg,
                    code="missing_embedder",
                )
            if self._warn_on_load:
                warn_deprecated(
                    "QuestionSpecificTemplate without an embedder",
                    use="QuestionSpecificTemplate.from_pretrained(model_name)",
                    stacklevel=None,
                )
            self._embedder = _load_embedder(self.model_name, self.device)
        return self._embedder

    def _validate_and_embed_templates(self):
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

        logger.debug("Validated and embedded %d example questions", len(all_examples))

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

        similarities = [
            _cosine(example_embedding, question_embedding)
            for example_embedding in self._example_embeddings["embeddings"]
        ]

        # Find the best similarity, the first on a tie
        best_example_idx = max(range(len(similarities)), key=similarities.__getitem__)
        best_similarity = float(similarities[best_example_idx])
        best_template_idx = self._example_embeddings["template_indices"][best_example_idx]

        logger.debug("Question matched template %d (similarity: %.3f)", best_template_idx, best_similarity)

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

        logger.debug("Using default template (no templates configured)")
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
        # An embedder given replaces the saved model.
        if self._embedder is None and "model_name" in state:
            self.model_name = state["model_name"]
        if self._embedder is None and "device" in state:
            self.device = state["device"]

        # Embed the loaded examples, with the embedder given, else the saved model
        self._example_embeddings = None
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
        logger.debug("Added template (total: %d)", len(self.templates))

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
        logger.debug("Removed template at index %d (total: %d)", index, len(self.templates))


def _load_embedder(model_name: str, device: str | None) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        msg = (
            "Question-specific templates need sentence-transformers. Install it with pip install sentence-transformers."
        )
        raise ConfigurationError(
            msg,
            code="missing_dependency",
        ) from e

    started = time.perf_counter()
    with _load_errors_translated(model_name):
        embedder = SentenceTransformer(model_name, device=device)
    _log_model_loaded(logger, model_name, str(embedder.device), started)
    return embedder


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """The cosine similarity of two vectors, and 0 if either is zero."""
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b, strict=False)) / norms if norms else 0.0
