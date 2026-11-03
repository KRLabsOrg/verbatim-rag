"""
Extractors for identifying relevant spans in documents (RAG-agnostic).

This copy avoids importing vector-store specific types; accepts any objects
with a `.text` attribute as search results.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence, Tuple, Type, TypeVar

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError

from ._deprecation import deprecated, warn_deprecated
from ._locate import locate
from ._logging import content_logger
from .errors import ConfigurationError, InvalidInputError, UpstreamError
from .llm_client import Completion, LLMClient
from .prompts import load_prompt, render_prompt
from .types import Chunk, ExtractedSpan, Extraction, ResultWarning, Usage

logger = logging.getLogger(__name__)


class ExtractConfig(BaseModel):
    """Settings of an extractor. Each extractor has its own subclass.

    The constructor of an extractor takes its default config, and a config passed to `extract`
    replaces it for that call.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


_ConfigT = TypeVar("_ConfigT", bound=ExtractConfig)


class SpanExtractor(ABC):
    """Base class for span extractors.

    A subclass implements `extract`, and `aextract` when it has a native async path. Otherwise
    `aextract` runs `extract` in a worker thread.

    A subclass written against the old interface implements only `extract_spans`. It gets
    `extract` and `aextract` from this class, which locate each span text in its chunk and warn
    that the subclass should implement `extract`. A span text that can't be located is dropped
    and reported in the result's warnings with the code `span_text_not_found`.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not (_overrides(cls, "extract") or _overrides(cls, "extract_spans")):
            raise TypeError(f"{cls.__qualname__} must implement extract() or extract_spans()")

    def extract(
        self, question: str, chunks: Sequence[Chunk], *, config: ExtractConfig | None = None
    ) -> Extraction:
        """
        Extract the spans of the chunks that are relevant to the question.

        :param question: The query or question
        :param chunks: Chunks with unique ids
        :param config: Replaces the extractor's default config for this call
        :return: The spans, with the usage and warnings of the call
        :raises InvalidInputError: If two chunks share an id
        """
        _check_old_interface_call(type(self), chunks, config)
        return _locate_span_texts(chunks, self.extract_spans(question, chunks))

    async def aextract(
        self, question: str, chunks: Sequence[Chunk], *, config: ExtractConfig | None = None
    ) -> Extraction:
        """Async version of `extract`."""
        if _overrides(type(self), "extract"):
            return await asyncio.to_thread(self.extract, question, chunks, config=config)

        _check_old_interface_call(type(self), chunks, config)
        if _overrides(type(self), "extract_spans_async"):
            span_texts = await self.extract_spans_async(question, chunks)
        else:
            span_texts = await asyncio.to_thread(self.extract_spans, question, chunks)
        return _locate_span_texts(chunks, span_texts)

    @deprecated("SpanExtractor.extract_spans", use="extract()")
    def extract_spans(self, question: str, search_results: List[Any]) -> Dict[str, List[str]]:
        """
        Extract relevant spans from search results.

        :param question: The query or question
        :param search_results: List of search results to extract from
        :return: Dictionary mapping result text to list of relevant spans
        """
        chunks = _chunks_from_results(search_results)
        return _span_texts_by_chunk_text(chunks, self.extract(question, chunks))

    @deprecated("SpanExtractor.extract_spans_async", use="aextract()")
    async def extract_spans_async(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """Async version of `extract_spans`."""
        if not _overrides(type(self), "extract"):
            return await asyncio.to_thread(self.extract_spans, question, search_results)

        chunks = _chunks_from_results(search_results)
        return _span_texts_by_chunk_text(chunks, await self.aextract(question, chunks))


def _overrides(cls: type, name: str) -> bool:
    return getattr(cls, name) is not getattr(SpanExtractor, name)


def _default_config(config: _ConfigT | None, config_class: Type[_ConfigT]) -> _ConfigT:
    """The default config an extractor's constructor was given, or the default of its class."""
    if config is None:
        return config_class()
    if not isinstance(config, config_class):
        raise ConfigurationError(f"config must be a {config_class.__name__}.", code="invalid_value")
    return config


def _config_from_settings(config_class: Type[_ConfigT], settings: Dict[str, Any]) -> _ConfigT:
    """The config that the deprecated setting arguments of an extractor's constructor describe."""
    try:
        return config_class(**settings)
    except ValidationError as e:
        names = ", ".join(str(error["loc"][0]) for error in e.errors())
        raise ConfigurationError(
            f"Invalid value for {names}. See {config_class.__name__} for the valid values.",
            code="invalid_value",
        ) from e


def _call_config(
    config: _ConfigT | None, default: _ConfigT, config_class: Type[_ConfigT]
) -> _ConfigT:
    """The config of one call: the one the call passed, else the extractor's default."""
    if config is None:
        return default
    if not isinstance(config, config_class):
        raise InvalidInputError(
            f"config must be a {config_class.__name__}.", code="invalid_value", param="config"
        )
    return config


def _check_unique_ids(chunks: Sequence[Chunk]) -> None:
    positions: Dict[str, int] = {}
    for position, chunk in enumerate(chunks):
        if chunk.id in positions:
            raise InvalidInputError(
                f"chunks[{position}] has the same id as chunks[{positions[chunk.id]}]. "
                "Chunk ids must be unique within a call.",
                code="invalid_value",
                param=f"chunks[{position}].id",
            )
        positions[chunk.id] = position


def _check_old_interface_call(cls: type, chunks: Sequence[Chunk], config: Any) -> None:
    """Check a call of the extract_spans implementation of an extractor, and warn about it."""
    implementation = _extract_spans_implementation(cls)
    _check_unique_ids(chunks)
    if config is not None:
        raise InvalidInputError(
            f"{implementation.__qualname__} implements only extract_spans(), which takes no config.",
            code="invalid_value",
            param="config",
        )
    warn_deprecated(
        f"Implementing only extract_spans() in {implementation.__qualname__}",
        use="extract()",
        stacklevel=4,
    )


def _extract_spans_implementation(cls: type) -> type:
    """The class that implements extract_spans for `cls`, which may be `cls` or a parent."""
    for parent in cls.__mro__:
        if parent is SpanExtractor:
            break
        if "extract_spans" in vars(parent):
            return parent
    raise NotImplementedError(
        f"SpanExtractor.extract() needs an extract_spans() implementation, and {cls.__qualname__} "
        "has none. Implement extract() without calling the base class."
    )


def _locate_span_texts(chunks: Sequence[Chunk], span_texts: Dict[str, List[str]]) -> Extraction:
    """Turn today's span texts, keyed by chunk text, into an extraction."""
    texts = [
        (position, text)
        for position, chunk in enumerate(chunks)
        for text in span_texts.get(chunk.text, [])
    ]
    spans, warnings = _locate_texts(
        chunks, texts, fuzzy_threshold=0.8, what="span text", code="span_text_not_found"
    )
    # The extractor doesn't say which model it runs.
    usage = Usage(
        model_id=None,
        model_version=None,
        units={"chunks": len(chunks), "characters": sum(len(chunk.text) for chunk in chunks)},
    )
    return Extraction(spans=spans, usage=[usage], warnings=warnings)


def _locate_texts(
    chunks: Sequence[Chunk],
    texts: Sequence[Tuple[int, str]],
    *,
    fuzzy_threshold: float | None,
    what: str,
    code: str,
) -> Tuple[List[ExtractedSpan], List[ResultWarning]]:
    """Locate texts without offsets in their chunks. Identical spans collapse into one.

    Empty texts are skipped: they point at nothing, so they are not missing from the chunk.

    :param texts: Each text with the position of its chunk
    :param what: What the texts are, for the messages, such as "quote"
    :param code: The code of the warning for a text that can't be located
    :return: The spans, and a warning for each text that can't be located
    """
    spans: List[ExtractedSpan] = []
    warnings = []
    dropped = []
    for position, text in texts:
        if not text.strip():
            continue
        span = locate(text, chunks[position], fuzzy_threshold=fuzzy_threshold)
        if span is None:
            dropped.append(text)
            warnings.append(
                ResultWarning(
                    code=code, message=f"A {what} for chunks[{position}] is not in the chunk."
                )
            )
        elif span not in spans:
            spans.append(span)

    if dropped:
        logger.warning(
            "Dropped %d %ss that are not in their chunks",
            len(dropped),
            what,
            extra={"event": "fallback_used", "warning_code": code},
        )
        content_logger.debug("Dropped %ss: %r", what, dropped)
    return spans, warnings


def _chunks_from_results(search_results: List[Any]) -> List[Chunk]:
    return [
        Chunk(id=str(position), text=getattr(result, "text", ""))
        for position, result in enumerate(search_results)
    ]


def _span_texts_by_chunk_text(
    chunks: Sequence[Chunk], extraction: Extraction
) -> Dict[str, List[str]]:
    """Today's shape of an extraction. Of two chunks with the same text, the last one wins."""
    texts_by_id: Dict[str, List[str]] = {chunk.id: [] for chunk in chunks}
    for span in extraction.spans:
        texts_by_id[span.chunk_id].append(span.text)
    return {chunk.text: texts_by_id[chunk.id] for chunk in chunks}


class ModelSpanExtractor(SpanExtractor):
    """Extract spans using a Verbatim-RAG ModernBERT model.

    Supports two model formats and auto-detects which one ``model_path`` is:

    - **highlighter** (`KRLabsOrg/verbatim-rag-modern-bert-v2`,
      `KRLabsOrg/acl-verbatim-modernbert`): a token classifier loaded via
      ``AutoModel.from_pretrained(..., trust_remote_code=True)`` that exposes
      a ``.process()`` method returning character spans. This is the current
      Verbatim-RAG model family.
    - **qa_model** (`KRLabsOrg/verbatim-rag-modern-bert-v1` and earlier
      checkpoints): the legacy custom ``QAModel`` sentence classifier.

    Detection is done at load time by checking the model config for an
    ``auto_map`` entry pointing to a ``.process()``-capable class. Both
    formats yield the same return shape: ``dict[chunk_text, list[span_text]]``.
    """

    DEFAULT_MODEL = "KRLabsOrg/verbatim-rag-modern-bert-v2"
    _FORMAT_HIGHLIGHTER = "highlighter"
    _FORMAT_QA_MODEL = "qa_model"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        device: str | None = None,
        threshold: float = 0.2,
        extraction_mode: str = "individual",
        max_display_spans: int = 5,
        # Highlighter-format-only knobs (ignored for the legacy qa_model path):
        min_span_chars: int = 30,
        merge_gap_chars: int = 20,
        max_length: int = 8192,
        doc_stride: int = 256,
    ):
        """
        :param model_path: Path or HuggingFace id. Defaults to v2 generic.
        :param device: ``"cuda"`` / ``"mps"`` / ``"cpu"`` or None to auto-detect.
        :param threshold: Probability cutoff. Default 0.2 matches v2's
            published headline; raise to 0.5 for the legacy v1 sentence model.
        :param extraction_mode: Reserved for future use.
        :param max_display_spans: Reserved for future use.
        :param min_span_chars: (highlighter only) drop predicted spans shorter
            than this many characters.
        :param merge_gap_chars: (highlighter only) merge adjacent spans
            separated by ≤ this many characters.
        :param max_length: (highlighter only) max tokens per sliding window.
        :param doc_stride: (highlighter only) token overlap between windows.
        """
        import torch

        self.model_path = model_path
        self.threshold = threshold
        self.min_span_chars = min_span_chars
        self.merge_gap_chars = merge_gap_chars
        self.max_length = max_length
        self.doc_stride = doc_stride
        self._torch = torch

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif (
                getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self._format = self._detect_format(model_path)
        logger.info("Loading model from %s as format=%s on %s", model_path, self._format, device)
        if self._format == self._FORMAT_HIGHLIGHTER:
            self._init_highlighter(model_path)
        else:
            self._init_qa_model(model_path)

    @staticmethod
    def _detect_format(model_path: str) -> str:
        """Return ``"highlighter"`` if the model exposes a ``.process()`` API
        via ``trust_remote_code``, else ``"qa_model"`` (legacy)."""
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            auto_map = getattr(config, "auto_map", None) or {}
            target = auto_map.get("AutoModel") or auto_map.get("AutoModelForTokenClassification")
            if target and "Highlighter" in target:
                return ModelSpanExtractor._FORMAT_HIGHLIGHTER
        except Exception as exc:
            logger.warning("Highlighter detection failed for %s: %s", model_path, exc)
        return ModelSpanExtractor._FORMAT_QA_MODEL

    def _init_highlighter(self, model_path: str) -> None:
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = None  # the highlighter loads its tokenizer internally

    def _init_qa_model(self, model_path: str) -> None:
        from transformers import AutoTokenizer

        from .extractor_models.dataset import (
            Document as DatasetDocument,
        )
        from .extractor_models.dataset import (
            QADataset,
            QASample,
        )
        from .extractor_models.dataset import (
            Sentence as DatasetSentence,
        )
        from .extractor_models.model import QAModel

        self.QADataset = QADataset
        self.DatasetSentence = DatasetSentence
        self.DatasetDocument = DatasetDocument
        self.QASample = QASample

        self.model = QAModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            logger.warning("Could not load tokenizer from %s: %s", model_path, e)
            base_model = getattr(self.model.config, "model_name", "answerdotai/ModernBERT-base")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)

    def _split_into_sentences(self, text: str) -> list[str]:
        """Simple sentence splitting."""
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def extract_spans(self, question: str, search_results: List[Any]) -> Dict[str, List[str]]:
        """Extract spans using the loaded model. Dispatches by format."""
        if self._format == self._FORMAT_HIGHLIGHTER:
            return self._extract_highlighter(question, search_results)
        return self._extract_qa_model(question, search_results)

    def _extract_highlighter(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        relevant: Dict[str, List[str]] = {}
        for result in search_results:
            context = getattr(result, "text", "")
            if not context.strip():
                relevant[context] = []
                continue
            try:
                out = self.model.process(
                    question=question,
                    context=context,
                    threshold=self.threshold,
                    min_span_chars=self.min_span_chars,
                    merge_gap_chars=self.merge_gap_chars,
                    max_length=self.max_length,
                    doc_stride=self.doc_stride,
                )
                relevant[context] = [
                    sp["text"] for sp in out.get("spans", []) if sp.get("text", "").strip()
                ]
            except Exception as exc:
                logger.error("Highlighter extraction failed: %s", exc)
                relevant[context] = []
        return relevant

    def _extract_qa_model(self, question: str, search_results: List[Any]) -> Dict[str, List[str]]:
        relevant_spans: Dict[str, List[str]] = {}

        for result in search_results:
            raw_text = getattr(result, "text", "")
            raw_sentences = self._split_into_sentences(raw_text)
            if not raw_sentences:
                relevant_spans[raw_text] = []
                continue

            dataset_sentences = [
                self.DatasetSentence(text=sent, relevant=False, sentence_id=f"s{i}")
                for i, sent in enumerate(raw_sentences)
            ]
            dataset_doc = self.DatasetDocument(sentences=dataset_sentences)

            qa_sample = self.QASample(
                question=question,
                documents=[dataset_doc],
                split="test",
                dataset_name="inference",
                task_type="qa",
            )

            dataset = self.QADataset([qa_sample], self.tokenizer, max_length=512)
            if len(dataset) == 0:
                relevant_spans[raw_text] = []
                continue

            encoding = dataset[0]
            input_ids = encoding["input_ids"].unsqueeze(0).to(self.device)
            attention_mask = encoding["attention_mask"].unsqueeze(0).to(self.device)

            with self._torch.no_grad():
                predictions = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sentence_boundaries=[encoding["sentence_boundaries"]],
                )

            spans = []
            if len(predictions) > 0 and len(predictions[0]) > 0:
                sentence_preds = self._torch.nn.functional.softmax(predictions[0], dim=1)
                for i, pred in enumerate(sentence_preds):
                    if i < len(raw_sentences) and pred[1] > self.threshold:
                        spans.append(raw_sentences[i])

            relevant_spans[raw_text] = spans

        return relevant_spans


class SemanticHighlightExtractor(SpanExtractor):
    """
    Extract spans using Zilliz semantic highlighting model.

    This extractor uses a pre-trained model (zilliz/semantic-highlight-bilingual-v1)
    specifically designed for RAG highlighting tasks. It supports two output modes:

    - "sentences": Returns complete sentences using the model's built-in sentence
      splitting (cleaner boundaries, language-aware)
    - "spans": Returns token-level spans for finer granularity (may cross sentence
      boundaries or return partial sentences)

    Unlike ModelSpanExtractor which requires training, this uses a pre-trained
    model that works out of the box for English and Chinese text.
    """

    def __init__(
        self,
        model_name: str = "zilliz/semantic-highlight-bilingual-v1",
        device: str | None = None,
        threshold: float = 0.5,
        output_mode: str = "sentences",
        language: str = "auto",
        min_span_tokens: int = 3,
        merge_gap: int = 2,
        max_tokens: int = 4096,
    ):
        """
        Initialize the semantic highlight extractor.

        :param model_name: HuggingFace model name
        :param device: Device to run on (auto-detects if None)
        :param threshold: Probability threshold (0-1)
        :param output_mode: "sentences" for complete sentences, "spans" for token-level
        :param language: Language hint ("en", "zh", or "auto")
        :param min_span_tokens: Minimum tokens for a valid span (spans mode only)
        :param merge_gap: Merge spans separated by <= N tokens (spans mode only)
        :param max_tokens: Token limit of the extractor model
        """
        import torch
        from transformers import AutoModel

        if output_mode not in ("sentences", "spans"):
            raise ValueError(f"output_mode must be 'sentences' or 'spans', got {output_mode!r}")

        self.threshold = threshold
        self.output_mode = output_mode
        self.language = language
        self.min_span_tokens = min_span_tokens
        self.merge_gap = merge_gap
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading semantic highlight model: %s...", model_name)
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, max_length=max_tokens
        )
        self.tokenizer = self.model.tokenizer
        self.max_tokens = max_tokens

    def extract_spans(self, question: str, search_results: List[Any]) -> Dict[str, List[str]]:
        """
        Extract relevant spans from search results.

        :param question: The query or question
        :param search_results: List of search results (objects with .text attribute)
        :return: Dictionary mapping result text to list of relevant spans
        """
        relevant_spans: Dict[str, List[str]] = {}
        for result in search_results:
            context = getattr(result, "text", "")
            if not context.strip():
                relevant_spans[context] = []
                continue

            try:
                if self.output_mode == "sentences":
                    spans = self._extract_sentences(question, context)
                else:
                    spans = self._extract_token_spans(question, context)
                relevant_spans[context] = spans
            except Exception as e:
                logger.error("Semantic highlight extraction failed: %s", e)
                relevant_spans[context] = []

        return relevant_spans

    def _extract_sentences(self, question: str, context: str) -> List[str]:
        """Extract complete sentences using the model's built-in processing."""
        result = self.model.process(
            question=question,
            context=context,
            threshold=self.threshold,
            language=self.language,
            return_sentence_metrics=False,
        )
        sentences = result.get("highlighted_sentences", [])
        # Clean up whitespace from sentences
        return [s.strip() for s in sentences if s.strip()]

    def _extract_token_spans(self, question: str, context: str) -> List[str]:
        """Extract spans based on token-level probabilities."""
        import re

        # Get token-level scores
        raw = self.model.get_raw_predictions(question, [context])
        token_probs = raw.pruning_probs

        if not raw.context_ranges:
            return []

        start_idx, end_idx = raw.context_ranges[0]
        context_probs = token_probs[start_idx:end_idx]

        # Tokenize to get offset mapping
        encoding = self.tokenizer(
            context,
            return_offsets_mapping=True,
            add_special_tokens=False,
            max_length=self.max_tokens,
        )
        offset_mapping = encoding["offset_mapping"]

        # Validate lengths match
        if len(context_probs) != len(offset_mapping):
            logger.warning(
                "Length mismatch probs=%d tokens=%d", len(context_probs), len(offset_mapping)
            )
            min_len = min(len(context_probs), len(offset_mapping))
            context_probs = context_probs[:min_len]
            offset_mapping = offset_mapping[:min_len]

        # Find contiguous high-probability regions
        span_regions = self._find_span_regions(context_probs)

        # Convert token regions to text spans
        spans = []
        for token_start, token_end in span_regions:
            token_end = min(token_end, len(offset_mapping))
            if token_start >= len(offset_mapping):
                continue

            char_start = offset_mapping[token_start][0]
            char_end = offset_mapping[token_end - 1][1]
            span_text = context[char_start:char_end]

            # Clean up leading/trailing punctuation
            span_text = re.sub(r"^[\s.,;:!?\-–—]+", "", span_text)
            span_text = re.sub(r"[\s.,;:!?\-–—]+$", "", span_text)
            span_text = span_text.strip()

            if span_text:
                spans.append(span_text)

        return spans

    def _find_span_regions(self, probs) -> List[tuple]:
        """Find contiguous regions of high-probability tokens."""
        above_threshold = probs > self.threshold

        regions = []
        in_span = False
        span_start = 0

        for i, is_relevant in enumerate(above_threshold):
            if is_relevant and not in_span:
                span_start = i
                in_span = True
            elif not is_relevant and in_span:
                if i - span_start >= self.min_span_tokens:
                    regions.append((span_start, i))
                in_span = False

        if in_span and len(above_threshold) - span_start >= self.min_span_tokens:
            regions.append((span_start, len(above_threshold)))

        # Merge nearby spans
        if regions:
            merged = [regions[0]]
            for start, end in regions[1:]:
                prev_start, prev_end = merged[-1]
                if start - prev_end <= self.merge_gap:
                    merged[-1] = (prev_start, end)
                else:
                    merged.append((start, end))
            regions = merged

        return regions


class LLMExtractConfig(ExtractConfig):
    """Settings of `LLMSpanExtractor`."""

    # "batch" sends up to batch_size chunks in one LLM call, "individual" one chunk per call, and
    # "auto" one batch when there are at most batch_size chunks and one chunk per call otherwise.
    extraction_mode: Literal["batch", "individual", "auto"] = "auto"
    batch_size: PositiveInt = 5
    # "exact" locates a quote only where the chunk contains it, "fuzzy" also where it differs a
    # little, for example in spacing or punctuation.
    span_match_mode: Literal["exact", "fuzzy"] = "exact"
    # How closely, from 0 to 1, a quote must match the chunk text to be located in fuzzy mode.
    fuzzy_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    # A Jinja2 prompt with the variables `question` and `documents`, instead of the bundled one.
    extraction_prompt: str | None = None
    system_prompt: str | None = None  # sent only with extraction_prompt


# Marks the deprecated constructor arguments that the caller didn't pass.
_UNSET: Any = object()


class LLMSpanExtractor(SpanExtractor):
    """Extract spans with an LLM that quotes the text of each chunk that answers the question.

    Each quote is located in its chunk: the first exact occurrence, else in fuzzy match mode the
    best fuzzy match, the earliest on a tie. A span's text is the chunk's text at the located
    offsets, never the quote. A quote that can't be located is dropped and reported in the
    result's warnings with the code `quote_not_found`. Every span has the score 1.0, "extracted
    without a score".

    A failed LLM call raises, and the whole extraction fails.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        model: str = _UNSET,
        extraction_mode: str = _UNSET,
        max_display_spans: int = _UNSET,
        batch_size: int = _UNSET,
        span_match_mode: str = _UNSET,
        fuzzy_threshold: float = _UNSET,
        extraction_prompt: str | None = _UNSET,
        system_prompt: str | None = _UNSET,
        *,
        config: LLMExtractConfig | None = None,
    ):
        """
        :param llm_client: The client that makes the LLM calls
        :param config: The settings of a call, unless the call passes its own

        The other arguments are deprecated. Without ``llm_client`` the extractor builds an
        ``LLMClient(model)``. The settings from ``extraction_mode`` to ``system_prompt`` belong in
        ``config``, and ``max_display_spans`` is unused.
        :raises ConfigurationError: If ``config`` is not an ``LLMExtractConfig``, is passed
            together with the deprecated settings, or a deprecated setting has an invalid value
        """
        settings = {
            "extraction_mode": extraction_mode,
            "batch_size": batch_size,
            "span_match_mode": span_match_mode,
            "fuzzy_threshold": fuzzy_threshold,
            "extraction_prompt": extraction_prompt,
            "system_prompt": system_prompt,
        }
        settings = {name: value for name, value in settings.items() if value is not _UNSET}
        if settings and config is not None:
            raise ConfigurationError(
                "Pass the settings of LLMSpanExtractor either in config or as arguments, not both.",
                code="invalid_value",
            )
        if settings:
            warn_deprecated(
                f"Passing {', '.join(settings)} to LLMSpanExtractor",
                use="config=LLMExtractConfig(...)",
            )
            config = _config_from_settings(LLMExtractConfig, settings)
        if max_display_spans is not _UNSET:
            warn_deprecated(
                "Passing max_display_spans to LLMSpanExtractor",
                use="TransformConfig(max_display_spans=...)",
            )
        if llm_client is None:
            warn_deprecated(
                "LLMSpanExtractor without an llm_client", use="LLMSpanExtractor(LLMClient(model))"
            )
            llm_client = LLMClient("gpt-4o-mini" if model is _UNSET else model)
        elif model is not _UNSET:
            warn_deprecated("Passing model to LLMSpanExtractor", use="the model of the llm_client")

        self.llm_client = llm_client
        self.config = _default_config(config, LLMExtractConfig)

    def extract(
        self, question: str, chunks: Sequence[Chunk], *, config: LLMExtractConfig | None = None
    ) -> Extraction:
        """
        Extract the spans of the chunks that are relevant to the question.

        :param question: The query or question
        :param chunks: Chunks with unique ids
        :param config: Replaces the extractor's default config for this call
        :return: The spans, with the usage and warnings of the call
        :raises InvalidInputError: If two chunks share an id, or the config is of the wrong type
        :raises UpstreamError: If an LLM call fails or its reply can't be read
        :raises ConfigurationError: If the LLM service rejects the key, the model or the quota
        """
        config = _call_config(config, self.config, LLMExtractConfig)
        _check_unique_ids(chunks)
        calls = _plan_llm_calls(question, chunks, config)
        logger.debug("Extracting spans from %d chunks in %d LLM calls", len(chunks), len(calls))
        completions = [_complete(self.llm_client, call) for call in calls]
        return _extraction_from_replies(chunks, calls, completions, config, self._model_id())

    async def aextract(
        self, question: str, chunks: Sequence[Chunk], *, config: LLMExtractConfig | None = None
    ) -> Extraction:
        """Async version of `extract`. It makes its LLM calls concurrently."""
        config = _call_config(config, self.config, LLMExtractConfig)
        _check_unique_ids(chunks)
        calls = _plan_llm_calls(question, chunks, config)
        logger.debug("Extracting spans from %d chunks in %d LLM calls", len(chunks), len(calls))
        completions = await asyncio.gather(*(_acomplete(self.llm_client, call) for call in calls))
        return _extraction_from_replies(chunks, calls, completions, config, self._model_id())

    @deprecated("LLMSpanExtractor.extract_spans", use="extract()")
    def extract_spans(self, question: str, search_results: List[Any]) -> Dict[str, List[str]]:
        """
        Extract relevant spans from search results.

        Unlike `extract`, it tolerates failed LLM calls. When a batch call fails, it retries the
        chunks of the batch one by one, and a chunk whose call fails yields no spans.

        :param question: The query or question
        :param search_results: List of search results to extract from
        :return: Dictionary mapping result text to list of relevant spans
        """
        chunks = _chunks_from_results(search_results)
        batched, groups = _call_groups(len(chunks), self.config)
        spans = [
            span
            for group in groups
            for span in self._extract_group(question, chunks, group, batched)
        ]
        return _span_texts_by_chunk_text(chunks, Extraction(spans=spans))

    @deprecated("LLMSpanExtractor.extract_spans_async", use="aextract()")
    async def extract_spans_async(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """Async version of `extract_spans`."""
        chunks = _chunks_from_results(search_results)
        batched, groups = _call_groups(len(chunks), self.config)
        group_spans = await asyncio.gather(
            *(self._aextract_group(question, chunks, group, batched) for group in groups)
        )
        spans = [span for spans in group_spans for span in spans]
        return _span_texts_by_chunk_text(chunks, Extraction(spans=spans))

    def _extract_group(
        self, question: str, chunks: List[Chunk], group: range, batched: bool
    ) -> List[ExtractedSpan]:
        """The spans of the chunks one LLM call covers, retried one by one if a batch call fails."""
        if batched:
            try:
                batch = chunks[group.start : group.stop]
                return self.extract(question, batch, config=self._batch_config).spans
            except UpstreamError as error:
                _log_retry(group, error)
        spans = []
        for position in group:
            try:
                config = self._individual_config
                spans += self.extract(question, [chunks[position]], config=config).spans
            except UpstreamError as error:
                _log_failed_chunk(position, error)
        return spans

    async def _aextract_group(
        self, question: str, chunks: List[Chunk], group: range, batched: bool
    ) -> List[ExtractedSpan]:
        """Async version of `_extract_group`."""
        if batched:
            try:
                extraction = await self.aextract(
                    question, chunks[group.start : group.stop], config=self._batch_config
                )
                return extraction.spans
            except UpstreamError as error:
                _log_retry(group, error)
        spans = []
        for position in group:
            try:
                extraction = await self.aextract(
                    question, [chunks[position]], config=self._individual_config
                )
                spans += extraction.spans
            except UpstreamError as error:
                _log_failed_chunk(position, error)
        return spans

    @property
    def _batch_config(self) -> LLMExtractConfig:
        return self.config.model_copy(update={"extraction_mode": "batch"})

    @property
    def _individual_config(self) -> LLMExtractConfig:
        return self.config.model_copy(update={"extraction_mode": "individual"})

    def _model_id(self) -> str | None:
        return getattr(self.llm_client, "model", None)


@dataclass(frozen=True)
class _LLMCall:
    """One LLM call of an extraction: the chunks it covers and the prompt that sends them."""

    positions: range  # the positions of its chunks in the input
    keys: Tuple[str, ...]  # the key of each of its chunks in the prompt and in the reply
    prompt: str
    system_prompt: str | None


def _call_groups(chunk_count: int, config: LLMExtractConfig) -> Tuple[bool, List[range]]:
    """Whether the chunks go to the LLM in batches, and the positions of the chunks of each call."""
    batched = config.extraction_mode == "batch" or (
        config.extraction_mode == "auto" and chunk_count <= config.batch_size
    )
    size = config.batch_size if batched else 1
    groups = [range(start, min(start + size, chunk_count)) for start in range(0, chunk_count, size)]
    return batched, groups


def _plan_llm_calls(
    question: str, chunks: Sequence[Chunk], config: LLMExtractConfig
) -> List[_LLMCall]:
    batched, groups = _call_groups(len(chunks), config)
    calls = []
    for group in groups:
        if batched:
            keys = tuple(f"doc_{i}" for i in range(len(group)))
        else:
            # As before 0.3.0: a single chunk is "doc", or "doc_0" in a custom prompt.
            keys = ("doc_0",) if config.extraction_prompt else ("doc",)
        documents = {key: chunks[position].text for key, position in zip(keys, group)}
        calls.append(
            _LLMCall(
                positions=group,
                keys=keys,
                prompt=_extraction_prompt(question, documents, config),
                system_prompt=config.system_prompt if config.extraction_prompt else None,
            )
        )
    return calls


def _complete(client: LLMClient, call: _LLMCall) -> Completion:
    if _replaces(client, "complete", "completion"):
        text = client.complete(call.prompt, json_mode=True, system_prompt=call.system_prompt)
        return Completion(text=text)
    return client.completion(call.prompt, json_mode=True, system_prompt=call.system_prompt)


async def _acomplete(client: LLMClient, call: _LLMCall) -> Completion:
    if _replaces(client, "complete_async", "completion_async"):
        text = await client.complete_async(
            call.prompt, json_mode=True, system_prompt=call.system_prompt
        )
        return Completion(text=text)
    return await client.completion_async(
        call.prompt, json_mode=True, system_prompt=call.system_prompt
    )


def _replaces(client: Any, old: str, new: str) -> bool:
    """Whether a client replaces LLMClient's old method but not the new one.

    Clients written before 0.3.0 replace `complete`, for example to call another provider. They
    are called that way, and their usage has no model version or tokens.
    """
    client_class = type(client)
    if not hasattr(client_class, new):
        return True
    if getattr(client_class, new) is not getattr(LLMClient, new):
        return False
    return getattr(client_class, old) is not getattr(LLMClient, old)


def _extraction_prompt(question: str, documents: Dict[str, str], config: LLMExtractConfig) -> str:
    if config.extraction_prompt:
        formatted = "\n\n".join(f"[{key}]\n{text}" for key, text in documents.items())
        return render_prompt(config.extraction_prompt, question=question, documents=formatted)
    return load_prompt(
        "extraction/default", question=question, documents=json.dumps(documents, indent=2)
    )


def _extraction_from_replies(
    chunks: Sequence[Chunk],
    calls: List[_LLMCall],
    completions: Sequence[Completion],
    config: LLMExtractConfig,
    model_id: str | None,
) -> Extraction:
    quotes = []
    for call, completion in zip(calls, completions):
        quotes_by_key = _parse_reply(completion.text, call.keys)
        quotes += [
            (position, quote)
            for key, position in zip(call.keys, call.positions)
            for quote in quotes_by_key[key]
        ]
    fuzzy_threshold = config.fuzzy_threshold if config.span_match_mode == "fuzzy" else None
    spans, warnings = _locate_texts(
        chunks, quotes, fuzzy_threshold=fuzzy_threshold, what="quote", code="quote_not_found"
    )
    return Extraction(
        spans=spans, usage=_llm_usage(chunks, calls, completions, model_id), warnings=warnings
    )


def _parse_reply(text: str, keys: Sequence[str]) -> Dict[str, List[str]]:
    """The quotes in an LLM reply, a JSON object that maps chunk keys to lists of quotes.

    A chunk the reply leaves out has no quotes.
    """
    try:
        reply = json.loads(text)
    except json.JSONDecodeError as e:
        raise _invalid_reply() from e
    if not isinstance(reply, dict):
        raise _invalid_reply()
    quotes = {key: reply.get(key, []) for key in keys}
    for key_quotes in quotes.values():
        if not isinstance(key_quotes, list) or not all(isinstance(q, str) for q in key_quotes):
            raise _invalid_reply()
    return quotes


def _invalid_reply() -> UpstreamError:
    return UpstreamError(
        "The LLM sent a reply that is not a JSON object of quote lists. Retry, and check that a "
        "custom extraction prompt asks for that format.",
        code="invalid_reply",
        service="llm",
        retryable=True,
    )


def _llm_usage(
    chunks: Sequence[Chunk],
    calls: List[_LLMCall],
    completions: Sequence[Completion],
    model_id: str | None,
) -> List[Usage]:
    """One usage entry per model version the LLM service named, normally one."""
    calls_by_version: Dict[str | None, List[Tuple[_LLMCall, Completion]]] = {}
    for call, completion in zip(calls, completions):
        calls_by_version.setdefault(completion.model, []).append((call, completion))

    usage = []
    for version, version_calls in calls_by_version.items() or [(None, [])]:
        texts = [chunks[position].text for call, _ in version_calls for position in call.positions]
        units = {
            "chunks": len(texts),
            "characters": sum(len(text) for text in texts),
            "llm_calls": len(version_calls),
        }
        for unit in ("input_tokens", "output_tokens"):
            counts = [getattr(completion, unit) for _, completion in version_calls]
            # Counted only when every call reported them.
            if counts and None not in counts:
                units[unit] = sum(counts)
        usage.append(Usage(model_id=model_id, model_version=version, units=units))
    return usage


def _log_retry(group: range, error: UpstreamError) -> None:
    logger.info(
        "Retrying chunks[%d:%d] one by one after a failed LLM call (%s)",
        group.start,
        group.stop,
        error.code,
    )


def _log_failed_chunk(position: int, error: UpstreamError) -> None:
    logger.warning(
        "No spans for chunks[%d]: its LLM call failed (%s)",
        position,
        error.code,
        exc_info=True,
        extra={"event": "item_failed", "error_code": error.code},
    )
