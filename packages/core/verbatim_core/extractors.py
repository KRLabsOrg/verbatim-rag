"""
Extractors for identifying relevant spans in documents (RAG-agnostic).

This copy avoids importing vector-store specific types; accepts any objects
with a `.text` attribute as search results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence, Tuple, Type, TypeVar

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

from ._checks import _call_config, _check_unique_ids, _config_from_settings, _default_config
from ._deprecation import _UNSET, deprecated, warn_deprecated
from ._loading import _load_errors_translated, _log_model_loaded
from ._locate import UNSCORED, locate
from ._logging import content_logger
from .errors import ConfigurationError, InvalidInputError, UpstreamError
from .llm_client import Completion, LLMClient, client_completion, client_completion_async
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
    usage = _model_usage(None, None, chunks)
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
            warnings.append(_not_found(position, what, code))
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


class ModelExtractConfig(ExtractConfig):
    """Settings of `ModelSpanExtractor`."""

    # The lowest probability of a span. 0.2 suits the highlighter models, 0.5 the v1 sentence model.
    threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    # Highlighter models only:
    min_span_chars: NonNegativeInt = 30  # shorter spans are dropped
    merge_gap_chars: NonNegativeInt = 20  # spans at most this many characters apart are merged
    max_length: PositiveInt = 8192  # tokens per window
    doc_stride: NonNegativeInt = 256  # tokens shared by neighbouring windows


class ModelSpanExtractor(SpanExtractor):
    """Extract spans with a Verbatim-RAG ModernBERT model.

    Supports two model formats:

    - **highlighter** (`KRLabsOrg/verbatim-rag-modern-bert-v2`,
      `KRLabsOrg/acl-verbatim-modernbert`): a token classifier with a ``.process()`` method that
      returns character spans with their probabilities. This is the current model family.
    - **sentence model** (`KRLabsOrg/verbatim-rag-modern-bert-v1` and earlier checkpoints): the
      legacy ``QAModel`` sentence classifier, which needs its tokenizer. Its spans are the
      sentences of a chunk, scored with their probability.

    Load a model with `from_pretrained`. The constructor takes a loaded model and loads nothing.
    """

    DEFAULT_MODEL = "KRLabsOrg/verbatim-rag-modern-bert-v2"
    _FORMAT_HIGHLIGHTER = "highlighter"
    _FORMAT_QA_MODEL = "qa_model"

    def __init__(
        self,
        model: Any = None,
        device: str | None = _UNSET,
        threshold: float = _UNSET,
        extraction_mode: str = _UNSET,
        max_display_spans: int = _UNSET,
        min_span_chars: int = _UNSET,
        merge_gap_chars: int = _UNSET,
        max_length: int = _UNSET,
        doc_stride: int = _UNSET,
        *,
        config: ModelExtractConfig | None = None,
        tokenizer: Any = None,
        model_id: str | None = None,
        model_version: str | None = None,
        model_path: str = _UNSET,
    ):
        """
        :param model: A loaded highlighter model, or a loaded v1 sentence model
        :param config: The settings of a call, unless the call passes its own
        :param tokenizer: The tokenizer of a v1 sentence model
        :param model_id: The model's name or path, for the usage of a call
        :param model_version: The exact version of the model's weights, for the usage of a call

        The other arguments are deprecated. Without ``model``, or with ``model_path``, the
        extractor loads the model with `from_pretrained`. The settings from ``threshold`` to
        ``doc_stride`` belong in ``config``, and ``extraction_mode`` and ``max_display_spans``
        are unused.
        :raises ConfigurationError: If the arguments contradict each other or have invalid
            values, or a package the model needs is missing
        """
        settings = {
            "threshold": threshold,
            "min_span_chars": min_span_chars,
            "merge_gap_chars": merge_gap_chars,
            "max_length": max_length,
            "doc_stride": doc_stride,
        }
        config = _config_with_old_settings(type(self), ModelExtractConfig, config, settings)
        _warn_unused(
            type(self), extraction_mode=extraction_mode, max_display_spans=max_display_spans
        )

        if model_path is not _UNSET or model is None or isinstance(model, str):
            if model_path is not _UNSET and model is not None:
                raise ConfigurationError(
                    "Pass a loaded model or model_path to ModelSpanExtractor, not both.",
                    code="invalid_value",
                )
            model_id = model if isinstance(model, str) else self.DEFAULT_MODEL
            model_id = model_id if model_path is _UNSET else model_path
            warn_deprecated(
                "Loading a model in the ModelSpanExtractor constructor",
                use="ModelSpanExtractor.from_pretrained(model_id)",
            )
            loaded = type(self).from_pretrained(
                model_id,
                None if device is _UNSET else device,
                model_version=model_version,
                config=config,
            )
            vars(self).update(vars(loaded))
            return
        if device is not _UNSET:
            raise ConfigurationError(
                "ModelSpanExtractor got a loaded model and a device. Pass the device to "
                "from_pretrained, or move the model before passing it.",
                code="invalid_value",
            )

        self.model = model
        self.config = config
        self.model_id = model_id
        self.model_version = model_version
        self._format = (
            self._FORMAT_HIGHLIGHTER if hasattr(model, "process") else self._FORMAT_QA_MODEL
        )
        if self._format == self._FORMAT_QA_MODEL:
            if tokenizer is None:
                raise ConfigurationError(
                    "A v1 sentence model needs its tokenizer: "
                    "ModelSpanExtractor(model, tokenizer=tokenizer).",
                    code="invalid_value",
                )
            self.tokenizer = tokenizer
            self._qa_dataset = _import_qa_dataset()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str | None = None,
        *,
        model_version: str | None = None,
        config: ModelExtractConfig | None = None,
    ) -> ModelSpanExtractor:
        """
        Download and load a model, or load it from a local path.

        :param model_id: A Hugging Face model id or the path of a local model
        :param device: ``"cuda"``, ``"mps"`` or ``"cpu"``, or None to choose the best available
        :param model_version: The version of weights from a local path. A model from Hugging
            Face records its resolved commit instead.
        :param config: The default settings of a call
        :raises ConfigurationError: If a package the model needs is missing, or the model can't
            be downloaded or read
        """
        model, tokenizer, resolved_version = _load_model_span_model(model_id, device)
        return cls(
            model,
            config=config,
            tokenizer=tokenizer,
            model_id=model_id,
            model_version=resolved_version or model_version,
        )

    @staticmethod
    def _detect_format(model_id: str) -> str:
        """Return ``"highlighter"`` if the model exposes a ``.process()`` API
        via ``trust_remote_code``, else ``"qa_model"`` (legacy)."""
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            auto_map = getattr(config, "auto_map", None) or {}
            target = auto_map.get("AutoModel") or auto_map.get("AutoModelForTokenClassification")
            if target and "Highlighter" in target:
                return ModelSpanExtractor._FORMAT_HIGHLIGHTER
        # The config's remote code may raise anything. The legacy loader then reports the problem.
        except Exception as exc:
            logger.warning("Highlighter detection failed for %s: %s", model_id, exc)
        return ModelSpanExtractor._FORMAT_QA_MODEL

    def extract(
        self, question: str, chunks: Sequence[Chunk], *, config: ModelExtractConfig | None = None
    ) -> Extraction:
        """
        Extract the spans of the chunks that are relevant to the question.

        :param question: The query or question
        :param chunks: Chunks with unique ids
        :param config: Replaces the extractor's default config for this call
        :return: The spans, with the usage and warnings of the call
        :raises InvalidInputError: If two chunks share an id, or the config is of the wrong type
        """
        config = _call_config(config, self.config, ModelExtractConfig)
        _check_unique_ids(chunks)
        logger.debug("Extracting spans from %d chunks with a %s model", len(chunks), self._format)
        spans: List[ExtractedSpan] = []
        warnings: List[ResultWarning] = []
        for position, chunk in enumerate(chunks):
            if not chunk.text.strip():
                continue
            if self._format == self._FORMAT_HIGHLIGHTER:
                spans += self._highlight(question, chunk, config)
            else:
                chunk_spans, chunk_warnings = self._classify_sentences(
                    question, chunk, position, config
                )
                spans += chunk_spans
                warnings += chunk_warnings
        _log_result_warnings(warnings)
        usage = _model_usage(self.model_id, self.model_version, chunks)
        return Extraction(spans=spans, usage=[usage], warnings=warnings)

    def _highlight(
        self, question: str, chunk: Chunk, config: ModelExtractConfig
    ) -> List[ExtractedSpan]:
        output = self.model.process(
            question=question,
            context=chunk.text,
            threshold=config.threshold,
            min_span_chars=config.min_span_chars,
            merge_gap_chars=config.merge_gap_chars,
            max_length=config.max_length,
            doc_stride=config.doc_stride,
        )
        return [
            _model_span(chunk, span["start"], span["end"], span.get("score", UNSCORED))
            for span in output.get("spans", [])
            if chunk.text[span["start"] : span["end"]].strip()
        ]

    def _classify_sentences(
        self, question: str, chunk: Chunk, position: int, config: ModelExtractConfig
    ) -> Tuple[List[ExtractedSpan], List[ResultWarning]]:
        sentences = _sentence_offsets(chunk.text)
        probabilities = self._sentence_probabilities(
            question, [chunk.text[start:end] for start, end in sentences]
        )
        spans = [
            _model_span(chunk, start, end, probability)
            for (start, end), probability in zip(sentences, probabilities)
            if probability > config.threshold
        ]
        if len(probabilities) == len(sentences):
            return spans, []
        return spans, [_truncated(position, len(probabilities), len(sentences), "sentences")]

    def _sentence_probabilities(self, question: str, sentences: List[str]) -> List[float]:
        """The probability of each sentence that fits the model's input, in order."""
        import torch

        dataset_module = self._qa_dataset
        document = dataset_module.Document(
            sentences=[
                dataset_module.Sentence(text=text, relevant=False, sentence_id=f"s{i}")
                for i, text in enumerate(sentences)
            ]
        )
        sample = dataset_module.QASample(
            question=question,
            documents=[document],
            split="test",
            dataset_name="inference",
            task_type="qa",
        )
        dataset = dataset_module.QADataset([sample], self.tokenizer, max_length=512)
        if len(dataset) == 0:
            return []

        encoding = dataset[0]
        device = next(self.model.parameters()).device
        with torch.no_grad():
            predictions = self.model(
                input_ids=encoding["input_ids"].unsqueeze(0).to(device),
                attention_mask=encoding["attention_mask"].unsqueeze(0).to(device),
                sentence_boundaries=[encoding["sentence_boundaries"]],
            )
        if len(predictions) == 0 or len(predictions[0]) == 0:
            return []
        probabilities = torch.nn.functional.softmax(predictions[0], dim=1)[:, 1]
        return [float(probability) for probability in probabilities][: len(sentences)]


class SemanticHighlightConfig(ExtractConfig):
    """Settings of `SemanticHighlightExtractor`."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)  # the lowest probability of a span
    # "sentences" gives the complete sentences the model highlights, "spans" token-level spans
    # that may cross sentence boundaries.
    output_mode: Literal["sentences", "spans"] = "sentences"
    language: str = "auto"  # the language code of the sentence splitter, or "auto"
    # Spans mode only:
    min_span_tokens: PositiveInt = 3  # shorter spans are dropped
    merge_gap: NonNegativeInt = 2  # spans at most this many tokens apart are merged


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

    Load the model with `from_pretrained`. The constructor takes a loaded model and loads nothing.
    """

    DEFAULT_MODEL = "zilliz/semantic-highlight-bilingual-v1"

    def __init__(
        self,
        model: Any = None,
        device: str | None = _UNSET,
        threshold: float = _UNSET,
        output_mode: str = _UNSET,
        language: str = _UNSET,
        min_span_tokens: int = _UNSET,
        merge_gap: int = _UNSET,
        max_tokens: int = _UNSET,
        *,
        config: SemanticHighlightConfig | None = None,
        model_id: str | None = None,
        model_version: str | None = None,
        model_name: str = _UNSET,
    ):
        """
        :param model: The loaded semantic highlight model
        :param config: The settings of a call, unless the call passes its own
        :param model_id: The model's name or path, for the usage of a call
        :param model_version: The exact version of the model's weights, for the usage of a call

        The other arguments are deprecated. Without ``model``, or with ``model_name``, the
        extractor loads the model with `from_pretrained`, which also takes ``device`` and
        ``max_tokens``. The settings from ``threshold`` to ``merge_gap`` belong in ``config``.
        :raises ConfigurationError: If the arguments contradict each other or have invalid
            values, or a package the model needs is missing
        """
        settings = {
            "threshold": threshold,
            "output_mode": output_mode,
            "language": language,
            "min_span_tokens": min_span_tokens,
            "merge_gap": merge_gap,
        }
        config = _config_with_old_settings(type(self), SemanticHighlightConfig, config, settings)

        if model_name is not _UNSET or model is None or isinstance(model, str):
            if model_name is not _UNSET and model is not None:
                raise ConfigurationError(
                    "Pass a loaded model or model_name to SemanticHighlightExtractor, not both.",
                    code="invalid_value",
                )
            model_id = model if isinstance(model, str) else self.DEFAULT_MODEL
            model_id = model_id if model_name is _UNSET else model_name
            warn_deprecated(
                "Loading a model in the SemanticHighlightExtractor constructor",
                use="SemanticHighlightExtractor.from_pretrained(model_id)",
            )
            loaded = type(self).from_pretrained(
                model_id,
                None if device is _UNSET else device,
                model_version=model_version,
                max_tokens=4096 if max_tokens is _UNSET else max_tokens,
                config=config,
            )
            vars(self).update(vars(loaded))
            return
        if device is not _UNSET or max_tokens is not _UNSET:
            raise ConfigurationError(
                "SemanticHighlightExtractor got a loaded model with device or max_tokens. Pass "
                "them to from_pretrained instead.",
                code="invalid_value",
            )

        self.model = model
        self.config = config
        self.model_id = model_id
        self.model_version = model_version

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str | None = None,
        *,
        model_version: str | None = None,
        max_tokens: int = 4096,
        config: SemanticHighlightConfig | None = None,
    ) -> SemanticHighlightExtractor:
        """
        Download and load the model, or load it from a local path.

        :param model_id: A Hugging Face model id or the path of a local model
        :param device: ``"cuda"``, ``"mps"`` or ``"cpu"``, or None to choose the best available
        :param model_version: The version of weights from a local path. A model from Hugging
            Face records its resolved commit instead.
        :param max_tokens: The token limit of the model
        :param config: The default settings of a call
        :raises ConfigurationError: If a package the model needs is missing, or the model can't
            be downloaded or read
        """
        model, resolved_version = _load_semantic_highlight_model(model_id, device, max_tokens)
        return cls(
            model,
            config=config,
            model_id=model_id,
            model_version=resolved_version or model_version,
        )

    def extract(
        self,
        question: str,
        chunks: Sequence[Chunk],
        *,
        config: SemanticHighlightConfig | None = None,
    ) -> Extraction:
        """
        Extract the spans of the chunks that are relevant to the question.

        :param question: The query or question
        :param chunks: Chunks with unique ids
        :param config: Replaces the extractor's default config for this call
        :return: The spans, with the usage and warnings of the call
        :raises InvalidInputError: If two chunks share an id, or the config is of the wrong type
        """
        config = _call_config(config, self.config, SemanticHighlightConfig)
        _check_unique_ids(chunks)
        logger.debug("Extracting spans from %d chunks in %s mode", len(chunks), config.output_mode)
        spans: List[ExtractedSpan] = []
        warnings: List[ResultWarning] = []
        for position, chunk in enumerate(chunks):
            if not chunk.text.strip():
                continue
            if config.output_mode == "sentences":
                chunk_spans, chunk_warnings = self._sentences(question, chunk, position, config)
            else:
                chunk_spans, chunk_warnings = self._token_spans(question, chunk, position, config)
            spans += chunk_spans
            warnings += chunk_warnings
        _log_result_warnings(warnings)
        usage = _model_usage(self.model_id, self.model_version, chunks)
        return Extraction(spans=spans, usage=[usage], warnings=warnings)

    def _sentences(
        self, question: str, chunk: Chunk, position: int, config: SemanticHighlightConfig
    ) -> Tuple[List[ExtractedSpan], List[ResultWarning]]:
        """The sentences the model highlights, scored with their probability."""
        result = self.model.process(
            question=question,
            context=chunk.text,
            threshold=config.threshold,
            language=config.language,
            return_sentence_metrics=True,
            always_select_title=False,
            # Explicit preprocessing settings keep the model from reading environment variables.
            show_progress=False,
            preprocess_workers=0,
            preprocess_batch_size=32,
        )
        sentences = result.get("highlighted_sentences", [])
        # A sentence is highlighted when its probability is above the threshold, so the
        # probabilities above it belong to the highlighted sentences, in order.
        probabilities = [
            probability
            for probability in result.get("sentence_probabilities", [])
            if probability > config.threshold
        ]
        if len(probabilities) != len(sentences):
            probabilities = [UNSCORED] * len(sentences)

        spans = []
        warnings = []
        cursor = 0
        for sentence, probability in zip(sentences, probabilities):
            text = sentence.strip()
            if not text:
                continue
            start = chunk.text.find(text, cursor)
            if start < 0:
                start = chunk.text.find(text)
            if start < 0:
                warnings.append(
                    _not_found(position, "sentence the model highlighted", "span_text_not_found")
                )
                continue
            cursor = start + len(text)
            spans.append(_model_span(chunk, start, cursor, probability))
        return spans, warnings

    def _token_spans(
        self, question: str, chunk: Chunk, position: int, config: SemanticHighlightConfig
    ) -> Tuple[List[ExtractedSpan], List[ResultWarning]]:
        """Spans of contiguous tokens whose probability is above the threshold."""
        raw = self.model.get_raw_predictions(question, [chunk.text])
        if not raw.context_ranges:
            return [], []

        start_index, end_index = raw.context_ranges[0]
        probabilities = raw.pruning_probs[start_index:end_index]
        encoding = self.model.tokenizer(
            chunk.text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            max_length=getattr(self.model, "max_length", None),
        )
        offsets = encoding["offset_mapping"]
        warnings = []
        if len(probabilities) != len(offsets):
            token_count = min(len(probabilities), len(offsets))
            total = max(len(probabilities), len(offsets))
            warnings.append(_truncated(position, token_count, total, "tokens"))
            probabilities = probabilities[:token_count]
            offsets = offsets[:token_count]

        spans = []
        for token_start, token_end in _token_regions(probabilities, config):
            region = [float(probabilities[i]) for i in range(token_start, token_end)]
            start, end = _trim_punctuation(
                chunk.text, offsets[token_start][0], offsets[token_end - 1][1]
            )
            if end > start:
                spans.append(_model_span(chunk, start, end, sum(region) / len(region)))
        return spans, warnings


def _config_with_old_settings(
    extractor: type,
    config_class: Type[_ConfigT],
    config: _ConfigT | None,
    settings: Dict[str, Any],
) -> _ConfigT:
    """The default config of an extractor, built from its deprecated setting arguments if passed."""
    settings = {name: value for name, value in settings.items() if value is not _UNSET}
    if settings and config is not None:
        raise ConfigurationError(
            f"Pass the settings of {extractor.__name__} either in config or as arguments, not "
            "both.",
            code="invalid_value",
        )
    if settings:
        warn_deprecated(
            f"Passing {', '.join(settings)} to {extractor.__name__}",
            use=f"config={config_class.__name__}(...)",
            stacklevel=4,
        )
        return _config_from_settings(config_class, settings)
    return _default_config(config, config_class)


# What replaces the extractors' constructor arguments that have no effect.
_UNUSED_ARGUMENTS = {
    "extraction_mode": "nothing in its place, as it has no effect",
    "max_display_spans": "TransformConfig(max_display_spans=...)",
}


def _warn_unused(extractor: type, **arguments: Any) -> None:
    for name, value in arguments.items():
        if value is not _UNSET:
            warn_deprecated(
                f"Passing {name} to {extractor.__name__}",
                use=_UNUSED_ARGUMENTS[name],
                stacklevel=4,
            )


def _import_model_packages(needed_by: str) -> Tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as e:
        raise ConfigurationError(
            f"{needed_by} needs torch and transformers. Install them with "
            "pip install 'verbatim-core[model]'.",
            code="missing_dependency",
        ) from e
    return torch, transformers


def _import_qa_dataset() -> Any:
    _import_model_packages("The v1 sentence model")
    from .extractor_models import dataset

    return dataset


def _best_device(torch: Any) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model_span_model(model_id: str, device: str | None) -> Tuple[Any, Any, str | None]:
    """Load a ModelSpanExtractor model: the model, its tokenizer if it needs one, and its commit."""
    torch, transformers = _import_model_packages("ModelSpanExtractor")
    device = device or _best_device(torch)
    started = time.perf_counter()
    with _load_errors_translated(model_id):
        if ModelSpanExtractor._detect_format(model_id) == ModelSpanExtractor._FORMAT_HIGHLIGHTER:
            model = transformers.AutoModel.from_pretrained(model_id, trust_remote_code=True)
            # The highlighter loads its tokenizer on first use. Loading it now keeps the download
            # out of the first extraction.
            if hasattr(model, "_get_tokenizer"):
                model._get_tokenizer()
            tokenizer = None  # the highlighter holds its tokenizer itself
        else:
            from .extractor_models.model import QAModel

            model = QAModel.from_pretrained(model_id)
            tokenizer = _load_qa_tokenizer(transformers, model_id, model)
    model.to(device)
    model.eval()
    _log_model_loaded(logger, model_id, device, started)
    return model, tokenizer, _commit(model)


def _load_qa_tokenizer(transformers: Any, model_id: str, model: Any) -> Any:
    try:
        return transformers.AutoTokenizer.from_pretrained(model_id)
    except (OSError, ValueError) as e:
        base_model = getattr(model.config, "model_name", "answerdotai/ModernBERT-base")
        logger.warning("Loading the tokenizer of %s from %s: %s", model_id, base_model, e)
        return transformers.AutoTokenizer.from_pretrained(base_model)


def _load_semantic_highlight_model(
    model_id: str, device: str | None, max_tokens: int
) -> Tuple[Any, str | None]:
    """Load a SemanticHighlightExtractor model: the model and its commit."""
    torch, transformers = _import_model_packages("SemanticHighlightExtractor")
    device = device or _best_device(torch)
    started = time.perf_counter()
    with _load_errors_translated(model_id):
        model = transformers.AutoModel.from_pretrained(
            model_id, trust_remote_code=True, max_length=max_tokens
        )
    model.to(device)
    model.eval()
    _log_model_loaded(logger, model_id, device, started)
    return model, _commit(model)


def _commit(model: Any) -> str | None:
    """The Hugging Face commit the model was loaded from. None for a local path."""
    return getattr(getattr(model, "config", None), "_commit_hash", None)


def _model_span(chunk: Chunk, start: int, end: int, probability: float) -> ExtractedSpan:
    return ExtractedSpan(
        chunk_id=chunk.id,
        start=start,
        end=end,
        text=chunk.text[start:end],
        # Rounding can put a probability just outside 0 to 1.
        score=min(max(float(probability), 0.0), 1.0),
    )


def _model_usage(model_id: str | None, model_version: str | None, chunks: Sequence[Chunk]) -> Usage:
    return Usage(
        model_id=model_id,
        model_version=model_version,
        units={"chunks": len(chunks), "characters": sum(len(chunk.text) for chunk in chunks)},
    )


def _sentence_offsets(text: str) -> List[Tuple[int, int]]:
    """The offsets of the sentences of a text, without their surrounding whitespace."""
    offsets = []
    start = 0
    for boundary in [*re.finditer(r"(?<=[.!?])\s+", text), None]:
        end = boundary.start() if boundary else len(text)
        sentence = text[start:end]
        if sentence.strip():
            leading = len(sentence) - len(sentence.lstrip())
            trailing = len(sentence) - len(sentence.rstrip())
            offsets.append((start + leading, end - trailing))
        start = boundary.end() if boundary else len(text)
    return offsets


def _token_regions(probabilities: Any, config: SemanticHighlightConfig) -> List[Tuple[int, int]]:
    """Runs of tokens above the threshold, of at least min_span_tokens, merged across gaps."""
    regions = []
    region_start = None
    for i, probability in enumerate([*probabilities, 0.0]):
        if probability > config.threshold and region_start is None:
            region_start = i
        elif probability <= config.threshold and region_start is not None:
            if i - region_start >= config.min_span_tokens:
                regions.append((region_start, i))
            region_start = None

    merged: List[Tuple[int, int]] = []
    for start, end in regions:
        if merged and start - merged[-1][1] <= config.merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


_PUNCTUATION = r"[\s.,;:!?\-–—]*"


def _trim_punctuation(text: str, start: int, end: int) -> Tuple[int, int]:
    """Offsets without the whitespace and punctuation at either end of text[start:end]."""
    start += re.match(_PUNCTUATION, text[start:end]).end()
    end = start + re.search(_PUNCTUATION + "$", text[start:end]).start()
    return start, end


def _truncated(position: int, examined: int, total: int, unit: str) -> ResultWarning:
    return ResultWarning(
        code="chunk_truncated",
        message=f"The model examined only {examined} of {total} {unit} of chunks[{position}].",
    )


def _not_found(position: int, what: str, code: str) -> ResultWarning:
    return ResultWarning(code=code, message=f"A {what} for chunks[{position}] is not in the chunk.")


def _log_result_warnings(warnings: List[ResultWarning]) -> None:
    """One WARNING record per warning code of a result, with the number of its warnings."""
    for code in sorted({warning.code for warning in warnings}):
        count = sum(warning.code == code for warning in warnings)
        logger.warning(
            "Added %d result warnings with the code %s",
            count,
            code,
            extra={"event": "fallback_used", "warning_code": code},
        )


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
        self.config = _config_with_old_settings(type(self), LLMExtractConfig, config, settings)
        _warn_unused(type(self), max_display_spans=max_display_spans)
        if llm_client is None:
            warn_deprecated(
                "LLMSpanExtractor without an llm_client", use="LLMSpanExtractor(LLMClient(model))"
            )
            llm_client = LLMClient("gpt-4o-mini" if model is _UNSET else model)
        elif model is not _UNSET:
            warn_deprecated("Passing model to LLMSpanExtractor", use="the model of the llm_client")

        self.llm_client = llm_client

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
    return client_completion(client, call.prompt, json_mode=True, system_prompt=call.system_prompt)


async def _acomplete(client: LLMClient, call: _LLMCall) -> Completion:
    return await client_completion_async(
        client, call.prompt, json_mode=True, system_prompt=call.system_prompt
    )


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
