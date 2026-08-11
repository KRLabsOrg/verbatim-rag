"""
Extractors for identifying relevant spans in documents (RAG-agnostic).

This copy avoids importing vector-store specific types; accepts any objects
with a `.text` attribute as search results.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from rapidfuzz.fuzz import partial_ratio_alignment

from .llm_client import LLMClient

logger = logging.getLogger(__name__)

_FUZZY_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class _NormalizedTokenText:
    text: str
    original_token_offsets: List[tuple[int, int]]
    normalized_token_offsets: List[tuple[int, int]]


class SpanExtractor(ABC):
    """Abstract base class for span extractors."""

    @abstractmethod
    def extract_spans(self, question: str, search_results: List[Any]) -> Dict[str, List[str]]:
        """
        Extract relevant spans from search results.

        :param question: The query or question
        :param search_results: List of search results to extract from
        :return: Dictionary mapping result text to list of relevant spans
        """
        raise NotImplementedError

    async def extract_spans_async(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """Default async implementation that delegates to sync version."""
        import asyncio

        return await asyncio.to_thread(self.extract_spans, question, search_results)


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


class LLMSpanExtractor(SpanExtractor):
    """Extract spans using an LLM with centralized client and batch processing."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        model: str = "gpt-4o-mini",
        extraction_mode: str = "auto",
        max_display_spans: int = 5,
        batch_size: int = 5,
        span_match_mode: str = "exact",
        fuzzy_threshold: float = 0.8,
        extraction_prompt: str | None = None,
        system_prompt: str | None = None,
    ):
        """
        Initialize the LLM span extractor.

        :param llm_client: LLM client for extraction (creates one if None)
        :param model: The LLM model to use (if creating new client)
        :param extraction_mode: "batch", "individual", or "auto"
        :param max_display_spans: Maximum spans to prioritize for display
        :param batch_size: Maximum documents to process in batch mode
        :param span_match_mode: "exact" for substring match, "fuzzy" for fuzzy matching
        :param fuzzy_threshold: Minimum score (0-1) for fuzzy span matching
        :param extraction_prompt: Custom prompt template with {question} and {documents} placeholders
        :param system_prompt: Optional system prompt to use with custom extraction_prompt
        """
        if span_match_mode not in ("exact", "fuzzy"):
            raise ValueError(f"span_match_mode must be 'exact' or 'fuzzy', got {span_match_mode!r}")
        self.llm_client = llm_client or LLMClient(model)
        self.extraction_mode = extraction_mode
        self.max_display_spans = max_display_spans
        self.batch_size = batch_size
        self.span_match_mode = span_match_mode
        self.fuzzy_threshold = fuzzy_threshold
        self.extraction_prompt = extraction_prompt
        self.system_prompt = system_prompt

    def _build_custom_prompt(self, question: str, documents: Dict[str, str]) -> str:
        """Build a prompt from the custom extraction_prompt template.

        Uses Jinja2 rendering (same as the prompt bank) so custom prompts
        can use {{ variable }}, {% if %} conditionals, and literal braces
        without escaping issues.

        :param question: The user's question
        :param documents: Dictionary mapping doc IDs to document text
        :return: Formatted prompt string
        """
        from .prompts import render_prompt

        docs_formatted = "\n\n".join(f"[{doc_id}]\n{text}" for doc_id, text in documents.items())
        return render_prompt(
            self.extraction_prompt,
            question=question,
            documents=docs_formatted,
        )

    def extract_spans(self, question: str, search_results: List[Any]) -> Dict[str, List[str]]:
        """
        Extract spans using LLM with mode selection.

        :param question: The query or question
        :param search_results: List of search results to extract from
        :return: Dictionary mapping result text to list of relevant spans
        """
        if not search_results:
            return {}

        # Decide on processing mode
        should_batch = self.extraction_mode == "batch" or (
            self.extraction_mode == "auto" and len(search_results) <= self.batch_size
        )

        if should_batch:
            return self._extract_spans_batch(question, search_results)
        else:
            return self._extract_spans_individual(question, search_results)

    async def extract_spans_async(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """
        Async version of span extraction.

        :param question: The query or question
        :param search_results: List of search results to extract from
        :return: Dictionary mapping result text to list of relevant spans
        """
        if not search_results:
            return {}

        should_batch = self.extraction_mode == "batch" or (
            self.extraction_mode == "auto" and len(search_results) <= self.batch_size
        )

        if should_batch:
            return await self._extract_spans_batch_async(question, search_results)
        else:
            return await self._extract_spans_individual_async(question, search_results)

    def _extract_spans_batch(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """
        Extract spans from multiple documents using batch processing.

        Iterates in chunks of batch_size so all documents are processed.
        """
        logger.info("Extracting spans (batch mode)...")

        all_verified: Dict[str, List[str]] = {}

        for batch_start in range(0, len(search_results), self.batch_size):
            batch = search_results[batch_start : batch_start + self.batch_size]

            # Build document mapping for this chunk
            documents_text = {}
            for i, result in enumerate(batch):
                documents_text[f"doc_{i}"] = getattr(result, "text", "")

            try:
                if self.extraction_prompt:
                    # Custom prompt path
                    prompt = self._build_custom_prompt(question, documents_text)
                    response = self.llm_client.complete(
                        prompt, json_mode=True, system_prompt=self.system_prompt
                    )
                    extracted_data = json.loads(response)
                else:
                    # Use LLMClient for extraction
                    extracted_data = self.llm_client.extract_spans(question, documents_text)

                # Map back to original search results and verify spans
                for i, result in enumerate(batch):
                    doc_key = f"doc_{i}"
                    result_text = getattr(result, "text", "")
                    if doc_key in extracted_data:
                        verified = self._verify_spans(extracted_data[doc_key], result_text)
                        all_verified[result_text] = verified
                    else:
                        all_verified[result_text] = []

            except Exception as e:
                logger.warning(
                    "Batch extraction failed for chunk starting at %d, "
                    "falling back to individual: %s",
                    batch_start,
                    e,
                )
                # Fall back to individual for this chunk
                for result in batch:
                    result_text = getattr(result, "text", "")
                    try:
                        if self.extraction_prompt:
                            single_docs = {"doc_0": result_text}
                            prompt = self._build_custom_prompt(question, single_docs)
                            response = self.llm_client.complete(
                                prompt, json_mode=True, system_prompt=self.system_prompt
                            )
                            extracted = json.loads(response).get("doc_0", [])
                        else:
                            extracted = self.llm_client.extract_relevant_spans(
                                question, result_text
                            )
                        all_verified[result_text] = self._verify_spans(extracted, result_text)
                    except Exception as inner_e:
                        logger.error("Individual fallback extraction failed: %s", inner_e)
                        all_verified[result_text] = []

        return all_verified

    async def _extract_spans_batch_async(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """
        Async batch extraction.

        Iterates in chunks of batch_size so all documents are processed.
        """
        logger.info("Extracting spans (async batch mode)...")

        all_verified: Dict[str, List[str]] = {}

        for batch_start in range(0, len(search_results), self.batch_size):
            batch = search_results[batch_start : batch_start + self.batch_size]

            documents_text = {}
            for i, result in enumerate(batch):
                documents_text[f"doc_{i}"] = getattr(result, "text", "")

            try:
                if self.extraction_prompt:
                    prompt = self._build_custom_prompt(question, documents_text)
                    response = await self.llm_client.complete_async(
                        prompt, json_mode=True, system_prompt=self.system_prompt
                    )
                    extracted_data = json.loads(response)
                else:
                    extracted_data = await self.llm_client.extract_spans_async(
                        question, documents_text
                    )

                for i, result in enumerate(batch):
                    doc_key = f"doc_{i}"
                    result_text = getattr(result, "text", "")
                    if doc_key in extracted_data:
                        verified = self._verify_spans(extracted_data[doc_key], result_text)
                        all_verified[result_text] = verified
                    else:
                        all_verified[result_text] = []

            except Exception as e:
                logger.warning(
                    "Async batch extraction failed for chunk starting at %d, "
                    "falling back to individual: %s",
                    batch_start,
                    e,
                )
                for result in batch:
                    result_text = getattr(result, "text", "")
                    try:
                        if self.extraction_prompt:
                            single_docs = {"doc_0": result_text}
                            prompt = self._build_custom_prompt(question, single_docs)
                            response = await self.llm_client.complete_async(
                                prompt, json_mode=True, system_prompt=self.system_prompt
                            )
                            extracted = json.loads(response).get("doc_0", [])
                        else:
                            extracted = await self.llm_client.extract_relevant_spans_async(
                                question, result_text
                            )
                        all_verified[result_text] = self._verify_spans(extracted, result_text)
                    except Exception as inner_e:
                        logger.error("Async individual fallback extraction failed: %s", inner_e)
                        all_verified[result_text] = []

        return all_verified

    def _extract_spans_individual(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """
        Extract spans from documents individually (one at a time).

        :param question: The query or question
        :param search_results: List of search results to process
        :return: Dictionary mapping result text to list of relevant spans
        """
        logger.info("Extracting spans (individual mode)...")
        all_spans = {}

        for result in search_results:
            result_text = getattr(result, "text", "")
            try:
                if self.extraction_prompt:
                    single_docs = {"doc_0": result_text}
                    prompt = self._build_custom_prompt(question, single_docs)
                    response = self.llm_client.complete(
                        prompt, json_mode=True, system_prompt=self.system_prompt
                    )
                    extracted_spans = json.loads(response).get("doc_0", [])
                else:
                    extracted_spans = self.llm_client.extract_relevant_spans(question, result_text)
                verified = self._verify_spans(extracted_spans, result_text)
                all_spans[result_text] = verified
            except Exception as e:
                logger.error("Individual extraction failed for document: %s", e)
                all_spans[result_text] = []

        return all_spans

    async def _extract_spans_individual_async(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """
        Async individual extraction using concurrent asyncio.gather.
        """
        import asyncio

        logger.info("Extracting spans (async individual mode)...")

        async def _extract_one(result: Any) -> tuple[str, List[str]]:
            result_text = getattr(result, "text", "")
            try:
                if self.extraction_prompt:
                    single_docs = {"doc_0": result_text}
                    prompt = self._build_custom_prompt(question, single_docs)
                    response = await self.llm_client.complete_async(
                        prompt, json_mode=True, system_prompt=self.system_prompt
                    )
                    extracted = json.loads(response).get("doc_0", [])
                else:
                    extracted = await self.llm_client.extract_relevant_spans_async(
                        question, result_text
                    )
                return result_text, self._verify_spans(extracted, result_text)
            except Exception as e:
                logger.error("Async individual extraction failed for document: %s", e)
                return result_text, []

        results = await asyncio.gather(*[_extract_one(r) for r in search_results])
        return dict(results)

    def _verify_spans(self, spans: List[str], document_text: str) -> List[str]:
        """
        Verify that extracted spans actually exist in the document text.

        When span_match_mode is "fuzzy", uses rapidfuzz to locate spans that
        may not match exactly (e.g. due to encoding issues, OCR artifacts,
        or minor differences in whitespace/punctuation). In fuzzy mode, the
        returned spans are the actual text from the document, not the LLM's
        version, ensuring correct character offsets for highlighting.

        :param spans: List of spans to verify
        :param document_text: Original document text
        :return: List of verified spans that exist in the document
        """
        if self.span_match_mode == "fuzzy":
            return self._verify_spans_fuzzy(spans, document_text)

        verified = []
        for span in spans:
            if span.strip() and span.strip() in document_text:
                verified.append(span.strip())
            else:
                logger.warning("Span not found verbatim in document: '%s...'", span[:100])
        return verified

    def _verify_spans_fuzzy(self, spans: List[str], document_text: str) -> List[str]:
        """
        Verify spans using fuzzy matching against the document text.

        Returns the actual matched text from the document (not the LLM's
        version), so that downstream highlight offsets are always correct.

        :param spans: List of spans to verify
        :param document_text: Original document text
        :return: List of verified spans from the document text
        """
        verified = []
        for span in spans:
            span = span.strip()
            if not span:
                continue

            # Try exact match first (fast path)
            if span in document_text:
                verified.append(span)
                continue

            # Fall back to fuzzy matching
            score, matched_text = self._find_fuzzy_span_match(span, document_text)
            if score >= self.fuzzy_threshold and matched_text:
                verified.append(matched_text)
            else:
                logger.warning(
                    "Span not found in document (best fuzzy score: %.2f): '%s...'",
                    score,
                    span[:100],
                )
        return verified

    @classmethod
    def _find_fuzzy_span_match(cls, span: str, document_text: str) -> tuple[float, str]:
        """
        Locate a fuzzy span in the document and return document text on token boundaries.

        RapidFuzz alignment offsets are character-level and can land in the
        middle of words. Matching normalized token text keeps OCR/punctuation
        tolerance while mapping the final slice back to original token offsets.
        """
        normalized_span = cls._normalize_for_fuzzy_span_match(span)
        normalized_document = cls._normalize_for_fuzzy_span_match(document_text)

        if not normalized_span.text or not normalized_document.text:
            return 0.0, ""

        result = partial_ratio_alignment(normalized_span.text, normalized_document.text)
        score = result.score / 100.0
        matched_text = cls._slice_original_text_from_normalized_offsets(
            document_text,
            normalized_document,
            result.dest_start,
            result.dest_end,
        )
        return score, matched_text

    @staticmethod
    def _normalize_for_fuzzy_span_match(text: str) -> _NormalizedTokenText:
        parts: List[str] = []
        original_token_offsets: List[tuple[int, int]] = []
        normalized_token_offsets: List[tuple[int, int]] = []
        normalized_pos = 0

        for match in _FUZZY_TOKEN_RE.finditer(text):
            token = unicodedata.normalize("NFKC", match.group(0)).casefold()
            if not token:
                continue

            if parts:
                parts.append(" ")
                normalized_pos += 1

            token_start = normalized_pos
            parts.append(token)
            normalized_pos += len(token)

            original_token_offsets.append((match.start(), match.end()))
            normalized_token_offsets.append((token_start, normalized_pos))

        return _NormalizedTokenText(
            text="".join(parts),
            original_token_offsets=original_token_offsets,
            normalized_token_offsets=normalized_token_offsets,
        )

    @staticmethod
    def _slice_original_text_from_normalized_offsets(
        original_text: str,
        normalized_text: _NormalizedTokenText,
        normalized_start: int,
        normalized_end: int,
    ) -> str:
        first_token_idx = None
        last_token_idx = None

        for idx, (token_start, token_end) in enumerate(normalized_text.normalized_token_offsets):
            if token_end <= normalized_start:
                continue
            if token_start >= normalized_end:
                break

            if first_token_idx is None:
                first_token_idx = idx
            last_token_idx = idx

        if first_token_idx is None or last_token_idx is None:
            return ""

        original_start = normalized_text.original_token_offsets[first_token_idx][0]
        original_end = normalized_text.original_token_offsets[last_token_idx][1]
        return original_text[original_start:original_end]
