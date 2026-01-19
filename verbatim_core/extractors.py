"""
Extractors for identifying relevant spans in documents (RAG-agnostic).

This copy avoids importing vector-store specific types; accepts any objects
with a `.text` attribute as search results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Dict

from .llm_client import LLMClient


class SpanExtractor(ABC):
    """Abstract base class for span extractors."""

    @abstractmethod
    def extract_spans(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
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
    """Extract spans using a fine-tuned QA sentence classification model."""

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        threshold: float = 0.5,
        extraction_mode: str = "individual",
        max_display_spans: int = 5,
    ):
        """
        Initialize the model-based span extractor.

        :param model_path: Path to the trained QA model
        :param device: Device to run the model on (auto-detects if None)
        :param threshold: Threshold for sentence classification
        :param extraction_mode: Not used for model extractor
        :param max_display_spans: Not used for model extractor
        """
        # Lazy-import heavy deps so importing this module doesn't require them
        import torch  # noqa: WPS433 (allow local import)
        from transformers import AutoTokenizer  # noqa: WPS433
        from .extractor_models.model import QAModel  # noqa: WPS433
        from .extractor_models.dataset import (  # noqa: WPS433
            QADataset,
            Sentence as DatasetSentence,
            Document as DatasetDocument,
            QASample,
        )

        self.model_path = model_path
        self.threshold = threshold
        self._torch = torch
        self.device = device or ("cuda" if self._torch.cuda.is_available() else "cpu")

        # Cache dataset classes for reuse without re-import
        self.QADataset = QADataset
        self.DatasetSentence = DatasetSentence
        self.DatasetDocument = DatasetDocument
        self.QASample = QASample

        print(f"Loading model from {model_path}...")

        # Load the model
        self.model = QAModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Load the tokenizer
        try:
            print(f"Loading tokenizer from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            print("Tokenizer loaded successfully.")
        except Exception as e:
            print(f"Could not load tokenizer from {model_path}: {e}")
            # Fall back to base model
            base_model = getattr(
                self.model.config, "model_name", "answerdotai/ModernBERT-base"
            )
            print(f"Trying to load tokenizer from base model: {base_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            print(f"Loaded tokenizer from {base_model}")

    def _split_into_sentences(self, text: str) -> list[str]:
        """Simple sentence splitting."""
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def extract_spans(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """
        Extract spans using the trained model.

        :param question: The query or question
        :param search_results: List of search results to extract from
        :return: Dictionary mapping result text to list of relevant spans
        """
        relevant_spans = {}

        for result in search_results:
            raw_text = getattr(result, "text", "")

            # Split text into sentences
            raw_sentences = self._split_into_sentences(raw_text)
            if not raw_sentences:
                relevant_spans[raw_text] = []
                continue

            # Create dataset objects for model processing
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

            # Extract spans based on predictions
            spans = []
            if len(predictions) > 0 and len(predictions[0]) > 0:
                sentence_preds = self._torch.nn.functional.softmax(
                    predictions[0], dim=1
                )
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
        """
        import torch
        from transformers import AutoModel

        if output_mode not in ("sentences", "spans"):
            raise ValueError(
                f"output_mode must be 'sentences' or 'spans', got {output_mode!r}"
            )

        self.threshold = threshold
        self.output_mode = output_mode
        self.language = language
        self.min_span_tokens = min_span_tokens
        self.merge_gap = merge_gap
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading semantic highlight model: {model_name}...")
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.tokenizer = self.model.tokenizer

    def extract_spans(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
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
                print(f"Semantic highlight extraction failed: {e}")
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
        )
        offset_mapping = encoding["offset_mapping"]

        # Validate lengths match
        if len(context_probs) != len(offset_mapping):
            print(
                f"Warning: length mismatch probs={len(context_probs)} "
                f"tokens={len(offset_mapping)}"
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
    ):
        """
        Initialize the LLM span extractor.

        :param llm_client: LLM client for extraction (creates one if None)
        :param model: The LLM model to use (if creating new client)
        :param extraction_mode: "batch", "individual", or "auto"
        :param max_display_spans: Maximum spans to prioritize for display
        :param batch_size: Maximum documents to process in batch mode
        """
        self.llm_client = llm_client or LLMClient(model)
        self.extraction_mode = extraction_mode
        self.max_display_spans = max_display_spans
        self.batch_size = batch_size

    def extract_spans(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
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
        """
        print("Extracting spans (batch mode)...")

        # Limit to batch_size to avoid prompt size issues
        top_results = search_results[: self.batch_size]

        # Build document mapping for LLMClient
        documents_text = {}
        for i, result in enumerate(top_results):
            documents_text[f"doc_{i}"] = getattr(result, "text", "")

        try:
            # Use LLMClient for extraction
            extracted_data = self.llm_client.extract_spans(question, documents_text)

            # Map back to original search results and verify spans
            verified_spans = {}

            # Process documents that were included in batch
            for i, result in enumerate(top_results):
                doc_key = f"doc_{i}"
                result_text = getattr(result, "text", "")
                if doc_key in extracted_data:
                    verified = self._verify_spans(extracted_data[doc_key], result_text)
                    verified_spans[result_text] = verified
                else:
                    verified_spans[result_text] = []

            # Handle remaining documents (beyond batch_size) with empty spans
            for i in range(self.batch_size, len(search_results)):
                verified_spans[getattr(search_results[i], "text", "")] = []

            return verified_spans

        except Exception as e:
            print(f"Batch extraction failed, falling back to individual: {e}")
            return self._extract_spans_individual(question, search_results)

    async def _extract_spans_batch_async(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """
        Async batch extraction.
        """
        print("Extracting spans (async batch mode)...")

        top_results = search_results[: self.batch_size]

        documents_text = {}
        for i, result in enumerate(top_results):
            documents_text[f"doc_{i}"] = getattr(result, "text", "")

        try:
            extracted_data = await self.llm_client.extract_spans_async(
                question, documents_text
            )

            verified_spans = {}

            for i, result in enumerate(top_results):
                doc_key = f"doc_{i}"
                result_text = getattr(result, "text", "")
                if doc_key in extracted_data:
                    verified = self._verify_spans(extracted_data[doc_key], result_text)
                    verified_spans[result_text] = verified
                else:
                    verified_spans[result_text] = []

            for i in range(self.batch_size, len(search_results)):
                verified_spans[getattr(search_results[i], "text", "")] = []

            return verified_spans

        except Exception as e:
            print(f"Async batch extraction failed, falling back to individual: {e}")
            return await self._extract_spans_individual_async(question, search_results)

    def _extract_spans_individual(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """
        Extract spans from documents individually (one at a time).

        :param question: The query or question
        :param search_results: List of search results to process
        :return: Dictionary mapping result text to list of relevant spans
        """
        print("Extracting spans (individual mode)...")
        all_spans = {}

        for result in search_results:
            result_text = getattr(result, "text", "")
            try:
                extracted_spans = self.llm_client.extract_relevant_spans(
                    question, result_text
                )
                verified = self._verify_spans(extracted_spans, result_text)
                all_spans[result_text] = verified
            except Exception as e:
                print(f"Individual extraction failed for document: {e}")
                all_spans[result_text] = []

        return all_spans

    async def _extract_spans_individual_async(
        self, question: str, search_results: List[Any]
    ) -> Dict[str, List[str]]:
        """
        Async individual extraction.
        """
        print("Extracting spans (async individual mode)...")
        all_spans = {}

        for result in search_results:
            result_text = getattr(result, "text", "")
            try:
                extracted_spans = await self.llm_client.extract_relevant_spans_async(
                    question, result_text
                )
                verified = self._verify_spans(extracted_spans, result_text)
                all_spans[result_text] = verified
            except Exception as e:
                print(f"Async individual extraction failed for document: {e}")
                all_spans[result_text] = []

        return all_spans

    def _verify_spans(self, spans: List[str], document_text: str) -> List[str]:
        """
        Verify that extracted spans actually exist in the document text.

        :param spans: List of spans to verify
        :param document_text: Original document text
        :return: List of verified spans that exist in the document
        """
        verified = []
        for span in spans:
            if span.strip() and span.strip() in document_text:
                verified.append(span.strip())
            else:
                print(
                    f"Warning: Span not found verbatim in document: '{span[:100]}...'"
                )
        return verified
