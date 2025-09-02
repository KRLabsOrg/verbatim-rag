"""
Extractors for identifying relevant spans in documents (RAG-agnostic).

This copy avoids importing vector-store specific types; accepts any objects
with a `.text` attribute as search results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Dict
import torch
from transformers import AutoTokenizer

from .llm_client import LLMClient
from .extractor_models.model import QAModel
from .extractor_models.dataset import (
    QADataset,
    Sentence as DatasetSentence,
    Document as DatasetDocument,
    QASample,
)


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
        self.model_path = model_path
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

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
                DatasetSentence(text=sent, relevant=False, sentence_id=f"s{i}")
                for i, sent in enumerate(raw_sentences)
            ]
            dataset_doc = DatasetDocument(sentences=dataset_sentences)

            qa_sample = QASample(
                question=question,
                documents=[dataset_doc],
                split="test",
                dataset_name="inference",
                task_type="qa",
            )

            dataset = QADataset([qa_sample], self.tokenizer, max_length=512)
            if len(dataset) == 0:
                relevant_spans[raw_text] = []
                continue

            encoding = dataset[0]

            input_ids = encoding["input_ids"].unsqueeze(0).to(self.device)
            attention_mask = encoding["attention_mask"].unsqueeze(0).to(self.device)

            with torch.no_grad():
                predictions = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sentence_boundaries=[encoding["sentence_boundaries"]],
                )

            # Extract spans based on predictions
            spans = []
            if len(predictions) > 0 and len(predictions[0]) > 0:
                sentence_preds = torch.nn.functional.softmax(predictions[0], dim=1)
                for i, pred in enumerate(sentence_preds):
                    if i < len(raw_sentences) and pred[1] > self.threshold:
                        spans.append(raw_sentences[i])

            relevant_spans[raw_text] = spans

        return relevant_spans


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
