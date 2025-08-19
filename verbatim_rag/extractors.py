"""
Extractors for identifying relevant spans in documents.

This module provides interfaces for extracting relevant spans from documents,
allowing for easy implementation of different extraction methods.
"""

from abc import ABC, abstractmethod
import torch
from transformers import AutoTokenizer

from verbatim_rag.vector_stores import SearchResult
from verbatim_rag.llm_client import LLMClient
from verbatim_rag.extractor_models.model import QAModel
from verbatim_rag.extractor_models.dataset import (
    QADataset,
    Sentence as DatasetSentence,
    Document as DatasetDocument,
    QASample,
)


class SpanExtractor(ABC):
    """Abstract base class for span extractors."""

    @abstractmethod
    def extract_spans(
        self, question: str, search_results: list[SearchResult]
    ) -> dict[str, list[str]]:
        """
        Extract relevant spans from search results based on a question.

        :param question: The query or question to extract spans for
        :param search_results: List of search results to extract spans from
        :return: Dictionary mapping result text to list of relevant spans
        """
        pass


class ModelSpanExtractor(SpanExtractor):
    """Extract spans using a fine-tuned QA sentence classification model."""

    def __init__(
        self,
        model_path: str,
        device: str = None,
        threshold: float = 0.5,
        extraction_mode: str = "individual",  # Compatible parameter (not used)
        max_display_spans: int = 5,  # Compatible parameter (not used)
    ):
        """
        Initialize the model-based span extractor.

        :param model_path: Path to the saved model (local path or HuggingFace model ID)
        :param device: Device to run the model on ('cpu', 'cuda', etc). If None, will use CUDA if available.
        :param threshold: Confidence threshold for considering a span relevant (0.0-1.0)
        :param extraction_mode: Ignored for ModelSpanExtractor (for API compatibility)
        :param max_display_spans: Ignored for ModelSpanExtractor (for API compatibility)
        """
        self.model_path = model_path
        self.threshold = threshold

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading model from {model_path}...")

        # Load model using HuggingFace's standard methods
        self.model = QAModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Load tokenizer using HuggingFace's standard methods
        try:
            print(f"Loading tokenizer from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            print("Tokenizer loaded successfully.")
        except Exception as e:
            print(f"Could not load tokenizer from {model_path}: {e}")
            # Try to get base model name from model config
            base_model = getattr(
                self.model.config, "model_name", "answerdotai/ModernBERT-base"
            )
            print(f"Trying to load tokenizer from base model: {base_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            print(f"Loaded tokenizer from {base_model}")

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using simple rules.

        :param text: The text to split
        :return: List of sentences
        """
        import re

        # Simple rule-based sentence splitting (can be improved)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def extract_spans(
        self, question: str, search_results: list[SearchResult]
    ) -> dict[str, list[str]]:
        """
        Extract relevant spans using the trained model.

        :param question: The query or question
        :param search_results: List of search results to extract from
        :return: Dictionary mapping result text to list of relevant spans
        """
        relevant_spans = {}

        for result in search_results:
            # Split the result text into sentences
            raw_sentences = self._split_into_sentences(result.text)
            if not raw_sentences:
                relevant_spans[result.text] = []
                continue

            # Create Dataset objects to use the same processing logic as training
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

            # Use the QADataset class to process the data just like during training
            dataset = QADataset([qa_sample], self.tokenizer, max_length=512)

            # Skip if dataset processing didn't yield any results
            if len(dataset) == 0:
                relevant_spans[result.text] = []
                continue

            encoding = dataset[0]

            input_ids = encoding["input_ids"].unsqueeze(0).to(self.device)
            attention_mask = encoding["attention_mask"].unsqueeze(0).to(self.device)

            # Make prediction with the model
            with torch.no_grad():
                predictions = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sentence_boundaries=[encoding["sentence_boundaries"]],
                )

            # Extract relevant sentences
            spans = []
            if len(predictions) > 0 and len(predictions[0]) > 0:
                sentence_preds = torch.nn.functional.softmax(predictions[0], dim=1)

                for i, pred in enumerate(sentence_preds):
                    if i < len(raw_sentences) and pred[1] > self.threshold:
                        spans.append(raw_sentences[i])

            relevant_spans[result.text] = spans

        return relevant_spans


class LLMSpanExtractor(SpanExtractor):
    """Extract spans using an LLM with centralized client and batch processing."""

    def __init__(
        self,
        llm_client: LLMClient = None,
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
        self, question: str, search_results: list[SearchResult]
    ) -> dict[str, list[str]]:
        """
        Extract relevant spans using centralized LLM client.

        :param question: The query or question
        :param search_results: List of search results to extract from
        :return: Dictionary mapping result text to list of relevant spans
        """
        if not search_results:
            return {}

        # Determine extraction method
        should_batch = self.extraction_mode == "batch" or (
            self.extraction_mode == "auto" and len(search_results) <= self.batch_size
        )

        if should_batch:
            return self._extract_spans_batch(question, search_results)
        else:
            return self._extract_spans_individual(question, search_results)

    async def extract_spans_async(
        self, question: str, search_results: list[SearchResult]
    ) -> dict[str, list[str]]:
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
        self, question: str, search_results: list[SearchResult]
    ) -> dict[str, list[str]]:
        """
        Extract spans from multiple documents using batch processing.
        """
        print("Extracting spans (batch mode)...")

        # Limit to batch_size to avoid prompt size issues
        top_results = search_results[: self.batch_size]

        # Build document mapping for LLMClient
        documents_text = {}
        for i, result in enumerate(top_results):
            documents_text[f"doc_{i}"] = result.text

        try:
            # Use LLMClient for extraction
            extracted_data = self.llm_client.extract_spans(question, documents_text)

            # Map back to original search results and verify spans
            verified_spans = {}

            # Process documents that were included in batch
            for i, result in enumerate(top_results):
                doc_key = f"doc_{i}"
                if doc_key in extracted_data:
                    verified = self._verify_spans(extracted_data[doc_key], result.text)
                    verified_spans[result.text] = verified
                else:
                    verified_spans[result.text] = []

            # Add empty entries for remaining documents (beyond batch_size)
            for i in range(self.batch_size, len(search_results)):
                verified_spans[search_results[i].text] = []

            return verified_spans

        except Exception as e:
            print(
                f"Batch extraction failed: {e}, falling back to individual extraction"
            )
            return self._extract_spans_individual(question, search_results)

    async def _extract_spans_batch_async(
        self, question: str, search_results: list[SearchResult]
    ) -> dict[str, list[str]]:
        """
        Async batch extraction.
        """
        print("Extracting spans (async batch mode)...")

        top_results = search_results[: self.batch_size]

        documents_text = {}
        for i, result in enumerate(top_results):
            documents_text[f"doc_{i}"] = result.text

        try:
            extracted_data = await self.llm_client.extract_spans_async(
                question, documents_text
            )

            verified_spans = {}

            for i, result in enumerate(top_results):
                doc_key = f"doc_{i}"
                if doc_key in extracted_data:
                    verified = self._verify_spans(extracted_data[doc_key], result.text)
                    verified_spans[result.text] = verified
                else:
                    verified_spans[result.text] = []

            for i in range(self.batch_size, len(search_results)):
                verified_spans[search_results[i].text] = []

            return verified_spans

        except Exception as e:
            print(
                f"Async batch extraction failed: {e}, falling back to individual extraction"
            )
            return await self._extract_spans_individual_async(question, search_results)

    def _extract_spans_individual(
        self, question: str, search_results: list[SearchResult]
    ) -> dict[str, list[str]]:
        """
        Extract spans from each document individually.
        """
        print("Extracting spans (individual mode)...")
        relevant_spans = {}

        for result in search_results:
            try:
                # Use LLMClient with individual document
                doc_dict = {"doc_0": result.text}
                extracted_data = self.llm_client.extract_spans(question, doc_dict)
                spans = extracted_data.get("doc_0", [])

                # Verify all spans exist verbatim in the source text
                verified_spans = self._verify_spans(spans, result.text)
                relevant_spans[result.text] = verified_spans

            except Exception as e:
                print(f"Individual extraction failed for document: {e}")
                relevant_spans[result.text] = []

        return relevant_spans

    async def _extract_spans_individual_async(
        self, question: str, search_results: list[SearchResult]
    ) -> dict[str, list[str]]:
        """
        Async individual extraction.
        """
        print("Extracting spans (async individual mode)...")
        relevant_spans = {}

        for result in search_results:
            try:
                doc_dict = {"doc_0": result.text}
                extracted_data = await self.llm_client.extract_spans_async(
                    question, doc_dict
                )
                spans = extracted_data.get("doc_0", [])

                verified_spans = self._verify_spans(spans, result.text)
                relevant_spans[result.text] = verified_spans

            except Exception as e:
                print(f"Async individual extraction failed for document: {e}")
                relevant_spans[result.text] = []

        return relevant_spans

    def _verify_spans(self, spans: list[str], document_text: str) -> list[str]:
        """
        Verify that all extracted spans exist verbatim in the source document.
        This is critical for preventing hallucination.

        :param spans: List of extracted spans
        :param document_text: Original document text
        :return: List of verified spans that exist in the document
        """
        verified = []
        for span in spans:
            if span and span.strip() and span in document_text:
                verified.append(span)
            else:
                print(
                    f"Warning: Span not found verbatim in document: '{span[:100]}...'"
                )
        return verified
