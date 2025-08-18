"""
Extractors for identifying relevant spans in documents.

This module provides interfaces for extracting relevant spans from documents,
allowing for easy implementation of different extraction methods.
"""

from abc import ABC, abstractmethod
import torch
from transformers import AutoTokenizer

import json
import openai

from verbatim_rag.vector_stores import SearchResult
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
        max_display_spans: int = 5           # Compatible parameter (not used)
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
    """Extract spans using an LLM with JSON output and optional batch processing."""

    def __init__(
        self, 
        model: str = "gpt-4o-mini", 
        extraction_mode: str = "batch", 
        max_display_spans: int = 5
    ):
        """
        Initialize the LLM span extractor.

        :param model: The LLM model to use for extraction
        :param extraction_mode: "batch" for single API call or "individual" for per-document calls
        :param max_display_spans: Maximum spans to prioritize for display
        """
        self.model = model
        self.extraction_mode = extraction_mode
        self.max_display_spans = max_display_spans
        # System prompts for different modes
        self.individual_system_prompt = """
You are a Q&A text extraction system. Extract EXACT verbatim text spans from the document that answer the question.

# Rules
1. Extract **only** text that explicitly addresses the question
2. Never paraphrase, modify, or add to the original text
3. Preserve original wording, capitalization, and punctuation
4. Order spans by relevance - MOST RELEVANT FIRST
5. Include complete sentences or paragraphs for context

# Output Format
Return a JSON object with spans ordered by relevance:
{
  "spans": ["most relevant exact text", "next most relevant text", ...]
}

If no relevant information, return: {"spans": []}

# Your Task
Question: {QUESTION}
Document: {DOCUMENT}

Extract verbatim spans ordered by relevance:
"""

        self.batch_system_prompt = """
You are a Q&A text extraction system. Extract EXACT verbatim text spans from multiple documents that answer the question.

# Rules
1. Extract **only** text that explicitly addresses the question
2. Never paraphrase, modify, or add to the original text
3. Preserve original wording, capitalization, and punctuation
4. Order spans within each document by relevance - MOST RELEVANT FIRST
5. Include complete sentences or paragraphs for context

# Output Format
Return a JSON object mapping document IDs to span arrays ordered by relevance:
{
  "doc_0": ["most relevant span", "next most relevant span"],
  "doc_1": ["most relevant from doc 1"],
  "doc_2": []
}

If no relevant information in a document, use empty array.
"""

    def extract_spans(
        self, question: str, search_results: list[SearchResult]
    ) -> dict[str, list[str]]:
        """
        Extract relevant spans using JSON output with batch or individual processing.

        :param question: The query or question
        :param search_results: List of search results to extract from
        :return: Dictionary mapping result text to list of relevant spans
        """
        if self.extraction_mode == "batch":
            return self._extract_spans_batch(question, search_results)
        else:
            return self._extract_spans_individual(question, search_results)

    def _extract_spans_batch(
        self, question: str, search_results: list[SearchResult]
    ) -> dict[str, list[str]]:
        """
        Extract spans from multiple documents in a single API call.
        """
        if not search_results:
            return {}

        print("Extracting spans...")
        # Limit to top 5 documents to avoid prompt size issues
        top_results = search_results[:5]
        
        # Build document mapping
        documents_text = {}
        for i, result in enumerate(top_results):
            documents_text[f"doc_{i}"] = result.text

        # Create batch prompt
        prompt = f"""
{self.batch_system_prompt}

# Your Task
Question: {question}

Documents:
{json.dumps(documents_text, indent=2)}

Extract verbatim spans from each document:
"""

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                #temperature=0.0,
            )

            print("Processing response...")
            extracted_data = json.loads(response.choices[0].message.content)
            
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
            
            # Add empty entries for remaining documents (beyond top 5)
            for i in range(5, len(search_results)):
                verified_spans[search_results[i].text] = []

            return verified_spans

        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Batch extraction failed: {e}, falling back to individual extraction")
            return self._extract_spans_individual(question, search_results)

    def _extract_spans_individual(
        self, question: str, search_results: list[SearchResult]
    ) -> dict[str, list[str]]:
        """
        Extract spans from each document individually (original approach with JSON).
        """
        relevant_spans = {}

        for result in search_results:
            prompt = self.individual_system_prompt.replace("{QUESTION}", question).replace(
                "{DOCUMENT}", result.text
            )

            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    #temperature=0.0,
                )

                extracted_data = json.loads(response.choices[0].message.content)
                spans = extracted_data.get("spans", [])
                
                # Verify all spans exist verbatim in the source text
                verified_spans = self._verify_spans(spans, result.text)
                relevant_spans[result.text] = verified_spans

            except (json.JSONDecodeError, KeyError, Exception) as e:
                print(f"Individual extraction failed for document: {e}")
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
                print(f"Warning: Span not found verbatim in document: '{span[:100]}...'")
        return verified

