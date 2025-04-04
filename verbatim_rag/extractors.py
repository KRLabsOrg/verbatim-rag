"""
Extractors for identifying relevant spans in documents.

This module provides interfaces for extracting relevant spans from documents,
allowing for easy implementation of different extraction methods.
"""

from abc import ABC, abstractmethod
import torch
from transformers import AutoTokenizer

import openai

from verbatim_rag.document import Document
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
        self, question: str, documents: list[Document]
    ) -> dict[str, list[str]]:
        """
        Extract relevant spans from documents based on a question.

        :param question: The query or question to extract spans for
        :param documents: List of documents to extract spans from
        :return: Dictionary mapping document content to list of relevant spans
        """
        pass


class ModelSpanExtractor(SpanExtractor):
    """Extract spans using a fine-tuned QA sentence classification model."""

    def __init__(self, model_path: str, device: str = None, threshold: float = 0.5):
        """
        Initialize the model-based span extractor.

        :param model_path: Path to the saved model (local path or HuggingFace model ID)
        :param device: Device to run the model on ('cpu', 'cuda', etc). If None, will use CUDA if available.
        :param threshold: Confidence threshold for considering a span relevant (0.0-1.0)
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
        self, question: str, documents: list[Document]
    ) -> dict[str, list[str]]:
        """
        Extract relevant spans using the trained model.

        :param question: The query or question
        :param documents: List of documents to extract from
        :return: Dictionary mapping document content to list of relevant spans
        """
        relevant_spans = {}

        for doc in documents:
            # Split the document into sentences
            raw_sentences = self._split_into_sentences(doc.content)
            if not raw_sentences:
                relevant_spans[doc.content] = []
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
                relevant_spans[doc.content] = []
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

            relevant_spans[doc.content] = spans

        return relevant_spans


class LLMSpanExtractor(SpanExtractor):
    """Extract spans using an LLM with XML tagging approach."""

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the LLM span extractor.

        :param model: The LLM model to use for extraction
        """
        self.model = model
        self.system_prompt = """
You are a Q&A text extraction system. Your task is to identify and mark EXACT verbatim text spans from the provided document that is relevant to answer the user's question.

# Rules
1. Mark **only** text that explicitly addresses the question
2. Never paraphrase, modify, or add to the original text
3. Preserve original wording, capitalization, and punctuation
4. Mark all relevant segments - even if they're non-consecutive
5. If there is no relevant information, don't add any tags.

# Output Format
Wrap each relevant text span with <relevant> tags. 
Return ONLY the marked document text - no explanations or summaries.

# Example
Question: What causes climate change?
Document: "Scientists agree that carbon emissions (CO2) from burning fossil fuels are the primary driver of climate change. Deforestation also contributes significantly."
Marked: "Scientists agree that <relevant>carbon emissions (CO2) from burning fossil fuels</relevant> are the primary driver of climate change. <relevant>Deforestation also contributes significantly</relevant>."

# Your Task
Question: {QUESTION}
Document: {DOCUMENT}

Mark the relevant text:
"""

    def extract_spans(
        self, question: str, documents: list[Document]
    ) -> dict[str, list[str]]:
        """
        Extract relevant spans using an LLM with XML tagging.

        :param question: The query or question
        :param documents: List of documents to extract from
        :return: Dictionary mapping document content to list of relevant spans
        """
        relevant_spans = {}

        for doc in documents:
            prompt = self.system_prompt.replace("{QUESTION}", question).replace(
                "{DOCUMENT}", doc.content
            )

            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )

            marked_text = response.choices[0].message.content

            # Extract spans between <relevant> tags
            spans = []
            start_tag = "<relevant>"
            end_tag = "</relevant>"

            start_pos = 0
            while True:
                start_idx = marked_text.find(start_tag, start_pos)
                if start_idx == -1:
                    break

                end_idx = marked_text.find(end_tag, start_idx)
                if end_idx == -1:
                    break

                span = marked_text[start_idx + len(start_tag) : end_idx]
                spans.append(span)
                start_pos = end_idx + len(end_tag)

            relevant_spans[doc.content] = spans

        return relevant_spans
