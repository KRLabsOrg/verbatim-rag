from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


@dataclass
class Sentence:
    text: str
    relevant: bool
    sentence_id: str

    def to_json(self) -> dict:
        return {
            "text": self.text,
            "relevant": self.relevant,
            "sentence_id": self.sentence_id,
        }

    @classmethod
    def from_json(cls, json_dict: dict) -> "Sentence":
        return cls(
            text=json_dict["text"],
            relevant=json_dict["relevant"],
            sentence_id=json_dict["sentence_id"],
        )


@dataclass
class Document:
    sentences: list[Sentence]

    def to_json(self) -> list[dict]:
        return [sentence.to_json() for sentence in self.sentences]

    @classmethod
    def from_json(cls, json_dict: dict) -> "Document":
        return cls(sentences=[Sentence.from_json(sentence) for sentence in json_dict])


@dataclass
class QASample:
    question: str
    documents: list[Document]
    split: Literal["train", "dev", "test"]
    dataset_name: str
    task_type: str

    def to_json(self) -> dict:
        return {
            "question": self.question,
            "documents": [document.to_json() for document in self.documents],
            "split": self.split,
            "task_type": self.task_type,
            "dataset_name": self.dataset_name,
        }

    @classmethod
    def from_json(cls, json_dict: dict) -> "QASample":
        return cls(
            question=json_dict["question"],
            documents=[
                Document.from_json(document) for document in json_dict["documents"]
            ],
            split=json_dict["split"],
            task_type=json_dict["task_type"],
            dataset_name=json_dict["dataset_name"],
        )


@dataclass
class QAData:
    samples: list[QASample]

    def to_json(self) -> dict:
        return [sample.to_json() for sample in self.samples]

    @classmethod
    def from_json(cls, json_dict: dict) -> "QAData":
        return cls(samples=[QASample.from_json(sample) for sample in json_dict])


class QADataset(Dataset):
    """Dataset for extracting relevant sentences from a question and a list of documents.
    We train a sentence-level binary classifier to predict whether a sentence is relevant to the question.
    We process the whole sample in one class to get context for each sentence, then, each sentence is classified
    independently.
    """

    def __init__(
        self,
        samples: list[QASample],
        tokenizer: AutoTokenizer,
        max_length: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples = []

        for sample in samples:
            for doc in sample.documents:
                if doc.sentences:
                    self.samples.append((sample.question, doc))

    @classmethod
    def encode_question_and_sentences_with_offsets(
        cls,
        question: str,
        sentences: list[Sentence],
        tokenizer: AutoTokenizer,
        max_length: int = 4096,
    ) -> dict:
        """
        Build a single input sequence:
        [CLS] question [SEP] S1 [SEP] S2 [SEP] ... [SEP]
        *and* collect offset mappings so you can retrieve partial text from each sentence.

        :param question: The question to encode
        :param sentences: List of Sentence objects to encode
        :param tokenizer: The tokenizer to use
        :param max_length: The maximum length of the input sequence
        """
        max_length = max_length - 2
        # -------------------------------------------------------------------------
        # 1) Encode the question with special tokens
        # -------------------------------------------------------------------------
        encoded_question = tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            return_offsets_mapping=True,
            max_length=max_length,
            truncation=True,
        )
        question_ids = encoded_question["input_ids"]
        question_attn_mask = encoded_question["attention_mask"]
        question_offsets = encoded_question["offset_mapping"]

        if len(question_ids) > 1 and question_ids[-1] == tokenizer.sep_token_id:
            question_ids.pop()
            question_attn_mask.pop()
            question_offsets.pop()

        input_ids = question_ids[:]
        attention_mask = question_attn_mask[:]
        offset_mapping = question_offsets[:]

        sentence_boundaries = []
        sentence_offset_mappings = []
        included_labels = []  # Track labels for included sentences

        # -------------------------------------------------------------------------
        # 2) Encode each sentence and check if it fits within max_length
        # -------------------------------------------------------------------------
        for sent in sentences:
            # First check if adding this sentence would exceed max_length
            # Encode the sentence to check its length
            encoded_sent = tokenizer.encode_plus(
                sent.text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                max_length=max_length,
                truncation=True,
            )
            sent_ids = encoded_sent["input_ids"]
            sent_offsets = encoded_sent["offset_mapping"]

            # +1 for [SEP] token
            if len(input_ids) + len(sent_ids) + 1 > max_length:
                # If this sentence won't fit, stop processing more sentences
                break

            # If we get here, we can add the sentence
            # Insert [SEP] for boundary
            input_ids.append(tokenizer.sep_token_id)
            attention_mask.append(1)
            offset_mapping.append((0, 0))

            sent_start_idx = len(input_ids)

            # Add the sentence tokens
            input_ids.extend(sent_ids)
            attention_mask.extend([1] * len(sent_ids))
            offset_mapping.extend(sent_offsets)

            sent_end_idx = len(input_ids) - 1  # inclusive end

            # Mark this sentence boundary and store its offsets and label
            sentence_boundaries.append((sent_start_idx, sent_end_idx))
            sentence_offset_mappings.append(sent_offsets)
            included_labels.append(sent.relevant)

        # Add final [SEP] if there's room
        if len(input_ids) < max_length:
            input_ids.append(tokenizer.sep_token_id)
            attention_mask.append(1)
            offset_mapping.append((0, 0))

        # -------------------------------------------------------------------------
        # 3) Handle truncation by only including complete sentences
        # -------------------------------------------------------------------------
        if len(input_ids) > max_length:
            # Find the last complete sentence that fits
            last_valid_idx = 0
            for i, (start, end) in enumerate(sentence_boundaries):
                if end < max_length:
                    last_valid_idx = i
                else:
                    break

            # Keep only the tokens up to the last valid sentence
            if last_valid_idx >= 0:
                last_token = sentence_boundaries[last_valid_idx][1]
                input_ids = input_ids[: last_token + 1]  # +1 to include the last [SEP]
                attention_mask = attention_mask[: last_token + 1]
                offset_mapping = offset_mapping[: last_token + 1]
                sentence_boundaries = sentence_boundaries[: last_valid_idx + 1]
                sentence_offset_mappings = sentence_offset_mappings[
                    : last_valid_idx + 1
                ]

        # Add labels for included sentences
        labels = [sent.relevant for sent in sentences[: len(sentence_boundaries)]]

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,
            "sentence_boundaries": sentence_boundaries,
            "sentence_offset_mappings": sentence_offset_mappings,
            "labels": labels,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> list[dict]:
        """Get a sample from the dataset.

        :param index: The index of the sample to get
        :return: A list of dictionaries, one per document, each containing:
            - input_ids (torch.Tensor): [seq_length]
            - attention_mask (torch.Tensor): [seq_length]
            - offset_mapping (List[Tuple[int,int]]): Character offsets for each token
            - sentence_boundaries (List[Tuple[int,int]]): Token indices for each sentence
            - sentence_offset_mappings (List[List[Tuple[int,int]]]): Character offsets per sentence
            - labels (torch.Tensor): [num_sentences]
        """
        question, doc = self.samples[index]

        encoding = self.encode_question_and_sentences_with_offsets(
            question, doc.sentences, self.tokenizer, self.max_length
        )

        return encoding
