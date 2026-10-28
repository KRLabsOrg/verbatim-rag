"""Convert the RAGBench dataset into the extractor training data format."""

import argparse
import json
from pathlib import Path
from typing import Literal

from datasets import load_dataset
from tqdm import tqdm
from verbatim_core.extractor_models.dataset import Document, QAData, QASample, Sentence


def load_data(hugging_dir: str) -> dict:
    """Load the RAGBench data from the Hugging Face dataset."""
    ragbench = {}
    for dataset in [
        "covidqa",
        "cuad",
        "delucionqa",
        "emanual",
        "expertqa",
        "finqa",
        "hagrid",
        "hotpotqa",
        "msmarco",
        "pubmedqa",
        "tatqa",
        "techqa",
    ]:
        ragbench[dataset] = load_dataset(hugging_dir, dataset)

    return ragbench


def create_sample(
    sample: dict,
    dataset_name: str,
    split: Literal["train", "dev", "test"],
) -> QASample:
    """Create a QASample from the RAGBench dataset.

    :param sample: A dictionary containing the RAGBench dataset sample
    :param dataset_name: The name of the dataset
    :param split: The split of the dataset
    :return: A QASample object
    """
    question = sample["question"]
    documents = []

    # Create a dictionary to map sentence_ids to their positions for quick lookup
    sentence_map = {}

    if sample["documents_sentences"] is None:
        return None

    for doc_idx, document in enumerate(sample["documents_sentences"]):
        sentences = []
        for sentence_id, sentence in document:
            sentences.append(Sentence(text=sentence, relevant=False, sentence_id=sentence_id))
            # Store the location of this sentence for quick access later
            sentence_map[sentence_id] = (doc_idx, len(sentences) - 1)
        documents.append(Document(sentences=sentences))

    for relevant_sentence_key in sample["all_relevant_sentence_keys"]:
        if relevant_sentence_key in sentence_map:
            doc_idx, sent_idx = sentence_map[relevant_sentence_key]
            documents[doc_idx].sentences[sent_idx].relevant = True

    return QASample(
        question=question,
        documents=documents,
        split=split,
        task_type="ragbench",
        dataset_name=dataset_name,
    )


def main(input_dir: str, output_dir: Path, dataset_name: str | None = None) -> None:
    """Load the RAGBench data and save it.

    :param input_dir: Path to the Hugging Face dataset
    :param output_dir: Path to save the RAGBench data
    :param dataset_name: The name of the dataset to process, if not provided, all datasets will be processed
    """
    dataset = load_data(input_dir)
    qa_data = QAData(samples=[])

    if dataset_name and dataset_name in dataset:
        dataset = {dataset_name: dataset[dataset_name]}

    for name in dataset:
        for split_name in ["train", "test", "validation"]:
            data_split = dataset[name][split_name]
            split = "dev" if split_name == "validation" else split_name
            for raw_sample in tqdm(data_split, desc=f"Processing {split} split"):
                qa_sample = create_sample(raw_sample, name, split)
                if qa_sample is not None:
                    qa_data.samples.append(qa_sample)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ragbench_data.json").write_text(json.dumps(qa_data.to_json(), indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=False,
        default=None,
        help="The name of the dataset to process, if not provided, all datasets will be processed",
    )
    args = parser.parse_args()
    main(args.input_dir, Path(args.output_dir), args.dataset_name)
