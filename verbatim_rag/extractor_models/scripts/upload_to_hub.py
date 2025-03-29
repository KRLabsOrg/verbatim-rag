#!/usr/bin/env python
"""
Script to upload saved models to Hugging Face Hub.
This script loads a saved model and tokenizer from a local directory
and uploads them to the Hugging Face Hub.
"""

import argparse
import logging
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer

from verbatim_rag.extractor_models.model import QAModel

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def prepare_model_for_hub(
    model_path: Path,
    tokenizer_path: Path,
    output_dir: Path,
) -> None:
    """
    Prepare the model and tokenizer for uploading to the Hugging Face Hub.

    Args:
        model_path: Path to the saved model weights
        tokenizer_path: Path to the saved tokenizer
        output_dir: Directory to save the model and tokenizer for hub upload
    """
    logger.info(f"Loading model from {model_path}")

    # Load the model based on whether it's a directory or a file
    if model_path.is_dir():
        # If it's already a directory containing a saved model in HF format
        model = QAModel.from_pretrained(model_path)
    else:
        # If it's a .pt file containing state_dict
        model = QAModel()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

    # Create output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Save model in the format expected by HF
    model.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

    # Copy tokenizer files
    logger.info(f"Copying tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Tokenizer saved to {output_dir}")

    # Create a simple model card
    model_card = f"""
# Verbatim RAG QA Model

This model is designed to extract relevant sentences from documents for question answering.

## Model Details

- **Developed by:** Verbatim RAG Team
- **Model type:** Question-Answering Sentence Classifier
- **Base model:** {model.model_name}

## Intended Use
This model predicts whether sentences in a document are relevant to a given question.

## Training Data
The model was trained on a custom dataset of questions and relevant/irrelevant sentence pairs.

## Performance
For detailed performance metrics, please refer to the accompanying documentation.
"""

    with open(output_dir / "README.md", "w") as f:
        f.write(model_card)

    logger.info("Model preparation completed successfully")


def upload_to_hub(
    prepared_model_dir: Path,
    repo_name: str,
    organization: str = None,
    private: bool = False,
) -> None:
    """
    Upload the prepared model to the Hugging Face Hub.

    Args:
        prepared_model_dir: Directory with the prepared model and tokenizer
        repo_name: Name of the repository on the Hub
        organization: Optional organization name
        private: Whether to create a private repository
    """
    # Initialize the Hugging Face API
    api = HfApi()

    # Create the full repository name
    if organization:
        full_repo_name = f"{organization}/{repo_name}"
    else:
        full_repo_name = repo_name

    logger.info(f"Creating repository: {full_repo_name}")

    # Create or get the repository
    create_repo(
        repo_id=full_repo_name,
        exist_ok=True,
        private=private,
    )

    logger.info(f"Uploading model to {full_repo_name}")

    # Upload the model and tokenizer
    api.upload_folder(
        folder_path=prepared_model_dir,
        repo_id=full_repo_name,
        commit_message="Upload model and tokenizer",
    )

    logger.info(f"Model successfully uploaded to {full_repo_name}")
    logger.info(f"View your model at: https://huggingface.co/{full_repo_name}")


def main():
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model weights (.pt file)",
    )

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the saved tokenizer directory",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the prepared model",
    )

    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="Name of the repository on the Hub",
    )

    parser.add_argument(
        "--organization",
        type=str,
        default=None,
        help="Organization name (if uploading to an organization)",
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )

    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only prepare the model without uploading",
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)
    tokenizer_path = Path(args.tokenizer_path)
    output_dir = Path(args.output_dir)

    # Prepare the model
    prepare_model_for_hub(model_path, tokenizer_path, output_dir)

    # Upload to Hub if not prepare_only
    if not args.prepare_only:
        upload_to_hub(
            prepared_model_dir=output_dir,
            repo_name=args.repo_name,
            organization=args.organization,
            private=args.private,
        )


if __name__ == "__main__":
    main()
