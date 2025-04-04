#!/usr/bin/env python3
"""
Tool to upload a trained QAModel to the HuggingFace Hub.

This script uses the standard HuggingFace methods to upload a model to the Hub.
"""

import argparse
import os

from transformers import AutoTokenizer
from huggingface_hub import login

from verbatim_rag.extractor_models.model import QAModel


def upload_model_to_hub(
    model_path: str,
    repo_id: str,
    description: str = None,
    private: bool = False,
    token: str = None,
):
    """
    Upload a trained model to HuggingFace Hub using standard methods.

    Args:
        model_path: Path to the saved model directory
        repo_id: Repository ID on HuggingFace (username/model-name)
        description: Short description of the model
        private: Whether the repository should be private
        token: HuggingFace API token
    """
    print(f"Loading model from {model_path}...")
    model = QAModel.from_pretrained(model_path)

    try:
        print(f"Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        print("Loading default ModernBERT tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    # Set up credentials
    if token:
        login(token=token)
    else:
        env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if env_token:
            login(token=env_token)
        else:
            print(
                "No HuggingFace token provided. Using stored credentials if available."
            )

    # Set model card metadata
    model_card = {
        "language": "en",
        "license": "apache-2.0",
        "tags": ["question-answering", "sentence-classification", "verbatim-rag"],
        "datasets": ["custom"],
        "metrics": ["f1"],
    }

    if description:
        model_card["model-index"] = [
            {
                "name": repo_id.split("/")[-1],
                "results": [],
                "metadata": {"description": description},
            }
        ]

    # Push to Hub with standard methods
    print(f"Uploading model to {repo_id}...")
    model.push_to_hub(
        repo_id=repo_id,
        private=private,
        commit_message="Upload model files",
        use_auth_token=token or True,
    )

    print(f"Uploading tokenizer to {repo_id}...")
    tokenizer.push_to_hub(
        repo_id=repo_id,
        private=private,
        commit_message="Upload tokenizer files",
        use_auth_token=token or True,
    )

    print(
        f"Model and tokenizer uploaded successfully to https://huggingface.co/{repo_id}"
    )
    print("To use this model, you can initialize a ModelSpanExtractor with:")
    print(f"extractor = ModelSpanExtractor(model_path='{repo_id}', threshold=0.5)")

    return repo_id


def main():
    parser = argparse.ArgumentParser(
        description="Upload a trained QAModel to HuggingFace Hub"
    )

    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        required=True,
        help="Path to the saved model directory",
    )
    parser.add_argument(
        "--repo_id",
        "-r",
        type=str,
        required=True,
        help="Repository ID on HuggingFace (username/model-name)",
    )
    parser.add_argument(
        "--description", "-d", type=str, help="Short description of the model"
    )
    parser.add_argument(
        "--private", "-p", action="store_true", help="Make the repository private"
    )
    parser.add_argument(
        "--token",
        "-t",
        type=str,
        help="HuggingFace API token (or set HF_TOKEN environment variable)",
    )

    args = parser.parse_args()

    # Upload the model
    upload_model_to_hub(
        model_path=args.model_path,
        repo_id=args.repo_id,
        description=args.description,
        private=args.private,
        token=args.token,
    )


if __name__ == "__main__":
    main()
