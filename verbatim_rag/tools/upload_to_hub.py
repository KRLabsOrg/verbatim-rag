#!/usr/bin/env python3
"""
Tool to upload a trained QAModel to the HuggingFace Hub.

This script prepares a QAModel and tokenizer for upload to HuggingFace Hub,
ensuring all required files are in place.
"""

import argparse
import json
import shutil
from pathlib import Path
import os
import sys
from datetime import datetime

import torch
from transformers import AutoTokenizer
from huggingface_hub import HfApi, login

from verbatim_rag.extractor_models.model import QAModel


def prepare_model_for_hub(
    model_path: str,
    output_dir: str,
    model_name: str,
    description: str = None,
    readme_template: str = None,
):
    """
    Prepare a trained QAModel for upload to HuggingFace Hub.

    Args:
        model_path: Path to the saved model directory
        output_dir: Directory to save the prepared files
        model_name: Name for the model on HuggingFace
        description: Short description of the model
        readme_template: Path to a README template file

    Returns:
        Path to the prepared directory
    """
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}...")
    model = QAModel.from_pretrained(model_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Loaded tokenizer from model directory")
    except:
        # Try to load from the base model
        try:
            config_path = model_path / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                    base_model = config.get("model_name", "answerdotai/ModernBERT-base")
                    tokenizer = AutoTokenizer.from_pretrained(base_model)
                    print(f"Loaded tokenizer from base model: {base_model}")
            else:
                # Last resort default
                tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
                print("Loaded default ModernBERT tokenizer")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            sys.exit(1)

    # Get metadata if available
    metadata = {}
    model_config_path = model_path / "model_config.json"
    if model_config_path.exists():
        with open(model_config_path, "r") as f:
            metadata = json.load(f)

    # Create the config.json
    config = model.get_config()
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save the model in safetensors format only
    try:
        from safetensors.torch import save_file

        # Convert state dict to CPU
        state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        save_file(state_dict_cpu, output_dir / "model.safetensors")
        print("Saved model in safetensors format")
    except ImportError:
        print("safetensors not available, saving in PyTorch format instead")
        torch.save(model.state_dict(), output_dir / "pytorch_model.bin")

    # Save the tokenizer - only the essential files
    tokenizer.save_pretrained(output_dir)

    essential_files = [
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "README.md",
        ".gitattributes",
    ]

    # Remove any non-essential files
    for file in output_dir.glob("*"):
        filename = file.name
        if file.is_file() and filename not in essential_files:
            if not filename.startswith(".") or (
                filename.startswith(".") and filename != ".gitattributes"
            ):
                file.unlink()
                print(f"Removed non-essential file: {filename}")
        elif file.is_dir():
            # Remove all directories (like 'bert')
            shutil.rmtree(file)
            print(f"Removed directory: {filename}")

    print("Saved essential files matching HuggingFace model repository structure")

    # Create or copy README.md
    if readme_template and Path(readme_template).exists():
        with open(readme_template, "r") as f:
            readme_content = f.read()
    else:
        best_f1 = metadata.get("best_f1", "N/A")
        if isinstance(best_f1, float):
            best_f1 = f"{best_f1:.4f}"

        timestamp = metadata.get(
            "timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        readme_content = f"""# {model_name}

{description or "A sentence classification model for extracting relevant spans from documents based on a question."}

## Model Details
- Base model: {config.get("model_name", "answerdotai/ModernBERT-base")}
- Hidden dimension: {config.get("hidden_dim", 768)}
- Number of labels: {config.get("num_labels", 2)}
- Best validation F1: {best_f1}
- Saved on: {timestamp}

## Usage

```python
from transformers import AutoTokenizer
from verbatim_rag.extractor_models.model import QAModel
from verbatim_rag.extractors import ModelSpanExtractor
from verbatim_rag.document import Document

# Initialize the extractor
extractor = ModelSpanExtractor(
    model_path="{model_name}",
    threshold=0.5
)

# Create documents
documents = [
    Document(
        content="Climate change is a significant issue. Rising sea levels threaten coastal areas.",
        metadata={{"source": "example"}}
    )
]

# Extract relevant spans
question = "What are the effects of climate change?"
results = extractor.extract_spans(question, documents)

# Print the results
for doc_content, spans in results.items():
    for span in spans:
        print(f"- {{span}}")
```

## Training Data

This model was trained on {metadata.get("dataset", "a QA dataset")} to classify sentences as relevant or not relevant to a given question.

## Limitations

- The model works at the sentence level and may miss relevant spans that cross sentence boundaries
- Performance depends on the quality and relevance of the training data
- The model is designed for English text only
"""

    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)

    # Create .gitattributes for LFS
    gitattributes = """
# Declare files that will always have LFS/text/eol=lf
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
"""

    with open(output_dir / ".gitattributes", "w") as f:
        f.write(gitattributes)

    # List the files being prepared
    print("\nPrepared files for HuggingFace Hub:")
    for file in sorted(output_dir.glob("*")):
        print(f"- {file.name}")

    print(f"\nModel prepared for HuggingFace Hub in {output_dir}")
    return output_dir


def upload_to_hub(
    prepared_dir: str, repo_id: str, private: bool = False, token: str = None
):
    """
    Upload the prepared model to HuggingFace Hub.

    Args:
        prepared_dir: Directory with the prepared model files
        repo_id: Repository ID on HuggingFace (username/model_name)
        private: Whether the repository should be private
        token: HuggingFace API token
    """
    # If token is provided, use it to login
    if token:
        login(token=token)
    else:
        # Check if token is available in environment
        env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if env_token:
            login(token=env_token)
        else:
            print(
                "No HuggingFace token provided. Using stored credentials if available."
            )

    api = HfApi()

    print(f"Creating/updating repository: {repo_id}")
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

    print(f"Uploading files to {repo_id}...")
    api.upload_folder(
        folder_path=prepared_dir, repo_id=repo_id, commit_message="Upload model files"
    )

    print(f"Model uploaded successfully to https://huggingface.co/{repo_id}")


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
        help="Repository ID on HuggingFace (username/model_name)",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="hf_upload",
        help="Directory to save the prepared files",
    )
    parser.add_argument(
        "--description", "-d", type=str, help="Short description of the model"
    )
    parser.add_argument("--readme", type=str, help="Path to a README template file")
    parser.add_argument(
        "--private", "-p", action="store_true", help="Make the repository private"
    )
    parser.add_argument(
        "--token",
        "-t",
        type=str,
        help="HuggingFace API token (or set HF_TOKEN environment variable)",
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only prepare the files, don't upload to HuggingFace",
    )

    args = parser.parse_args()

    # Extract model name from repo_id
    model_name = args.repo_id.split("/")[-1] if "/" in args.repo_id else args.repo_id

    # Prepare the model
    prepared_dir = prepare_model_for_hub(
        model_path=args.model_path,
        output_dir=args.output_dir,
        model_name=model_name,
        description=args.description,
        readme_template=args.readme,
    )

    # Upload if not prepare_only
    if not args.prepare_only:
        upload_to_hub(
            prepared_dir=prepared_dir,
            repo_id=args.repo_id,
            private=args.private,
            token=args.token,
        )
    else:
        print(f"Files prepared in {prepared_dir}. Use --prepare_only to upload.")


if __name__ == "__main__":
    main()
