#!/usr/bin/env python3
"""
upload_model_to_hub.py

Creates (if needed) a Hugging-Face repo and pushes your
QAModel + tokenizer + model card to the Hub.
"""

import argparse
import os
from pathlib import Path

import torch
from huggingface_hub import HfApi, Repository, login
from transformers import AutoTokenizer

from verbatim_rag.extractor_models.model import QAModel

def upload_model_to_hub(
    model_path: Path,
    repo_id: str,
    description: str = None,
    private: bool = False,
    token: str = None,
):
    model_path = Path(model_path)
    assert model_path.exists(), f"{model_path} not found"

    # 1) authenticate
    hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token is None:
        raise ValueError("No HuggingFace token found; set --token or HF_TOKEN env var")
    login(token=hf_token)

    api = HfApi()
    # 2) create repo if missing
    try:
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    except Exception as e:
        # exist_ok=True means we'll only get an error on name conflict
        print(f"⚠️  Repo creation warning: {e}")

    # 3) load your model & tokenizer
    print(f"▶️  Loading QAModel from {model_path}")
    model = QAModel.from_pretrained(model_path)
    try:
        print(f"▶️  Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except:
        print("⚠️  Could not load tokenizer from output — falling back to default ModernBERT-base")
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    # 4) clone & push
    repo_local = Path(f"/tmp/{repo_id.replace('/','_')}")
    if repo_local.exists():
        # fresh clone each time
        import shutil
        shutil.rmtree(repo_local)
    repo = Repository(
        local_dir=repo_local,
        clone_from=repo_id,
        use_auth_token=hf_token,
    )

    print("📦  Saving model + tokenizer into repo …")
    model.save_pretrained(repo_local)
    tokenizer.save_pretrained(repo_local)

    # 5) write a minimal README.md / model card
    card = f"# {repo_id}\n\n"
    card += description or "My sentence-relevance extractor.\n"
    card += "\n---\n"
    card += "### How to use\n"
    card += "```python\n"
    card += f"from transformers import AutoTokenizer\n"
    card += f"from verbatim_rag.extractor_models.model import QAModel\n\n"
    card += f"tok = AutoTokenizer.from_pretrained('{repo_id}')\n"
    card += f"model = QAModel.from_pretrained('{repo_id}')\n```"
    (repo_local / "README.md").write_text(card, encoding="utf-8")

    # 6) commit & push
    print(f"🚀  Pushing to https://huggingface.co/{repo_id} …")
    repo.push_to_hub(commit_message="Upload sentence-relevance model", blocking=True)

    print("✅ Done! Your model is live at:")
    print(f"   https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload your trained QAModel to the HuggingFace Hub"
    )
    parser.add_argument(
        "--model_path", "-m", type=str, required=True,
        help="Local directory where your model/tokenizer were saved"
    )
    parser.add_argument(
        "--repo_id", "-r", type=str, required=True,
        help="HuggingFace repo (e.g. username/my-sentence-relevance)"
    )
    parser.add_argument(
        "--description", "-d", type=str,
        help="Short markdown description (goes in the README)"
    )
    parser.add_argument(
        "--private", "-p", action="store_true",
        help="Create a private repo"
    )
    parser.add_argument(
        "--token", "-t", type=str,
        help="Your HF API token (or set HF_TOKEN env var)"
    )
    args = parser.parse_args()

    upload_model_to_hub(
        model_path=args.model_path,
        repo_id=args.repo_id,
        description=args.description,
        private=args.private,
        token=args.token,
    )
