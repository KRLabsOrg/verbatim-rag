#!/usr/bin/env python
# scripts/upload_model.py

import argparse
from pathlib import Path
from huggingface_hub import notebook_login

from configs.config import repo_name, hf_username, hf_token, repo_dir
from verbatim_rag.inference.upload_model_to_hg import upload_model_to_hg


def main():
    parser = argparse.ArgumentParser(description="Push a local model directory to the HF Hub.")
    parser.add_argument(
        "--local-dir", type=Path, required=True,
        help="Path to your local model + tokenizer files"
    )
    parser.add_argument(
        "--commit-msg", type=str, default="Upload model",
        help="Hub commit message"
    )
    args = parser.parse_args()

    # Notebook style login prompt once, then token is cached
    notebook_login()

    # call your core function
    upload_model_to_hg(
        local_model_dir=args.local_dir,
        repo_name=repo_name,
        hf_username=hf_username,
        hf_token=hf_token,
        repo_dir=repo_dir,
        commit_message=args.commit_msg,
    )


if __name__ == "__main__":
    main()

