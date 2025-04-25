# src/verbatim_rag/inference/push_model.py

import shutil
from pathlib import Path
from huggingface_hub import Repository


def upload_model_to_hg(
        local_model_dir: Path,
        repo_name: str,
        hf_username: str,
        hf_token: str,
        repo_dir: Path,
        commit_message: str = "Upload model and tokenizer"
):
    """
    Clone (or open) your HF repo, copy all files from local_model_dir
    into it, then commit & push.
    """
    full_repo_id = f"{hf_username}/{repo_name}"
    repo = Repository(local_dir=repo_dir, clone_from=full_repo_id, token=hf_token)
    for file in local_model_dir.iterdir():
        shutil.copy(file, repo_dir / file.name)
    repo.push_to_hub(commit_message=commit_message)
