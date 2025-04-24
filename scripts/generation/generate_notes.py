#!/usr/bin/env python
# coding: utf-8
"""
note_generation.py

Generate synthetic EHR note excerpts in batch, using OpenAI’s Chat API.
"""

import sys                              # to modify path
import re                               # regex for text cleaning
import time                             # sleep for retries
import random                           # sampling for examples
import logging                          # logging instead of print
from pathlib import Path                # filesystem paths
from datetime import datetime          # timestamp filenames
import argparse                         # CLI argument parsing

import pandas as pd                     # dataframes and CSV I/O
from tqdm import tqdm                   # progress bars for loops
import openai                           # OpenAI ChatCompletion API

# add configs directory to path to import API tokens
sys.path.append("../../configs")
from hf_config import openai_token     # secure token storage (do not commit credentials)


# -----------------------------------------------------------------------------
# ---------------------------- Configuration constants ------------------------
# -----------------------------------------------------------------------------


# (unused VLLM_URL stub retained for future support)
VLLM_URL    = "http://localhost:8000/v1/completions"
# directory containing few-shot prompt templates
PROMPTS_DIR = Path(__file__).parent / "prompts"
# base directory for saving synthetic note CSVs
OUTPUT_DIR  = Path("../../data/synthetic/note-excerpts")


# -----------------------------------------------------------------------------
# ------------------------------- Logging setup -------------------------------
# -----------------------------------------------------------------------------


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%(Y-%m-%d %H:%M:%S",
)


# -----------------------------------------------------------------------------
# ----------------------------- OpenAI LLM wrapper ----------------------------
# -----------------------------------------------------------------------------


def generate_with_openai(
    prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 0.95,
    n: int = 1,
    batch_size: int = 1,
    retries: int = 3,
    backoff: float = 2.0,
) -> list[str]:
    """
    Generate `n` completions for the given prompt, in batches of up to `batch_size`.
    Retries on API errors with exponential backoff.
    Returns exactly `n` generated strings.
    """
    results: list[str] = []
    # Loop until we have gathered n outputs
    while len(results) < n:
        to_request = min(batch_size, n - len(results))
        # Retry loop for resilience
        for attempt in range(1, retries + 1):
            try:
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a clinical-note generator."},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=to_request,
                )
                # Extract and trim the responses
                chunk = [choice.message.content.strip() for choice in resp.choices]
                results.extend(chunk)
                break  # success -> exit retry loop
            except Exception as e:
                logging.warning(f"OpenAI API error (attempt {attempt}/{retries}): {e}")
                if attempt == retries:
                    # On final failure, re-raise
                    raise
                time.sleep(backoff ** attempt)
    return results


# -----------------------------------------------------------------------------
# ------------------------------ Main workflow --------------------------------
# -----------------------------------------------------------------------------


def main(
    prompt_file: Path,
    output_dir: Path,
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    total: int,
    batch_size: int,
    seed: int,
):
    # Set the OpenAI API key (from hf_config)
    openai.api_key = openai_token

    logging.info("Loading prompt template from %s", prompt_file)
    prompt_template = load_prompt(prompt_file)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = output_dir / f"synthetic_notes_{timestamp}.csv"

    # Generate raw notes via LLM
    logging.info("Generating %d notes (batch_size=%d)", total, batch_size)
    raw_notes = generate_with_openai(
        prompt_template,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=total,
        batch_size=batch_size,
    )

    # Clean each note and sample a few for logging
    logging.info("Cleaning generated notes")
    cleaned = [clean_note(n) for n in raw_notes]
    random.seed(seed)
    sample = random.sample(cleaned, min(10, len(cleaned)))
    for i, note in enumerate(sample, 1):
        logging.info("Sample %d: %s...", i, note.replace("\n", " ")[:80])

    # Save all cleaned notes to CSV
    logging.info("Saving all %d notes to %s", len(cleaned), out_csv)
    df = pd.DataFrame({"note_excerpt": cleaned})
    df.to_csv(out_csv, index=False)
    logging.info("Done.")

    
# -----------------------------------------------------------------------------
# ----------------------------- CLI entrypoint --------------------------------
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic EHR note excerpts."
    )
    parser.add_argument(
        "--prompt-file", type=Path,
        default=PROMPTS_DIR / "generate_notes-few-shot.txt",
        help="Path to your few-shot prompt template"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=OUTPUT_DIR,
        help="Directory where CSVs will be stored"
    )
    parser.add_argument(
        "--model", default="gpt-4o",
        help="OpenAI model name"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.9,
        help="sampling temperature"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024,
        help="max new tokens per completion"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95,
        help="nucleus sampling p"
    )
    parser.add_argument(
        "--total", type=int, default=1200,
        help="total number of completions to generate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="how many notes per API call"
    )
    parser.add_argument(
        "--seed", type=int, default=1050,
        help="random seed for sampling examples"
    )
    args = parser.parse_args()

    main(
        prompt_file=args.prompt_file,
        output_dir=args.output_dir,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        total=args.total,
        batch_size=args.batch_size,
        seed=args.seed,
    )  
