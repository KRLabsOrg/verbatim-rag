#!/usr/bin/env python
# coding: utf-8
"""
generate_questions.py

Given a CSV of note excerpts, generate synthetic patient+clinician Q&A with supporting sentence indices.
"""

import ast
import time
import logging
import argparse
from pathlib import Path
import pandas as pd
import nltk
from tqdm import tqdm
from datetime import datetime

import openai
import requests

from verbatim_rag.util.text_processing_util import clean_text_df, split_sentences_by_delim, \
    postprocess_synthetic_question
from verbatim_rag.util.generation_util import format_few_shot_examples, is_valid_generation
from verbatim_rag.util.generation_util import load_prompt

# -----------------------------------------------------------------------------
# CONFIGURATIONS
# -----------------------------------------------------------------------------


# HTTP endpoint for a local VLLM service (optional alternative to OpenAI)
VLLM_URL = "http://localhost:8000/v1/completions"
# Directory containing your few-shot prompt templates
PROMPTS_DIR = Path("prompts")
# Base data directory for synthetic notes and output questions
DATA_DIR = Path("../data")
NOTES_DIR = DATA_DIR / "synthetic" / "note-excerpts"
OUTPUT_DIR = DATA_DIR / "synthetic" / "questions"
# Prompt template file for question generation
PROMPT_FILE = PROMPTS_DIR / "generate_questions.txt"
# Name of the CSV file with generated notes
NOTE_FILE_NAME = "few_shot_gpt4_separated_V2.csv"

# Ensure the output directory exists (create if missing)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# SETUP LOGGING
# -----------------------------------------------------------------------------


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


# -----------------------------------------------------------------------------
# LLM WRAPPERS
# -----------------------------------------------------------------------------


def generate_with_openai(
        prompt: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.9,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        n: int = 1,
        batch_size: int = 1,
        retries: int = 3,
        backoff: float = 2.0,
) -> list[str]:
    """
    Generate `n` completions using the OpenAI Chat API, with retry/backoff logic.
    Returns a list of `n` response strings.
    """
    results: list[str] = []
    # Keep requesting until we hit n outputs
    while len(results) < n:
        to_req = min(batch_size, n - len(results))
        for attempt in range(1, retries + 1):
            try:
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a clinical-QA generator."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=to_req,
                    stop=["\n\n"],  # stop at blank line
                )
                # Extract each choice's content
                results.extend([c.message.content.strip() for c in resp.choices])
                break
            except Exception as e:
                logging.warning(f"OpenAI error (attempt {attempt}/{retries}): {e}")
                if attempt == retries:
                    raise
                time.sleep(backoff ** attempt)
    return results


def generate_with_vllm(
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        retries: int = 3,
        backoff: float = 2.0,
        url: str = VLLM_URL,
) -> str:
    """
    Generate a single completion via a local VLLM HTTP endpoint, with retry/backoff.
    Returns the raw text.
    """
    payload = {
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["text"].strip()
        except Exception as e:
            logging.warning("VLLM error (%d/%d): %s", attempt, retries, e)
            time.sleep(backoff ** attempt)
    raise RuntimeError("VLLM failed too many times")


# -----------------------------------------------------------------------------
# MAIN WORKFLOW
# -----------------------------------------------------------------------------


def main(args):
    # Load synthetic notes CSV
    notes_df = pd.read_csv(args.note_file)
    notes_df["note_excerpt"] = clean_text_df(notes_df, text_columns=["note_excerpt"])
    # Split into sentences list
    notes_df["sentences"] = notes_df["note_excerpt"].apply(split_sentences_by_delim)

    # Build few-shot examples from dev dataset
    arch = pd.read_csv(args.dev_csv)
    arch["sentences"] = arch["sentences"].apply(ast.literal_eval)
    arch["labels"] = arch["labels"].apply(ast.literal_eval)
    few = arch[arch.case_id.isin(args.case_ids)].head(len(args.case_ids))
    few = clean_text_df(few, text_columns=["question", "clinician_question"], list_columns=["sentences"])
    few_shot_block = format_few_shot_examples(few)

    # Read the prompt template
    prompt_tmpl = load_prompt(PROMPT_FILE)

    # Set API key if using OpenAI
    openai.api_key = args.openai_key

    # Iterate over notes and generate Q&A
    output_rows = []
    for idx, row in tqdm(notes_df.iterrows(), total=len(notes_df), desc="Gen QAs"):
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(row.sentences))

        prompt = prompt_tmpl.format(example_qas=few_shot_block, note=numbered)

        # Choose API
        if args.backend == "openai":
            gens = generate_with_openai(
                prompt,
                args.model,
                args.temperature,
                args.max_tokens,
                args.top_p,
                n=args.n_completions,
                batch_size=args.batch_size,
            )
        else:
            # VLLM returns a single string
            gens = [generate_with_vllm(
                prompt,
                args.temperature,
                args.top_p,
                args.max_tokens,
            )]

        # Parse and collect valid outputs
        for G in gens:
            if not is_valid_generation(G):
                logging.warning("Invalid gen @ idx %d: %s", idx, G)
                continue
            output_rows.append({
                "id": idx,
                "note_excerpt": row.note_excerpt,
                "sentences": row.sentences,
                "output": postprocess_synthetic_question(G),
            })

    # Post-process into final DataFrame
    out_df = pd.DataFrame(output_rows)
    out_df = out_df.rename(columns={"output": "full_qablock"})
    # Extract fields from the QA block
    pat, cli, rel = out_df["full_qablock"].str.extract(
        r"Patient Question:\s*(.*?)\nClinician Question:\s*(.*?)\nRelevant Sentences:\s*(.*)"
    ).T.values
    out_df["patient_question"] = pat
    out_df["clinician_question"] = cli
    out_df["relevant_sentences"] = [ast.literal_eval(r) for r in rel]
    # Build binary label lists for each sentence
    out_df["labels"] = out_df.apply(
        lambda r: [1 if i + 1 in r.relevant_sentences else 0 for i in range(len(r.sentences))],
        axis=1
    )
    out_df.drop(columns=["full_qablock", "relevant_sentences"], inplace=True)

    # Final text cleaning
    text_cols = ["patient_question", "clinician_question", "note_excerpt"]
    list_cols = ["sentences"]
    out_df = clean_text_df(out_df, text_columns=text_cols, list_columns=list_cols)

    # Reorder and save
    cols = ["patient_question", "clinician_question", "sentences", "note_excerpt", "labels"]
    out_df = out_df[cols]
    out_path = Path(args.output_dir) / f"questions_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out_df.to_csv(out_path, index=False)
    logging.info("Wrote %d rows to %s", len(out_df), out_path)


# -----------------------------------------------------------------------------
# CLI ENTRYPOINT
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate QA from note excerpts.")
    parser.add_argument("--note-file", type=Path, required=True,
                        help="CSV of note excerpts")
    parser.add_argument("--prompt-file", type=Path, default=PROMPT_FILE,
                        help="Few-shot prompt template")
    parser.add_argument("--dev-csv", type=Path, required=True,
                        help="CSV of real dev examples for few-shot")
    parser.add_argument("--case-ids", nargs="+", type=int, default=[1, 3, 14, 19],
                        help="Which case_ids to sample for few-shot")
    parser.add_argument("--backend", choices=["openai", "vllm"], default="openai",
                        help="Which LLM backend to use")
    parser.add_argument("--openai-key", default=None,
                        help="OpenAI API key override (fallback to env if missing)")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI model name")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p nucleus sampling")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max tokens per completion")
    parser.add_argument("--n-completions", type=int, default=1,
                        help="Number of completions per note")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for OpenAI calls")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Directory to write questions CSV")

    args = parser.parse_args()
    # If provided, override API key
    if args.openai_key:
        openai.api_key = args.openai_key

    nltk.download('punkt', quiet=True)  # ensure sentence tokenizer is ready
    main(args)
