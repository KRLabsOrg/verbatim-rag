#!/usr/bin/env python3
"""
generate_notes_with_labels.py

Generate synthetic discharge notes along with clinician questions and relevant sentence labels.
This script:
1. Takes example notes and questions from ArchehrData
2. Uses an LLM to generate synthetic notes, questions, and relevant sentence labels
3. Outputs JSONL with notes, questions, and sentence relevance annotations

Usage:
python generate_notes_with_labels.py \
    --num_examples 100 \
    --model gpt-4o \
    --output_file data/synthetic/labeled_notes.jsonl \
    --seed_data data/archehr/dev/archehr-qa_processed.json
"""

import argparse
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openai
from tqdm import tqdm

from scripts.archehr.preprocess import ArchehrData

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Configurable settings ────────────────────────────────────────────────────
MODEL = os.getenv("LLM_MODEL", "google/gemma-3-27b-it")  # override via env
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
TEMPERATURE = float(os.getenv("LLM_TEMP", "0.5"))  # Increased for more variety
MAX_TOKENS = 4000
DEFAULT_THREADS = 20  # default number of parallel workers

# Initialize OpenAI client
client = openai.OpenAI(
    base_url=BASE_URL,
    # api_key  = os.getenv("OPENAI_API_KEY") or "EMPTY",
    api_key="EMPTY",
    timeout=30,  # s; tune to taste
)

PROMPT_TEMPLATE = """
You are a medical expert who writes EHR discharge notes. You will be given an example including a patient narrative, clinician question, and note excerpt with relevant sentences.

Your task will be to generate {examples_count} similar examples with: 
1. Patient narrative
2. Clinician question
3. Note excerpt 
4. Relevant sentences

Here is an example:

== EXAMPLE ==
PATIENT NARRATIVE:
{patient_narrative}

CLINICIAN QUESTION: 
{question}

NOTE EXCERPT:
{note}

RELEVANT SENTENCES:
{relevant_sentences}
== END EXAMPLE ==

Your output should have {examples_count} similar but NEW examples with the following format:

PATIENT NARRATIVE:
[A brief narrative describing the patient's situation and concerns]

CLINICIAN QUESTION: 
[A question from a clinician about a patient]

NOTE EXCERPT:
[A complete medical discharge note excerpt that contains information relevant to the question]

RELEVANT SENTENCES:
[List the exact sentences from your note that are relevant to answering the question. Copy them word-for-word, one per line]

Make sure the note excerpt is realistic, detailed, and contains relevant medical terminology. The generated question should be clinically meaningful. The relevant sentences should be exact copies of sentences from your note that directly answer the question.
"""


def load_seed_examples(
    file_path: str, max_examples: Optional[int] = None
) -> List[Dict]:
    """Load seed examples from ArchehrData JSON file."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            archehr_data = ArchehrData.from_json(data)

        examples = []
        for case in archehr_data.cases:
            if (
                not case.clinician_question
                or not case.sentences
                or not case.patient_narrative
            ):
                continue

            # Get the full note text and relevant sentences
            note_text = " ".join([s.sentence_text for s in case.sentences])

            # Get relevant sentences
            relevant_sentences = []
            for sentence in case.sentences:
                if sentence.relevance in ["essential", "supplementary"]:
                    sentence_text = sentence.sentence_text.replace("\n", " ")
                    relevant_sentences.append(sentence_text)

            if not relevant_sentences:
                continue

            examples.append(
                {
                    "patient_narrative": case.patient_narrative,
                    "question": case.clinician_question,
                    "note": note_text,
                    "relevant_sentences": relevant_sentences,
                }
            )

        # Limit the number of examples if specified
        if max_examples and len(examples) > max_examples:
            examples = random.sample(examples, max_examples)

        logger.info(f"Loaded {len(examples)} seed examples from {file_path}")
        return examples

    except Exception as e:
        logger.error(f"Error loading seed examples: {e}")
        return []


def generate_with_llm(
    prompt: str,
    model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    """Generate text using OpenAI LLM."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a board‑certified physician who can generate clinical notes and answer questions about them.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating with LLM: {e}")
        # Add exponential backoff if rate limited
        if "rate_limit" in str(e).lower():
            wait_time = 10
            logger.info(f"Rate limited. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            return generate_with_llm(prompt, model, max_tokens, temperature)
        return ""


def parse_generated_output(output: str) -> List[Tuple[str, str, str, List[str]]]:
    """Parse the generated output into multiple examples."""
    examples = []

    # Split by "PATIENT NARRATIVE:" to get separate examples
    parts = output.split("PATIENT NARRATIVE:")
    # Skip the first part if it doesn't contain an example
    parts = [p for p in parts if p.strip()]

    for part in parts:
        patient_narrative = ""
        question = ""
        note = ""
        relevant_sentences = []

        # Extract the patient narrative
        narrative_end = part.find("CLINICIAN QUESTION:")
        if narrative_end != -1:
            patient_narrative = part[:narrative_end].strip()

        # Extract the question
        question_start = part.find("CLINICIAN QUESTION:") + len("CLINICIAN QUESTION:")
        question_end = part.find("NOTE EXCERPT:", question_start)
        if question_end != -1:
            question = part[question_start:question_end].strip()

        # Extract the note
        if "NOTE EXCERPT:" in part:
            note_start = part.find("NOTE EXCERPT:") + len("NOTE EXCERPT:")
            note_end = part.find("RELEVANT SENTENCES:", note_start)
            if note_end != -1:
                note = part[note_start:note_end].strip()

        # Extract the relevant sentences
        if "RELEVANT SENTENCES:" in part:
            rel_start = part.find("RELEVANT SENTENCES:") + len("RELEVANT SENTENCES:")
            rel_text = part[rel_start:].strip()
            # Split by newlines and clean up
            relevant_sentences = [
                line.strip() for line in rel_text.split("\n") if line.strip()
            ]

        if patient_narrative and question and note and relevant_sentences:
            examples.append((patient_narrative, question, note, relevant_sentences))

    return examples


def generate_examples_task(
    task_id: int,
    seed_examples: List[Dict],
    model: str,
    examples_per_prompt: int,
    temperature: float,
) -> List[Dict]:
    """Task to generate examples for threading."""
    try:
        # Select a random seed example
        seed_example = random.choice(seed_examples)

        # Format prompt
        prompt = PROMPT_TEMPLATE.format(
            examples_count=examples_per_prompt,
            patient_narrative=seed_example["patient_narrative"],
            question=seed_example["question"],
            note=seed_example["note"],
            relevant_sentences="\n".join(seed_example["relevant_sentences"]),
        )

        # Generate output
        generated_output = generate_with_llm(
            prompt=prompt, model=model, temperature=temperature
        )

        if not generated_output:
            logger.warning(f"Empty output for task {task_id}. Skipping.")
            return []

        # Parse output
        parsed_examples = parse_generated_output(generated_output)

        if not parsed_examples:
            logger.warning(f"Failed to parse output for task {task_id}. Skipping.")
            return []

        # Create examples
        result = []
        for patient_narrative, question, note, relevant_sentences in parsed_examples:
            example = {
                "patient_narrative": patient_narrative,
                "clinician_question": question,
                "note_text": note,
                "relevant_sentences": relevant_sentences,
                "seed_question": seed_example["question"],
                "model": model,
            }
            result.append(example)

        logger.info(f"Task {task_id}: Generated {len(result)} examples")
        return result

    except Exception as e:
        logger.error(f"Error in task {task_id}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic labeled notes")
    parser.add_argument(
        "--num_examples", type=int, default=100, help="Number of examples to generate"
    )
    parser.add_argument("--model", type=str, default=MODEL, help="LLM model to use")
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/synthetic/labeled_notes.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--seed_data", type=str, required=True, help="Path to ArchehrData JSON file"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--examples_per_prompt",
        type=int,
        default=1,
        help="Number of examples to generate per prompt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="Number of threads for parallel processing",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Load seed examples
    seed_examples = load_seed_examples(args.seed_data)
    if not seed_examples:
        logger.error("No seed examples found. Exiting.")
        return

    # Calculate number of tasks needed
    num_prompts = (
        args.num_examples + args.examples_per_prompt - 1
    ) // args.examples_per_prompt

    logger.info(
        f"Generating {args.num_examples} examples using {num_prompts} prompts with {args.threads} threads..."
    )

    # Create temporary file for periodic saving
    out_dir = Path(os.path.dirname(args.output_file))
    temp_file = out_dir / f"{os.path.basename(args.output_file)}.temp"

    # Generate examples in parallel
    generated_examples = []
    total_successful = 0
    total_failed = 0
    last_save_time = time.time()

    try:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = {
                executor.submit(
                    generate_examples_task,
                    i,
                    seed_examples,
                    args.model,
                    args.examples_per_prompt,
                    args.temperature,
                ): i
                for i in range(num_prompts)
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Generating examples"
            ):
                task_id = futures[future]
                try:
                    task_examples = future.result()
                    generated_examples.extend(task_examples)
                    total_successful += 1

                    # Show progress
                    logger.info(
                        f"Task {task_id} completed. Total examples: {len(generated_examples)}"
                    )

                    # Save periodically (every 5 tasks or 60 seconds)
                    current_time = time.time()
                    if total_successful % 5 == 0 or (
                        current_time - last_save_time > 60
                    ):
                        with open(temp_file, "w") as f:
                            for example in generated_examples:
                                f.write(json.dumps(example) + "\n")
                        last_save_time = current_time
                        logger.info(
                            f"Saved {len(generated_examples)} examples to temporary file"
                        )

                except Exception as e:
                    total_failed += 1
                    logger.error(f"Task {task_id} failed: {e}")

                # Show progress percentage
                progress = (total_successful + total_failed) / num_prompts * 100
                logger.info(
                    f"Progress: {progress:.1f}% ({total_successful} success, {total_failed} failed)"
                )

    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Saving collected examples...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}. Saving collected examples...")
    finally:
        # Save all collected examples
        if generated_examples:
            with open(args.output_file, "w") as f:
                for example in generated_examples:
                    f.write(json.dumps(example) + "\n")
            logger.info(
                f"Saved {len(generated_examples)} examples to {args.output_file}"
            )

            # Remove temp file if it exists
            if temp_file.exists():
                os.remove(temp_file)
                logger.info(f"Removed temporary file {temp_file}")

    logger.info(
        f"Generated {len(generated_examples)} examples. Saved to {args.output_file}"
    )


if __name__ == "__main__":
    main()
