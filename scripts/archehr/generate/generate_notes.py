import os
import random
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

import openai

# ── Configurable settings ────────────────────────────────────────────────────
MODEL = os.getenv(
    "LLM_MODEL", "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
)  # override via env
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
N_REQUESTS = 10000  # how many notes total
THREADS = 30  # parallel workers
TEMPERATURE = float(os.getenv("LLM_TEMP", "0.7"))  # Increased for more variety
MAX_TOKENS = 2000

# Path to the archehr examples file
ARCHEHR_EXAMPLES_FILE = os.getenv(
    "ARCHEHR_EXAMPLES_FILE", "data/archehr/dev/archehr-qa_processed.json"
)


# Load example notes from the archehr file
def load_example_notes():
    """Load and parse example notes from the archehr-qa_processed.json file."""
    try:
        with open(ARCHEHR_EXAMPLES_FILE, "r") as f:
            data = json.load(f)
            examples = []

            # The file structure follows the ArchehrData class from preprocess.py
            for case in data.get("cases", []):
                note_excerpt = case.get("note_excerpt", "").strip()
                examples.append(note_excerpt)

            print(f"Loaded {len(examples)} example notes from dataset")
            return examples
    except Exception as e:
        print(f"Warning: Could not load example notes: {e}")
        return []


# Load the examples at module level
EXAMPLE_NOTES = load_example_notes()

# Base prompt template
PROMPT_TEMPLATE = """
Generate a synthetic discharge note excerpt.

Choose **one** of the following headers *at random* and begin your paragraph immediately after it  
 • Brief Hospital Course  
 • Major Procedures  
 • Discharge Summary  


(Optionally append one clinical hashtag, e.g. `# COPD`, `# Sepsis`.)

Write **8 – 25** sentences or fragments in a rushed, semi‑structured style that clinicians might type into an EHR.

**Include in your paragraph (choose a few of the following)**

* Admission reason, key findings, major interventions, and discharge plan  
* ≥ 1 lab / image / procedure reference (e.g. "CT head small SAH," "elevated LFTs," "PCI done")  
* Common abbreviations (HTN, DM2, CAD, WNL, BP, HR, SpO₂, CXR, CT, ICU, OR, PO, IV, BiPAP, CATH)  
* Randomised demographics only when useful: "A ##‑year‑old M/F" (ages 20‑90)  
* Varied clinical scenarios: COPD, pneumonia, HF, MI, stroke, GI bleed, sepsis, post‑op ortho/neurosurg, DKA, ARF, obstetrics, transplant, etc.  
* "Noise" in ≈ 15 % of sentences: missing commas, double spaces, dangling fragments  
* Mixed sentence lengths; include very short (2 – 3 words) and long run‑ons (≈ 25 words)

**Output rules**

* One paragraph only—no lists, explanations, or extra text  
* End with the exact token: `***END NOTE***`

{example_section}

Your output:
"""


# ── OpenAI client ────────────────────────────────────────────────────────────
client = openai.OpenAI(
    base_url=BASE_URL,
    # api_key  = os.getenv("OPENAI_API_KEY") or "EMPTY",
    api_key="EMPTY",
    timeout=30,  # s; tune to taste
)


def generate_note(seed: int | None = None) -> str:
    """Call the chat model once and return the synthetic note string."""
    example_note = random.choice(EXAMPLE_NOTES)

    # Format the example section
    example_section = f"""
== EXAMPLE ==

{example_note}

== END EXAMPLE ==
"""
    # Generate the prompt with the selected example
    prompt = PROMPT_TEMPLATE.format(example_section=example_section)

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": "You are a board‑certified physician composing a *single* narrative paragraph for an inpatient discharge (clinical) note.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


# ── Run in parallel ──────────────────────────────────────────────────────────
def main() -> None:
    notes: list[str] = []

    print(f"Loaded {len(EXAMPLE_NOTES)} example notes from dataset")

    # Create output directory if it doesn't exist
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # Define output file
    timestamp = import_time().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"synthetic_discharge_notes_{timestamp}.jsonl"
    temp_file = out_dir / f"synthetic_discharge_notes_{timestamp}_temp.jsonl"

    print(f"Will save notes to {out_file}")

    # Track progress
    total_successful = 0
    total_failed = 0
    last_save_time = time.time()

    try:
        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            futures = {pool.submit(generate_note, i): i for i in range(N_REQUESTS)}

            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    note = fut.result()
                    notes.append(note)
                    total_successful += 1

                    # Show detailed progress
                    print(
                        f"[{idx:03}/{N_REQUESTS}] ✅ Generated note ({len(note)} chars)"
                    )

                    # Save periodically (every 5 notes or 60 seconds)
                    current_time = time.time()
                    if total_successful % 5 == 0 or (
                        current_time - last_save_time > 60
                    ):
                        save_notes_to_file(notes, temp_file)
                        last_save_time = current_time
                        print(f"💾 Saved {len(notes)} notes to temporary file")

                except Exception as exc:
                    total_failed += 1
                    print(f"[{idx:03}/{N_REQUESTS}] ❌ Error: {exc}")

                # Show progress percentage
                progress = (total_successful + total_failed) / N_REQUESTS * 100
                print(
                    f"Progress: {progress:.1f}% ({total_successful} success, {total_failed} failed)"
                )

    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user. Saving collected notes...")
    except Exception as e:
        print(f"\n⚠️ Unexpected error: {e}. Saving collected notes...")
    finally:
        # Save all collected notes even if interrupted
        if notes:
            save_notes_to_file(notes, out_file)
            print(f"\nSaved {len(notes)} notes → {out_file.resolve()}")

            # Remove temp file if it exists
            if temp_file.exists():
                temp_file.unlink()


def save_notes_to_file(notes: list[str], file_path: Path):
    """Save notes to a JSONL file."""
    with open(file_path, "w") as f:
        for note in notes:
            f.write(json.dumps({"text": note}) + "\n")


# Add time module for timestamps
def import_time():
    """Import time module dynamically."""
    from datetime import datetime

    return datetime.now()


if __name__ == "__main__":
    main()
