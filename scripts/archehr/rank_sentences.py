import argparse
import os
import json
import pandas as pd
from tqdm import tqdm
from sentence_transformers.cross_encoder import CrossEncoder


def rank_sentences(
    model_path,
    input_file,
    output_file,
    batch_size=32,
    max_sentences=None,
    threshold=None,
):
    """
    Rank sentences from clinical notes based on relevance to a question.

    Args:
        model_path: Path to the trained cross-encoder model
        input_file: Path to input CSV file with questions and notes
        output_file: Path to output JSON file with ranked sentences
        batch_size: Batch size for inference
        max_sentences: Maximum number of top sentences to include in output
        threshold: Minimum score threshold for sentences to include
    """
    print(f"Loading model from {model_path}")
    model = CrossEncoder(model_path)

    print(f"Loading input data from {input_file}")
    input_data = pd.read_csv(input_file)

    results = []

    for idx, row in tqdm(
        input_data.iterrows(), total=len(input_data), desc="Processing"
    ):
        question = (
            row["patient_question"]
            if "patient_question" in row
            else row["clinician_question"]
        )

        # Get the sentences
        if "note_text" in row:
            note_text = row["note_text"]

            # Parse sentences
            if isinstance(note_text, str):
                if note_text.startswith("[") and note_text.endswith("]"):
                    # Parse as JSON list
                    sentences = json.loads(note_text)
                else:
                    # Single text - treat as one sentence
                    sentences = [note_text]
            else:
                # Already a list
                sentences = note_text
        else:
            print(f"Skipping row {idx}: no note_text found")
            continue

        # Skip if no sentences
        if not sentences:
            print(f"Skipping row {idx}: no sentences found")
            continue

        # Prepare sentence pairs for scoring
        sentence_pairs = [[question, sentence] for sentence in sentences]

        # Score sentences with cross-encoder
        scores = model.predict(sentence_pairs, batch_size=batch_size)

        # Combine sentences and scores
        sentence_scores = list(zip(sentences, scores))

        # Sort by score in descending order
        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        # Apply threshold if specified
        if threshold is not None:
            sentence_scores = [item for item in sentence_scores if item[1] >= threshold]

        # Limit to max_sentences if specified
        if max_sentences is not None:
            sentence_scores = sentence_scores[:max_sentences]

        # Create result entry
        result = {
            "question": question,
            "ranked_sentences": [
                {"sentence": s, "score": float(score)} for s, score in sentence_scores
            ],
        }

        results.append(result)

    # Save results
    print(f"Saving ranked sentences to {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Processed {len(results)} questions.")


def main():
    parser = argparse.ArgumentParser(
        description="Rank sentences by relevance using cross-encoder"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained cross-encoder model"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file with questions and notes",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file with ranked sentences",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--max_sentences",
        type=int,
        default=None,
        help="Maximum number of top sentences to include in output",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum score threshold for sentences to include",
    )

    args = parser.parse_args()

    rank_sentences(
        model_path=args.model,
        input_file=args.input,
        output_file=args.output,
        batch_size=args.batch_size,
        max_sentences=args.max_sentences,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
