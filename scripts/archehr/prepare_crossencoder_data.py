import argparse
import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import csv
import ast


def read_synthetic_questions(file_path):
    """Read synthetic questions from CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully read {len(df)} synthetic questions")
        return df
    except Exception as e:
        print(f"Error reading synthetic questions: {e}")
        return None


def clean_string_list(text):
    """
    Clean and parse a string that looks like a Python list of strings.
    Example input: "['string1', 'string2', 'string3']"
    """
    try:
        # Use ast.literal_eval to safely parse the string as a Python literal
        return ast.literal_eval(text)
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing string list: {e}")
        # If parsing fails, return the input as a single-item list
        return [text]


def prepare_crossencoder_data(synthetic_data_path, output_dir, val_size=0.1):
    """Prepare data for cross-encoder training."""
    os.makedirs(output_dir, exist_ok=True)

    # Initialize counters
    total_examples = 0
    positive_examples = 0
    negative_examples = 0
    error_count = 0

    # Initialize dataset lists
    data_pairs = []

    # Process synthetic data
    print(f"Processing synthetic data from {synthetic_data_path}")
    synthetic_df = read_synthetic_questions(synthetic_data_path)

    if synthetic_df is not None:
        # Print column names to help debugging
        print(f"Available columns: {synthetic_df.columns.tolist()}")

        for idx, row in tqdm(
            synthetic_df.iterrows(),
            total=len(synthetic_df),
            desc="Processing synthetic data",
        ):
            patient_question = (
                row["patient_question"] if "patient_question" in row else None
            )
            clinician_question = (
                row["clinician_question"] if "clinician_question" in row else None
            )

            # Use either patient or clinician question, prioritizing clinician question
            question = clinician_question if clinician_question else patient_question

            if not question:
                continue

            try:
                # First, try to use the sentences column directly
                if "sentences" in synthetic_df.columns:
                    # Use the sentences column if it exists
                    sentences_data = row["sentences"]
                    # Replace double "" with single '
                    sentences_data = sentences_data.replace('"', "'")
                    if (
                        isinstance(sentences_data, str)
                        and sentences_data.startswith("['")
                        and sentences_data.endswith("']")
                    ):
                        # Parse the string representation of a Python list
                        sentences = clean_string_list(sentences_data)
                    elif isinstance(sentences_data, list):
                        # Already a list
                        sentences = sentences_data
                    else:
                        # Treat as a single sentence
                        sentences = (
                            [sentences_data] if isinstance(sentences_data, str) else []
                        )
                else:
                    print(f"Row {idx}: No 'sentences' or 'note_text' column found")
                    continue

                if not sentences:
                    print(f"Row {idx}: No sentences found after parsing")
                    continue

                # Handle labels
                if "labels" in row:
                    labels = row["labels"]
                    if (
                        isinstance(labels, str)
                        and labels.startswith("[")
                        and labels.endswith("]")
                    ):
                        # This looks like a string representation of a Python list
                        try:
                            relevance = clean_string_list(labels)
                        except:
                            print(f"Row {idx}: Failed to parse labels: {labels}")
                            continue
                    elif isinstance(labels, (int, float)):
                        # Single numeric label
                        relevance = [int(labels)]
                        # If we have a single label but multiple sentences, just use the first sentence
                        if len(sentences) > 1:
                            sentences = [sentences[0]]
                    elif isinstance(labels, list):
                        # Already a list of labels
                        relevance = labels
                    else:
                        print(f"Row {idx}: Unexpected labels type: {type(labels)}")
                        continue
                else:
                    print(f"Row {idx}: No 'labels' column found")
                    continue

                # Ensure sentences and labels have the same length
                if len(sentences) != len(relevance):
                    print(
                        f"Row {idx}: Mismatch - {len(sentences)} sentences vs {len(relevance)} labels"
                    )
                    # skip this row
                    continue

                # Create examples
                for sentence, is_relevant in zip(sentences, relevance):
                    if not isinstance(sentence, str):
                        print(
                            f"Row {idx}: Skipping non-string sentence: {type(sentence)}"
                        )
                        continue

                    # Convert label to binary (0 or 1)
                    try:
                        label = 1 if int(is_relevant) else 0
                    except (ValueError, TypeError):
                        print(f"Row {idx}: Invalid relevance value: {is_relevant}")
                        continue

                    data_pairs.append(
                        {"question": question, "sentence": sentence, "label": label}
                    )

                    total_examples += 1
                    if label == 1:
                        positive_examples += 1
                    else:
                        negative_examples += 1

            except Exception as e:
                error_count += 1
                print(f"Row {idx}: Error processing row: {e}")
                # Print a sample of the row data for the first few errors
                if error_count <= 3:
                    print(f"Question: {question[:50]}...")
                    if "sentences" in synthetic_df.columns:
                        print(f"sentences sample: {str(row['sentences'])[:100]}...")
                    if "labels" in row:
                        print(f"labels: {str(row['labels'])}")
                continue

    # Convert to DataFrame
    df = pd.DataFrame(data_pairs)

    # Print statistics
    if total_examples > 0:
        print(f"Total examples: {total_examples}")
        print(
            f"Positive examples: {positive_examples} ({positive_examples / total_examples * 100:.2f}%)"
        )
        print(
            f"Negative examples: {negative_examples} ({negative_examples / total_examples * 100:.2f}%)"
        )
        print(f"Total errors: {error_count}")
    else:
        print("No examples found. Please check the input data.")
        return

    # Split into train and validation sets
    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=42, stratify=df["label"]
    )

    # Print split sizes
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")

    # Save datasets as TSV files with headers
    train_path = os.path.join(output_dir, "train.tsv")
    val_path = os.path.join(output_dir, "val.tsv")

    train_df.to_csv(train_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    val_df.to_csv(val_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Saved datasets to {output_dir}")
    print(f"  - Train: {train_path}")
    print(f"  - Validation: {val_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for cross-encoder training"
    )
    parser.add_argument(
        "--synthetic_data",
        type=str,
        required=True,
        help="Path to synthetic questions CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/crossencoder",
        help="Output directory for prepared data",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Validation set size (fraction of total data)",
    )

    args = parser.parse_args()

    prepare_crossencoder_data(
        synthetic_data_path=args.synthetic_data,
        output_dir=args.output_dir,
        val_size=args.val_size,
    )


if __name__ == "__main__":
    main()
