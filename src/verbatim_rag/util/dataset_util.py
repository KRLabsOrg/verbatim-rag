import ast
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset
from torch.utils.data import DataLoader


# mask sentences, tokenize, batch setup
def prepare_dataset(df, tokenizer, context_length, window_size, batch_size):

    # mask the data
    dataset = Dataset.from_pandas(df)

    # set up progress bar
    progress_bar = tqdm(total=len(dataset), desc="Tokenizing", position=0)

    def tokenize_batch(batch):
        encodings = tokenizer(
            batch["question"],
            batch["context"],
            padding="max_length",
            truncation=True,
            max_length=context_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encodings["input_ids"].tolist(),
            "attention_mask": encodings["attention_mask"].tolist(),
            "labels": batch["label"]
        }

    def tokenize_with_progress(batch):
        out = tokenize_batch(batch)
        progress_bar.update(len(batch["question"]))
        return out

    # tokenize
    tokenized = dataset.map(tokenize_with_progress, batched=True, batch_size=batch_size)
    progress_bar.close()

    # format + wrap in loader
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return DataLoader(tokenized, batch_size=batch_size)


def mask_on_sentence_level(df, window=0, sep=". ", use_clinician_question=False):
    """
    Expand each note into one row per sentence, optionally including context.

    Parameters:
    - df: DataFrame with columns ["patient_question", "clinician_question", "sentences", "labels"]
    - window: int, number of neighbor sentences to include on each side (0 = only target sentence)
    - sep: str, delimiter to join context sentences into a single string
    - use_clinician_question: bool, if True use clinician_question as prompt, otherwise patient_question

    Returns:
    - DataFrame with columns:
        "question"        : the selected question text
        "context"         : the masked context string (with [START]/[END] around target)
        "target_sentence" : the original sentence at index i
        "target_index"    : integer index of the target sentence in the original list
        "label"           : binary label indicating relevance (0 or 1)
    """
    expanded = []

    df["sentences"] = df["sentences"].apply(safe_eval)
    df["labels"] = df["labels"].apply(safe_eval)

    # Iterate over each note / QA example
    for _, row in df.iterrows():
        # Choose question type based on flag
        question = row["clinician_question"] if use_clinician_question else row["patient_question"]
        sentences = row["sentences"]
        labels = row["labels"]

        # Walk through each sentence in the note
        for i, (sentence, label) in enumerate(zip(sentences, labels)):
            if window == 0:
                # No surrounding context: context is only the target sentence
                context = sentence
            else:
                # Determine slice bounds for context window
                start = max(0, i - window)
                end = min(len(sentences), i + window + 1)
                # Copy the slice of sentences
                context_slice = sentences[start:end].copy()
                # Compute local index of the target within the slice
                target_local_idx = i - start
                # Mark the target sentence with special tokens
                context_slice[target_local_idx] = (
                    f"[START] {context_slice[target_local_idx]} [END]"
                )
                # Join the slice back into a single context string
                context = sep.join(str(s) for s in context_slice if pd.notnull(s))

            # Add one row per sentence/context combination
            expanded.append({
                "question": question,  # which question to use
                "context": context,  # masked or unmasked context
                "target_sentence": sentence,  # the actual sentence text
                "target_index": i,  # position in the original note
                "label": label  # relevance label (0/1)
            })

    # Return as a new DataFrame
    return pd.DataFrame(expanded)


def balance_ratio(df, label_col='label', neg_to_pos=1.0, seed=1050):
    """
    Returns a new DataFrame with negatives sampled to achieve
    `neg_to_pos` negatives per positive.

    neg_to_pos: float, how many negatives per positive you want
                (1.0 → 1:1, 65/35≈1.857→65:35, etc.)
    """
    pos = df[df[label_col] == 1]
    neg = df[df[label_col] == 0]
    n_pos = len(pos)
    n_neg = int(n_pos * neg_to_pos)
    neg_sample = neg.sample(n=n_neg, random_state=seed)
    return pd.concat([pos, neg_sample]) \
        .sample(frac=1, random_state=seed) \
        .reset_index(drop=True)


def load_question_sentence_pairs(path: Path, question_type="patient_question"):
    """
    Load a dataframe with sentence-level labels and return
    a list of question+sentence pairs and labels.
    """
    df = pd.read_csv(path)
    df["sentences"] = df["sentences"].apply(ast.literal_eval)
    df["labels"] = df["labels"].apply(ast.literal_eval)

    texts = []
    labels = []
    for _, row in df.iterrows():
        question = row[question_type]
        for sentence, label in zip(row["sentences"], row["labels"]):
            combined = f"{question} [SEP] {sentence}"
            texts.append(combined)
            labels.append(label)
    return texts, labels


def safe_eval(x):
    return ast.literal_eval(x) if isinstance(x, str) else x
