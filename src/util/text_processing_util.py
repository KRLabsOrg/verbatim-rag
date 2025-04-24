import pandas as pd
import string
import unicodedata
from collections import defaultdict
import re
import ast
import nltk
from tqdm import tqdm


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
                "question": question,           # which question to use
                "context": context,             # masked or unmasked context
                "target_sentence": sentence,    # the actual sentence text
                "target_index": i,              # position in the original note
                "label": label                  # relevance label (0/1)
            })

    # Return as a new DataFrame
    return pd.DataFrame(expanded)


def _clean_text_column(text: str) -> str:
    """
    Minimal cleaning before feeding into BertTokenizer:
      1. Unicode normalize (NFKC)
      2. Collapse runs of whitespace to single spaces
      3. Strip leading/trailing spaces
    """
    # 1) normalize
    txt = unicodedata.normalize("NFKC", text)
    # 2+3) collapse whitespace
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _clean_sentence_list_column(series: pd.Series) -> pd.Series:
    """
    Take a Series of lists of sentences, clean each sentence,
    and return a Series of cleaned lists.
    """
    def _clean_list(sent_list):
        if not isinstance(sent_list, list):
            return []
        cleaned = []
        for s in sent_list:
            c = clean_text(s)
            if c:
                cleaned.append(c)
        return cleaned

    return series.apply(_clean_list)

def clean_text_df(df: pd.DataFrame,
                  text_columns=["question", "sentence"],
                  list_columns=["sentences"]) -> pd.DataFrame:
    """
    Extend your cleaning function to also handle columns
    which are lists of strings (e.g. your sentences-column).
    """
    df = df.copy()

    # Option-1: clean normal text columns
    for col in text_columns:
        df[col] = df[col].astype(str).apply(_clean_text_column)
    
    # Option-2: clean list of text columns
    for col in list_columns:
        df[col] = _clean_sentence_list_column(df[col])

    return df


def postprocess_synthetic_question(raw: str) -> str:
    """
    Extract only the portion starting from 'Patient Question:'.
    Raises if the marker is missing.
    """
    parts = raw.split("Patient Question:", 1)
    if len(parts) < 2:
        raise ValueError("No 'Patient Question:' found in generation.")
    return "Patient Question:" + parts[1].strip()


def postprocess_synthetic_note(raw: str) -> str:
    """
    Apply both intro stripping and end-tag removal to raw LLM output.
    """
    # note: missing closing paren fixed
    return _remove_end_tag(_strip_example_intro(raw))


def _strip_example_intro(text: str) -> str:
    """
    Remove any leading example header blocks (e.g. '--- Example X ---')
    up to the first blank line.
    """
    parts = re.split(r"^\s*[-—]{3,}.*\n", text, flags=re.MULTILINE)
    return parts[-1].lstrip()


def _remove_end_tag(raw: str, end_token: str = "***END NOTE***") -> str:
    """
    Trim the output at the special end token, if present.
    """
    idx = raw.find(end_token)
    return raw[:idx].strip() if idx != -1 else raw.strip()


def split_sentences_by_delim(note: str, delim: str = "|") -> list[str]:
    """
    Split a note excerpt string into list of sentences using a delimiter (e.g. '|').
    Strip whitespace and drop empty fragments.
    """
    return [s.strip() for s in note.split(delim) if s.strip()]


def aggregate_sentences_by_question(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates sentence-level rows into question-level examples.

    For each question, groups:
    - all its sentences into a list
    - their corresponding labels into a list

    Returns:
        pd.DataFrame with columns: question, sentences, labels
    """
    grouped = defaultdict(lambda: {"sentences": [], "labels": []})

    for _, row in df.iterrows():
        q = row["question"]
        grouped[q]["sentences"].append(row["sentence"])
        grouped[q]["labels"].append(row["label"])

    return pd.DataFrame([
        {"question": q, "sentences": v["sentences"], "labels": v["labels"]}
        for q, v in grouped.items()
    ])


def aggregate_sentences_by_question_and_context(df) -> pd.DataFrame:
    grouped = df.groupby(["question", "context"])
    aggregated = grouped.agg({
        "target_sentence": list,
        "label": list
    }).reset_index()
    
    return aggregated.rename(columns={
        "target_sentence": "sentences",
        "label": "labels"
    })


