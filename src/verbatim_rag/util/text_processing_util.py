import re
import unicodedata
import pandas as pd
from collections import defaultdict


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
            c = _clean_text_column(s)
            if c:
                cleaned.append(c)
        return cleaned

    return series.apply(_clean_list)


def clean_text_df(df: pd.DataFrame, text_columns: list[str] = None, list_columns: list[str] = None) -> pd.DataFrame:
    """
    Extend your cleaning function to also clean columns which are lists of strings
    (e.g. your context or sentence columns).
    """

    df = df.copy()

    # Option-1: clean normal text columns
    if text_columns:
        for col in text_columns:
            df[col] = df[col].astype(str).apply(_clean_text_column)

    # Option-2: clean list of text columns
    if list_columns:
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
    return _light_clean(_remove_end_tag(_strip_example_intro(raw)))


def _light_clean(text):
    text = text.strip()
    text = text.replace("\n\n", "\n")  # reduce double newlines
    text = " ".join(text.split())  # collapse multiple spaces
    return text


def _strip_example_intro(text: str) -> str:
    """
    Remove leading example headers (e.g., '--- Example X ---' or '***Example***') up to the first blank line.
    """
    parts = re.split(r"^\s*[-–—*_ ]{2,}.*?\n\s*\n", text, flags=re.MULTILINE | re.DOTALL)
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
