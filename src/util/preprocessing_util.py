import pandas as pd
import string
import unicodedata
from collections import defaultdict
import re
import ast


def safe_literal_eval_column(df, column):
    """Parse stringified Python objects in a column."""
    df = df.copy()
    df[column] = df[column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df


def transform_dataset_pubmedQA(df):
    """
    Transforms a raw PubMedQA-style DataFrame into (question, sentence, label) rows.

    - Parses 'documents_sentences' from string.
    - Matches sentence keys to relevance labels.

    Returns:
        DataFrame with columns: question, sentence, label
    """
    data_rows = []

    for _, row in df.iterrows():
        question = row["question"]
        relevant_keys = set(row["all_relevant_sentence_keys"])

        # Parse string to list of sentence lists
        doc_sentences = row["documents_sentences"]
        if isinstance(doc_sentences, str):
            doc_sentences = ast.literal_eval(doc_sentences)

        for sent_list in doc_sentences:
            for sentence in sent_list:
                if len(sentence) == 2:
                    sentence_key, sentence_text = sentence
                    label = 1 if sentence_key in relevant_keys else 0
                    data_rows.append((question, sentence_text, label))

    return pd.DataFrame(data_rows, columns=["question", "sentence", "label"])

def clean_text_df(df: pd.DataFrame, columns=("question", "sentence")) -> pd.DataFrame:
    """Apply cleaning to specific columns in a DataFrame."""
    df = df.copy()
    for col in columns:
        df[col] = clean_text_column(df[col])
    return df


def clean_text_column(series: pd.Series) -> pd.Series:
    """Apply standard text preprocessing steps to a Pandas Series (column)."""
    def clean_whitespace(text):
        return re.sub(r'\s+', ' ', text).strip()

    def normalize_unicode(text):
        return unicodedata.normalize("NFKC", text)

    def remove_punctuation(text):
        return text.translate(str.maketrans("", "", string.punctuation))

    return (
        series.astype(str)
              .str.lower()
              .apply(clean_whitespace)
              .apply(normalize_unicode)
              .apply(remove_punctuation)
    )

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