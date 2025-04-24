import pandas as pd
import string
import unicodedata
from collections import defaultdict
import re
import ast
import nltk
from tqdm.notebook import tqdm


def safe_literal_eval_column(df, column):
    """Parse stringified Python objects in a column."""
    df = df.copy()
    df[column] = df[column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df


def transform_dataset_emrqa(df):
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        context = row["context"]
        question = row["question"]
        answer_texts = row["answers"]["text"]

        if not answer_texts:
            continue

        answer_text = answer_texts[0]
        sentences = nltk.sent_tokenize(context)

        for i, sent in enumerate(sentences):
            label = int(answer_text in sent)
            records.append({
                "question": question,
                "context": context,
                "target_sentence": sent,
                "target_index": i,
                "label": label
            })

    return pd.DataFrame(records)


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

def clean_text(text: str) -> str:
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


def clean_sentence_list_column(series: pd.Series) -> pd.Series:
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

    # 1) clean normal text columns
    for col in text_columns:
        df[col] = df[col].astype(str).apply(clean_text)

    # 2) clean list-of-strings columns
    for col in list_columns:
        df[col] = clean_sentence_list_column(df[col])

    return df

def mask_on_sentence_level(df, window=0, sep=". ", use_clinician_question = False):
    expanded = []

    for _, row in df.iterrows():
        if use_clinician_question:
            question = row["clinician_question"]
        else:
            question = row["patient_question"]
        sentences = row["sentences"]
        labels = row["labels"]

        for i, (sentence, label) in enumerate(zip(sentences, labels)):
            if window == 0:
                # No context, just the target sentence as context
                context = sentence
            else:
                # Get surrounding context
                start = max(0, i - window)
                end = min(len(sentences), i + window + 1)
                context_slice = sentences[start:end].copy()
                target_local_idx = i - start
                context_slice[target_local_idx] = f"[START] {context_slice[target_local_idx]} [END]"
                context = sep.join(str(s) for s in context_slice if pd.notnull(s))

            expanded.append({
                "question": question,
                "context": context,
                "target_sentence": sentence,
                "target_index": i,
                "label": label
            })

    return pd.DataFrame(expanded)
