import re
import ast
import nltk
import unicodedata
import pandas as pd
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