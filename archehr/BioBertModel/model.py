from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from typing import List
import ast


def safe_eval(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


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
        question = (
            row["clinician_question"]
            if use_clinician_question
            else row["patient_question"]
        )
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
            expanded.append(
                {
                    "question": question,  # which question to use
                    "context": context,  # masked or unmasked context
                    "target_sentence": sentence,  # the actual sentence text
                    "target_index": i,  # position in the original note
                    "label": label,  # relevance label (0/1)
                }
            )

    # Return as a new DataFrame
    return pd.DataFrame(expanded)


class ClinicalBERTModel:
    def __init__(
        self,
        model_name="flackojodye/Verbatim-BioMedBert",
        context_size: int = 1,
        device: str = "auto",
        max_length: int = 512,
        threshold: float = 0.3,
    ):
        """
        :param model_name: HF checkpoint dir or model ID
        :param context_size: number of neighbor sentences on each side
        :param device: "auto", "cpu" or "cuda"
        :param max_length: tokenizer/model max seq‐length
        :param threshold: prob threshold for positive class
        """
        # load tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device).eval()

        self.context_size = context_size
        self.max_length = max_length
        self.threshold = threshold

    def predict(
        self,
        patient_question: str,
        clinician_question: str,
        sentences: List[str],
        sep=". ",
        use_clinician_question=True,
    ) -> List[bool]:
        """
        Returns one bool per sentence in `sentences`, in order.
        """

        # 2) one‐row DataFrame to expand
        df = pd.DataFrame(
            [
                {
                    "patient_question": patient_question,
                    "clinician_question": clinician_question,
                    "sentences": sentences,
                    "labels": [0] * len(sentences),  # dummy labels
                }
            ]
        )

        # 3) expand into one row per sentence + context-window
        expanded = mask_on_sentence_level(
            df,
            window=self.context_size,
            sep=sep,
            use_clinician_question=use_clinician_question,
        )

        # 4) batch‐tokenize all (question, context) pairs
        enc = self.tokenizer(
            expanded["question"].tolist(),
            expanded["context"].tolist(),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        # 5) forward → logits → probs
        with torch.no_grad():
            logits = self.model(**enc).logits
        if logits.ndim == 2 and logits.size(1) == 2:
            probs = torch.softmax(logits, dim=1)[:, 1]
        else:
            probs = torch.sigmoid(logits.squeeze(-1))

        # 6) threshold → bool list
        preds = (probs >= self.threshold).long().cpu().tolist()
        return [bool(x) for x in preds]
