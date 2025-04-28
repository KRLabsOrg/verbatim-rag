from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from typing import List
from pathlib import Path
from verbatim_rag.util.dataset_util import mask_on_sentence_level
from configs.config import hf_token

MODEL_DIR = Path("../../../models")


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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token)
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
            use_clinician_question=True
    ) -> List[bool]:
        """
        Returns one bool per sentence in `sentences`, in order.
        """

        # 2) one‐row DataFrame to expand
        df = pd.DataFrame([{
            "patient_question": patient_question,
            "clinician_question": clinician_question,
            "sentences": sentences,
            "labels": [0] * len(sentences),  # dummy labels
        }])

        # 3) expand into one row per sentence + context-window
        expanded = mask_on_sentence_level(
            df,
            window=self.context_size,
            sep=sep,
            use_clinician_question=use_clinician_question
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
