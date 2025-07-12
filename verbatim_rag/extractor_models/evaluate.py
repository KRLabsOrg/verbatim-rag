#!/usr/bin/env python
"""
evaluate_test.py

Load a pretrained sentence‐relevance extractor and evaluate it on a held‐out test split.
"""
import os
# 1) force HuggingFace into offline/local‐only mode
os.environ["HUGGINGFACE_HUB_OFFLINE"] = "1"

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from verbatim_rag.extractor_models.dataset import QADataset, QASample
from verbatim_rag.extractor_models.model import QAModel
from verbatim_rag.extractor_models.trainer import qa_collate_fn, Trainer


def load_jsonl(path: Path) -> list[QASample]:
    """Read a JSONL of QASample records and return a list of QASample objects."""
    samples = []
    for line in path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        samples.append(QASample.from_json(json.loads(line)))
    return samples


def evaluate_test(test_path: Path, model_dir: Path, batch_size: int = 16):
    """
    Load the test split and a pretrained model, run evaluation, and save metrics.

    :param test_path:   Path to the test.jsonl file
    :param model_dir:   Directory containing the pretrained model + tokenizer
    :param batch_size:  Batch size for DataLoader during evaluation
    """
    # ----- Device -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Load Test Samples -----
    print(f"→ Loading TEST data from {test_path}")
    test_samples = load_jsonl(test_path)

    # ----- Tokenizer & Dataset -----
    abs_dir = str(model_dir.resolve())
    print(f"→ Loading tokenizer from {abs_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        abs_dir,
        local_files_only=True
    )
    dataset = QADataset(test_samples, tokenizer, max_length=4096)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=qa_collate_fn,
    )

    # ----- Load Model -----
    print(f"→ Loading model from {abs_dir}")
    model = QAModel.from_pretrained(
        abs_dir,
        local_files_only=True
    )
    model.to(device)

    # ----- Evaluate -----
    print("→ Running evaluation on TEST split …")
    # **IMPORTANT** pass a real dataset so Trainer.__init__ can build its samplers
    trainer = Trainer(
        model=model,
        train_dataset=dataset,      # <— cannot be None!
        dev_dataset=None,           # unused here
        tokenizer=tokenizer,        # not used in _evaluate, but safe
        batch_size=batch_size,
        lr=0.0,                     # won't train, so lr doesn't matter
        epochs=0,                   # won't train 
        device=device,
        output_dir=None
    )
    metrics = trainer._evaluate(loader)

    # ----- Print & Save -----
    print("\n***** TEST RESULTS *****")
    for key in ["loss", "precision", "recall", "f1", "accuracy"]:
        print(f"{key.capitalize():<10}: {metrics[key]:.4f}")

    out = {f"test_{k}": v for k, v in metrics.items()}
    out_path = model_dir / "test_metrics.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"→ Saved metrics to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate sentence‐relevance extractor on a test split"
    )
    parser.add_argument(
        "--test_path", type=Path, required=True,
        help="Path to your test.jsonl"
    )
    parser.add_argument(
        "--model_dir", type=Path, required=True,
        help="Directory containing the pretrained model + tokenizer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for evaluation"
    )
    args = parser.parse_args()

    evaluate_test(
        test_path=args.test_path,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
