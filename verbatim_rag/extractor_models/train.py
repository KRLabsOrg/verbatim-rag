# train.py
import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    Trainer as HfTrainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from tqdm import tqdm

from verbatim_rag.extractor_models.dataset import QAData, QADataset, QASample
from verbatim_rag.extractor_models.model import QAModel
from verbatim_rag.extractor_models.trainer import Trainer, qa_collate_fn  # you can still use your collate

logger = logging.getLogger(__name__)


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

 
def load_jsonl(path: Path) -> list[QASample]:
    """Utility to load a JSONL of QASample records."""
    raw = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error(f"Line {i} in {path} is invalid JSON: {e}")
                raise
    return [QASample.from_json(r) for r in raw]


def train(args):
    set_seed(args.seed)

    # ---------------- Device ----------------
    if args.cpu_only:
        device = torch.device("cpu")
        logger.info("Forcing CPU usage")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ---------------- Tokenizer ----------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    logger.info(f"Loaded tokenizer: {args.model_name}")

    # ---------------- Load Train & Dev Samples ----------------
    train_path = Path(args.data_path)
    if not train_path.exists():
        logger.error(f"Train file not found: {train_path}")
        return

    # load train.jsonl
    logger.info(f"Loading TRAIN data from {train_path}")
    train_samples = load_jsonl(train_path)

    # optionally load separate dev file
    if args.dev_data_path:
        dev_path = Path(args.dev_data_path)
        if not dev_path.exists():
            logger.error(f"Dev file not found: {dev_path}")
            return
        logger.info(f"Loading DEV data from {dev_path}")
        dev_samples = load_jsonl(dev_path)
    else:
        # fallback to in-file split tags
        logger.info("No --dev_data_path passed; falling back to split=='dev'")
        qa_data = QAData(samples=train_samples)
        dev_samples = [s for s in qa_data.samples if s.split == "dev"]
        # leave train_samples as only those marked 'train'
        train_samples = [s for s in qa_data.samples if s.split == "train"]

     # ───── DEBUG MODE: SLICE DATA & DISABLE SAVING ─────────
    if args.debug:
        DEBUG_N = 50
        logger.warning(f"DEBUG mode ON → slicing to first {DEBUG_N} samples and disabling saving")
        train_samples = train_samples[:DEBUG_N]
        dev_samples   = dev_samples[:DEBUG_N]

    # always pull out any test‐split if present
    if not args.dev_data_path:
        test_samples = [s for s in qa_data.samples if s.split == "test"]
    else:
        # if user passed a dev file, we only test‐split via tags in the train_path file
        qa_data_full = QAData(samples=train_samples + dev_samples)
        test_samples  = [s for s in qa_data_full.samples if s.split == "test"]

    logger.info(f"Sample counts → train: {len(train_samples)}, dev: {len(dev_samples)}, test: {len(test_samples)}")

    if not train_samples:
        logger.error("No training samples after split.")
        return

    # ---------------- Build Datasets ----------------
    train_dataset = QADataset(train_samples, tokenizer, max_length=args.max_seq_length)
    logger.info(f"Train dataset size: {len(train_dataset)}")

    # count how many items __getitem__ returns None
    none_count = sum(1 for i in range(len(train_dataset)) if train_dataset[i] is None)
    logger.info(f"→ {none_count}/{len(train_dataset)} train samples returned None and will be skipped")

    dev_dataset = None
    if dev_samples:
        dev_dataset = QADataset(dev_samples, tokenizer, max_length=args.max_seq_length)
        logger.info(f"Dev dataset size: {len(dev_dataset)}")
    else:
        logger.warning("No dev samples provided; skipping validation")

    test_dataset = None
    if test_samples:
        test_dataset = QADataset(test_samples, tokenizer, max_length=args.max_seq_length)
        logger.info(f"Test dataset size: {len(test_dataset)}")

    # ---------------- Model & Trainer ----------------
    model = QAModel(model_name=args.model_name)
    logger.info(f"Initialized model: {args.model_name}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    batch_size = 1 if args.debug else args.batch_size
    logger.info(f"Batch size: {batch_size}, Eval batch size: {args.eval_batch_size}")

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        lr=args.learning_rate,
        epochs=args.num_epochs,
        device=device,
        output_dir=output_dir,
    )

    # ---------------- Run Training ----------------
    logger.info("***** Starting training *****")
    best_f1 = trainer.train()

    # ---------------- Save final model ----------------
    if not args.debug and args.save_final_model:
        logger.info("Saving final model state")
        metadata = {
            "best_f1": float(best_f1),
            "timestamp": datetime.now().isoformat(),
            "final_model": True,
            "model_name": args.model_name,
            "max_seq_length": args.max_seq_length,
        }
        final_dir = output_dir / "final_state"
        model.save_pretrained(final_dir, tokenizer=tokenizer, metadata=metadata)
        logger.info(f"Final model saved to {final_dir}")

    # ---------------- Final Test Eval ----------------
    if test_dataset:
        logger.info("***** Evaluating on TEST split *****")
        test_loader = DataLoader(
            test_dataset,
            batch_size=1 if args.debug else args.eval_batch_size,
            shuffle=False,
            collate_fn=qa_collate_fn,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        tm = trainer._evaluate(test_loader)
        logger.info(f"Test Loss:      {tm['loss']:.4f}")
        logger.info(f"Test Precision: {tm['precision']:.4f}")
        logger.info(f"Test Recall:    {tm['recall']:.4f}")
        logger.info(f"Test F1:        {tm['f1']:.4f}")
        logger.info(f"Test Acc:       {tm['accuracy']:.4f}")

        results_path = output_dir / "test_metrics.json"
        with results_path.open("w") as f:
            json.dump({**tm, "timestamp": datetime.now().isoformat()}, f, indent=2)
        logger.info(f"Saved test metrics to {results_path}")

    logger.info("Training run complete.")


def main():
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to train.jsonl (or combined train/dev/test JSONL)"
    )
    parser.add_argument(
        "--dev_data_path", type=str, default=None,
        help="Optional: separate dev.jsonl for validation"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Where to write checkpoints & metrics"
    )

    # Model args
    parser.add_argument(
        "--model_name", type=str,
        default="answerdotai/ModernBERT-base",
        help="Hugging-Face name or local path"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=4096,
        help="Max tokens for QA inputs"
    )

    # Training args
    parser.add_argument("--batch_size", type=int, default=8,  help="Train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Eval batch size")
    parser.add_argument("--num_epochs",      type=int, default=3,  help="Epochs")
    parser.add_argument("--learning_rate",   type=float, default=2e-5, help="LR")
    parser.add_argument("--seed",            type=int,   default=42,   help="Random seed")
    parser.add_argument(
        "--save_final_model", action="store_true",
        help="Also persist a final_state/ folder at the end"
    )
    parser.add_argument(
        "--cpu_only", action="store_true", help="Disable GPU even if available"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode → batch size=1, extra logging"
    )

    parser.add_argument(
        "--class_weight",
        type=float,
        default=1.0,
        help="Static weight for the positive class in loss",
    )
    parser.add_argument(
        "--dynamic_weighting",
        action="store_true",
        default=False,
        help="Compute class weight based on train split",
    )

    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode ON")
        logger.debug(f"Args: {args}")

    train(args)


if __name__ == "__main__":
    main()
