import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from verbatim_rag.extractor_models.dataset import QAData, QADataset, QASample
from verbatim_rag.extractor_models.model import QAModel
from verbatim_rag.extractor_models.trainer import Trainer, qa_collate_fn

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    set_seed(args.seed)

    if args.cpu_only:
        device = torch.device("cpu")
        logger.info("Forcing CPU usage as requested")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    logger.info(f"Loaded tokenizer: {args.model_name}")

    try:
        # ----------------- Data Loading -----------------
        logger.info(f"Loading data from {args.data_path}")
        data_path = Path(args.data_path)

        if not data_path.exists():
            logger.error(f"Data file {args.data_path} does not exist")
            return

        # Handle JSONL vs JSON
        if data_path.suffix.lower() == ".jsonl":
            raw_samples = []
            with data_path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw_samples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.error(f"Line {i}: invalid JSON: {e}")
                        return
            logger.info(f"Detected JSONL format: loaded {len(raw_samples)} raw samples")
            try:
                samples = [QASample.from_json(rec) for rec in raw_samples]
            except Exception as e:
                logger.error(f"Error converting raw dict to QASample: {e}")
                return
            qa_data = QAData(samples=samples)
        else:
            data_text = data_path.read_text(encoding="utf-8")
            logger.info(f"Data file size: {len(data_text)} bytes")
            try:
                data_json = json.loads(data_text)
                logger.info("Successfully parsed JSON data")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON: {e}")
                return
            qa_data = QAData.from_json(data_json)
        logger.info(f"Loaded {len(qa_data.samples)} total samples")
        # ------------------------------------------------

        # DEBUGGING: Validate a few samples
        if qa_data.samples:
            sample = qa_data.samples[0]
            logger.info(f"Sample split: {sample.split}")
            logger.info(f"Sample question: {sample.question}")
            logger.info(f"Sample dataset: {sample.dataset_name}")
            logger.info(f"Sample documents count: {len(sample.documents)}")
            if sample.documents and sample.documents[0].sentences:
                first_sentence = sample.documents[0].sentences[0]
                logger.info(
                    f"First sentence: '{first_sentence.text[:100]}...' (relevant: {first_sentence.relevant})"
                )

        # Split into train/dev/test
        train_samples = [s for s in qa_data.samples if s.split == "train"]
        dev_samples = [s for s in qa_data.samples if s.split == "dev"]
        test_samples = [s for s in qa_data.samples if s.split == "test"]
        logger.info(
            f"Split sizes → Train: {len(train_samples)}, Dev: {len(dev_samples)}, Test: {len(test_samples)}"
        )

        if not train_samples:
            logger.error("No training samples found. Check data path and split names.")
            return

        # Create datasets
        train_dataset = QADataset(train_samples, tokenizer, max_length=args.max_seq_length)
        logger.info(f"Train dataset: {len(train_dataset)} examples")

        # count how many items __getitem__ returns None
        none_count = 0
        for i in range(len(train_dataset)):
            if train_dataset[i] is None:
                none_count += 1
        
        logging.info(f"Out of {len(train_dataset)} examples, {none_count} returned None and will be skipped.")

        dev_dataset = None
        if dev_samples:
            dev_dataset = QADataset(dev_samples, tokenizer, max_length=args.max_seq_length)
            logger.info(f"Dev dataset: {len(dev_dataset)} examples")
        else:
            logger.warning("No dev samples found; skipping validation.")

        test_dataset = None
        if test_samples:
            test_dataset = QADataset(test_samples, tokenizer, max_length=args.max_seq_length)
            logger.info(f"Test dataset: {len(test_dataset)} examples")

        # ---------------- Model & Trainer Setup ----------------
        model = QAModel(model_name=args.model_name)
        logger.info(f"Initialized model: {args.model_name}")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        batch_size = 1 if args.debug else args.batch_size
        logger.info(f"Using batch size: {batch_size}")

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

        # -------------------- Run Training --------------------
        logger.info("***** Starting training *****")
        best_f1 = trainer.train()

        # Optionally save final model
        if args.save_final_model:
            logger.info("Saving final model state")
            metadata = {
                "best_f1": float(best_f1),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "final_model": True,
                "model_name": args.model_name,
                "max_seq_length": args.max_seq_length,
            }
            final_dir = output_dir / "final_state"
            model.save_pretrained(final_dir, tokenizer=tokenizer, metadata=metadata)
            logger.info(f"Final model saved to {final_dir}")

        # ---------------- Final Evaluation on Test ----------------
        if test_dataset:
            logger.info("***** Evaluating on test set *****")
            test_loader = DataLoader(
                test_dataset,
                batch_size=1 if args.debug else args.eval_batch_size,
                shuffle=False,
                collate_fn=qa_collate_fn,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )
            test_metrics = trainer._evaluate(test_loader)
            logger.info(f"Test Loss:      {test_metrics['loss']:.4f}")
            logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
            logger.info(f"Test Recall:    {test_metrics['recall']:.4f}")
            logger.info(f"Test F1:        {test_metrics['f1']:.4f}")
            logger.info(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")

            results_path = output_dir / "test_metrics.json"
            with open(results_path, "w") as f:
                json.dump({**test_metrics, "timestamp": datetime.now().isoformat()}, f, indent=2)
            logger.info(f"Saved test metrics to {results_path}")

        logger.info("Training complete.")

    except Exception as e:
        logger.error("Error in training", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser()

    # Data arguments
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the data file (JSON or JSONL)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="Model name or path",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=4096, help="Maximum sequence length"
    )

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="Evaluation batch size for test set",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (batch size=1, extra logging)",
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Force CPU usage even if CUDA is available",
    )
    parser.add_argument(
        "--save_final_model",
        action="store_true",
        help="Save the final model state after training",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
        import sys
        logger.debug(f"Python version: {sys.version}")
        logger.debug(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            logger.debug(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.debug("Arguments:")
        for k, v in vars(args).items():
            logger.debug(f"  {k}: {v}")

    train(args)


if __name__ == "__main__":
    main()
