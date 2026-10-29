"""Command-line training entry point for the extractor model."""

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from verbatim_core.extractor_models.dataset import QAData, QADataset
from verbatim_core.extractor_models.model import QAModel
from verbatim_core.extractor_models.trainer import Trainer, qa_collate_fn

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - global seed kept for reproducibility of legacy numpy users
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> None:  # noqa: PLR0912, PLR0915
    """Train the extractor model with the given command-line arguments."""
    set_seed(args.seed)

    if args.cpu_only:
        device = torch.device("cpu")
        logger.info("Forcing CPU usage as requested")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    logger.info("Loaded tokenizer: %s", args.model_name)

    try:
        logger.info("Loading data from %s", args.data_path)
        data_path = Path(args.data_path)

        if not data_path.exists():
            logger.error("Data file %s does not exist", args.data_path)
            return

        data_text = data_path.read_text()
        logger.info("Data file size: %s bytes", len(data_text))

        try:
            data_json = json.loads(data_text)
            logger.info("Successfully parsed JSON data")
        except json.JSONDecodeError:
            logger.exception("Error parsing JSON")
            return

        qa_data = QAData.from_json(data_json)
        logger.info("Loaded %s samples", len(qa_data.samples))

        # DEBUGGING: Validate a few samples
        if len(qa_data.samples) > 0:
            sample = qa_data.samples[0]
            logger.info("Sample split: %s", sample.split)
            logger.info("Sample question: %s", sample.question)
            logger.info("Sample dataset: %s", sample.dataset_name)
            logger.info("Sample documents count: %s", len(sample.documents))
            if sample.documents and len(sample.documents) > 0:
                first_doc = sample.documents[0]
                logger.info("First document sentences count: %s", len(first_doc.sentences))
                if first_doc.sentences and len(first_doc.sentences) > 0:
                    first_sentence = first_doc.sentences[0]
                    logger.info(
                        "First sentence: '%s...' (relevant: %s)",
                        first_sentence.text[:100],
                        first_sentence.relevant,
                    )

        train_samples = [sample for sample in qa_data.samples if sample.split == "train"]
        dev_samples = [sample for sample in qa_data.samples if sample.split == "dev"]
        test_samples = [sample for sample in qa_data.samples if sample.split == "test"]

        logger.info(
            "Train: %d, Dev: %d, Test: %d",
            len(train_samples),
            len(dev_samples),
            len(test_samples),
        )
        if not train_samples:
            logger.error("No training samples found. Check data path and split names.")
            return
        if not dev_samples:
            logger.warning(
                "No development samples found. Training will proceed without evaluation.",
            )
            dev_dataset = None
        else:
            logger.info("Creating dev dataset with max_length=%s", args.max_seq_length)
            dev_dataset = QADataset(dev_samples, tokenizer, max_length=args.max_seq_length)

            if len(dev_dataset) > 0:
                logger.info("Dev dataset length: %s", len(dev_dataset))
                try:
                    sample_item = dev_dataset[0]
                    logger.info("Dev dataset item keys: %s", sample_item.keys())
                    logger.info("Dev dataset input_ids shape: %s", sample_item["input_ids"].shape)
                except Exception:
                    logger.exception("Error accessing dev dataset")

        # Create datasets
        logger.info("Creating train dataset with max_length=%s", args.max_seq_length)
        train_dataset = QADataset(train_samples, tokenizer, max_length=args.max_seq_length)

        # Validate train dataset
        if len(train_dataset) > 0:
            logger.info("Train dataset length: %s", len(train_dataset))
            try:
                sample_item = train_dataset[0]
                logger.info("Train dataset item keys: %s", sample_item.keys())
                logger.info("Train dataset input_ids shape: %s", sample_item["input_ids"].shape)
                logger.info(
                    "Train dataset has %d sentence boundaries",
                    len(sample_item["sentence_boundaries"]),
                )
            except Exception:
                logger.exception("Error accessing train dataset")

        if test_samples:
            logger.info("Creating test dataset with max_length=%s", args.max_seq_length)
            test_dataset = QADataset(test_samples, tokenizer, max_length=args.max_seq_length)
        else:
            test_dataset = None

        # Create model
        logger.info("Creating QAModel with %s", args.model_name)
        model = QAModel(model_name=args.model_name)

        # Use output directory directly instead of creating run_timestamp subdirectory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", output_dir)

        # Use a very small batch size if debugging is enabled
        batch_size = 1 if args.debug else args.batch_size
        logger.info("Using batch size: %s", batch_size)

        # Initialize Trainer
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

        # Start training
        logger.info("***** Starting training using Trainer *****")
        best_f1 = trainer.train()

        if args.save_final_model:
            logger.info("Saving final model state")
            final_metadata = {
                "best_f1": float(best_f1),
                "timestamp": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "final_model": True,
                "model_name": args.model_name,
                "max_seq_length": args.max_seq_length,
            }
            # Save with final model suffix to distinguish from best model
            final_dir = output_dir / "final_state"
            model.save_pretrained(final_dir, tokenizer=tokenizer, metadata=final_metadata)
            logger.info("Final model state saved to %s", final_dir)

        # Final evaluation on test set
        if test_dataset:
            logger.info("***** Final evaluation on test set *****")
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=1 if args.debug else args.eval_batch_size,
                shuffle=False,
                collate_fn=qa_collate_fn,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )
            test_metrics = trainer._evaluate(test_dataloader)  # noqa: SLF001 - no public evaluation API
            logger.info("Test Loss: %.4f", test_metrics["loss"])
            logger.info("Test Precision: %.4f", test_metrics["precision"])
            logger.info("Test Recall: %.4f", test_metrics["recall"])
            logger.info("Test F1: %.4f", test_metrics["f1"])
            logger.info("Test Accuracy: %.4f", test_metrics["accuracy"])

            results_path = output_dir / "test_metrics.json"
            with results_path.open("w") as f:
                test_results = {
                    "loss": float(test_metrics["loss"]),
                    "precision": float(test_metrics["precision"]),
                    "recall": float(test_metrics["recall"]),
                    "f1": float(test_metrics["f1"]),
                    "accuracy": float(test_metrics["accuracy"]),
                    "timestamp": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                }
                json.dump(test_results, f, indent=4)
            logger.info("Saved test results to %s", results_path)
        else:
            logger.info("No test dataset found, skipping final test evaluation.")

        logger.info("Training finished.")
    except Exception:
        logger.exception("Error in training function")
        raise


def main() -> None:
    """Parse command-line arguments and run training."""
    parser = argparse.ArgumentParser()

    # Data arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="Model name or path",
    )
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="Evaluation batch size for test set",
    )
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debugging mode with batch size 1 and extra logging",
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Force using CPU even if CUDA is available",
    )
    parser.add_argument(
        "--save_final_model",
        action="store_true",
        help="Save the final model state after training",
    )

    args = parser.parse_args()

    if args.debug:
        # Set logging to DEBUG level
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

        logger.debug("Python version: %s", sys.version)
        logger.debug("PyTorch version: %s", torch.__version__)
        if torch.cuda.is_available():
            logger.debug("CUDA available: %s", torch.cuda.is_available())
            logger.debug("CUDA version: %s", torch.version.cuda)
            logger.debug("CUDA device: %s", torch.cuda.get_device_name(0))

        logger.debug("Arguments:")
        for arg, value in vars(args).items():
            logger.debug("  %s: %s", arg, value)

    train(args)


if __name__ == "__main__":
    main()
