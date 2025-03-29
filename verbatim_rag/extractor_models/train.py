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

from verbatim_rag.extractor_models.dataset import QAData, QADataset
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
    # Set random seed
    set_seed(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )  # Added trust_remote_code=True as in snippet
    logger.info(f"Loaded tokenizer: {args.model_name}")

    # Load data
    logger.info(f"Loading data from {args.data_path}")
    data_path = Path(args.data_path)
    # Use QAData loading method
    qa_data = QAData.from_json(json.loads(data_path.read_text()))
    logger.info(f"Loaded {len(qa_data.samples)} samples")

    # Split data into train, dev, test
    train_samples = [sample for sample in qa_data.samples if sample.split == "train"]
    dev_samples = [sample for sample in qa_data.samples if sample.split == "dev"]
    test_samples = [
        sample for sample in qa_data.samples if sample.split == "test"
    ]  # Keep test split for potential later use

    logger.info(
        f"Train: {len(train_samples)}, Dev: {len(dev_samples)}, Test: {len(test_samples)}"
    )
    if not train_samples:
        logger.error("No training samples found. Check data path and split names.")
        return
    if not dev_samples:
        logger.warning(
            "No development samples found. Training will proceed without evaluation."
        )
        # Fallback or error based on Trainer's requirement
        dev_dataset = None  # Set to None, Trainer needs to handle this
    else:
        dev_dataset = QADataset(dev_samples, tokenizer, max_length=args.max_seq_length)

    # Create datasets
    train_dataset = QADataset(train_samples, tokenizer, max_length=args.max_seq_length)
    # dev_dataset created above
    # test_dataset might be needed if Trainer handles final testing
    test_dataset = (
        QADataset(test_samples, tokenizer, max_length=args.max_seq_length)
        if test_samples
        else None
    )

    # Create model
    # Use QAModel directly
    model = QAModel(model_name=args.model_name)
    # Note: Trainer likely handles moving model to device

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        epochs=args.num_epochs,
        device=device,
        output_dir=output_dir,  # Pass output_dir for best model checkpointing
    )

    # Start training
    logger.info("***** Starting training using Trainer *****")
    trainer.train()

    # Save the final model and tokenizer
    logger.info("Saving final model")
    model_save_path = output_dir / "model-final.pt"
    tokenizer_save_dir = output_dir / "tokenizer"
    tokenizer_save_dir.mkdir(exist_ok=True)

    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(tokenizer_save_dir)
    logger.info(f"Final model saved to {model_save_path}")
    logger.info(f"Tokenizer saved to {tokenizer_save_dir}")

    # Final evaluation on test set
    if test_dataset:
        logger.info("***** Final evaluation on test set *****")
        # Create a DataLoader for the test set
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=qa_collate_fn,
        )
        test_metrics = trainer._evaluate(test_dataloader)
        logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
        logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
        logger.info(f"Test F1: {test_metrics['f1']:.4f}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

        results_path = output_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(test_metrics, f, indent=4)
        logger.info(f"Saved test results to {results_path}")
    else:
        logger.info("No test dataset found, skipping final test evaluation.")

    logger.info("Training finished.")


def main():
    parser = argparse.ArgumentParser()

    # Data arguments
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the data file"
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
        "--max_seq_length", type=int, default=512, help="Maximum sequence length"
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

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
