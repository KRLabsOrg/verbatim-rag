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
        logger.info(f"Loading data from {args.data_path}")
        data_path = Path(args.data_path)

        if not data_path.exists():
            logger.error(f"Data file {args.data_path} does not exist")
            return

        data_text = data_path.read_text()
        logger.info(f"Data file size: {len(data_text)} bytes")

        try:
            data_json = json.loads(data_text)
            logger.info("Successfully parsed JSON data")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            return

        qa_data = QAData.from_json(data_json)
        logger.info(f"Loaded {len(qa_data.samples)} samples")

        # DEBUGGING: Validate a few samples
        if len(qa_data.samples) > 0:
            sample = qa_data.samples[0]
            logger.info(f"Sample split: {sample.split}")
            logger.info(f"Sample question: {sample.question}")
            logger.info(f"Sample dataset: {sample.dataset_name}")
            logger.info(f"Sample documents count: {len(sample.documents)}")
            if sample.documents and len(sample.documents) > 0:
                first_doc = sample.documents[0]
                logger.info(
                    f"First document sentences count: {len(first_doc.sentences)}"
                )
                if first_doc.sentences and len(first_doc.sentences) > 0:
                    first_sentence = first_doc.sentences[0]
                    logger.info(
                        f"First sentence: '{first_sentence.text[:100]}...' (relevant: {first_sentence.relevant})"
                    )

        train_samples = [
            sample for sample in qa_data.samples if sample.split == "train"
        ]
        dev_samples = [sample for sample in qa_data.samples if sample.split == "dev"]
        test_samples = [sample for sample in qa_data.samples if sample.split == "test"]

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
            dev_dataset = None
        else:
            logger.info(f"Creating dev dataset with max_length={args.max_seq_length}")
            dev_dataset = QADataset(
                dev_samples, tokenizer, max_length=args.max_seq_length
            )

            if len(dev_dataset) > 0:
                logger.info(f"Dev dataset length: {len(dev_dataset)}")
                try:
                    sample_item = dev_dataset[0]
                    logger.info(f"Dev dataset item keys: {sample_item.keys()}")
                    logger.info(
                        f"Dev dataset input_ids shape: {sample_item['input_ids'].shape}"
                    )
                except Exception as e:
                    logger.error(f"Error accessing dev dataset: {e}")

        # Create datasets
        logger.info(f"Creating train dataset with max_length={args.max_seq_length}")
        train_dataset = QADataset(
            train_samples, tokenizer, max_length=args.max_seq_length
        )

        # Validate train dataset
        if len(train_dataset) > 0:
            logger.info(f"Train dataset length: {len(train_dataset)}")
            try:
                sample_item = train_dataset[0]
                logger.info(f"Train dataset item keys: {sample_item.keys()}")
                logger.info(
                    f"Train dataset input_ids shape: {sample_item['input_ids'].shape}"
                )
                logger.info(
                    f"Train dataset has {len(sample_item['sentence_boundaries'])} sentence boundaries"
                )
            except Exception as e:
                logger.error(f"Error accessing train dataset: {e}")

        if test_samples:
            logger.info(f"Creating test dataset with max_length={args.max_seq_length}")
            test_dataset = QADataset(
                test_samples, tokenizer, max_length=args.max_seq_length
            )
        else:
            test_dataset = None

        # Create model
        logger.info(f"Creating QAModel with {args.model_name}")
        model = QAModel(model_name=args.model_name)

        # Use output directory directly instead of creating run_timestamp subdirectory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Use a very small batch size if debugging is enabled
        batch_size = 1 if args.debug else args.batch_size
        logger.info(f"Using batch size: {batch_size}")

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
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "final_model": True,
                "model_name": args.model_name,
                "max_seq_length": args.max_seq_length,
            }
            # Save with final model suffix to distinguish from best model
            final_dir = output_dir / "final_state"
            model.save_pretrained(
                final_dir, tokenizer=tokenizer, metadata=final_metadata
            )
            logger.info(f"Final model state saved to {final_dir}")

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
            test_metrics = trainer._evaluate(test_dataloader)
            logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
            logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
            logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
            logger.info(f"Test F1: {test_metrics['f1']:.4f}")
            logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

            results_path = output_dir / "test_metrics.json"
            with open(results_path, "w") as f:
                test_results = {
                    "loss": float(test_metrics["loss"]),
                    "precision": float(test_metrics["precision"]),
                    "recall": float(test_metrics["recall"]),
                    "f1": float(test_metrics["f1"]),
                    "accuracy": float(test_metrics["accuracy"]),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                json.dump(test_results, f, indent=4)
            logger.info(f"Saved test results to {results_path}")
        else:
            logger.info("No test dataset found, skipping final test evaluation.")

        logger.info("Training finished.")
    except Exception as e:
        logger.error(f"Error in training function: {e}", exc_info=True)
        raise


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

        # Print some system info
        import sys

        logger.debug(f"Python version: {sys.version}")
        logger.debug(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            logger.debug(f"CUDA available: {torch.cuda.is_available()}")
            logger.debug(f"CUDA version: {torch.version.cuda}")
            logger.debug(f"CUDA device: {torch.cuda.get_device_name(0)}")

        logger.debug("Arguments:")
        for arg, value in vars(args).items():
            logger.debug(f"  {arg}: {value}")

    train(args)


if __name__ == "__main__":
    main()
