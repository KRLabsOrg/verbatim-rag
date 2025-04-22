import argparse
import os
import logging
import pandas as pd
import torch
from datetime import datetime
from sentence_transformers import LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderEvaluator

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[LoggingHandler()],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train a cross-encoder model for sentence relevance classification"
    )
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to training data (TSV)"
    )
    parser.add_argument(
        "--val_data", type=str, required=True, help="Path to validation data (TSV)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/crossencoder",
        help="Directory to save the model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="KRLabsOrg/chiliground-base-modernbert-v1",
        help="Base model to use for fine-tuning",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=32, help="Evaluation batch size"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=500, help="Number of warmup steps"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--fp16", action="store_true", help="Use mixed precision training"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=1000, help="Evaluate after this many steps"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load training dataset
    train_data = pd.read_csv(args.train_data, sep="\t")
    logger.info(f"Loaded {len(train_data)} training examples")

    # Get validation data
    val_data = pd.read_csv(args.val_data, sep="\t")
    logger.info(f"Loaded {len(val_data)} validation examples")

    # Prepare validation evaluator
    logger.info("Preparing validation evaluator")
    val_samples = []
    for _, row in val_data.iterrows():
        val_samples.append([row["question"], row["sentence"], row["label"]])

    # Create evaluator for validation
    evaluator = CrossEncoderEvaluator.from_input_examples(
        val_samples, name="val_evaluator", batch_size=args.eval_batch_size
    )

    # Initialize the CrossEncoder model
    model = CrossEncoder(
        args.model_name,
        num_labels=1,  # Binary classification
        max_length=args.max_length,
        device=device,
    )

    # Prepare training dataset
    train_samples = []
    for _, row in train_data.iterrows():
        train_samples.append([row["question"], row["sentence"], row["label"]])

    # Define warmup steps
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps > 0
        else int(len(train_samples) * args.num_epochs * 0.1)
    )

    # Train the model
    logger.info("Starting training...")
    model.fit(
        train_dataloader_kwargs={"batch_size": args.batch_size, "shuffle": True},
        train_samples=train_samples,
        evaluator=evaluator,
        epochs=args.num_epochs,
        evaluation_steps=args.eval_steps,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate, "weight_decay": args.weight_decay},
        use_amp=args.fp16,
        output_path=os.path.join(args.output_dir, "checkpoints"),
    )

    # Save the final model
    model_save_path = os.path.join(
        args.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    model.save(model_save_path)
    logger.info(f"Model saved to {model_save_path}")

    # Final evaluation on validation set
    logger.info("Final evaluation on validation set...")
    final_scores = evaluator(model)
    for metric, score in final_scores.items():
        logger.info(f"Validation {metric}: {score:.4f}")

    # Save final evaluation results
    with open(os.path.join(model_save_path, "val_results.txt"), "w") as f:
        for metric, score in final_scores.items():
            f.write(f"{metric}: {score:.4f}\n")

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
