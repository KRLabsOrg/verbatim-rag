import argparse
import pandas as pd
from pathlib import Path
import torch
import wandb

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
)

from verbatim_rag.training.trainer import WeightedLossTrainer
from verbatim_rag.util.dataset_util import prepare_dataset, mask_on_sentence_level, balance_ratio
from verbatim_rag.util.training_util import compute_class_weights
from verbatim_rag.evaluation.metrics import compute_metrics

from configs.config import hf_token


def main(args):

    # Initialize W&B
    wandb.login()

    # Load and preprocess
    data = pd.read_csv(args.train_file)

    data = mask_on_sentence_level(data, args.window_size)

    print(data.columns)
    """
    ratio = 1.0
    data_masked = balance_ratio(data_masked, neg_to_pos=ratio)
    data = (
        data[["case_id", "patient_question", "sentence_text", "relevance"]]
        .groupby("case_id")
        .apply(lambda g: pd.Series({
            "question": g["patient_question"].iloc[0],
            "sentences": g["sentence_text"].tolist(),
            "labels": [1 if r in ["essential", "relevant"] else 0 for r in g["relevance"]]
        }))
        .reset_index()
    )"""

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(args.model,
                                                               num_labels=2, token=hf_token)

    # Resize model embeddings to match new special tokens
    if args.window_size > 0:
        special_tokens = {"additional_special_tokens": ["[START]", "[END]"]}
        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))

    # Check if model parameters are trainable (Ensure We Are Training End-to-End)
    for param in model.parameters():
        param.requires_grad = True  # Ensure the entire model is updated

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare dataset
    train_loader = prepare_dataset(
        df=data,
        tokenizer=tokenizer,
        context_length=args.context_length,
        window_size=args.window_size,
        batch_size=args.batch_size,
    )

    # Compute class weights
    class_weights = compute_class_weights(data["label"])

    '''
    def compute_class_weights(labels: list[int]) -> torch.Tensor:
        classes = np.array([0, 1], dtype=int)
        weights = compute_class_weight("balanced", classes=classes, y=labels)
        return torch.tensor(weights, dtype=torch.float)
    '''

    # Training arguments
    training_args = TrainingArguments(
        output_dir="../../checkpoints",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_dir="./logs",
        logging_steps=100,
        report_to="wandb",
        run_name=args.run_name,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )

    # W&B init
    wandb.init(
        project=args.project_name,
        name=args.run_name,
        config=vars(args),
    )

    # Trainer
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=train_loader.dataset,  # self-validation for now
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    # Train
    trainer.train()

    # Save final model + tokenizer
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    print(f"✅ Model saved to {args.model_dir}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    MODEL_DIR = "../../models"
    TRAIN_DATA_DIR = Path("../../data/synthetic/questions")

    models = {"ModernBERT": "answerdotai/ModernBERT-base",
              "BioMedBERT-Abstract": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"}

    # required
    parser.add_argument("--train-file", type=Path, help="Name of the training CSV")
    parser.add_argument("--run-name", type=str, help="WandB run name")
    # optional
    parser.add_argument("--model", type=str, default="BioMedBERT-Abstract",
                        help="Pretrained model to fine-tune (ModernBERT, "
                             "BioMedBERT-Abstract)")
    parser.add_argument("--project-name", type=str, default="Synthetic-Med")

    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=1)

    args = parser.parse_args()

    training_file = "questions_V1.csv"
    run_name = "Script_RUN01"
    args.train_file = TRAIN_DATA_DIR /training_file
    args.run_name = run_name

    if not args.train_file:
        args.train_file = TRAIN_DATA_DIR / input("Name of the training CSV: ")
    if not args.run_name:
        args.run_name = input("Name of this training run: ")


    args.model = models[args.model]
    args.model_dir = MODEL_DIR
    main(args)
