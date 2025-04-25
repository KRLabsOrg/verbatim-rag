import os
import torch
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(model, dataloader, device, threshold=0.3, return_preds=False):
    """
    Run model on a dataloader and compute classification report.

    Args:
        model: HuggingFace model.
        dataloader: torch DataLoader containing test data.
        device: torch.device("cuda") or "cpu".
        threshold: sigmoid threshold (used for binary classification).
        return_preds: if True, also return predictions and labels.

    Returns:
        report: pd.DataFrame
        (optionally) preds, labels
    """
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"][:, 1].to(device)  # Binary target (1 = relevant)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).squeeze()
            preds = (probs > threshold).long()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    report = classification_report(all_labels, all_preds, digits=4, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    if return_preds:
        return report_df, all_preds, all_labels
    return report_df


def compute_metrics(eval_pred):
    """
    HF Trainer callback for computing metrics during training.
    """
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
    }


def log_classification_results(report_df: pd.DataFrame, preds, labels, run_name: str, out_dir: str = "./results"):
    """
    Save report + confusion matrix to disk and log them to Weights & Biases.

    Args:
        report_df: pd.DataFrame returned from evaluate_model
        preds: list of predictions
        labels: list of true labels
        run_name: wandb run name or identifier
        out_dir: directory to save artifacts
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save report
    report_path = os.path.join(out_dir, "classification_report.csv")
    report_df.to_csv(report_path)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    cm_path = os.path.join(out_dir, "confusion_matrix.png")

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    # W&B logging
    wandb.log({
        "eval/classification_report": wandb.Table(dataframe=report_df),
        "eval/confusion_matrix": wandb.Image(cm_path)
    })

    print(f"✅ Saved report to {report_path}")
    print(f"✅ Saved confusion matrix to {cm_path}")
