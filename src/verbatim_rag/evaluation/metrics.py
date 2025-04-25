import numpy as np
from sklearn.metrics import classification_report
import torch
from tqdm import tqdm
import os
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


def evaluate_model(model, dataloader, device, threshold=0.3):
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"][:, 1].to(device)  # extract relevant label

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).squeeze()
            preds = (probs > threshold).long()

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    report = classification_report(all_labels, all_preds, digits=4, output_dict=True)
    return pd.DataFrame(report).transpose()


def log_classification_results(report: pd.DataFrame, run_name: str, labels, preds, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # Save classification report
    report.to_csv(os.path.join(out_dir, "classification_report.csv"))

    # Plot + save confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(path)
    plt.close()

    # Log to W&B
    wandb.log({
        "eval/classification_report": wandb.Table(dataframe=report),
        "eval/confusion_matrix": wandb.Image(path)
    })
