import argparse
from datetime import datetime
import pandas as pd
from pathlib import Path
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from verbatim_rag.util.dataset_util import prepare_dataset, mask_on_sentence_level
from verbatim_rag.evaluation.metrics import evaluate_model

TEST_DATA_DIR = Path("../../data/dev/processed/")
MODEL_DIR = Path("../../models")
RESULTS_DIR = Path("../../results")


def main(args):

    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load and preprocess the test set
    test_data = pd.read_csv(args.test_file)
    test_data["sentences"] = test_data["sentences"].apply(eval)
    test_data["labels"] = test_data["labels"].apply(eval)
    test_data = mask_on_sentence_level(test_data, window=0)

    dataloader = prepare_dataset(
        df=test_data,
        tokenizer=tokenizer,
        context_length=args.context_length,
        window_size=args.window_size,
        batch_size=args.batch_size,
    )

    # Evaluate
    report, preds, labels = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        threshold=args.threshold,
        return_preds=True
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = RESULTS_DIR / f"classification_report_{ts}.csv"
    report.to_csv(fname)
    print(f"✅ Saved classification report to {fname}")

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    figname = RESULTS_DIR / f"confusion_matrix_{ts}.png"
    plt.savefig(figname)
    print(f"✅ Saved confusion matrix to {figname}")
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--test-file", type=Path,  help="Path to test dataset CSV")
    parser.add_argument("--model", type=Path, help="Path to trained model")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.3)

    args = parser.parse_args()

    test_file_name = "arch-dev.csv"
    model_name = "chillyground"

    if not args.test_file:
        args.test_file = TEST_DATA_DIR / input("Name of the training CSV: ")
    if not args.model:
        args.model = MODEL_DIR / input("Name of the Model: ")

    args.model = MODEL_DIR / model_name
    args.test_file = TEST_DATA_DIR / test_file_name

    main(args)
