import argparse
import pandas as pd
from pathlib import Path
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from verbatim_rag.util.dataset_util import prepare_dataset
from verbatim_rag.evaluation.metrics import evaluate_model


def main(args):

    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load and preprocess the test set
    test_data = pd.read_csv(args.test_path)
    test_data["sentences"] = test_data["sentences"].apply(eval)
    test_data["labels"] = test_data["labels"].apply(eval)

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

    # Save report
    if args.output_path:
        out_dir = Path(args.output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output_path)
        print(f"✅ Saved classification report to {args.output_path}")

    # Plot and save confusion matrix
    if args.cm_path:
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(args.cm_path)
        print(f"✅ Saved confusion matrix to {args.cm_path}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test-path", type=Path, required=True, help="Path to test dataset CSV")
    parser.add_argument("--model-checkpoint", type=Path, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output-path", type=Path, help="Where to save classification report CSV")
    parser.add_argument("--cm-path", type=Path, help="Where to save confusion matrix PNG")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.3)

    args = parser.parse_args()
    main(args)
