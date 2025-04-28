import joblib
import argparse
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

from verbatim_rag.util.dataset_util import load_question_sentence_pairs


def train_baseline(train_path: Path, test_path: Path, model_path: Path):
    print("📥 Loading training and test data...")
    X_train, y_train = load_question_sentence_pairs(train_path)
    X_test, y_test = load_question_sentence_pairs(test_path)

    print("🧠 Training logistic regression baseline...")
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    clf.fit(X_train, y_train)

    # Save Model
    model_path.mkdir(parents=True, exist_ok=True)
    saved_model_path = str(model_path / "baseline_classifier.joblib")
    joblib.dump(clf, saved_model_path)
    print(f"✅ Saved model to {saved_model_path}")

    print("🔍 Evaluating on (internal) test set...")
    y_pred = clf.predict(X_test)
    report = pd.DataFrame(classification_report(y_test, y_pred, digits=4, output_dict=True)).transpose()
    print(report)


def evaluate_baseline(model_path: Path, test_path: Path, output_dir: Path):
    print("📥 Loading test data...")
    X_test, y_test = load_question_sentence_pairs(test_path)

    print(f"📦 Loading saved model from {model_path}")
    model = joblib.load(model_path)

    print("🔍 Running evaluation...")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "baseline_eval_report.csv"
    df_report.to_csv(report_path)

    print(f"✅ Evaluation report saved to: {report_path}")
    print(df_report.head())


if __name__ == "__main__":

    TRAIN_DATA_DIR = Path("../../data/synthetic/questions")
    TEST_DATA_DIR = Path("../../data/dev/processed/")
    MODEL_DIR = Path("../../models/baseline")
    RESULTS_DIR = Path("../../results/baseline")

    parser = argparse.ArgumentParser(description="Train or evaluate a logistic regression baseline model.")
    parser.add_argument("--train-file", type=Path, help="Name of the training CSV")
    parser.add_argument("--test-file", type=Path, help="Name of the test CSV")

    args = parser.parse_args()

    args.train_file = TRAIN_DATA_DIR / "questions_V1.csv"
    args.test_file = TEST_DATA_DIR / "arch-dev.csv"

    if not args.train_file:
        args.train_file = TRAIN_DATA_DIR / input("Name of the training CSV: ")
    if not args.test_file:
        args.test_file = TEST_DATA_DIR / input("Name of the test CSV: ")

    # train
    train_baseline(args.train_file, args.test_file, MODEL_DIR)

    # eval
    evaluate_baseline(MODEL_DIR / "baseline_classifier.joblib", args.test_file, RESULTS_DIR)
