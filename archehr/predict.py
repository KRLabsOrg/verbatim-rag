import argparse
from pathlib import Path
import json
from archehr.preprocess import ArchehrData, Case
from archehr.models import LLMModel, ArchehrModel, LLMModelGenerate

from typing import NamedTuple
from rich import print

CASE_SENTENCE_TO_PREDICTION: dict[tuple[int, int], bool] = {}


class Metrics(NamedTuple):
    precision: float
    recall: float
    f1: float
    TP: int
    FP: int
    TN: int
    FN: int

    def __str__(self) -> str:
        return f"Precision: {self.precision}, Recall: {self.recall}, F1: {self.f1}, TP: {self.TP}, FP: {self.FP}, TN: {self.TN}, FN: {self.FN}"


def calculate_metrics(y_true: list[int], y_pred: list[int]) -> Metrics:
    """Calculate precision, recall, F1 by manually counting TP, FP, TN, FN.

    This manual calculation ensures that the metrics are consistent with the counts.

    :param y_true: Ground truth labels (0 or 1)
    :param y_pred: Predicted labels (0 or 1)
    :return: Metrics object with precision, recall, F1, TP, FP, TN, FN
    """
    # Manual count of TP, FP, TN, FN
    TP = FP = TN = FN = 0

    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            TP += 1
        elif true == 0 and pred == 1:
            FP += 1
        elif true == 0 and pred == 0:
            TN += 1
        elif true == 1 and pred == 0:
            FN += 1

    # Calculate precision, recall, and F1 directly from counts
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return Metrics(
        precision=precision,
        recall=recall,
        f1=f1,
        TP=TP,
        FP=FP,
        TN=TN,
        FN=FN,
    )


def evaluate_case(case: Case, predictions: list[bool]) -> tuple[list[int], list[int]]:
    """Evaluate a case by comparing the predictions to the ground truth.

    :param case: The case to evaluate.
    :param predictions: The predictions for the case.
    :return: A tuple containing the ground truth and the predictions.
    """
    y_true = []
    y_pred = []

    for i, sentence in enumerate(case.sentences):
        y_true.append(1 if sentence.relevance in ["essential", "supplementary"] else 0)
        y_pred.append(1 if predictions[i] else 0)

    return y_true, y_pred


def predict_case(
    model: ArchehrModel,
    patient_narrative: str,
    clinician_question: str,
    note_excerpt: str,
    sentences: list[str],
) -> list[bool]:
    return model.predict(patient_narrative, clinician_question, note_excerpt, sentences)


def process_case(case: Case, model: ArchehrModel, mode: str):
    """Process a single case: generate predictions and evaluate if in dev mode.

    :param case: The case to process
    :param model: The model to use for prediction
    :param mode: Whether we're in dev or test mode
    :return: A tuple containing (case_result, y_true, y_pred) where the latter two are None in test mode
    """
    # Generate predictions for all sentences in the case
    case_predictions = predict_case(
        model,
        case.patient_narrative,
        case.clinician_question,
        case.note_excerpt,
        [sentence.sentence_text for sentence in case.sentences],
    )

    for sentence, prediction in zip(case.sentences, case_predictions):
        CASE_SENTENCE_TO_PREDICTION[(case.case_id, sentence.sentence_id)] = prediction

    if mode == "dev":
        # Evaluate in dev mode
        y_true, y_pred = evaluate_case(case, case_predictions)
        metrics = calculate_metrics(y_true, y_pred)

        case_result = {
            "case_id": case.case_id,
            "clinician_question": case.clinician_question,
            "metrics": metrics,
            "ground_truth": [
                {
                    "sentence_id": sentence.sentence_id,
                    "sentence_text": sentence.sentence_text,
                    "relevance": sentence.relevance,
                }
                for sentence in case.sentences
            ],
            "predictions": [
                {
                    "sentence_id": sentence.sentence_id,
                    "sentence_text": sentence.sentence_text,
                    "predicted_relevance": "relevant" if prediction else "not-relevant",
                    "true_relevance": sentence.relevance,
                }
                for sentence, prediction in zip(case.sentences, y_pred)
            ],
        }

        return case_result, y_true, y_pred
    else:
        # In test mode, just save predictions without evaluation
        case_result = {
            "case_id": case.case_id,
            "clinician_question": case.clinician_question,
            "predictions": [
                {
                    "sentence_id": sentence.sentence_id,
                    "sentence_text": sentence.sentence_text,
                    "predicted_relevance": "relevant"
                    if CASE_SENTENCE_TO_PREDICTION[(case.case_id, sentence.sentence_id)]
                    else "not-relevant",
                }
                for sentence in case.sentences
            ],
        }

        return case_result, None, None


def load_model(model_name: str) -> ArchehrModel:
    """Load the appropriate model based on the model name.

    :param model_name: The name of the model to load.
    :return: An instance of the model.
    """
    if model_name == "LLMModel":
        # return LLMModel(model_name="THUDM/GLM-4-32B-0414", zero_shot=True)
        return LLMModel(model_name="google/gemma-3-27b-it", zero_shot=True)
        # return LLMModel(model_name="mistralai/Mistral-Small-3.1-24B-Instruct-2503", zero_shot=True)
    elif model_name == "BERTModel":
        from archehr.models.encoder import BERTModel

        return BERTModel(
            model_name="models/bert_classifier_clinical/2025-04-26_17-30-07"
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def save_results(case_results: list[dict], output_dir: str, model_name: str, mode: str):
    """Save the case results to a file.

    :param case_results: The results for each case.
    :param output_dir: The directory to save the results to.
    :param model_name: The name of the model used.
    :param mode: The mode used (dev or test).
    """
    output_path = Path(output_dir) / f"case_results_{model_name}_{mode}.json"
    print(f"Saving case results to {output_path}")

    with open(output_path, "w") as f:
        json.dump(case_results, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--model", type=str, required=True, choices=["LLMModel", "BERTModel"]
    )
    parser.add_argument("--output_dir", type=str, default="case_results")
    # If test, we don't have ground truth, so skip evaluation
    parser.add_argument("--mode", type=str, choices=["dev", "test"], default="dev")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model = load_model(args.model)
    data = ArchehrData.from_json(json.load(open(args.data_dir)))

    all_y_true = []
    all_y_pred = []
    case_results = []

    for case in data.cases:
        case_result, y_true, y_pred = process_case(case, model, args.mode)
        case_results.append(case_result)

        print(
            f"Case {case.case_id}/{len(data.cases)}: "
            + (f"{case_result['metrics']}" if args.mode == "dev" else "Processed")
        )

        if args.mode == "dev" and y_true and y_pred:
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)

    # Only output overall metrics in dev mode
    if args.mode == "dev":
        metrics = calculate_metrics(all_y_true, all_y_pred)
        print(f"Overall: {metrics}")

    save_results(case_results, args.output_dir, args.model, args.mode)

    if args.generate:
        # generate_model = LLMModelGenerate(model_name="mistralai/Mistral-Small-3.1-24B-Instruct-2503")
        # generate_model = LLMModelGenerate(model_name="THUDM/GLM-4-32B-0414")
        generate_model = LLMModelGenerate(model_name="google/gemma-3-27b-it")
        case_id_to_answer = {}

        for case in data.cases:
            print(f"Generating answer for case {case.case_id}")
            sentence_relevancy = [
                CASE_SENTENCE_TO_PREDICTION[(case.case_id, sentence.sentence_id)]
                for sentence in case.sentences
            ]
            case_id_to_answer[case.case_id] = generate_model.generate(
                case.patient_narrative,
                case.clinician_question,
                case.note_excerpt,
                [sentence.sentence_text for sentence in case.sentences],
                sentence_relevancy,
            )

        print(
            f"Saving results to {Path(args.output_dir) / f'case_results_{args.model}_generate.json'}"
        )

        # Answers should be sentence per line
        with open(
            Path(args.output_dir) / f"case_results_{args.model}_generate.json", "w"
        ) as f:
            json.dump(case_id_to_answer, f, indent=4)


if __name__ == "__main__":
    main()
