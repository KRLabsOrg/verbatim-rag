from preprocess import ArchehrData
import argparse
import json
from verbatim_rag.extractors import ModelSpanExtractor
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from rich import print


def evaluate_bert_extraction(case, sentences, extracted_spans):
    """
    Evaluate BERT model extraction by direct sentence matching.

    When using extract_spans_from_sentences, the model returns the exact
    sentences from the input list that it classified as relevant.
    """
    y_true = []
    y_pred = []

    # Create sentence => prediction mapping
    sentence_is_relevant = {s: False for s in sentences}
    for span in extracted_spans:
        if span in sentence_is_relevant:
            sentence_is_relevant[span] = True

    # Generate predictions and ground truth
    for i, sentence in enumerate(case.sentences):
        # Ground truth: sentences marked as "essential" or "supplementary" are relevant
        is_relevant_truth = (
            sentence.relevance in ["essential", "supplementary"]
            if sentence.relevance
            else False
        )
        y_true.append(1 if is_relevant_truth else 0)

        # Prediction: check if this exact sentence was in the extracted spans
        text = sentence.sentence_text
        is_relevant_pred = sentence_is_relevant.get(text, False)
        y_pred.append(1 if is_relevant_pred else 0)

    return y_true, y_pred


def calculate_metrics(y_true, y_pred):
    """Calculate precision, recall, and F1 score."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="data/archehr/dev/archehr-qa_processed.json"
    )
    parser.add_argument(
        "--num_cases",
        type=int,
        default=None,
        help="Number of cases to evaluate (None for all)",
    )
    parser.add_argument(
        "--output_file", type=str, help="Path to save evaluation results"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="KRLabsOrg/chiliground-base-modernbert-v1",
        help="Path to the model to use for extraction",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for considering a span relevant",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the model on ('cpu', 'cuda', etc). If None, will use CUDA if available.",
    )
    parser.add_argument(
        "--detailed_report",
        action="store_true",
        help="Print detailed classification report",
    )
    args = parser.parse_args()

    with open(args.data_dir, "r") as f:
        archehr_data = ArchehrData.from_json(json.load(f))

    all_y_true = []
    all_y_pred = []
    case_results = []

    # Initialize the model extractor
    model_extractor = ModelSpanExtractor(
        model_path=args.model_path, device=args.device, threshold=args.threshold
    )

    print(f"Using BERT model: {args.model_path}")
    print(f"Threshold: {args.threshold}")

    cases_to_evaluate = (
        archehr_data.cases[: args.num_cases] if args.num_cases else archehr_data.cases
    )
    print(f"Evaluating {len(cases_to_evaluate)} cases...")

    for case_idx, case in enumerate(cases_to_evaluate):
        try:
            # Extract the sentences from the case
            sentences = [sentence.sentence_text for sentence in case.sentences]

            # Use the model to extract relevant sentences
            extracted_spans = model_extractor.extract_spans_from_sentences(
                case.clinician_question, sentences
            )

            # Evaluate the results
            y_true, y_pred = evaluate_bert_extraction(case, sentences, extracted_spans)
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)

            # Calculate per-case metrics
            case_metrics = calculate_metrics(y_true, y_pred)
            print(f"Case {case_idx + 1}/{len(cases_to_evaluate)}: {case_metrics}")

            # Display relevant spans for debugging
            print(
                f"Extracted {len(extracted_spans)} relevant sentences out of {len(sentences)} total:"
            )
            for i, span in enumerate(extracted_spans[:3]):  # Show first 3 spans
                print(
                    f"  {i + 1}. {span[:80]}..."
                    if len(span) > 80
                    else f"  {i + 1}. {span}"
                )
            if len(extracted_spans) > 3:
                print(f"  ... and {len(extracted_spans) - 3} more")

            # Save case results
            case_results.append(
                {
                    "case_id": case.case_id,
                    "clinician_question": case.clinician_question,
                    "metrics": case_metrics,
                    "num_sentences": len(sentences),
                    "num_relevant": len(extracted_spans),
                    "extracted_spans": extracted_spans,
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
                            "predicted_relevance": "relevant"
                            if pred == 1
                            else "not-relevant",
                            "true_relevance": sentence.relevance,
                        }
                        for sentence, pred in zip(case.sentences, y_pred)
                    ],
                }
            )

        except Exception as e:
            print(f"Error processing case {case_idx + 1}: {e}")
            import traceback

            traceback.print_exc()

    # Calculate overall metrics
    if all_y_true and all_y_pred:
        overall_metrics = calculate_metrics(all_y_true, all_y_pred)
        print("\nOverall metrics:")
        print(f"Precision: {overall_metrics['precision']:.4f}")
        print(f"Recall: {overall_metrics['recall']:.4f}")
        print(f"F1 Score: {overall_metrics['f1']:.4f}")

        if args.detailed_report:
            print("\nDetailed Classification Report:")
            print(
                classification_report(
                    all_y_true,
                    all_y_pred,
                    target_names=["not-relevant", "relevant"],
                    digits=4,
                )
            )

        # Save results to file if requested
        if args.output_file:
            output_data = {
                "overall_metrics": overall_metrics,
                "model_path": args.model_path,
                "threshold": args.threshold,
                "case_results": case_results,
            }
            with open(args.output_file, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"Results saved to {args.output_file}")
    else:
        print("No cases were successfully evaluated.")


if __name__ == "__main__":
    main()
