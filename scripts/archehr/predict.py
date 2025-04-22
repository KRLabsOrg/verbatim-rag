from preprocess import ArchehrData
import argparse
import json
from verbatim_rag.extractors import (
    LLMSpanExtractor,
    FewShotLLMSpanExtractor,
)
from verbatim_rag.document import Document
from sklearn.metrics import precision_score, recall_score, f1_score
import openai

from rich import print


def create_examples_from_archehr(archehr_data):
    """Create examples from ArchehrData for few-shot learning."""
    examples = []

    # Iterate through cases to find ones with well-labeled sentences
    for case in archehr_data.cases:
        if not case.sentences:
            continue

        # Get relevant sentences
        relevant_sentences = []
        non_relevant_sentences = []

        for sentence in case.sentences:
            if sentence.relevance in ["essential", "supplementary"]:
                relevant_sentences.append(sentence)
            elif sentence.relevance == "not-relevant":
                non_relevant_sentences.append(sentence)

        # Skip cases without any relevant sentences or without clear labels
        if not relevant_sentences:
            continue

        # Create marked text by inserting <relevant> tags around relevant sentences
        # Sort sentences by their position in the document
        all_sentences = relevant_sentences + non_relevant_sentences
        all_sentences.sort(key=lambda s: int(s.start_char_index))

        # Build the marked document
        original_text = case.note_excerpt
        marked_text = ""
        last_end = 0

        for sentence in all_sentences:
            start_idx = int(sentence.start_char_index)
            end_idx = start_idx + int(sentence.length)

            # Add text before this sentence
            if start_idx > last_end:
                marked_text += original_text[last_end:start_idx]

            # Add the sentence, with tags if relevant
            if sentence in relevant_sentences:
                marked_text += (
                    "<relevant>" + original_text[start_idx:end_idx] + "</relevant>"
                )
            else:
                marked_text += original_text[start_idx:end_idx]

            last_end = end_idx

        # Add any remaining text
        if last_end < len(original_text):
            marked_text += original_text[last_end:]

        # Add to examples
        examples.append(
            {
                "question": case.clinician_question,
                "document": case.note_excerpt,
                "marked_text": marked_text,
            }
        )

    return examples


def get_embedding(text):
    """Get OpenAI embedding for a text."""
    text = text.replace("\n", " ")
    try:
        response = openai.embeddings.create(model="text-embedding-3-small", input=text)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Return a dummy embedding
        return [0.0] * 1536


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5
    if magnitude_a == 0 or magnitude_b == 0:
        return 0
    return dot_product / (magnitude_a * magnitude_b)


class SmartExampleSelector:
    """Class to select relevant examples based on semantic similarity."""

    def __init__(self, examples):
        self.examples = examples
        # Cache for embeddings to avoid recalculating
        self.embedding_cache = {}

    def _get_cached_embedding(self, text):
        """Get embedding with caching."""
        if text not in self.embedding_cache:
            self.embedding_cache[text] = get_embedding(text)
        return self.embedding_cache[text]

    def select_examples(self, question, num_examples=2):
        """Select most relevant examples for a question based on semantic similarity."""
        if not self.examples:
            return []

        # Get embedding for the current question
        question_embedding = self._get_cached_embedding(question)

        # Calculate similarity with all examples
        similarities = []
        for i, example in enumerate(self.examples):
            example_embedding = self._get_cached_embedding(example["question"])
            similarity = cosine_similarity(question_embedding, example_embedding)
            similarities.append((similarity, i))

        # Sort by similarity (descending)
        similarities.sort(reverse=True)

        # Select top examples
        selected_examples = []
        for _, idx in similarities[:num_examples]:
            selected_examples.append(self.examples[idx])

        return selected_examples


class EnhancedFewShotLLMSpanExtractor(FewShotLLMSpanExtractor):
    """Enhanced few-shot extractor with smart example selection."""

    def __init__(
        self, model: str = "gpt-4o-mini", examples=None, example_selector=None
    ):
        """
        Initialize the enhanced few-shot LLM span extractor.

        :param model: The LLM model to use for extraction
        :param examples: Optional list of examples to use for few-shot learning
        :param example_selector: Optional selector to choose relevant examples
        """
        super().__init__(model=model, examples=examples)
        self.example_selector = example_selector

    def _select_examples(self, question, num_examples=2):
        """Select relevant examples for the question using the selector if available."""
        if self.example_selector:
            return self.example_selector.select_examples(question, num_examples)
        return super()._select_examples(question, num_examples)


def check_span_in_sentence(
    span_text, span_start, span_end, sentence_text, sentence_start
):
    """Check if a span overlaps with a sentence."""
    span_end = span_start + len(span_text)
    sentence_end = sentence_start + len(sentence_text)

    # Check if there's any overlap between the span and the sentence
    return not (span_end <= sentence_start or span_start >= sentence_end)


def evaluate_case(case, extracted_spans_text):
    """Evaluate a single case by matching extracted spans to sentences."""
    y_true = []
    y_pred = []

    for sentence in case.sentences:
        # Ground truth: sentences marked as "essential" or "supplementary" are relevant
        is_relevant_truth = (
            sentence.relevance in ["essential", "supplementary"]
            if sentence.relevance
            else False
        )
        y_true.append(1 if is_relevant_truth else 0)

        # Prediction: sentence is relevant if any extracted span is contained in the sentence or vice versa
        is_relevant_pred = False
        sentence_text = sentence.sentence_text.strip()

        for span_text in extracted_spans_text:
            span_text = span_text.strip()
            span_text = span_text.replace("\n", " ").lower()
            sentence_text = sentence_text.replace("\n", " ").lower()

            # Check if the span is in the sentence or if the sentence is in the span
            if span_text in sentence_text or sentence_text in span_text:
                is_relevant_pred = True
                break

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
        "--extractor",
        type=str,
        default="fewshot",
        choices=["llm", "fewshot"],
        help="Which extractor to use: 'llm' for standard LLMSpanExtractor or 'fewshot' for FewShotLLMSpanExtractor",
    )
    parser.add_argument(
        "--smart_examples", action="store_true", help="Use smart example selection"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=2,
        help="Number of examples to use for few-shot learning",
    )
    args = parser.parse_args()

    with open(args.data_dir, "r") as f:
        archehr_data = ArchehrData.from_json(json.load(f))

    all_y_true = []
    all_y_pred = []
    case_results = []

    # Generate examples from ArchehrData for few-shot learning
    examples = create_examples_from_archehr(archehr_data)
    print(f"Generated {len(examples)} examples from ArchehrData")

    # Create example selector if using smart selection
    example_selector = SmartExampleSelector(examples) if args.smart_examples else None

    # Select the appropriate extractor based on the argument
    if args.extractor == "llm":
        extractor = LLMSpanExtractor(model="gpt-4o-mini")
    else:  # fewshot
        extractor = EnhancedFewShotLLMSpanExtractor(
            model="gpt-4o", examples=None, example_selector=None
        )

    print(f"Using {args.extractor.upper()} extractor")
    if args.extractor == "fewshot" and args.smart_examples:
        print("With smart example selection")

    cases_to_evaluate = (
        archehr_data.cases[: args.num_cases] if args.num_cases else archehr_data.cases
    )
    print(f"Evaluating {len(cases_to_evaluate)} cases...")

    for case_idx, case in enumerate(cases_to_evaluate):
        document = Document(content=case.note_excerpt)
        try:
            # Get spans from the selected extractor
            spans_output = extractor.extract_spans(case.clinician_question, [document])

            # Get the list of extracted spans
            doc_key = list(spans_output.keys())[0]
            extracted_spans_text = spans_output[doc_key]

            y_true, y_pred = evaluate_case(case, extracted_spans_text)
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)

            # Calculate per-case metrics
            case_metrics = calculate_metrics(y_true, y_pred)
            print(f"Case {case_idx + 1}/{len(cases_to_evaluate)}: {case_metrics}")

            # Save case results
            case_results.append(
                {
                    "case_id": case.case_id,
                    "clinician_question": case.clinician_question,
                    "metrics": case_metrics,
                    "extracted_spans": extracted_spans_text,
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

        # Save results to file if requested
        if args.output_file:
            output_data = {
                "overall_metrics": overall_metrics,
                "extractor_type": args.extractor,
                "smart_examples": args.smart_examples
                if args.extractor == "fewshot"
                else False,
                "num_examples": args.num_examples,
                "case_results": case_results,
            }
            with open(args.output_file, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"Results saved to {args.output_file}")
    else:
        print("No cases were successfully evaluated.")


if __name__ == "__main__":
    main()
