# clinical_bert_inference.py

import argparse
import json
from verbatim_rag.inference.model import ClinicalBERTModel
from verbatim_rag.inference.document import Document


def load_documents(documents_path):
    with open(documents_path, "r") as f:
        data = json.load(f)
    documents = []
    for item in data:
        documents.append(
            Document(
                content=item["content"],
                metadata=item.get("metadata", {})
            )
        )
    return documents


def main():
    parser = argparse.ArgumentParser(description="Run ClinicalBERT inference on clinical documents.")
    parser.add_argument("--documents", type=str, required=True, help="Path to JSON file with documents.")
    parser.add_argument("--patient_question", type=str, required=True, help="Patient's question.")
    parser.add_argument("--clinician_question", type=str, required=True, help="Clinician's question.")
    parser.add_argument("--context_size", type=int, default=1, help="Context size for model.")
    parser.add_argument("--max_length", type=int, default=512, help="Max token length.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold.")
    args = parser.parse_args()

    classifier = ClinicalBERTModel(
        context_size=args.context_size,
        max_length=args.max_length,
        threshold=args.threshold,
    )

    documents = load_documents(args.documents)

    results = {}
    for doc in documents:
        preds = classifier.predict(
            patient_question=args.patient_question,
            clinician_question=args.clinician_question,
            sentences=doc.content,
            sep=". ",
            use_clinician_question=True,
        )
        results[doc.metadata.get("id", "unknown_id")] = preds

    for doc_id, preds in results.items():
        print(f"\n>> {doc_id}")
        print(preds)


if __name__ == "__main__":
    main()
