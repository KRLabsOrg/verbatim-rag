"""
Example usage of the ModelSpanExtractor for extracting relevant spans.

This example demonstrates how to use the ModelSpanExtractor to extract
relevant spans from documents based on a question using a trained model.
"""

import argparse

from verbatim_rag.document import Document
from verbatim_rag.extractors import ModelSpanExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Extract relevant spans using a trained model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on ('cpu', 'cuda'). Default: auto-detect",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for considering a span relevant (0.0-1.0)",
    )
    args = parser.parse_args()

    # Initialize the extractor
    extractor = ModelSpanExtractor(model_path=args.model_path, device=args.device)

    # Example documents
    documents = [
        Document(
            content="""
            Climate change is a significant and lasting change in the statistical distribution of weather patterns.
            Global warming is the observed increase in the average temperature of the Earth's atmosphere and oceans.
            Greenhouse gases include water vapor, carbon dioxide, methane, nitrous oxide, and ozone.
            Human activities since the beginning of the Industrial Revolution have increased greenhouse gas levels.
            """,
            metadata={"source": "example_doc_1", "id": "climate_1"},
        ),
        Document(
            content="""
            Renewable energy comes from sources that are naturally replenished on a human timescale.
            Solar power is the conversion of energy from sunlight into electricity.
            Wind power is the use of wind to provide mechanical power or electricity.
            Hydropower is electricity generated from the energy of falling water.
            """,
            metadata={"source": "example_doc_2", "id": "energy_1"},
        ),
    ]

    # Example questions
    questions = [
        "What causes climate change?",
        "What are some types of renewable energy?",
        "How does solar power work?",
    ]

    # Extract and display results
    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 50)

        results = extractor.extract_spans(question, documents)

        for doc_idx, (content, spans) in enumerate(results.items()):
            doc = documents[doc_idx]
            print(f"\nDocument {doc_idx + 1} ({doc.metadata.get('id', 'unknown')}):")

            if spans:
                print("Relevant spans:")
                for i, span in enumerate(spans):
                    print(f"  {i + 1}. {span}")
            else:
                print("No relevant spans found.")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
