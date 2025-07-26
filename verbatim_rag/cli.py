"""
Command-line interface for the Verbatim RAG system.
"""

import argparse
import json
import os

from verbatim_rag import (
    TemplateManager,
    VerbatimIndex,
    VerbatimRAG,
)

try:
    from verbatim_rag.ingestion import DocumentProcessor

    DOCUMENT_PROCESSING_AVAILABLE = True
except ImportError:
    DocumentProcessor = None
    DOCUMENT_PROCESSING_AVAILABLE = False


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key using:")
        print("export OPENAI_API_KEY=your_api_key_here")
        return

    parser = argparse.ArgumentParser(description="Verbatim RAG Command-Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument(
        "--input", "-i", required=True, nargs="+", help="Paths to input documents"
    )
    index_parser.add_argument(
        "--output", "-o", required=True, help="Path to save the index"
    )

    # Template command
    template_parser = subparsers.add_parser("template", help="Template management")
    template_parser.add_argument(
        "--questions",
        "-q",
        required=True,
        nargs="+",
        help="Questions to create templates for",
    )
    template_parser.add_argument(
        "--output", "-o", required=True, help="Path to save templates"
    )
    template_parser.add_argument(
        "--model", "-m", default="gpt-4", help="OpenAI model to use"
    )

    query_parser = subparsers.add_parser("query", help="Query the Verbatim RAG system")
    query_parser.add_argument("--index", "-i", required=True, help="Path to the index")
    query_parser.add_argument("--question", "-q", required=True, help="Question to ask")
    query_parser.add_argument(
        "--num-docs", "-n", type=int, default=5, help="Number of documents to retrieve"
    )
    query_parser.add_argument("--templates", "-t", help="Path to templates file")
    query_parser.add_argument(
        "--model", "-m", default="gpt-4", help="OpenAI model to use"
    )
    query_parser.add_argument(
        "--output", "-o", help="Path to save the response as JSON"
    )

    args = parser.parse_args()

    if args.command == "index":
        if not DOCUMENT_PROCESSING_AVAILABLE:
            print("Error: Document processing not available. Install with:")
            print("pip install 'verbatim-rag[document-processing]'")
            return

        print(f"Processing documents from: {args.input}")
        processor = DocumentProcessor()
        documents = []

        for input_path in args.input:
            if os.path.isdir(input_path):
                documents.extend(processor.process_directory(input_path))
            elif os.path.isfile(input_path):
                documents.append(processor.process_file(input_path))
            else:
                print(f"Warning: {input_path} not found")

        print(f"Processed {len(documents)} documents")

        print("Creating and populating the index...")
        index = VerbatimIndex(
            dense_model="sentence-transformers/all-MiniLM-L6-v2", db_path=args.output
        )
        index.add_documents(documents)
        print("Indexing complete")

    elif args.command == "template":
        print(f"Generating {len(args.questions)} random templates...")
        template_manager = TemplateManager(model=args.model)

        # Generate random templates
        templates = template_manager.generate_random_templates(
            questions=args.questions,
            count=5,  # Generate 5 variations per question
        )
        template_manager.use_random_mode()

        # Save templates to file
        with open(args.output, "w") as f:
            json.dump({"mode": "random", "templates": templates}, f, indent=2)

        print(f"\nGenerated {len(templates)} templates")
        print(f"Templates saved to: {args.output}")

    elif args.command == "query":
        print(f"Loading index from: {args.index}")
        try:
            index = VerbatimIndex(
                dense_model="sentence-transformers/all-MiniLM-L6-v2", db_path=args.index
            )
        except Exception as e:
            print(f"Error loading index: {e}")
            return

        template_manager = TemplateManager(model=args.model)
        if args.templates:
            print(f"Loading templates from: {args.templates}")
            try:
                with open(args.templates, "r") as f:
                    template_data = json.load(f)
                    if template_data.get("mode") == "random":
                        template_manager.use_random_mode()
                        template_manager._random_templates = template_data["templates"]
            except Exception as e:
                print(f"Warning: Could not load templates: {e}")

        rag = VerbatimRAG(index, template_manager=template_manager)
        response = rag.query(args.question)

        print("\nQuestion:", response.question)
        print("\nAnswer:", response.answer)
        print(f"\nCitations: {len(response.structured_answer.citations)}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(response.model_dump(), f, indent=2)
            print(f"\nResponse saved to: {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
