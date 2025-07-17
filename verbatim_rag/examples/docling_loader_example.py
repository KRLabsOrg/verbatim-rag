import os
from verbatim_rag.docling_loader import DoclingLoader  # assume you saved it as loader_docling.py
from verbatim_rag.document import Document

def main():
    # Paths
    input_path = "../../data/acl_papers/"  # can be a single PDF file or folder
    tmp_output_path = "tmp/docling_output.json"

    # Ensure tmp dir exists
    os.makedirs(os.path.dirname(tmp_output_path), exist_ok=True)

    # Run Docling + load documents
    print(f"Running Docling on: {input_path}")
    try:
        documents = DoclingLoader.load_with_docling(input_path, tmp_output_path)
    except Exception as e:
        print(f"Docling failed: {e}")
        return

    # Print results
    print(f"\n✅ Loaded {len(documents)} documents from Docling.\n")
    for i, doc in enumerate(documents[:5]):  # show first 5 for inspection
        print(f"--- Document {i+1} ---")
        print(f"ID: {doc.doc_id}")
        print(f"Metadata: {doc.metadata}")
        print(f"Content (first 300 chars):\n{doc.content[:300]}...\n")

if __name__ == "__main__":
    main()
