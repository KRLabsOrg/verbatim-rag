import os
from verbatim_rag.docling_loader import DoclingLoader  # assume you saved it as loader_docling.py
from verbatim_rag.document import Document

def main():
    # Paths
    input_path = "../../data/acl_papers/"  # can be a single PDF file or folder
    tmp_output_path = "tmp/docling_output/"
    os.makedirs(tmp_output_path, exist_ok=True)


    # Run Docling + load documents
    print(f"Running Docling on: {input_path}")
    try:
        documents = DoclingLoader.load_with_docling(input_path, tmp_output_path)
    except Exception as e:
        print(f"Docling failed: {e}")
        return

    # Print results
    print(f"\n✅ Loaded {len(documents)} documents from Docling.\n")
    print(f"Type: {type(documents)}\n")
    
    for i, doc in enumerate(documents):
        print(f"--- Document {i} ---")
        print(f"ID         : {doc.id}")
        print(f"📄 Section   : {doc.metadata.get('section', '-')}")
        print(f"↪️  Parent    : {doc.metadata.get('parent', '-')}")
        print(f"📝 Content   : {doc.content[:120].strip()}...\n")

if __name__ == "__main__":
    main()
