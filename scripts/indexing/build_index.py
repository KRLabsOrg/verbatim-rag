#!/usr/bin/env python3
import argparse
from pathlib import Path

from verbatim_rag.loader import DocumentLoader
from verbatim_rag.text_splitter import TextSplitter
from verbatim_rag.index import VerbatimIndex

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    default_docs = repo_root / "examples" / "example_docs"
    default_out  = repo_root / "models"   / "index"

    parser = argparse.ArgumentParser(
        description="Chunk your docs & build a FAISS index."
    )
    parser.add_argument(
        "--docs-dir", type=Path, default=default_docs,
        help="Directory with source .txt/.csv files"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=default_out,
        help="Where to write models/index.* files"
    )
    parser.add_argument(
        "--chunk-size",   type=int, default=200, help="Max chars per chunk"
    )
    parser.add_argument(
        "--chunk-overlap",type=int, default=50,  help="Overlap chars between chunks"
    )
    parser.add_argument(
        "--embed-model",  type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        help="HuggingFace or OpenAI embedding model"
    )
    args = parser.parse_args()

    # 1) load & chunk
    docs   = DocumentLoader.load_directory(str(args.docs_dir), recursive=True)
    print(f"🔍 Found {len(docs)} source documents.")
    splitter = TextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks   = splitter.split_documents(docs)
    print(f"🗂️  Produced {len(chunks)} chunks.")

    # 2) index & save
    index = VerbatimIndex(embedding_model=args.embed_model)
    index.add_documents(chunks)
    print("🚀 Added all chunks to the index.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index.save(str(args.output_dir))
    print(f"💾 Index saved under {args.output_dir}/")

if __name__ == "__main__":
    main()
