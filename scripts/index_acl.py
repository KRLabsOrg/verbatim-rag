import argparse
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s",
)

from pathlib import Path
from tqdm import tqdm

from verbatim_rag import VerbatimIndex, VerbatimRAG
from verbatim_rag.embedding_providers import SpladeProvider
from verbatim_rag.ingestion import DocumentProcessor
from verbatim_rag.vector_stores import LocalMilvusStore


def index_acl(args):
    with open(args.metadata_file) as f:
        papers = {paper["url"].split("/")[-2]: paper for paper in json.load(f)}

    processor = DocumentProcessor()
    logging.info("loading and chunking documents...")
    documents = []
    for file_path in tqdm(Path(args.input_dir).rglob("*")):
        if file_path.suffix.lower() != ".md":
            logging.warning(f"skipping file because extension isn't md: {file_path}")
            continue
        paper_id = file_path.stem
        if paper_id not in papers:
            logging.warning(f"skipping paper not in metadata file: {paper_id}")
            continue

        document = processor.process_file(
            file_path, title=paper_id, metadata=papers[paper_id]
        )
        documents.append(document)

    logging.info("indexing documents...")

    sparse_provider = SpladeProvider(
        model_name="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
        device=args.device,
    )
    vector_store = LocalMilvusStore(
        db_path=args.index_file,
        collection_name=args.collection_name,
        enable_dense=False,
        enable_sparse=True,
    )

    index = VerbatimIndex(vector_store=vector_store, sparse_provider=sparse_provider)

    index.add_documents(documents)
    return index


def get_args():
    parser = argparse.ArgumentParser(description="Preprocess ACL Anthology papers")
    parser.add_argument(
        "--metadata-file", required=True, help="Path to paper metadata file"
    )
    parser.add_argument(
        "--input-dir", required=True, help="Directory for downloaded papers"
    )
    parser.add_argument("--index-file", required=True, help="File for storing index db")
    parser.add_argument("--collection-name", required=True, help="Name of collection")
    parser.add_argument(
        "--device", required=True, help="Device to use for embedding (e.g. cpu or cuda)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Dry run")

    return parser.parse_args()


def main():
    args = get_args()
    # Check dependencies
    index = index_acl(args)
    rag = VerbatimRAG(index)
    test_query = "What is 4lang?"
    logging.info(f"asking: {test_query}")
    response = rag.query(test_query)
    logging.info(f"answer: {response.answer}")


if __name__ == "__main__":
    main()
