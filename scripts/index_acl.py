import argparse
import json
import logging

from pathlib import Path
from tqdm import tqdm

from verbatim_rag import VerbatimIndex, VerbatimRAG
from verbatim_rag.config import create_default_config
from verbatim_rag.ingestion import DocumentProcessor

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s",
)
# log = logging.getLogger(__name__)


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
    index_config = create_default_config()
    index_config.sparse_embedding.model_name = (
        "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"
    )

    index_config.vector_db.db_path = args.index_file
    index_config.vector_db.collection_name = args.collection_name
    index_config.sparse_embedding.device = args.device
    index = VerbatimIndex(config=index_config)
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
    test_query = "Which paper describes 4lang?"
    logging.info(f"asking: {test_query}")
    response = rag.query(test_query)
    logging.info(f"answer: {response.answer}")


if __name__ == "__main__":
    main()
