import argparse
import logging


from verbatim_rag import VerbatimIndex, VerbatimRAG
from verbatim_rag.embedding_providers import SpladeProvider
from verbatim_rag.vector_stores import LocalMilvusStore


def get_args():
    parser = argparse.ArgumentParser(description="Preprocess ACL Anthology papers")
    parser.add_argument("--index-file", required=True, help="File for storing index db")
    parser.add_argument("--collection-name", required=True, help="Name of collection")
    parser.add_argument(
        "--device", required=True, help="Device to use for embedding (e.g. cpu or cuda)"
    )
    return parser.parse_args()


def main():
    args = get_args()
    store = LocalMilvusStore(
        db_path=args.index_file,
        collection_name=args.collection_name,
        enable_sparse=True,
    )
    sparse_provider = SpladeProvider(
        model_name="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
        device=args.device,
    )
    # chunker = MarkdownChunkerProvider()
    index = VerbatimIndex(
        vector_store=store, sparse_provider=sparse_provider
    )  # , chunker_provider=chunker

    rag = VerbatimRAG(index)
    while True:
        test_query = input(">")
        logging.info(f"asking: {test_query}")
        response = rag.query(test_query)
        logging.info(f"answer: {response.answer}")


if __name__ == "__main__":
    main()
