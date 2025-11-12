import argparse
import logging


from verbatim_rag import VerbatimIndex, VerbatimRAG
from verbatim_rag.embedding_providers import SentenceTransformersProvider
from verbatim_rag.vector_stores import LocalMilvusStore
from verbatim_rag.core import LLMClient


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

    llm_client = LLMClient(
        model="moonshotai/kimi-k2-instruct-0905",
        api_base="https://api.groq.com/openai/v1/",
    )

    dense_provider = SentenceTransformersProvider(
        model_name="ibm-granite/granite-embedding-english-r2", device=args.device
    )

    # Create vector store
    vector_store = LocalMilvusStore(
        db_path=args.index_file,
        collection_name=args.collection_name,
        enable_dense=True,
        enable_sparse=False,
        dense_dim=dense_provider.get_dimension(),
        nlist=16384,
    )

    # Create index
    index = VerbatimIndex(vector_store=vector_store, dense_provider=dense_provider)
    rag = VerbatimRAG(index, llm_client=llm_client)
    while True:
        test_query = input(">")
        logging.info(f"asking: {test_query}")
        response = rag.query(test_query)
        logging.info(f"answer: {response.answer}")


if __name__ == "__main__":
    main()
