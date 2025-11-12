import streamlit as st
import logging
from typing import Optional, List
import os
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from verbatim_rag import VerbatimIndex, VerbatimRAG
from verbatim_rag.embedding_providers import SentenceTransformersProvider
from verbatim_rag.vector_stores import LocalMilvusStore, SearchResult
from verbatim_rag.core import LLMClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@st.cache_resource
def initialize_rag(
    index_file: str,
    collection_name: str,
    device: str,
) -> tuple[VerbatimIndex, VerbatimRAG]:
    """Initialize the VerbatimIndex and VerbatimRAG with caching."""
    logger.info(f"Initializing VerbatimRAG from {index_file}")

    # Initialize LLM client
    llm_client = LLMClient(
        model="moonshotai/kimi-k2-instruct-0905",
        api_base="https://api.groq.com/openai/v1/",
    )

    # Initialize embedding provider
    dense_provider = SentenceTransformersProvider(
        model_name="ibm-granite/granite-embedding-english-r2", device=device
    )

    # Create vector store
    vector_store = LocalMilvusStore(
        db_path=index_file,
        collection_name=collection_name,
        enable_dense=True,
        enable_sparse=False,
        dense_dim=dense_provider.get_dimension(),
        nlist=16384
    )

    # Create index
    index = VerbatimIndex(
        vector_store=vector_store, dense_provider=dense_provider
    )

    # Create RAG
    rag = VerbatimRAG(index, llm_client=llm_client)

    logger.info("VerbatimRAG initialized successfully")
    return index, rag


@st.cache_resource
def load_reranker(device: str) -> CrossEncoder:
    """Load the CrossEncoder reranker model with caching."""
    logger.info("Loading reranker model")
    model = CrossEncoder("jinaai/jina-reranker-v3", device=device)
    logger.info("Reranker model loaded successfully")
    return model


def rerank_results(
    query: str,
    results: List[SearchResult],
    reranker: CrossEncoder,
) -> List[tuple[SearchResult, float]]:
    """Rerank search results using CrossEncoder."""
    if not results:
        return []

    # Extract passages from results
    passages = [result.text for result in results]

    # Rank using the reranker
    ranked = reranker.rank(query, passages, return_documents=True, batch_size=1)

    # Create list of (SearchResult, score) tuples, sorted by reranker score
    reranked_results = [
        (results[rank["corpus_id"]], rank["score"])
        for rank in ranked
    ]

    return reranked_results


def main():
    st.set_page_config(
        page_title="Verbatim Index Tester",
        layout="wide",
    )

    st.title("Verbatim Index Search Tester")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        index_file = st.text_input(
            "Index File Path",
            value=os.getenv("INDEX_FILE", "index.db"),
            help="Path to the Milvus index database"
        )

        collection_name = st.text_input(
            "Collection Name",
            value=os.getenv("COLLECTION_NAME", "acl"),
            help="Name of the vector store collection"
        )

        device = st.selectbox(
            "Device",
            options=["cpu", "cuda"],
            index=0,
            help="Device to use for embeddings"
        )

        top_k = st.slider(
            "Number of Top Results",
            min_value=1,
            max_value=100,
            value=5,
            help="How many search results to display"
        )


   # Initialize index
    try:
        index, rag = initialize_rag(
            index_file=index_file,
            collection_name=collection_name,
            device=device,
        )
        st.success("Index loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load index: {str(e)}")
        logger.error(f"Initialization error: {e}", exc_info=True)
        st.stop()

    # Clear reranked results when query changes
    query_key = "search_query_state"
    prev_query = st.session_state.get(query_key, "")

    # Search interface
    st.header("Search")

    # Query input
    query = st.text_input(
        "Enter your search query (optional):",
        placeholder="e.g., What is transformer architecture?",
        help="Leave empty to search by filter only"
    )

    # Filter input with help text
    with st.expander("Filter (Optional)", expanded=False):
        st.info(
            "Filter documents using Milvus syntax:\n\n"
            "**Exact matching:**\n"
            "- `metadata[\"document_id\"] == \"doc_id\"` - Get all chunks from a document\n"
            "- `metadata[\"title\"] == \"BERT\"` - Exact title match\n\n"
            "**Contains/Pattern matching (LIKE):**\n"
            "- `metadata[\"title\"] LIKE \"%BERT%\"` - Title contains \"BERT\"\n"
            "- `metadata[\"title\"] LIKE \"BERT%\"` - Title starts with \"BERT\"\n"
            "- `metadata[\"title\"] LIKE \"%2025%\"` - Title contains \"2025\"\n\n"
            "**Comparisons:**\n"
            "- `metadata[\"year\"] > 2020` - Papers after 2020\n"
            "- `metadata[\"year\"] == 2025` - Specific year\n\n"
            "Leave empty to skip filtering."
        )
        milvus_filter = st.text_area(
            "Filter expression:",
            placeholder="e.g., metadata[\"title\"] LIKE \"%BERT%\"",
            height=100,
            help="Optional filter"
        )

    # Determine which query to use
    effective_query = query if query else None
    use_vector_search = effective_query is not None and effective_query.strip() != ""
    use_filter = milvus_filter.strip() != ""

    # Search button
    search_button = st.button("Search", use_container_width=True)

    if search_button:
        if not (use_vector_search or use_filter):
            st.warning("Please enter a search query or filter expression")
        else:
            # Clear previous results and reranked results
            st.session_state.search_results = None
            st.session_state.reranked_results = None
            st.session_state.rerank_enabled = False
            st.session_state[query_key] = effective_query

            with st.spinner("Searching..."):
                try:
                    search_params = {"nprobe": 128000}

                    if use_vector_search and use_filter:
                        info_str = f"Vector search for '{effective_query}' + Filter: {milvus_filter}"
                    elif use_vector_search:
                        info_str = f"Vector search for '{effective_query}' with top_k={top_k}"
                    elif use_filter:
                        info_str = f"Filter-only query: {milvus_filter}"
                    st.write(info_str)

                    # Call index.query
                    results = index.query(
                        text=effective_query if use_vector_search else None,
                        k=top_k,
                        search_params=search_params if use_vector_search else None,
                        filter=milvus_filter.strip() if use_filter else None
                    )

                    st.session_state.search_results = results
                    st.success(f"Found {len(results)} results")

                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
                    logger.error(f"Search error: {e}", exc_info=True)

    if st.session_state.get("search_results"):
        results = st.session_state.search_results

        col1, col2 = st.columns([0.2, 0.8])
        with col1:
            enable_rerank = st.checkbox(
                "Rerank Results",
                value=st.session_state.get("rerank_enabled", False),
                key="rerank_enabled"
            )

        if enable_rerank:
            with st.spinner("Reranking results..."):
                try:
                    reranker = load_reranker(device)
                    st.session_state.reranked_results = rerank_results(query, results, reranker)
                    st.success("Results reranked!")
                except Exception as e:
                    st.error(f"Reranking failed: {str(e)}")
                    logger.error(f"Reranking error: {e}", exc_info=True)

        if st.session_state.get("reranked_results"):
            st.subheader(f"Reranked Results (Top {min(len(st.session_state.reranked_results), top_k)})")
            display_results = st.session_state.reranked_results

            for idx, (result, rerank_score) in enumerate(display_results, 1):
                with st.container(border=True):
                    col1, col2, col3 = st.columns([0.7, 0.15, 0.15])

                    with col1:
                        st.markdown(f"#### Result #{idx}")
                    with col2:
                        st.metric("Vector Score", f"{result.score:.4f}")
                    with col3:
                        st.metric("Rerank Score", f"{rerank_score:.4f}")

                    st.write(result.text)

                    if result.enhanced_text:
                        with st.expander("📝 Enhanced Text"):
                            st.write(result.enhanced_text)

                    if result.metadata:
                        st.caption("**Metadata:** " + " | ".join(
                            [f"{k}: {v}" for k, v in result.metadata.items()]
                        ))

                    st.caption(f"ID: {result.id}")
        else:
            st.subheader(f"Top {min(len(results), top_k)} Results")

            for idx, result in enumerate(results, 1):
                with st.container(border=True):
                    col1, col2 = st.columns([0.85, 0.15])

                    with col1:
                        st.markdown(f"#### Result #{idx}")
                    with col2:
                        st.metric("Score", f"{result.score:.4f}")

                    # Result content
                    st.write(result.text)

                    # Enhanced text (if available)
                    if result.enhanced_text:
                        with st.expander("📝 Enhanced Text"):
                            st.write(result.enhanced_text)

                    # Metadata
                    if result.metadata:
                        st.caption("**Metadata:** " + " | ".join(
                            [f"{k}: {v}" for k, v in result.metadata.items()]
                        ))

                    # Result ID
                    st.caption(f"ID: {result.id}")

if __name__ == "__main__":
    main()
