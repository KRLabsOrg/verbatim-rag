"""
Index class for the Verbatim RAG system.
"""

import os
import pickle

import faiss
import numpy as np

from verbatim_rag.document import Document


class VerbatimIndex:
    """
    A vector index for document retrieval using FAISS.
    """

    def __init__(self, embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize the VerbatimIndex.

        :param embedding_model: The embedding model to use.
                                Can be an OpenAI model ID or any Hugging-Face checkpoint.
        """
        self.embedding_model = embedding_model
        self.documents = []
        self.index = None
        self.document_ids = []

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts, using either a local HF model or the OpenAI API.

        :param texts: List of text strings to embed
        :return: Numpy array of embeddings, shape (len(texts), embedding_dim)
        """
        # If the model string is not an OpenAI embedding ID, run locally via HF
        if not self.embedding_model.startswith("text-embedding-"):
            from sentence_transformers import SentenceTransformer

            # Load the HF model and encode
            st = SentenceTransformer(self.embedding_model)
            hf_embeddings = st.encode(texts, show_progress_bar=True)
            return np.array(hf_embeddings, dtype=np.float32)

        # Otherwise assume it's an OpenAI model ID:
        import openai  # noqa: E402

        response = openai.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)

    def add_documents(self, documents: list[Document]) -> None:
        """
        Add documents to the index.

        :param documents: List of Document objects to add
        """
        if not documents:
            return

        texts = [doc.content for doc in documents]
        embeddings = self._get_embeddings(texts)

        start_idx = len(self.documents)
        self.documents.extend(documents)

        new_ids = list(range(start_idx, start_idx + len(documents)))
        self.document_ids.extend(new_ids)

        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
        else:
            self.index.add(embeddings)

    def search(self, query: str, k: int = 5) -> list[Document]:
        """
        Search for documents similar to the query.

        :param query: The search query
        :param k: Number of documents to retrieve
        :return: List of retrieved Document objects
        """
        if not self.index or not self.documents:
            return []

        query_embedding = self._get_embeddings([query])[0].reshape(1, -1)
        k = min(k, len(self.documents))
        distances, indices = self.index.search(query_embedding, k)

        return [self.documents[idx] for idx in indices[0]]

    def save(self, directory: str) -> None:
        """
        Save the index and documents to disk.

        :param directory: Directory to save the index in
        """
        os.makedirs(directory, exist_ok=True)

        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

        with open(os.path.join(directory, "document_ids.pkl"), "wb") as f:
            pickle.dump(self.document_ids, f)

        if self.index is not None:
            faiss.write_index(self.index, os.path.join(directory, "index.faiss"))

        with open(os.path.join(directory, "embedding_model.txt"), "w") as f:
            f.write(self.embedding_model)

    @classmethod
    def load(cls, directory: str) -> "VerbatimIndex":
        """
        Load an index from disk.

        :param directory: Directory containing the saved index
        :return: Loaded VerbatimIndex
        """
        with open(os.path.join(directory, "embedding_model.txt"), "r") as f:
            embedding_model = f.read().strip()

        index = cls(embedding_model=embedding_model)

        with open(os.path.join(directory, "documents.pkl"), "rb") as f:
            index.documents = pickle.load(f)

        with open(os.path.join(directory, "document_ids.pkl"), "rb") as f:
            index.document_ids = pickle.load(f)

        index.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        return index
