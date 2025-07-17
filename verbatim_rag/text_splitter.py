"""
Text splitter utility for the Verbatim RAG system.
"""

import re
from chonkie import chunk
from typing import List

from verbatim_rag.document import Document

class ChonkieSplitter:
    def __init__(self, max_tokens=512, stride=64):
        self.max_tokens = max_tokens
        self.stride = stride

    def split(self, docs: List[Document]) -> List[Document]:
        all_chunks = []

        for doc in docs:
            chunks = chunk(
                doc.content,
                max_tokens=self.max_tokens,
                stride=self.stride,
                include_overflow=False
            )

            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{doc.id}-chonkie_{i}"
                new_metadata = doc.metadata.copy()
                new_metadata["chunk_index"] = i
                new_doc = Document(content=chunk_text, doc_id=chunk_id, metadata=new_metadata)
                all_chunks.append(new_doc)

        return all_chunks

class TextSplitter:
    """
    A simple text splitter that chunks documents into smaller pieces.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the TextSplitter.

        :param chunk_size: Maximum size of each chunk in characters
        :param chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        """
        Split a text into chunks.

        :param text: The text to split
        :return: List of text chunks
        """
        # If the text is shorter than the chunk size, return it as is
        if len(text) <= self.chunk_size:
            return [text]

        # Split the text into chunks
        chunks = []
        start = 0
        while start < len(text):
            # Find the end of the chunk
            end = start + self.chunk_size

            # If we're not at the end of the text, try to find a good splitting point
            if end < len(text):
                # Try to split at paragraph, then sentence, then word boundary
                paragraph_match = re.search(r"\n\s*\n", text[end - 100 : end + 100])
                if paragraph_match:
                    end = end - 100 + paragraph_match.start() + 1
                else:
                    sentence_match = re.search(r"[.!?]\s+", text[end - 50 : end + 50])
                    if sentence_match:
                        end = end - 50 + sentence_match.start() + 1
                    else:
                        word_match = re.search(r"\s+", text[end - 20 : end + 20])
                        if word_match:
                            end = end - 20 + word_match.start() + 1

            # Add the chunk to the list
            chunks.append(text[start:end].strip())

            # Move the start position for the next chunk
            start = end - self.chunk_overlap

            # Make sure we're not stuck in a loop
            if start >= len(text) or start <= 0:
                break

        return chunks

    def split_document(self, document: Document) -> list[Document]:
        """
        Split a document into chunks.

        :param document: The document to split
        :return: List of document chunks
        """
        chunks = self.split_text(document.content)

        # Create a new document for each chunk
        documents = []
        for i, chunk in enumerate(chunks):
            # Copy the metadata and add chunk information
            metadata = document.metadata.copy()
            metadata["chunk"] = i
            metadata["chunk_of"] = len(chunks)

            # Create a new document
            documents.append(Document(chunk, metadata))

        return documents

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split multiple documents into chunks.

        :param documents: The documents to split
        :return: List of document chunks
        """
        chunks = []
        for document in documents:
            chunks.extend(self.split_document(document))

        return chunks
