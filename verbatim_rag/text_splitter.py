"""
Text splitter utility for the Verbatim RAG system.
"""

import re
from chonkie import SentenceChunker
from typing import List

from chonkie import RecursiveChunker
from verbatim_rag.document import Document


class ChonkieSplitter:
    
    def __init__(self, model_name="answerdotai/ModernBERT-base", max_tokens=512):
        self.tokenizer_name = model_name
        self.chunk_size     = max_tokens

        self.chunker = RecursiveChunker(
            tokenizer_or_token_counter = model_name,
            chunk_size     = self.chunk_size,
        )

    def split(self, texts: list[str]) -> list[str]:
        # Merge list of texts into one string with paragraph breaks
        merged_text = "\n\n".join(texts)
        chunks = self.chunker.chunk(merged_text)
        return [chunk.text for chunk in chunks]



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
