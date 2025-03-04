"""
Document class for the Verbatim RAG system.
"""

from typing import Any


class Document:
    """
    A simple document class that stores text content and associated metadata.
    """

    def __init__(self, content: str, metadata: dict[str, Any] | None = None):
        """
        Initialize a Document.

        :param content: The text content of the document
        :param metadata: Optional metadata associated with the document
        """
        self.content = content
        self.metadata = metadata or {}
