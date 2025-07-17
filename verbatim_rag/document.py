"""
Document class for the Verbatim RAG system.
"""

from typing import Any


class Document:
    """
    A simple document class that stores text content, a stable id, and associated metadata.
    """

    def __init__(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ):
        """
        :param content:  The text content of the document
        :param metadata: Optional metadata associated with the document
        :param     id:   A unique identifier for this document.  
                         If None, will look for metadata["id"], or remain None.
        """
        self.content = content
        # make a copy so we don’t mutate the user’s dict
        self.metadata = metadata.copy() if metadata else {}

        # determine self.id: explicit param > metadata["id"] > None
        if doc_id is not None:
            self.id = doc_id
        else:
            # fallback to any existing metadata["id"]
            self.id = self.metadata.get("id")

        # if we do have an id, always ensure it’s in metadata
        if self.id is not None:
            self.metadata["id"] = self.id
