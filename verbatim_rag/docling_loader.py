import os
import json
import subprocess
from typing import List

from verbatim_rag.document import Document
import logging

logger = logging.getLogger(__name__)

class DoclingLoader:
    """
    Load documents using Docling CLI.
    Assumes Docling outputs JSON chunks from input documents.
    """

    @staticmethod
    def run_docling(input_path: str, output_path: str):
        """
        Run Docling directly on a file or directory (no subcommand).
        """
        cmd = [
            "docling",
            input_path,
            "--output", output_path
        ]
    
        logger.info(f"Running Docling: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    @staticmethod
    def load_docling_output(json_path: str) -> List[Document]:
        """
        Load Docling's JSON output and convert to internal Document format.

        :param json_path: Path to the JSON output file
        :return: List of Document objects
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = []

        for i, entry in enumerate(data):
            content = entry.get("text", "")  # or "chunk", "body", etc. depending on format
            doc_id = entry.get("id", f"docling:chunk{i}")
            metadata = {
                "source": entry.get("source", json_path),
                "section": entry.get("section", ""),
                "title": entry.get("title", ""),
                "id": doc_id
            }

            documents.append(Document(content=content, doc_id=doc_id, metadata=metadata))

        return documents

    @classmethod
    def load_with_docling(cls, input_path: str, tmp_output_path: str) -> List[Document]:
        """
        Full pipeline: run Docling, load JSON output.

        :param input_path: Path to files or directory
        :param tmp_output_path: Temp path where Docling will store JSON output
        :return: List of Document objects
        """
        cls.run_docling(input_path, tmp_output_path)
        return cls.load_docling_output(tmp_output_path)
