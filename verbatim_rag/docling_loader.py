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
            "--output", output_path,
            "--to", "json"
        ]
    
        logger.info(f"Running Docling: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    @staticmethod
    def load_docling_output(json_dir: str) -> List[Document]:
        """
        Load Docling JSON output (with root.texts[].text structure) and convert to Document objects.
        """
        documents = []
    
        for filename in os.listdir(json_dir):
            if not filename.endswith(".json"):
                continue
    
            full_path = os.path.join(json_dir, filename)
            logger.info(f"Reading: {full_path}")
    
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            texts = data.get("texts", [])
            doc_title = data.get("name", filename)
            logger.info(f"Found {len(texts)} chunks in '{filename}'")
            
            
            for i, entry in enumerate(texts):
                content = entry.get("text", "")
                if not content:
                    continue  # skip empty chunks
    
                doc_id = f"{doc_title}-chunk_{i}"
                metadata = {
                    "source": filename,
                    "section": entry.get("label", ""),
                    "title": doc_title,
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
