import os
import json
import subprocess
from typing import List
from verbatim_rag.document import Document
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import re
from typing import List

HEADER_RX = re.compile(r"^(\d+(?:\.\d+)*)\s+(.*)$")

import re
from typing import List
from verbatim_rag.document import Document

import re
from typing import List
from verbatim_rag.document import Document

HEADER_RX = re.compile(r"^(\d+(?:\.\d+)*)\s+(.*)$")

def build_documents_from_sections(
    texts: List[dict],
    doc_title: str,
    filename: str
) -> List[Document]:
    """
    1) Lines starting with 'N Title' (no dot) become level‑1 headers
    2) Lines starting with 'N.M Subtitle' become level‑2 headers
    3) Everything before '1 Introduction' → section='Abstract'
    4) Detect literal 'References' as its own top‑level section
    5) Body chunks get:
       section = current_level1
       parent  = current_level2 (if any)
    """
    documents       = []
    section_prefix  = 0
    section_counter = 0

    current_level1  = "Abstract"
    current_level2  = None
    seen_intro      = False

    for entry in texts:
        text = (entry.get("text") or "").strip()
        if not text:
            continue

        # 0) Detect unnumbered "References" header
        if text.lower() == "references":
            current_level1  = "References"
            current_level2  = None
            section_prefix += 1
            section_counter = 0
            continue

        # 1) Detect numbered headers
        m = HEADER_RX.match(text)
        if m:
            num, title = m.groups()
            if "." not in num:
                # Top‑level: "3 Methods"
                current_level1 = f"{num} {title}"
                current_level2 = None
                section_prefix += 1
                section_counter = 0
                if num == "1":
                    seen_intro = True
            else:
                # Sub‑section: "3.2 Evidence Extraction"
                current_level2 = f"{num} {title}"
            continue

        # 2) Pre‑Introduction stays in Abstract
        if not seen_intro:
            current_level1 = "Abstract"

        # 3) Emit a body/text chunk
        doc_id = f"{doc_title}-sec{section_prefix}_chunk{section_counter}"
        meta = {
            "source":  filename,
            "title":   doc_title,
            "id":      doc_id,
            "section": current_level1,
        }
        if current_level2:
            meta["parent"] = current_level2

        documents.append(Document(content=text, doc_id=doc_id, metadata=meta))
        section_counter += 1

    return documents







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
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Docling failed: {e}")
            raise

    @staticmethod
    def load_docling_output(json_dir: str) -> List[Document]:
        logger.info(f"load_docling_output({json_dir})")
        documents = []

        for filename in os.listdir(json_dir):
            if not filename.endswith(".json"):
                continue

            full_path = os.path.join(json_dir, filename)
            logger.info(f"Reading: {full_path}")

            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to parse {full_path}: {e}")
                continue

            texts = data.get("texts", [])
            doc_title = data.get("name", filename)
            logger.info(f"Found {len(texts)} chunks in '{filename}'")

            documents.extend(
                build_documents_from_sections(texts, doc_title, filename)
            )

        return documents

    @classmethod
    def load_with_docling(cls, input_path: str, output_path: str) -> List[Document]:
        """
        Full pipeline: run Docling, load JSON output.

        :param input_path: Path to files or directory
        :param output_path: Temp path where Docling will store JSON output
        :return: List of Document objects
        """
        logger.info(f"load_with_docling({input_path}, {output_path})")
        cls.run_docling(input_path, output_path)
        return cls.load_docling_output(output_path)
