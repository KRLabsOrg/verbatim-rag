"""
Document loader utility for the Verbatim RAG system.
"""

import csv
import os

import pandas as pd

from verbatim_rag.document import Document


class DocumentLoader:
    """
    A utility for loading documents from various file formats.
    """

    @staticmethod
    def load_text(file_path: str) -> Document:
        """
        Load a text file.

        :param file_path: Path to the text file
        :return: Document object
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return Document(content=content, metadata={"source": file_path, "type": "text"})

    @staticmethod
    def load_csv(
        file_path: str, content_columns: list[str] | None = None
    ) -> list[Document]:
        """
        Load a CSV file.

        :param file_path: Path to the CSV file
        :param content_columns: Columns to include in the document content (if None, include all)
        :return: List of Document objects, one per row
        """
        documents = []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                if content_columns:
                    content_parts = []
                    for col in content_columns:
                        if col in row:
                            content_parts.append(f"{col}: {row[col]}")
                    content = "\n".join(content_parts)
                else:
                    content = "\n".join([f"{k}: {v}" for k, v in row.items()])

                # Create a document for this row
                documents.append(
                    Document(
                        content=content,
                        metadata={"source": file_path, "type": "csv", "row": i},
                    )
                )

        return documents

    @staticmethod
    def load_dataframe(
        df: pd.DataFrame, content_columns: list[str] | None = None
    ) -> list[Document]:
        """
        Load a pandas DataFrame.

        :param df: The DataFrame to load
        :param content_columns: Columns to include in the document content (if None, include all)
        :return: List of Document objects, one per row
        """
        documents = []

        for i, row in df.iterrows():
            if content_columns:
                content_parts = []
                for col in content_columns:
                    if col in df.columns:
                        content_parts.append(f"{col}: {row[col]}")
                content = "\n".join(content_parts)
            else:
                content = "\n".join([f"{k}: {v}" for k, v in row.items()])

            # Create a document for this row
            documents.append(
                Document(
                    content=content,
                    metadata={"source": "dataframe", "type": "dataframe", "row": i},
                )
            )

        return documents

    @classmethod
    def load_file(cls, file_path: str) -> list[Document]:
        """
        Load a file based on its extension.

        :param file_path: Path to the file
        :return: List of Document objects
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext == ".csv":
            return cls.load_csv(file_path)
        elif ext in [".txt", ".md", ".html", ".xml", ".json"]:
            return [cls.load_text(file_path)]
        else:
            # Default to text for unknown extensions
            return [cls.load_text(file_path)]

    @classmethod
    def load_directory(cls, directory: str, recursive: bool = True) -> list[Document]:
        """
        Load all files in a directory.

        :param directory: Path to the directory
        :param recursive: Whether to recursively load files in subdirectories
        :return: List of Document objects
        """
        documents = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    documents.extend(cls.load_file(file_path))
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

            if not recursive:
                break

        return documents
