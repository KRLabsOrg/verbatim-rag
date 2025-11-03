"""
Simple document processing pipeline following the pattern in create_sample_index.py
"""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path

try:
    from docling.document_converter import DocumentConverter

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

from ..document import Document, Chunk, ProcessedChunk, DocumentType, ChunkType
from ..chunker_providers import ChunkerProvider, MarkdownChunkerProvider


class DocumentProcessor:
    """
    Simple document processor using docling + chunker providers.

    Uses chunker_providers system for all text chunking operations.
    """

    def __init__(self, chunker_provider: Optional[ChunkerProvider] = None):
        if not DOCLING_AVAILABLE:
            raise ImportError("docling is required. Install with: pip install docling")

        self.converter = DocumentConverter()
        self.chunker_provider = chunker_provider or MarkdownChunkerProvider()

    def process_url(
        self, url: str, title: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """
        Process a document from URL (like PDF).

        Args:
            url: URL to the document
            title: Title of the document
            metadata: Optional metadata

        Returns:
            Processed Document with chunks
        """
        # Convert document using docling
        result = self.converter.convert(url)
        content_md = result.document.export_to_markdown()

        # Create Document
        document = Document(
            title=title,
            source=url,
            content_type=DocumentType.PDF,
            raw_content=content_md,
            metadata=metadata or {},
        )

        # Use chunker provider for structural chunking
        chunk_tuples = self.chunker_provider.chunk(content_md)

        # Process each chunk
        for i, (raw_text, struct_enhanced) in enumerate(chunk_tuples):
            # Add document metadata to enhanced content
            enhanced_content = self._add_document_metadata(struct_enhanced, document)

            # Create basic Chunk
            doc_chunk = Chunk(
                document_id=document.id,
                content=raw_text,
                chunk_number=i,
                chunk_type=ChunkType.PARAGRAPH,
            )

            # Create ProcessedChunk
            processed_chunk = ProcessedChunk(
                chunk_id=doc_chunk.id,
                enhanced_content=enhanced_content,
            )

            # Add to document
            doc_chunk.add_processed_chunk(processed_chunk)
            document.add_chunk(doc_chunk)

        return document

    def process_file(
        self,
        file_path: Union[str, Path],
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Process a local file.

        Args:
            file_path: Path to the file
            title: Optional title (defaults to filename)
            metadata: Optional metadata

        Returns:
            Processed Document with chunks
        """
        file_path = Path(file_path)

        # Convert document using docling
        result = self.converter.convert(file_path)
        content_md = result.document.export_to_markdown()

        # Create Document
        document = Document(
            title=title or file_path.stem,
            source=str(file_path),
            content_type=self._get_document_type(file_path.suffix),
            raw_content=content_md,
            metadata=metadata or {},
        )

        # Use chunker provider for structural chunking
        chunk_tuples = self.chunker_provider.chunk(content_md)

        # Process each chunk
        for i, (raw_text, struct_enhanced) in enumerate(chunk_tuples):
            # Add document metadata to enhanced content
            enhanced_content = self._add_document_metadata(struct_enhanced, document)

            # Create basic Chunk
            doc_chunk = Chunk(
                document_id=document.id,
                content=raw_text,
                chunk_number=i,
                chunk_type=ChunkType.PARAGRAPH,
            )

            # Create ProcessedChunk
            processed_chunk = ProcessedChunk(
                chunk_id=doc_chunk.id,
                enhanced_content=enhanced_content,
            )

            # Add to document
            doc_chunk.add_processed_chunk(processed_chunk)
            document.add_chunk(doc_chunk)

        return document

    def process_directory(
        self, directory_path: Union[str, Path], recursive: bool = True
    ) -> List[Document]:
        """
        Process all supported files in a directory.

        Args:
            directory_path: Path to directory
            recursive: Whether to search subdirectories

        Returns:
            List of processed Documents
        """
        directory_path = Path(directory_path)
        documents = []

        # Supported extensions
        supported_exts = {".pdf", ".docx", ".html", ".htm", ".txt", ".md"}

        if recursive:
            files = [
                f
                for f in directory_path.rglob("*")
                if f.suffix.lower() in supported_exts
            ]
        else:
            files = [
                f
                for f in directory_path.glob("*")
                if f.suffix.lower() in supported_exts
            ]

        for file_path in files:
            try:
                document = self.process_file(
                    file_path, metadata={"directory": str(directory_path)}
                )
                documents.append(document)
            except Exception as e:
                print(f"Warning: Failed to process {file_path}: {e}")
                continue

        return documents

    def extract_content_from_url(self, url: str) -> str:
        """
        Extract just the text content from a URL without creating Document/chunks.

        Args:
            url: The URL to process

        Returns:
            Raw text content as markdown string
        """
        result = self.converter.convert(url)
        return result.document.export_to_markdown()

    def extract_content_from_file(self, file_path: Union[str, Path]) -> str:
        """
        Extract just the text content from a file without creating Document/chunks.

        Args:
            file_path: Path to the file

        Returns:
            Raw text content as markdown string
        """
        file_path = Path(file_path)
        result = self.converter.convert(file_path)
        return result.document.export_to_markdown()

    def _get_document_type(self, extension: str) -> DocumentType:
        """Map file extension to DocumentType."""
        extension = extension.lower()
        if extension == ".pdf":
            return DocumentType.PDF
        elif extension in [".html", ".htm"]:
            return DocumentType.HTML
        elif extension == ".txt":
            return DocumentType.TXT
        elif extension == ".md":
            return DocumentType.MARKDOWN
        else:
            return DocumentType.UNKNOWN

    def _add_document_metadata(self, text: str, doc: Document) -> str:
        """Add document metadata footer to enhanced text."""
        parts = [text, "", "---"]
        parts.append(f"Document: {doc.title or 'Unknown'}")
        parts.append(f"Source: {doc.source or 'Unknown'}")

        if doc.metadata:
            skip_keys = {"user_id", "dataset_id", "userId"}
            for key, value in doc.metadata.items():
                if key not in skip_keys:
                    formatted_key = key.replace("_", " ").title()
                    parts.append(f"{formatted_key}: {value}")

        return "\n".join(parts)

    @classmethod
    def for_embeddings(cls, chunk_size: int = 512, overlap: int = 50):
        """Create processor optimized for embedding generation."""
        from ..chunker_providers import ChonkieChunkerProvider

        chunker = ChonkieChunkerProvider(
            recipe="default", chunk_size=chunk_size, chunk_overlap=overlap
        )
        return cls(chunker)

    @classmethod
    def for_qa(cls, sentence_chunks: int = 3, sentence_overlap: int = 1):
        """Create processor optimized for Q&A tasks."""
        from ..chunker_providers import ChonkieChunkerProvider

        chunker = ChonkieChunkerProvider(
            recipe="default", chunk_size=sentence_chunks, chunk_overlap=sentence_overlap
        )
        return cls(chunker)

    @classmethod
    def semantic(cls, chunk_size: int = 512):
        """Create processor with semantic chunking (uses ChonkieChunkerProvider)."""
        from ..chunker_providers import ChonkieChunkerProvider

        chunker = ChonkieChunkerProvider(
            recipe="default", chunk_size=chunk_size, chunk_overlap=50
        )
        return cls(chunker)

    @classmethod
    def markdown_recursive(
        cls,
        split_levels: tuple = (1, 2, 3, 4),
        include_preamble: bool = True,
    ):
        """Create processor with markdown hierarchical chunking."""
        from ..chunker_providers import MarkdownChunkerProvider

        chunker = MarkdownChunkerProvider(
            split_levels=split_levels, include_preamble=include_preamble
        )
        return cls(chunker)
