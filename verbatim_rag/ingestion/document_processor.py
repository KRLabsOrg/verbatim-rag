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
from ..chunking import ChunkingService, ChunkingConfig, ChunkingStrategy


class DocumentProcessor:
    """
    Simple document processor using docling + ChunkingService.

    Uses centralized ChunkingService for all text chunking operations.
    """

    def __init__(self, chunking_config: Optional[ChunkingConfig] = None):
        if not DOCLING_AVAILABLE:
            raise ImportError("docling is required. Install with: pip install docling")

        self.converter = DocumentConverter()
        self.chunking_service = ChunkingService(chunking_config)

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

        # Use ChunkingService for all chunking
        enhanced_chunks = self.chunking_service.chunk_document_enhanced(document)

        # Process each chunk
        for i, (original_text, enhanced_content) in enumerate(enhanced_chunks):
            # Create basic Chunk
            doc_chunk = Chunk(
                document_id=document.id,
                content=original_text,
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

        # Use ChunkingService for all chunking
        enhanced_chunks = self.chunking_service.chunk_document_enhanced(document)

        # Process each chunk
        for i, (original_text, enhanced_content) in enumerate(enhanced_chunks):
            # Create basic Chunk
            doc_chunk = Chunk(
                document_id=document.id,
                content=original_text,
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

    @classmethod
    def for_embeddings(cls, chunk_size: int = 512, overlap: int = 50):
        """Create processor optimized for embedding generation."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.TOKEN,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        return cls(config)

    @classmethod
    def for_qa(cls, sentence_chunks: int = 3, sentence_overlap: int = 1):
        """Create processor optimized for Q&A tasks."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SENTENCE,
            chunk_size=sentence_chunks,
            chunk_overlap=sentence_overlap,
        )
        return cls(config)

    @classmethod
    def semantic(cls, chunk_size: int = 512, merge_threshold: float = 0.7):
        """Create processor with semantic chunking."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SDPM,
            chunk_size=chunk_size,
            merge_threshold=merge_threshold,
            split_threshold=0.3,
        )
        return cls(config)

    @classmethod
    def markdown_recursive(
        cls,
        lang: str = "en",
        preserve_headings: bool = False,
        include_metadata: bool = True,
    ):
        """Create processor with recursive markdown chunking."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            recipe="markdown",
            lang=lang,
            preserve_headings=preserve_headings,
            include_metadata=include_metadata,
        )
        return cls(config)
