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

try:
    import chonkie

    CHONKIE_AVAILABLE = True
except ImportError:
    CHONKIE_AVAILABLE = False

from ..document import Document, Chunk, ProcessedChunk, DocumentType, ChunkType


class DocumentProcessor:
    """
    Simple document processor using docling + chonkie.

    Supports different chunking strategies:
    - "recursive" (default): RecursiveChunker with recipes
    - "token": TokenChunker
    - "sentence": SentenceChunker
    - "word": WordChunker
    - "sdpm": SDPMChunker (Semantic Double-Pass Merging)
    """

    def __init__(
        self,
        chunker_type: str = "recursive",
        chunker_recipe: str = "markdown",
        lang: str = "en",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        **chunker_kwargs,
    ):
        if not DOCLING_AVAILABLE:
            raise ImportError(
                "docling is required. Install with: pip install 'verbatim-rag[document-processing]'"
            )

        if not CHONKIE_AVAILABLE:
            raise ImportError(
                "chonkie is required. Install with: pip install 'verbatim-rag[document-processing]'"
            )

        self.converter = DocumentConverter()
        self.chunker_type = chunker_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunker_kwargs = chunker_kwargs

        # Create chunker based on type
        if chunker_type == "recursive":
            self.chunker = chonkie.RecursiveChunker.from_recipe(
                chunker_recipe, lang=lang
            )
        elif chunker_type == "token":
            self.chunker = chonkie.TokenChunker(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, **chunker_kwargs
            )
        elif chunker_type == "sentence":
            self.chunker = chonkie.SentenceChunker(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, **chunker_kwargs
            )
        elif chunker_type == "word":
            self.chunker = chonkie.WordChunker(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, **chunker_kwargs
            )
        elif chunker_type == "sdpm":
            self.chunker = chonkie.SDPMChunker(
                chunk_size=chunk_size,
                merge_threshold=chunker_kwargs.get("merge_threshold", 0.7),
                split_threshold=chunker_kwargs.get("split_threshold", 0.3),
            )
        else:
            raise ValueError(
                f"Unknown chunker type: {chunker_type}. "
                f"Supported: recursive, token, sentence, word, sdpm"
            )

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

        # Chunk with chonkie
        chunks = self.chunker(content_md)

        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Create basic Chunk
            doc_chunk = Chunk(
                document_id=document.id,
                content=chunk.text,
                chunk_number=i,
                chunk_type=ChunkType.PARAGRAPH,
            )

            # Create ProcessedChunk
            processed_chunk = ProcessedChunk(
                chunk_id=doc_chunk.id,
                enhanced_content=chunk.text,
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

        # Chunk with chonkie
        chunks = self.chunker(content_md)

        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Create basic Chunk
            doc_chunk = Chunk(
                document_id=document.id,
                content=chunk.text,
                chunk_number=i,
                chunk_type=ChunkType.PARAGRAPH,
            )

            # Create ProcessedChunk
            processed_chunk = ProcessedChunk(
                chunk_id=doc_chunk.id,
                enhanced_content=chunk.text,
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

    def _get_document_type(self, extension: str) -> DocumentType:
        """Map file extension to DocumentType."""
        extension = extension.lower()
        if extension == ".pdf":
            return DocumentType.PDF
        elif extension in [".html", ".htm"]:
            return DocumentType.HTML
        elif extension in [".txt", ".md"]:
            return DocumentType.TEXT
        else:
            return DocumentType.OTHER

    @classmethod
    def for_embeddings(cls, chunk_size: int = 512, overlap: int = 50):
        """Create processor optimized for embedding generation."""
        return cls(chunker_type="token", chunk_size=chunk_size, chunk_overlap=overlap)

    @classmethod
    def for_qa(cls, sentence_chunks: int = 3, sentence_overlap: int = 1):
        """Create processor optimized for Q&A tasks."""
        return cls(
            chunker_type="sentence",
            chunk_size=sentence_chunks,
            chunk_overlap=sentence_overlap,
        )

    @classmethod
    def semantic(cls, chunk_size: int = 512, merge_threshold: float = 0.7):
        """Create processor with semantic chunking."""
        return cls(
            chunker_type="sdpm",
            chunk_size=chunk_size,
            merge_threshold=merge_threshold,
            split_threshold=0.3,
        )

    @classmethod
    def markdown_recursive(cls, lang: str = "en"):
        """Create processor with recursive markdown chunking (default)."""
        return cls(chunker_type="recursive", chunker_recipe="markdown", lang=lang)
