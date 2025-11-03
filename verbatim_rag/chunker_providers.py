"""
Chunker providers for text chunking.

Follows the same provider pattern as embedding_providers and vector_stores.
All chunkers implement the ChunkerProvider interface.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import re


class ChunkerProvider(ABC):
    """Abstract base class for text chunkers."""

    @abstractmethod
    def chunk(self, text: str) -> List[Tuple[str, str]]:
        """
        Chunk text into (raw_text, enhanced_text) tuples.

        Enhanced text includes structural context (headings, etc.)
        but NOT document metadata (that's added by Index).

        Args:
            text: Raw text to chunk

        Returns:
            List of (raw_chunk, struct_enhanced_chunk) tuples where:
            - raw_chunk: Original text for extraction/display
            - struct_enhanced_chunk: Text with structural context (headings) for embedding
        """
        pass


class MarkdownChunkerProvider(ChunkerProvider):
    """
    Markdown chunker with ancestor heading injection.

    This implementation preserves the exact markdown structure while adding
    ancestor headings to provide hierarchical context for each chunk.
    """

    def __init__(
        self,
        split_levels: tuple = (1, 2, 3, 4),
        include_preamble: bool = True,
    ):
        """
        Initialize markdown chunker.

        Args:
            split_levels: Which header levels become chunks (default: H1-H4)
            include_preamble: Include text before first header as chunk
        """
        self.split_levels = split_levels
        self.include_preamble = include_preamble

    def chunk(self, text: str) -> List[Tuple[str, str]]:
        """Chunk markdown with ancestor heading injection."""
        chunks = self._md_chunker_with_ancestor_injection(
            text, self.split_levels, self.include_preamble
        )
        return [(c["raw_chunk"], c["enhanced_chunk"]) for c in chunks]

    def _md_chunker_with_ancestor_injection(
        self,
        md: str,
        split_levels: tuple = (1, 2, 3, 4),
        include_preamble: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Chunk markdown with ancestor heading injection.

        Returns list of chunks where each has:
          - raw_chunk: exact substring from original md (lossless)
          - enhanced_chunk: all ancestor header lines + blank line + raw_chunk
          - header_path: ['Title H1', 'Title H2', ... , 'This chunk title']

        Concatenating all raw_chunk in order reproduces the original file.
        """
        HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", flags=re.MULTILINE)

        n = len(md)
        starts = self._line_starts(md)

        # Gather headers with line numbers and their exact header lines
        headers = []
        for m in HEADER_RE.finditer(md):
            pos = m.start()
            ln = 0
            while ln + 1 < len(starts) and starts[ln + 1] <= pos:
                ln += 1
            # exact header line text (without trailing newline)
            line_end = md.find("\n", starts[ln])
            if line_end == -1:
                line_end = n
            exact_line = md[starts[ln] : line_end]
            headers.append(
                {
                    "level": len(m.group(1)),
                    "title": m.group(2).strip(),
                    "line_start": ln,
                    "exact_line": exact_line,
                }
            )

        # If no headers, return single chunk
        if not headers:
            return [
                {
                    "level": 0,
                    "title": "Document",
                    "header_path": ["Document"],
                    "start": 0,
                    "end": n,
                    "raw_chunk": md,
                    "enhanced_chunk": md,
                }
            ]

        # Compute the line-end boundary for each header block
        for i in range(len(headers)):
            headers[i]["line_end"] = (
                headers[i + 1]["line_start"] if i + 1 < len(headers) else len(starts)
            )

        chunks: List[Dict[str, Any]] = []

        # Optional preamble (before first header)
        if include_preamble and headers[0]["line_start"] > 0:
            s, e = self._span_of_lines(starts, 0, headers[0]["line_start"], n)
            raw = md[s:e]
            chunks.append(
                {
                    "level": 0,
                    "title": "Preamble",
                    "header_path": ["Preamble"],
                    "start": s,
                    "end": e,
                    "raw_chunk": raw,
                    "enhanced_chunk": raw,
                }
            )

        # Walk headers, maintain stack of ancestors
        stack: List[Dict[str, Any]] = []
        for h in headers:
            # pop until parent is lower level
            while stack and stack[-1]["level"] >= h["level"]:
                stack.pop()
            # current header becomes top of stack
            stack.append(h)

            # only chunk chosen levels
            if h["level"] not in split_levels:
                continue

            # compute exact raw span for this header block
            s, e = self._span_of_lines(starts, h["line_start"], h["line_end"], n)
            raw = md[s:e]

            # ancestors = every header in stack except the last (current one)
            ancestors = stack[:-1]
            ancestor_lines = [a["exact_line"] for a in ancestors]

            # enhanced_chunk = all ancestor header lines + blank line + raw
            if ancestor_lines:
                prefix = "\n".join(ancestor_lines) + "\n\n"
                enhanced = prefix + raw
            else:
                enhanced = raw

            chunks.append(
                {
                    "level": h["level"],
                    "title": h["title"],
                    "header_path": [x["title"] for x in stack],
                    "start": s,
                    "end": e,
                    "raw_chunk": raw,
                    "enhanced_chunk": enhanced,
                }
            )

        return chunks

    def _line_starts(self, s: str) -> List[int]:
        """Get starting positions of all lines."""
        idxs = [0]
        for i, ch in enumerate(s):
            if ch == "\n":
                idxs.append(i + 1)
        return idxs

    def _span_of_lines(
        self, starts: List[int], a: int, b_excl: int, n: int
    ) -> Tuple[int, int]:
        """Get character span from line a to line b (exclusive)."""
        start = starts[a]
        end = starts[b_excl] if b_excl < len(starts) else n
        return start, end


class ChonkieChunkerProvider(ChunkerProvider):
    """
    Chonkie-based chunking with optional hierarchy tracking.

    Uses the chonkie library for intelligent text chunking with support
    for various strategies (recursive, token, sentence, etc.).
    """

    def __init__(
        self,
        recipe: str = "markdown",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        preserve_hierarchy: bool = True,
    ):
        """
        Initialize chonkie-based chunker.

        Args:
            recipe: Chonkie recipe to use (e.g., "markdown", "default")
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            preserve_hierarchy: Track heading hierarchy for markdown
        """
        self.recipe = recipe
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_hierarchy = preserve_hierarchy
        self._chunker = self._create_chunker()

    def _create_chunker(self):
        """Create chonkie chunker with optional hierarchy tracking."""
        try:
            import chonkie
        except ImportError as e:
            raise ImportError(
                "ChonkieChunkerProvider requires chonkie. "
                "Install with: pip install chonkie"
            ) from e

        chunker = chonkie.RecursiveChunker.from_recipe(self.recipe)

        # Optionally wrap with hierarchy tracking for markdown
        if self.preserve_hierarchy and self.recipe == "markdown":
            try:
                from verbatim_rag.ingestion.hierarchical_chunker import (
                    HierarchicalWrapper,
                )

                chunker = HierarchicalWrapper(chunker)
            except ImportError:
                pass  # Use without hierarchy tracking

        return chunker

    def chunk(self, text: str) -> List[Tuple[str, str]]:
        """Chunk using chonkie with optional hierarchy."""
        chunks = self._chunker(text)

        results = []
        for chunk in chunks:
            raw_text = chunk.text

            # Build enhanced text with heading hierarchy if available
            if hasattr(chunk, "heading_path") and chunk.heading_path:
                enhanced = "\n".join(chunk.heading_path) + "\n\n" + raw_text
            else:
                enhanced = raw_text

            results.append((raw_text, enhanced))

        return results


class SimpleChunkerProvider(ChunkerProvider):
    """
    Simple fixed-size chunking with no dependencies.

    Uses a basic sliding window approach for chunking text.
    Useful as a fallback when no specialized chunker is available.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize simple chunker.

        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> List[Tuple[str, str]]:
        """Simple sliding window chunking."""
        if not text.strip():
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                # For simple chunker, raw and enhanced are the same
                chunks.append((chunk_text, chunk_text))

            start += self.chunk_size - self.chunk_overlap

            # Prevent infinite loop
            if start <= 0 or self.chunk_size <= self.chunk_overlap:
                break

        return chunks
