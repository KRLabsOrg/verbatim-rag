"""
Hierarchical wrapper that preserves heading tree structure for any chunker.

This wrapper can be applied to any chunker to add heading hierarchy tracking,
allowing us to prepend parent headings to maintain document structure context.
Primarily designed to work with RecursiveChunker using markdown recipe.
"""

import re
from typing import List, Optional
from dataclasses import dataclass

try:
    import chonkie

    CHONKIE_AVAILABLE = True
except ImportError:
    CHONKIE_AVAILABLE = False


@dataclass
class HeadingInfo:
    """Information about a heading in the document."""

    level: int  # 1-6 for h1-h6
    text: str  # Heading text without the # markers
    start_pos: int  # Character position in document


class HierarchicalChunk:
    """A chunk with hierarchical heading context."""

    def __init__(
        self,
        text: str,
        start_index: int,
        end_index: int,
        token_count: int,
        heading_path: List[str] = None,
    ):
        self.text = text
        self.start_index = start_index
        self.end_index = end_index
        self.token_count = token_count
        self.heading_path = heading_path or []  # List of ancestor headings


class HierarchicalWrapper:
    """
    Wrapper that adds heading hierarchy tracking to any chunker.

    This wrapper:
    1. Parses the markdown to identify all headings and their hierarchy
    2. Uses the provided base chunker to split the text
    3. For each chunk, determines which headings are its ancestors
    4. Returns chunks with heading_path metadata
    """

    def __init__(self, base_chunker):
        """
        Initialize the hierarchical wrapper.

        Args:
            base_chunker: Any chunker object (typically RecursiveChunker with markdown recipe)
        """
        self.base_chunker = base_chunker

    def __call__(self, text: str) -> List[HierarchicalChunk]:
        """Chunk the text while preserving heading hierarchy."""
        return self.chunk(text)

    def chunk(self, text: str) -> List[HierarchicalChunk]:
        """
        Chunk markdown text while tracking heading hierarchy.

        Args:
            text: Markdown text to chunk

        Returns:
            List of HierarchicalChunk objects with heading_path metadata
        """
        # First, parse all headings in the document
        headings = self._parse_headings(text)

        # Use base chunker to get the actual chunks
        base_chunks = self.base_chunker(text)

        # For each chunk, determine its heading hierarchy
        hierarchical_chunks = []
        for chunk in base_chunks:
            heading_path = self._get_heading_path(
                chunk.start_index, headings, chunk.text
            )

            hierarchical_chunk = HierarchicalChunk(
                text=chunk.text,
                start_index=chunk.start_index,
                end_index=chunk.end_index,
                token_count=chunk.token_count,
                heading_path=heading_path,
            )
            hierarchical_chunks.append(hierarchical_chunk)

        return hierarchical_chunks

    def _parse_headings(self, text: str) -> List[HeadingInfo]:
        """
        Parse all headings from markdown text.

        Args:
            text: Markdown text

        Returns:
            List of HeadingInfo objects sorted by position
        """
        headings = []

        # Regex to match markdown headings (# through ######)
        heading_pattern = r"^(#{1,6})\s+(.+)$"

        for match in re.finditer(heading_pattern, text, re.MULTILINE):
            level = len(match.group(1))  # Count # characters
            heading_text = match.group(2).strip()
            start_pos = match.start()

            headings.append(
                HeadingInfo(level=level, text=heading_text, start_pos=start_pos)
            )

        return headings

    def _get_heading_path(
        self, chunk_start: int, headings: List[HeadingInfo], chunk_text: str = ""
    ) -> List[str]:
        """
        Get the hierarchical path of headings for a chunk position.

        Args:
            chunk_start: Start position of the chunk in the document
            headings: List of all headings in the document
            chunk_text: Text content of the chunk (to avoid duplication)

        Returns:
            List of heading strings representing the path from root to current section
        """
        if not headings:
            return []

        # Find all headings that come before this chunk
        preceding_headings = [h for h in headings if h.start_pos < chunk_start]

        if not preceding_headings:
            return []

        # Build the hierarchical path
        path = []
        current_levels = {}  # level -> heading_text mapping

        for heading in preceding_headings:
            # Clear any deeper levels when we encounter a heading
            levels_to_remove = [
                lvl for lvl in current_levels.keys() if lvl >= heading.level
            ]
            for lvl in levels_to_remove:
                del current_levels[lvl]

            # Add this heading to the current path
            current_levels[heading.level] = heading

        # Check if chunk starts with a heading and exclude it from path to avoid duplication
        chunk_starts_with_heading_level = self._get_chunk_heading_level(chunk_text)

        # Convert to ordered path (h1 -> h2 -> h3 etc.)
        if current_levels:
            for level in sorted(current_levels.keys()):
                # Skip the heading level that appears at the start of the chunk content
                if level == chunk_starts_with_heading_level:
                    continue
                heading = current_levels[level]
                formatted_heading = "#" * level + " " + heading.text
                path.append(formatted_heading)

        return path

    def _get_chunk_heading_level(self, chunk_text: str) -> Optional[int]:
        """
        Check if chunk starts with a heading and return its level.

        Args:
            chunk_text: The text content of the chunk

        Returns:
            Heading level (1-6) if chunk starts with heading, None otherwise
        """
        if not chunk_text.strip():
            return None

        first_line = chunk_text.strip().split("\n")[0]
        heading_pattern = r"^(#{1,6})\s+"
        match = re.match(heading_pattern, first_line)

        return len(match.group(1)) if match else None


# Factory function for easy creation
def create_hierarchical_wrapper(base_chunker) -> HierarchicalWrapper:
    """Create a hierarchical wrapper around any base chunker."""
    return HierarchicalWrapper(base_chunker)
