"""
Context-enriched document processor for VerbatimRAG.

This processor enriches document chunks with their hierarchical context (section paths)
to improve RAG retrieval by embedding section information alongside content.
"""

import re
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field

from .document_processor import DocumentProcessor
from ..document import Document, Chunk, ProcessedChunk, ChunkType, DocumentType


@dataclass
class ContextEnrichedChunk(Chunk):
    """Extended Chunk with hierarchical context information."""
    
    # Hierarchical context
    section_path: List[str] = field(default_factory=list)  # ["2 Background", "2.1 Dataset"]
    section_numbers: List[str] = field(default_factory=list)  # ["2", "2.1"]
    context_string: str = ""  # "Section: 2 Background | Subsection: 2.1 Dataset"
    
    def get_enhanced_content(self, include_context: bool = True) -> str:
        """Get content with optional context prefix for embedding."""
        if not include_context or not self.context_string:
            return self.content
        return f"{self.context_string} | {self.content}"
    
    def get_citation_context(self) -> str:
        """Get formatted context for citations."""
        if not self.section_path:
            return ""
        return " → ".join(self.section_path)


class ContextEnrichedProcessor(DocumentProcessor):
    """
    Document processor that enriches chunks with hierarchical context.
    
    Creates chunks where each paragraph contains the full section path it belongs to,
    enabling better RAG retrieval through context-aware embeddings.
    """
    
    def __init__(
        self,
        chunker_type: str = "recursive",
        chunker_recipe: str = "markdown", 
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        context_separator: str = " | ",
        include_section_numbers: bool = True,
        **chunker_kwargs
    ):
        """
        Initialize context-enriched processor.
        
        Args:
            context_separator: Separator between context elements (default: " | ")
            include_section_numbers: Whether to include section numbers in context
            **kwargs: Arguments passed to base DocumentProcessor
        """
        super().__init__(
            chunker_type=chunker_type,
            chunker_recipe=chunker_recipe,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **chunker_kwargs
        )
        self.context_separator = context_separator
        self.include_section_numbers = include_section_numbers
        
        # Precompile regex patterns for performance
        self.header_pattern = re.compile(r'^##\s+(\d+(?:\.\d+)*)\s+(.+)$')
        self.section_pattern = re.compile(r'^(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z\s]+.*)$')
    
    def process_file(
        self,
        file_path: Union[str, Path],
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """Process a local file with context enrichment."""
        
        # Get base document from parent class
        base_document = super().process_file(file_path, title, metadata)
        
        # Apply context enrichment
        return self._enrich_document_with_context(base_document)
    
    def process_url(
        self, 
        url: str, 
        title: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Process a document from URL with context enrichment."""
        
        # Get base document from parent class  
        base_document = super().process_url(url, title, metadata)
        
        # Apply context enrichment
        return self._enrich_document_with_context(base_document)
    
    def _enrich_document_with_context(self, document: Document) -> Document:
        """
        Enrich document chunks with hierarchical context.
        
        Strategy:
        1. Detect section structure from raw markdown content
        2. Create context-enriched chunks for content under each section
        3. Fall back to original chunks if no sections found
        """
        
        # Detect section structure
        sections = self._detect_sections(document.raw_content)
        
        if not sections:
            # No sections found, return original document
            return document
        
        # Create context-enriched chunks
        enriched_chunks = self._create_context_enriched_chunks(
            document.raw_content, 
            document.id,
            sections,
            document.title or "Untitled Document"
        )
        
        # Replace original chunks with enriched ones
        document.chunks = enriched_chunks
        
        return document
    
    def _detect_sections(self, content: str) -> List[Dict[str, Any]]:
        """
        Detect section structure from markdown content.
        
        Returns:
            List of section info dicts with keys:
            - line_number: Line where section starts
            - section_number: "1", "2.1", etc. 
            - title: Section title
            - level: Hierarchy level (1, 2, 3...)
            - full_title: "2 Background", "2.1 Dataset"
        """
        lines = content.split('\n')
        sections = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Try header pattern first: "## 1 Introduction"
            match = self.header_pattern.match(line_stripped)
            if not match:
                # Try section pattern: "1 Introduction" 
                match = self.section_pattern.match(line_stripped)
            
            if match:
                section_num = match.group(1)
                title = match.group(2).strip()
                level = len(section_num.split('.'))
                
                # Create full title 
                full_title = f"{section_num} {title}"
                
                sections.append({
                    'line_number': i + 1,
                    'section_number': section_num,
                    'title': title,
                    'level': level,
                    'full_title': full_title
                })
        
        return sections
    
    def _create_context_enriched_chunks(
        self, 
        content: str, 
        document_id: str,
        sections: List[Dict[str, Any]],
        document_title: str
    ) -> List[ContextEnrichedChunk]:
        """Create context-enriched chunks from content and section structure."""
        
        lines = content.split('\n')
        enriched_chunks = []
        
        for i, section in enumerate(sections):
            # Determine content boundaries for this section
            start_line = section['line_number'] - 1  # Convert to 0-based
            if i + 1 < len(sections):
                end_line = sections[i + 1]['line_number'] - 1
            else:
                end_line = len(lines)
            
            # Extract section content (skip header line)
            section_lines = lines[start_line:end_line]
            if section_lines:
                header_line = section_lines[0]
                content_lines = section_lines[1:]  # Skip header
                section_content = '\n'.join(content_lines).strip()
            else:
                continue
            
            if not section_content:
                continue
            
            # Build section path for this section
            section_path = self._build_section_path(section, sections)
            
            # Chunk the section content using the base chunker
            if len(section_content) > self.chunk_size:
                # Use base chunker for large sections
                content_chunks = self._chunk_content(section_content)
            else:
                # Keep small sections as single chunk
                content_chunks = [section_content]
            
            # Create enriched chunks for each content chunk
            for chunk_idx, chunk_content in enumerate(content_chunks):
                if not chunk_content.strip():
                    continue
                
                # Create context string with document title
                context_string = self._build_context_string(section_path, document_title)
                
                # Determine chunk type
                chunk_type = ChunkType.SECTION if section['level'] == 1 else ChunkType.PARAGRAPH
                
                # Create enriched chunk
                enriched_chunk = ContextEnrichedChunk(
                    document_id=document_id,
                    content=chunk_content.strip(),
                    chunk_number=len(enriched_chunks),
                    chunk_type=chunk_type,
                    section_path=section_path,
                    section_numbers=[s.split()[0] for s in section_path],
                    context_string=context_string,
                    metadata={
                        'section_info': section,
                        'section_path': section_path,
                        'context_string': context_string,
                        'chunk_in_section': chunk_idx
                    }
                )
                
                # Create processed chunk with enhanced content
                processed_chunk = ProcessedChunk(
                    chunk_id=enriched_chunk.id,
                    enhanced_content=enriched_chunk.get_enhanced_content(include_context=True),
                    section_title=section['title'],
                    processing_metadata={
                        'context_enriched': True,
                        'section_path': section_path,
                        'context_string': context_string
                    }
                )
                
                enriched_chunk.add_processed_chunk(processed_chunk)
                enriched_chunks.append(enriched_chunk)
        
        return enriched_chunks
    
    def _build_section_path(
        self, 
        current_section: Dict[str, Any], 
        all_sections: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Build hierarchical path for a section.
        
        For section "2.1 Dataset", returns ["2 Background", "2.1 Dataset"]
        """
        section_path = []
        current_number = current_section['section_number']
        current_parts = current_number.split('.')
        
        # Add all parent sections
        for level in range(1, len(current_parts)):
            parent_number = '.'.join(current_parts[:level])
            
            # Find parent section
            for section in all_sections:
                if section['section_number'] == parent_number:
                    section_path.append(section['full_title'])
                    break
        
        # Add current section
        section_path.append(current_section['full_title'])
        
        return section_path
    
    def _build_context_string(self, section_path: List[str], document_title: str = "") -> str:
        """Build context string from section path and document title."""
        if not section_path and not document_title:
            return ""
        
        # Start with document title if provided
        context_parts = []
        if document_title:
            context_parts.append(document_title)
        
        # Add hierarchical section labels
        for i, section in enumerate(section_path):
            if i == 0:
                label = "Section"
            elif i == 1:
                label = "Subsection"  
            elif i == 2:
                label = "Subsubsection"
            else:
                label = f"Level-{i+1}"
            
            context_parts.append(f"{label}: {section}")
        
        return self.context_separator.join(context_parts)
    
    def _chunk_content(self, content: str) -> List[str]:
        """Chunk content using the base chunker."""
        try:
            # Use the inherited chunker from DocumentProcessor
            chunks = self.chunker(content)
            return [chunk.text for chunk in chunks]
        except Exception:
            # Fallback to simple paragraph splitting
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            return paragraphs if paragraphs else [content]
    
    @classmethod 
    def for_rag(
        cls, 
        chunk_size: int = 384,
        overlap: int = 50,
        context_separator: str = " | "
    ):
        """Create processor optimized for RAG with context enrichment."""
        return cls(
            chunker_type="token",
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            context_separator=context_separator,
            include_section_numbers=True
        )
    
    @classmethod
    def for_embeddings(
        cls,
        chunk_size: int = 512, 
        overlap: int = 50,
        context_separator: str = " | "
    ):
        """Create processor optimized for embedding generation with context."""
        return cls(
            chunker_type="recursive",
            chunk_size=chunk_size, 
            chunk_overlap=overlap,
            context_separator=context_separator,
            include_section_numbers=True
        )