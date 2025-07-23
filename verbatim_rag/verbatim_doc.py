"""
VerbatimDOC: Simple, clean document generation with embedded RAG queries.

Architecture:
1. Parser - extracts [!query=...] expressions from text
2. Processor - executes queries and generates responses
3. Replacer - substitutes queries with results
4. Interactive mode - allows user review/modification before replacement
"""

import re
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Union
from pathlib import Path

from .models import QueryResponse


@dataclass
class Query:
    """A single query extracted from a document"""

    text: str  # The actual query text
    start: int  # Start position in document
    end: int  # End position in document
    params: Dict[str, Any] = None  # Optional parameters

    def __post_init__(self):
        if self.params is None:
            self.params = {}

    @property
    def full_match(self) -> str:
        """The full matched text including [!query=...] syntax"""
        if self.params:
            param_str = "|" + ",".join(f"{k}={v}" for k, v in self.params.items())
            return f"[!query={self.text}{param_str}]"
        return f"[!query={self.text}]"


@dataclass
class QueryResult:
    """Result of executing a query"""

    query: Query
    result: str
    alternatives: List[str] = None
    approved: bool = False

    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []


class RAGInterface(ABC):
    """Simple interface for any RAG system"""

    @abstractmethod
    async def query(self, question: str) -> QueryResponse:
        """Execute a query and return response"""
        pass


class Parser:
    """Extracts queries from document text"""

    def __init__(self, pattern: str = r"\[!query=([^|\]]+)(?:\|([^\]]+))?\]"):
        self.pattern = re.compile(pattern, re.IGNORECASE)

    def extract_queries(self, text: str) -> List[Query]:
        """Extract all queries from text"""
        queries = []

        for match in self.pattern.finditer(text):
            query_text = match.group(1).strip()
            params_text = match.group(2) or ""

            # Parse parameters
            params = {}
            if params_text:
                for param in params_text.split(","):
                    if "=" in param:
                        key, value = param.split("=", 1)
                        params[key.strip()] = self._parse_value(value.strip())

            queries.append(
                Query(
                    text=query_text, start=match.start(), end=match.end(), params=params
                )
            )

        return queries

    def _parse_value(self, value: str) -> Any:
        """Parse parameter value to appropriate type"""
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        if value.isdigit():
            return int(value)
        if value.replace(".", "", 1).isdigit():
            return float(value)
        return value.strip("\"'")


class Processor:
    """Executes queries using a RAG system"""

    def __init__(self, rag: RAGInterface):
        self.rag = rag

    async def process_query(self, query: Query) -> QueryResult:
        """Process a single query"""
        try:
            response = await self.rag.query(query.text)
            result = self._format_result(response, query.params)

            return QueryResult(query=query, result=result)

        except Exception as e:
            return QueryResult(query=query, result=f"[Error: {str(e)}]")

    async def process_queries(self, queries: List[Query]) -> List[QueryResult]:
        """Process multiple queries concurrently"""
        tasks = [self.process_query(query) for query in queries]
        return await asyncio.gather(*tasks)

    def _format_result(self, response: QueryResponse, params: Dict[str, Any]) -> str:
        """Format response based on parameters"""
        result = response.answer

        # Apply formatting
        if params.get("format") == "bullet":
            sentences = result.split(". ")
            result = "\n".join(f"• {s.strip()}" for s in sentences if s.strip())
        elif params.get("format") == "short":
            result = result.split(".")[0] + "."

        # Apply length limit
        if "max_length" in params:
            max_len = params["max_length"]
            if len(result) > max_len:
                result = result[: max_len - 3] + "..."

        return result


class Replacer:
    """Replaces queries with results in document"""

    def replace(self, text: str, results: List[QueryResult]) -> str:
        """Replace queries with results"""
        # Sort by position (descending) to avoid position shifts
        sorted_results = sorted(results, key=lambda r: r.query.start, reverse=True)

        modified_text = text
        for result in sorted_results:
            if result.approved:  # Only replace approved queries
                modified_text = (
                    modified_text[: result.query.start]
                    + result.result
                    + modified_text[result.query.end :]
                )

        return modified_text


class VerbatimDOC:
    """Main VerbatimDOC class - simple and clean"""

    def __init__(self, rag: RAGInterface):
        self.parser = Parser()
        self.processor = Processor(rag)
        self.replacer = Replacer()

    async def process(self, text: str, auto_approve: bool = False) -> str:
        """Process document in one go (non-interactive)"""
        queries = self.parser.extract_queries(text)
        results = await self.processor.process_queries(queries)

        # Auto-approve if requested
        if auto_approve:
            for result in results:
                result.approved = True

        return self.replacer.replace(text, results)

    async def process_interactive(self, text: str) -> Tuple[str, List[QueryResult]]:
        """Process document for interactive review"""
        queries = self.parser.extract_queries(text)
        results = await self.processor.process_queries(queries)
        return text, results

    def finalize(self, text: str, results: List[QueryResult]) -> str:
        """Generate final document with approved results"""
        return self.replacer.replace(text, results)


class VerbatimRAGAdapter(RAGInterface):
    """Adapter for VerbatimRAG system"""

    def __init__(self, verbatim_rag):
        self.verbatim_rag = verbatim_rag

    async def query(self, question: str) -> QueryResponse:
        """Execute query using VerbatimRAG"""
        # If VerbatimRAG has async method, use it
        if hasattr(self.verbatim_rag, "query_async"):
            return await self.verbatim_rag.query_async(question)
        else:
            # Otherwise use sync method
            return self.verbatim_rag.query(question)


# Usage Examples
async def simple_example():
    """Simple non-interactive usage"""
    # Setup
    verbatim_rag = None  # Your VerbatimRAG instance
    rag_adapter = VerbatimRAGAdapter(verbatim_rag)
    doc_processor = VerbatimDOC(rag_adapter)

    # Process document
    template = "The company was founded [!query=when was apple founded] and is known for [!query=what is apple known for|format=bullet]"
    result = await doc_processor.process(template, auto_approve=True)
    print(result)


async def interactive_example():
    """Interactive usage with user review"""
    # Setup
    verbatim_rag = None  # Your VerbatimRAG instance
    rag_adapter = VerbatimRAGAdapter(verbatim_rag)
    doc_processor = VerbatimDOC(rag_adapter)

    # Process for review
    template = "The company was founded [!query=when was apple founded]"
    original_text, results = await doc_processor.process_interactive(template)

    # User reviews results
    for result in results:
        print(f"Query: {result.query.text}")
        print(f"Result: {result.result}")

        # User decides (in real UI, this would be interactive)
        user_approves = True  # User input
        result.approved = user_approves

    # Generate final document
    final_doc = doc_processor.finalize(original_text, results)
    print(final_doc)


# Utility functions
def load_template(file_path: Union[str, Path]) -> str:
    """Load template from file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def save_document(content: str, file_path: Union[str, Path]) -> None:
    """Save document to file"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


async def process_template_file(
    template_path: Union[str, Path],
    output_path: Union[str, Path],
    rag: RAGInterface,
    auto_approve: bool = False,
) -> None:
    """Process template file and save result"""
    template = load_template(template_path)
    doc_processor = VerbatimDOC(rag)

    result = await doc_processor.process(template, auto_approve=auto_approve)
    save_document(result, output_path)
