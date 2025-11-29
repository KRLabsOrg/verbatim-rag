"""
VerbatimDOC: Document generation with embedded RAG queries.

Standalone utility for processing templates with [!query=...] expressions.
Each query is executed independently with section context awareness.

Usage:
    from verbatim_rag.verbatim_doc import VerbatimDOC

    doc = VerbatimDOC(rag)  # Pass VerbatimRAG directly
    result = await doc.process(template, auto_approve=True)
"""

import re
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Union, Optional
from pathlib import Path


@dataclass
class Query:
    """A single query extracted from a document"""

    text: str
    start: int
    end: int
    params: Dict[str, Any] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class QueryResult:
    """Result of executing a query"""

    query: Query
    result: str
    alternatives: List[str] = None  # For interactive mode
    approved: bool = False

    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []


class Parser:
    """Extracts [!query=...] expressions from text"""

    PATTERN = re.compile(r"\[!query=([^|\]]+)(?:\|([^\]]+))?\]", re.IGNORECASE)

    def extract_queries(self, text: str) -> List[Query]:
        queries = []
        for match in self.PATTERN.finditer(text):
            query_text = match.group(1).strip()
            params_text = match.group(2) or ""

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
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        if value.isdigit():
            return int(value)
        if value.replace(".", "", 1).isdigit():
            return float(value)
        return value.strip("\"'")


class Processor:
    """Executes queries using a RAG system"""

    def __init__(self, rag, use_context: bool = True):
        """
        :param rag: VerbatimRAG instance (or any object with index/extractor)
        :param use_context: Include section context in queries
        """
        self.rag = rag
        self.use_context = use_context

    async def process_query(self, query: Query, template: str = "") -> QueryResult:
        try:
            question = self._build_question(query, template)

            # Use RAG's index and extractor directly (no template manager)
            answer = await self._execute_query(question)
            result = self._format_result(answer, query.params)

            return QueryResult(query=query, result=result)
        except Exception as e:
            return QueryResult(query=query, result=f"[Error: {str(e)}]")

    async def process_queries(
        self, queries: List[Query], template: str = ""
    ) -> List[QueryResult]:
        tasks = [self.process_query(query, template) for query in queries]
        return await asyncio.gather(*tasks)

    async def _execute_query(self, question: str) -> str:
        """Execute query using RAG's lower-level components directly."""
        # Get documents
        docs = self.rag.index.query(text=question, k=self.rag.k)

        # Extract spans
        spans_dict = await self.rag.extractor.extract_spans_async(question, docs)

        # Format as simple answer
        all_spans = []
        for doc_spans in spans_dict.values():
            all_spans.extend(doc_spans)

        if not all_spans:
            return "No relevant information found."

        # Format with citation numbers
        if len(all_spans) == 1:
            return all_spans[0]
        return "\n\n".join(f"[{i}] {span}" for i, span in enumerate(all_spans, 1))

    def _build_question(self, query: Query, template: str) -> str:
        if not self.use_context or not template:
            return query.text

        section = self._find_section(template, query.start)
        if section:
            return f"For the '{section}' section: {query.text}"
        return query.text

    def _find_section(self, text: str, position: int) -> Optional[str]:
        text_before = text[:position]
        for line in reversed(text_before.split("\n")):
            line = line.strip()
            if line.startswith("#"):
                header = line.lstrip("#").strip()
                return header.replace("**", "").replace("*", "").replace("`", "")
        return None

    def _format_result(self, answer: str, params: Dict[str, Any]) -> str:
        result = answer

        if params.get("format") == "bullet":
            sentences = result.split(". ")
            result = "\n".join(f"• {s.strip()}" for s in sentences if s.strip())
        elif params.get("format") == "short":
            result = result.split(".")[0] + "."

        if "max_length" in params:
            max_len = params["max_length"]
            if len(result) > max_len:
                result = result[: max_len - 3] + "..."

        return result


class Replacer:
    """Replaces queries with results in document"""

    def replace(self, text: str, results: List[QueryResult]) -> str:
        sorted_results = sorted(results, key=lambda r: r.query.start, reverse=True)

        for result in sorted_results:
            if result.approved:
                text = (
                    text[: result.query.start]
                    + result.result
                    + text[result.query.end :]
                )

        return text


class VerbatimDOC:
    """
    Document generation with embedded RAG queries.

    Usage:
        doc = VerbatimDOC(rag)
        result = await doc.process(template, auto_approve=True)
    """

    def __init__(self, rag, use_context: bool = True):
        """
        :param rag: VerbatimRAG instance
        :param use_context: Include section context in queries
        """
        self.rag = rag
        self.parser = Parser()
        self.processor = Processor(rag, use_context=use_context)
        self.replacer = Replacer()

    async def process(self, text: str, auto_approve: bool = False) -> str:
        """
        Process document with embedded queries.

        :param text: Template with [!query=...] expressions
        :param auto_approve: Auto-approve all queries
        :return: Filled document
        """
        queries = self.parser.extract_queries(text)
        results = await self.processor.process_queries(queries, template=text)

        if auto_approve:
            for result in results:
                result.approved = True

        return self.replacer.replace(text, results)

    async def process_interactive(self, text: str) -> Tuple[str, List[QueryResult]]:
        """Process for interactive review."""
        queries = self.parser.extract_queries(text)
        results = await self.processor.process_queries(queries, template=text)
        return text, results

    def finalize(self, text: str, results: List[QueryResult]) -> str:
        """Generate final document with approved results."""
        return self.replacer.replace(text, results)


# Utility functions
def load_template(file_path: Union[str, Path]) -> str:
    """Load template from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def save_document(content: str, file_path: Union[str, Path]) -> None:
    """Save document to file."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
