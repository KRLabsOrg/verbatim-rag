"""
VerbatimDOC: Document generation with embedded RAG queries.

Standalone utility for processing templates with [!query=...] expressions.
Each query is executed independently with section context awareness.

Returns QueryResponse with global citation numbering and document attribution.

Usage:
    from verbatim_rag.verbatim_doc import VerbatimDOC

    doc = VerbatimDOC(rag)  # Pass VerbatimRAG directly
    response = await doc.process(template, auto_approve=True)
    # response.answer = filled document
    # response.documents = all source documents
    # response.structured_answer.citations = global citations with doc_index
"""

import re
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Union, Optional, AsyncGenerator
from pathlib import Path

from verbatim_core.models import QueryResponse


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
class SpanWithDoc:
    """A span with its source document info"""

    text: str
    doc_index: int  # Index into the documents list
    doc_text: str  # Full text of source document


@dataclass
class QueryResult:
    """Result of executing a query"""

    query: Query
    result: str  # Formatted result for display
    spans: List[SpanWithDoc] = field(
        default_factory=list
    )  # Raw spans with doc attribution
    docs: List[Any] = field(default_factory=list)  # Source documents for this query
    alternatives: List[str] = field(default_factory=list)  # For interactive mode
    approved: bool = False


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

            # Execute query and get raw spans with doc attribution
            spans, docs = await self._execute_query_raw(question)

            # Format result for display (local numbering - will be renumbered globally later)
            result = self._format_spans_local(spans, query.params)

            return QueryResult(query=query, result=result, spans=spans, docs=docs)
        except Exception as e:
            return QueryResult(query=query, result=f"[Error: {str(e)}]")

    async def process_queries(
        self, queries: List[Query], template: str = ""
    ) -> List[QueryResult]:
        tasks = [self.process_query(query, template) for query in queries]
        return await asyncio.gather(*tasks)

    async def _execute_query_raw(
        self, question: str
    ) -> Tuple[List[SpanWithDoc], List[Any]]:
        """
        Execute query and return raw spans with document attribution.

        :return: Tuple of (spans_with_docs, source_documents)
        """
        # Get documents
        docs = self.rag.index.query(text=question, k=self.rag.k)

        # Extract spans
        spans_dict = await self.rag.extractor.extract_spans_async(question, docs)

        # Collect spans with document attribution
        spans_with_docs = []
        for i, doc in enumerate(docs):
            doc_text = getattr(doc, "text", "")
            doc_spans = spans_dict.get(doc_text, [])
            for span_text in doc_spans:
                spans_with_docs.append(
                    SpanWithDoc(text=span_text, doc_index=i, doc_text=doc_text)
                )

        return spans_with_docs, docs

    def _format_spans_local(
        self, spans: List[SpanWithDoc], params: Dict[str, Any]
    ) -> str:
        """Format spans with local numbering (for preview/interactive mode)."""
        if not spans:
            return "No relevant information found."

        texts = [s.text for s in spans]

        if len(texts) == 1:
            result = texts[0]
        else:
            result = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(texts, 1))

        return self._apply_format_params(result, params)

    def _apply_format_params(self, result: str, params: Dict[str, Any]) -> str:
        """Apply formatting parameters to result."""
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

    Returns QueryResponse with global citation numbering and document attribution.

    Usage:
        doc = VerbatimDOC(rag)
        response = await doc.process(template, auto_approve=True)
        # response.answer = filled document
        # response.documents = source documents
        # response.structured_answer.citations = citations with doc_index
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

    async def process(self, text: str, auto_approve: bool = False) -> QueryResponse:
        """
        Process document with embedded queries.

        :param text: Template with [!query=...] expressions
        :param auto_approve: Auto-approve all queries
        :return: QueryResponse with filled document and citations
        """
        queries = self.parser.extract_queries(text)
        results = await self.processor.process_queries(queries, template=text)

        if auto_approve:
            for result in results:
                result.approved = True

        return self._build_response(text, results)

    async def process_interactive(self, text: str) -> Tuple[str, List[QueryResult]]:
        """Process for interactive review."""
        queries = self.parser.extract_queries(text)
        results = await self.processor.process_queries(queries, template=text)
        return text, results

    def finalize(self, text: str, results: List[QueryResult]) -> QueryResponse:
        """Generate final document with approved results."""
        return self._build_response(text, results)

    async def stream_process(
        self, text: str, auto_approve: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream document processing with progress updates.

        Yields stages:
        - {"type": "queries_found", "count": N, "queries": [...]}
        - {"type": "query_start", "index": i, "query": "..."}
        - {"type": "query_complete", "index": i, "result": "...", "spans_count": N}
        - {"type": "document", "response": QueryResponse, "done": True}

        :param text: Template with [!query=...] expressions
        :param auto_approve: Auto-approve all queries
        """

        queries = self.parser.extract_queries(text)

        yield {
            "type": "queries_found",
            "count": len(queries),
            "queries": [q.text for q in queries],
        }

        if not queries:
            yield {
                "type": "document",
                "response": QueryResponse(
                    question="[VerbatimDOC]",
                    answer=text,
                    documents=[],
                    structured_answer=None,
                ),
                "done": True,
            }
            return

        results = []

        # Process queries one by one with progress updates
        for i, query in enumerate(queries):
            yield {
                "type": "query_start",
                "index": i,
                "total": len(queries),
                "query": query.text,
                "section": self.processor._find_section(text, query.start),
            }

            result = await self.processor.process_query(query, template=text)

            if auto_approve:
                result.approved = True

            results.append(result)

            yield {
                "type": "query_complete",
                "index": i,
                "total": len(queries),
                "query": query.text,
                "result_preview": result.result[:100] + "..."
                if len(result.result) > 100
                else result.result,
                "spans_count": len(result.spans),
                "approved": result.approved,
            }

        # Build final response
        response = self._build_response(text, results)

        yield {
            "type": "document",
            "response": response.model_dump(),
            "done": True,
        }

    def _build_response(
        self, template: str, results: List[QueryResult]
    ) -> QueryResponse:
        """
        Build QueryResponse with global citation numbering.

        Citations are numbered in query order (same order as they appear in the answer).

        :param template: Original template
        :param results: Query results with spans and docs
        :return: QueryResponse with filled answer and citations
        """
        from verbatim_core.models import (
            DocumentWithHighlights,
            Highlight,
            Citation,
            StructuredAnswer,
        )

        # Sort results by position in template (query order)
        sorted_results = sorted(results, key=lambda r: r.query.start)

        # Collect all unique documents across all queries (in query order)
        all_docs = []
        doc_text_to_global_idx = {}

        for result in sorted_results:
            if not result.approved:
                continue
            for doc in result.docs:
                doc_text = getattr(doc, "text", "")
                if doc_text and doc_text not in doc_text_to_global_idx:
                    doc_text_to_global_idx[doc_text] = len(all_docs)
                    all_docs.append(doc)

        # Build citations in query order (matching the answer's numbering)
        all_citations = []
        docs_highlights = {i: [] for i in range(len(all_docs))}
        citation_num = 1

        for result in sorted_results:
            if not result.approved:
                continue
            for span in result.spans:
                global_doc_idx = doc_text_to_global_idx.get(span.doc_text, 0)

                all_citations.append(
                    Citation(
                        text=span.text,
                        doc_index=global_doc_idx,
                        highlight_index=len(docs_highlights.get(global_doc_idx, [])),
                        number=citation_num,
                        type="display",
                    )
                )

                if global_doc_idx in docs_highlights:
                    docs_highlights[global_doc_idx].append(span.text)
                citation_num += 1

        # Build filled template with global citation numbering
        filled_template = self._fill_with_global_citations(
            template, results, doc_text_to_global_idx
        )

        # Build documents with highlights
        documents_with_highlights = []
        for i, doc in enumerate(all_docs):
            doc_text = getattr(doc, "text", "")
            highlights = []
            for span_text in docs_highlights.get(i, []):
                start = doc_text.find(span_text)
                if start >= 0:
                    highlights.append(
                        Highlight(
                            text=span_text,
                            start=start,
                            end=start + len(span_text),
                        )
                    )

            doc_metadata = getattr(doc, "metadata", {}) or {}
            documents_with_highlights.append(
                DocumentWithHighlights(
                    content=doc_text,
                    highlights=highlights,
                    title=getattr(doc, "title", "") or doc_metadata.get("title", ""),
                    source=getattr(doc, "source", "") or doc_metadata.get("source", ""),
                    metadata=doc_metadata,
                )
            )

        structured_answer = StructuredAnswer(
            text=filled_template, citations=all_citations
        )

        return QueryResponse(
            question="[VerbatimDOC]",
            answer=filled_template,
            documents=documents_with_highlights,
            structured_answer=structured_answer,
        )

    def _fill_with_global_citations(
        self,
        template: str,
        results: List[QueryResult],
        doc_text_to_global_idx: Dict[str, int],
    ) -> str:
        """
        Fill template with globally numbered citations.

        :param template: Original template with [!query=...] expressions
        :param results: Query results
        :param doc_text_to_global_idx: Map from doc text to global doc index
        :return: Filled template with global [1], [2]... citations
        """
        # Sort results by position (reversed for safe replacement)
        sorted_results = sorted(results, key=lambda r: r.query.start, reverse=True)

        # First pass: count total spans to assign global numbers
        global_num = 1
        result_to_start_num = {}

        # Process in forward order for numbering
        for result in sorted(results, key=lambda r: r.query.start):
            if result.approved and result.spans:
                result_to_start_num[id(result)] = global_num
                global_num += len(result.spans)

        # Second pass: replace in reverse order
        filled = template
        for result in sorted_results:
            if not result.approved:
                continue

            start_num = result_to_start_num.get(id(result), 1)
            replacement = self._format_spans_global(
                result.spans, start_num, doc_text_to_global_idx, result.query.params
            )
            filled = (
                filled[: result.query.start] + replacement + filled[result.query.end :]
            )

        return filled

    def _format_spans_global(
        self,
        spans: List[SpanWithDoc],
        start_num: int,
        doc_text_to_global_idx: Dict[str, int],
        params: Dict[str, Any],
    ) -> str:
        """Format spans with global citation numbers."""
        if not spans:
            return "No relevant information found."

        formatted = []
        for i, span in enumerate(spans):
            num = start_num + i
            formatted.append(f"[{num}] {span.text}")

        if len(formatted) == 1:
            result = formatted[0]
        else:
            result = "\n\n".join(formatted)

        return self.processor._apply_format_params(result, params)


# Utility functions
def load_template(file_path: Union[str, Path]) -> str:
    """Load template from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def save_document(content: str, file_path: Union[str, Path]) -> None:
    """Save document to file."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
