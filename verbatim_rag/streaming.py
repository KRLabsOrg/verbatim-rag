"""
Streaming interface for the Verbatim RAG system
Provides structured streaming of RAG processing stages
"""

from typing import AsyncGenerator, Dict, Any, List
from .models import (
    QueryResponse,
    DocumentWithHighlights,
    Citation,
    StructuredAnswer,
)
from .core import VerbatimRAG


class StreamingRAG:
    """
    Streaming wrapper for VerbatimRAG that provides step-by-step processing
    """

    def __init__(self, rag: VerbatimRAG):
        self.rag = rag

    async def stream_query(
        self, question: str, num_docs: int = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a query response in stages:
        1. Documents (without highlights)
        2. Documents with highlights
        3. Final answer

        Args:
            question: The user's question
            num_docs: Optional number of documents to retrieve

        Yields:
            Dictionary with type and data for each stage
        """
        try:
            # Set number of documents if specified
            if num_docs is not None:
                original_k = self.rag.k
                self.rag.k = num_docs

            # Step 1: Retrieve documents and send them without highlights
            docs = self.rag.index.search(question, k=self.rag.k)

            documents_without_highlights = [
                DocumentWithHighlights(
                    content=doc.text,
                    highlights=[],
                    title=doc.metadata.get("title", ""),
                    source=doc.metadata.get("source", ""),
                    metadata=doc.metadata,
                )
                for doc in docs
            ]

            yield {
                "type": "documents",
                "data": [doc.model_dump() for doc in documents_without_highlights],
            }

            # Step 2: Extract spans and create highlights
            relevant_spans = self.rag.extractor.extract_spans(question, docs)

            # Use the response builder to create properly formatted documents with highlights
            documents_with_highlights = []
            all_citations = []

            for doc_index, doc in enumerate(docs):
                doc_content = doc.text
                doc_spans = relevant_spans.get(doc_content, [])

                if doc_spans:
                    # Create highlights using the response builder logic
                    highlights = self.rag.response_builder._create_highlights(
                        doc_content, doc_spans
                    )

                    # Create citations
                    for highlight_index, highlight in enumerate(highlights):
                        citation = Citation(
                            text=highlight.text,
                            doc_index=doc_index,
                            highlight_index=highlight_index,
                        )
                        all_citations.append(citation)
                else:
                    highlights = []

                document_with_highlights = DocumentWithHighlights(
                    content=doc_content,
                    highlights=highlights,
                    title=doc.metadata.get("title", ""),
                    source=doc.metadata.get("source", ""),
                    metadata=doc.metadata,
                )
                documents_with_highlights.append(document_with_highlights)

            yield {
                "type": "highlights",
                "data": [doc.model_dump() for doc in documents_with_highlights],
            }

            # Step 3: Generate answer
            template = self.rag._generate_template(question)
            answer = self.rag._fill_template(template, relevant_spans.values())
            answer = self.rag.response_builder.clean_answer(answer)

            # Create structured answer
            structured_answer = StructuredAnswer(text=answer, citations=all_citations)

            # Create final result
            result = QueryResponse(
                question=question,
                answer=answer,
                structured_answer=structured_answer,
                documents=documents_with_highlights,
            )

            yield {"type": "answer", "data": result.model_dump(), "done": True}

            # Restore original k value if we changed it
            if num_docs is not None:
                self.rag.k = original_k

        except Exception as e:
            yield {"type": "error", "error": str(e), "done": True}

    def stream_query_sync(
        self, question: str, num_docs: int = None
    ) -> List[Dict[str, Any]]:
        """
        Synchronous version that returns all streaming stages as a list
        Useful for testing or when async is not needed
        """
        import asyncio

        async def collect_stream():
            stages = []
            async for stage in self.stream_query(question, num_docs):
                stages.append(stage)
            return stages

        return asyncio.run(collect_stream())
