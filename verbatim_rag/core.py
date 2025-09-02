"""Core implementation of the Verbatim RAG system.

Enhancement: Supports both legacy aggregate placeholders ([DISPLAY_SPANS], [CITATION_REFS])
and new per-fact placeholders of the form [FACT_1], [FACT_2], ... allowing templates
to interleave verbatim facts contextually. Citation-only facts (beyond the display
limit) expand to a numbered reference token (e.g. "[6]") without verbatim text.
"""

from typing import Optional
from verbatim_rag.extractors import LLMSpanExtractor, SpanExtractor
from verbatim_rag.index import VerbatimIndex
from verbatim_rag.models import QueryResponse
from verbatim_core.templates import TemplateManager
from verbatim_rag.response_builder import ResponseBuilder
from verbatim_rag.llm_client import LLMClient
from verbatim_rag.schema import DocumentSchema
from verbatim_rag.ingestion import schema_to_document

MARKING_SYSTEM_PROMPT = """
You are a Q&A text extraction system. Your task is to identify and mark EXACT verbatim text spans from the provided document that is relevant to answer the user's question.

# Rules
1. Mark **only** text that explicitly addresses the question
2. Never paraphrase, modify, or add to the original text
3. Preserve original wording, capitalization, and punctuation
4. Mark all relevant segments - even if they're non-consecutive
5. If there is no relevant information, don't add any tags.

# Output Format
Wrap each relevant text span with <relevant> tags. 
Return ONLY the marked document text - no explanations or summaries.

# Example
Question: What causes climate change?
Document: "Scientists agree that carbon emissions (CO2) from burning fossil fuels are the primary driver of climate change. Deforestation also contributes significantly."
Marked: "Scientists agree that <relevant>carbon emissions (CO2) from burning fossil fuels</relevant> are the primary driver of climate change. <relevant>Deforestation also contributes significantly</relevant>."

# Your Task
Question: {QUESTION}
Document: {DOCUMENT}

Mark the relevant text:
"""


class VerbatimRAG:
    """
    A RAG system that prevents hallucination by ensuring all generated content
    is explicitly derived from source documents.
    """

    def __init__(
        self,
        index: VerbatimIndex,
        model: str = "gpt-4o-mini",
        k: int = 5,
        template_manager: TemplateManager = None,
        extractor: SpanExtractor = None,
        max_display_spans: int = 5,
        template_mode: str = "contextual",  # "static", "contextual", "random"
        extraction_mode: str = "auto",  # "batch", "individual", "auto"
        llm_client: LLMClient = None,
    ):
        """
        Initialize the Verbatim RAG system with clean architecture.

        :param index: The index to search for relevant documents
        :param model: The LLM model to use (if creating new LLM client)
        :param k: The number of documents to retrieve
        :param template_manager: Optional template manager (creates one if None)
        :param extractor: Optional custom extractor (creates LLM extractor if None)
        :param max_display_spans: Maximum number of spans to display verbatim
        :param template_mode: Template mode ("static", "contextual", "random")
        :param extraction_mode: Extraction mode ("batch", "individual", "auto")
        :param llm_client: Optional LLM client (creates one if None)
        """
        self.index = index
        self.k = k
        self.max_display_spans = max_display_spans

        # Centralized LLM client
        self.llm_client = llm_client or LLMClient(model)

        # Initialize components with clean dependency injection
        self.extractor = extractor or LLMSpanExtractor(
            llm_client=self.llm_client,
            extraction_mode=extraction_mode,
            max_display_spans=max_display_spans,
        )

        self.template_manager = template_manager or TemplateManager(
            llm_client=self.llm_client, default_mode=template_mode
        )

        self.response_builder = ResponseBuilder()

    def _generate_template(
        self, question: str, display_spans: list[str] = None, citation_count: int = 0
    ) -> str:
        """
        Generate or select a template for the response.

        :param question: The user's question
        :param display_spans: Spans that will be displayed (for contextual templates)
        :param citation_count: Number of additional citations
        :return: A template string with placeholders
        """
        return self.template_manager.get_template(
            question, display_spans or [], citation_count
        )

    def _rank_and_split_spans(
        self, relevant_spans: dict[str, list[str]]
    ) -> tuple[list[dict], list[dict]]:
        """
        Split spans into display vs citation-only, trusting the extractor's ordering.

        :param relevant_spans: Dictionary mapping doc text to span lists (already ordered by relevance)
        :return: Tuple of (display_spans, citation_spans) with minimal metadata
        """
        # Flatten spans, preserving the order from the extractors
        all_spans = []
        for doc_text, spans in relevant_spans.items():
            for span in spans:
                all_spans.append({"text": span, "doc_text": doc_text})

        # Split into display and citation-only (trust the extractor's relevance ordering)
        display_spans = all_spans[: self.max_display_spans]
        citation_spans = all_spans[self.max_display_spans :]

        return display_spans, citation_spans

    def _fill_template_enhanced(
        self, template: str, display_spans: list[dict], citation_spans: list[dict]
    ) -> str:
        """
        Fill the template with display spans and citation references.

        Now delegates to the template manager's fill functionality.

        :param template: The template string with placeholders
        :param display_spans: Spans to display verbatim with metadata
        :param citation_spans: Spans for citation reference only
        :return: The filled template
        """
        return self.template_manager.fill_template(
            template, display_spans, citation_spans
        )

    def query(self, question: str, filter: Optional[str] = None) -> QueryResponse:
        """
        Process a query through the Verbatim RAG system with clean architecture.

        :param question: The user's question
        :param filter: Optional filter to narrow document search
        :return: A QueryResponse object containing the structured response
        """
        # Step 1: Retrieve documents
        search_results = self.index.query(text=question, k=self.k, filter=filter)

        # Step 2: Extract spans
        print("Extracting relevant spans...")
        all_relevant_spans = self.extractor.extract_spans(question, search_results)

        # Step 3: Split spans into display vs citation-only
        print("Processing spans...")
        display_spans, citation_spans = self._rank_and_split_spans(all_relevant_spans)

        # Step 4: Generate and fill template in one operation
        print("Generating response...")
        answer = self.template_manager.process(question, display_spans, citation_spans)

        # Step 5: Clean up the answer
        answer = self.response_builder.clean_answer(answer)

        # Step 6: Build the complete response
        return self.response_builder.build_response(
            question=question,
            answer=answer,
            search_results=search_results,
            relevant_spans=all_relevant_spans,
            display_span_count=len(display_spans),
        )

    async def query_async(
        self, question: str, filter: Optional[str] = None
    ) -> QueryResponse:
        """
        Async version of query method with clean architecture.

        :param question: The user's question
        :param filter: Optional filter to narrow document search
        :return: A QueryResponse object containing the structured response
        """
        # Step 1: Retrieve documents
        search_results = self.index.query(text=question, k=self.k, filter=filter)

        # Step 2: Extract spans (async)
        print("Extracting relevant spans (async)...")
        all_relevant_spans = await self.extractor.extract_spans_async(
            question, search_results
        )

        # Step 3: Split spans into display vs citation-only
        print("Processing spans...")
        display_spans, citation_spans = self._rank_and_split_spans(all_relevant_spans)

        # Step 4: Generate and fill template (async)
        print("Generating response (async)...")
        answer = await self.template_manager.process_async(
            question, display_spans, citation_spans
        )

        # Step 5: Clean up the answer
        answer = self.response_builder.clean_answer(answer)

        # Step 6: Build the complete response
        return self.response_builder.build_response(
            question=question,
            answer=answer,
            search_results=search_results,
            relevant_spans=all_relevant_spans,
            display_span_count=len(display_spans),
        )

    def add_document(self, document: DocumentSchema) -> str:
        """
        Add a single document using the DocumentSchema system.

        :param document: DocumentSchema instance with content and metadata
        :return: Document ID
        """
        # Convert schema → pre-chunked Document, then index
        prechunked = schema_to_document(document)
        self.index.add_documents([prechunked])
        return prechunked.id

    def add_documents_batch(self, documents: list[DocumentSchema]) -> list[str]:
        """
        Add multiple documents in batch using the DocumentSchema system.

        :param documents: List of DocumentSchema instances
        :return: List of document IDs
        """
        prechunked = [schema_to_document(d) for d in documents]
        self.index.add_documents(prechunked)
        return [d.id for d in prechunked]
