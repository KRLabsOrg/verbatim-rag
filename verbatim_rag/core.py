"""Core implementation of the Verbatim RAG system.

Enhancement: Supports both legacy aggregate placeholders ([DISPLAY_SPANS], [CITATION_REFS])
and new per-fact placeholders of the form [FACT_1], [FACT_2], ... allowing templates
to interleave verbatim facts contextually. Citation-only facts (beyond the display
limit) expand to a numbered reference token (e.g. "[6]") without verbatim text.
"""

from typing import Optional, Any
import asyncio
import logging
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
        intent_detector=None,
        reranker=None,
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
        self.intent_detector = intent_detector
        self.reranker = reranker

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

        # Ensure template manager has access to RAG system for structured mode
        self.template_manager.set_rag_system(self)

        self.response_builder = ResponseBuilder()

    def _build_short_circuit_response(
        self, question: str, answer: str
    ) -> QueryResponse:
        cleaned = self.response_builder.clean_answer(answer or "")
        return self.response_builder.build_response(
            question=question,
            answer=cleaned,
            search_results=[],
            relevant_spans={},
            display_span_count=0,
        )

    def _decision_field(self, decision, field: str, default=None):
        if isinstance(decision, dict):
            return decision.get(field, default)
        return getattr(decision, field, default)

    def _apply_reranker(self, question: str, results: list):
        if not self.reranker:
            return results
        try:
            return self.reranker.rerank(question, results)
        except Exception as exc:
            logging.warning(f"Reranker failed, using original order: {exc}")
            return results

    async def _apply_reranker_async(self, question: str, results: list):
        if not self.reranker:
            return results
        try:
            if hasattr(self.reranker, "rerank_async"):
                return await self.reranker.rerank_async(question, results)
            return await asyncio.to_thread(self.reranker.rerank, question, results)
        except Exception as exc:
            logging.warning(f"Async reranker failed, using original order: {exc}")
            return results

    def _detect_intent(self, question: str):
        if not self.intent_detector:
            return None
        if hasattr(self.intent_detector, "detect"):
            return self.intent_detector.detect(question)
        return None

    async def _detect_intent_async(self, question: str):
        if not self.intent_detector:
            return None
        if hasattr(self.intent_detector, "detect_async"):
            return await self.intent_detector.detect_async(question)
        if hasattr(self.intent_detector, "detect"):
            return await asyncio.to_thread(self.intent_detector.detect, question)
        return None

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

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        filter: Optional[str] = None,
        hybrid_weights: Optional[dict[str, float]] = None,
        rrf_k: int = 60,
        search_params: Optional[dict[str, Any]] = None,
        return_search_results: Optional[bool] = False,
    ) -> QueryResponse | tuple[QueryResponse, list]:
        """
        Process a query through the Verbatim RAG system.

        :param question: The user's question
        :param k: Number of documents to retrieve (uses self.k if not provided)
        :param filter: Optional filter to narrow document search
        :param hybrid_weights: Optional dict mapping method names to weights (e.g., {"dense": 0.5, "sparse": 0.3, "full_text": 0.2})
        :param rrf_k: RRF constant for hybrid search (default: 60)
        :return: A QueryResponse object containing the structured response
        """
        # Step 0: Optional intent detection
        decision = self._detect_intent(question)
        route = self._decision_field(decision, "route")
        if decision and route and route != "continue":
            answer = self._decision_field(decision, "answer", "") or ""
            return self._build_short_circuit_response(question, answer)

        # Step 1: Retrieve documents
        k = k or self.k
        search_results = self.index.query(
            text=question,
            k=k,
            filter=filter,
            hybrid_weights=hybrid_weights,
            rrf_k=rrf_k,
            search_params=search_params,
        )
        search_results = self._apply_reranker(question, search_results)

        # Step 2: Check mode and extract accordingly
        if self.template_manager.current_mode == "structured":
            answer, all_relevant_spans = self._process_structured(
                question, search_results
            )
        else:
            # Standard extraction flow
            print("Extracting relevant spans...")
            all_relevant_spans = self.extractor.extract_spans(question, search_results)

            print("Processing spans...")
            display_spans, citation_spans = self._rank_and_split_spans(
                all_relevant_spans
            )

            print("Generating response...")
            answer = self.template_manager.process(
                question, display_spans, citation_spans
            )

        # Clean up and build response
        answer = self.response_builder.clean_answer(answer)

        response = self.response_builder.build_response(
            question=question,
            answer=answer,
            search_results=search_results,
            relevant_spans=all_relevant_spans,
            display_span_count=len(all_relevant_spans),
        )

        if return_search_results:
            return response, search_results

        return response

    def _process_structured(
        self, question: str, search_results: list
    ) -> tuple[str, dict]:
        """
        Process query in structured mode - template controls extraction.

        :param question: The user's question
        :param search_results: Retrieved documents
        :return: Tuple of (filled_answer, spans_dict in response_builder format)
        """
        strategy = self.template_manager.strategies["structured"]
        template = strategy.template
        placeholders = strategy.get_placeholder_hints()

        # Get document texts
        doc_texts = [getattr(r, "text", str(r)) for r in search_results]

        # Structured extraction via LLM - returns {PLACEHOLDER: [{text, doc}, ...]}
        span_map = self.llm_client.extract_structured(
            question, template, placeholders, doc_texts
        )

        # Fill template with spans
        answer = strategy.fill_with_spans(span_map)

        # Convert to response_builder format: {doc_text: [spans]}
        relevant_spans = self._convert_structured_to_doc_spans(span_map, doc_texts)

        return answer, relevant_spans

    async def query_async(
        self,
        question: str,
        k: Optional[int] = None,
        filter: Optional[str] = None,
        hybrid_weights: Optional[dict[str, float]] = None,
        rrf_k: int = 60,
        search_params: Optional[dict[str, Any]] = None,
    ) -> QueryResponse:
        """
        Async version of query method.

        :param question: The user's question
        :param k: Number of documents to retrieve (uses self.k if not provided)
        :param filter: Optional filter to narrow document search
        :param hybrid_weights: Optional dict mapping method names to weights (e.g., {"dense": 0.5, "sparse": 0.3, "full_text": 0.2})
        :param rrf_k: RRF constant for hybrid search (default: 60)
        :return: A QueryResponse object containing the structured response
        """
        # Step 0: Optional intent detection
        decision = await self._detect_intent_async(question)
        route = self._decision_field(decision, "route")
        if decision and route and route != "continue":
            answer = self._decision_field(decision, "answer", "") or ""
            return self._build_short_circuit_response(question, answer)

        # Step 1: Retrieve documents
        k = k or self.k
        search_results = self.index.query(
            text=question,
            k=k,
            filter=filter,
            hybrid_weights=hybrid_weights,
            rrf_k=rrf_k,
            search_params=search_params,
        )
        search_results = await self._apply_reranker_async(question, search_results)

        # Step 2: Check mode and extract accordingly
        if self.template_manager.current_mode == "structured":
            answer, all_relevant_spans = await self._process_structured_async(
                question, search_results
            )
        else:
            # Standard extraction flow
            print("Extracting relevant spans (async)...")
            all_relevant_spans = await self.extractor.extract_spans_async(
                question, search_results
            )

            print("Processing spans...")
            display_spans, citation_spans = self._rank_and_split_spans(
                all_relevant_spans
            )

            print("Generating response (async)...")
            answer = await self.template_manager.process_async(
                question, display_spans, citation_spans
            )

        # Clean up and build response
        answer = self.response_builder.clean_answer(answer)

        return self.response_builder.build_response(
            question=question,
            answer=answer,
            search_results=search_results,
            relevant_spans=all_relevant_spans,
            display_span_count=len(all_relevant_spans),
        )

    async def _process_structured_async(
        self, question: str, search_results: list
    ) -> tuple[str, dict]:
        """
        Async structured mode processing.

        :param question: The user's question
        :param search_results: Retrieved documents
        :return: Tuple of (filled_answer, spans_dict in response_builder format)
        """
        strategy = self.template_manager.strategies["structured"]
        template = strategy.template
        placeholders = strategy.get_placeholder_hints()

        # Get document texts
        doc_texts = [getattr(r, "text", str(r)) for r in search_results]

        # Structured extraction via LLM - returns {PLACEHOLDER: [{text, doc}, ...]}
        span_map = await self.llm_client.extract_structured_async(
            question, template, placeholders, doc_texts
        )

        # Fill template with spans (pass the new format)
        answer = strategy.fill_with_spans(span_map)

        # Convert to response_builder format: {doc_text: [spans]}
        relevant_spans = self._convert_structured_to_doc_spans(span_map, doc_texts)

        return answer, relevant_spans

    def _convert_structured_to_doc_spans(self, span_map: dict, doc_texts: list) -> dict:
        """
        Convert structured span_map to response_builder format.

        :param span_map: {PLACEHOLDER: [{text, doc}, ...]}
        :param doc_texts: List of document texts
        :return: {doc_text: [spans]}
        """
        # Initialize with empty lists for all docs
        result = {text: [] for text in doc_texts}

        # Collect all spans by document
        for placeholder, items in span_map.items():
            for item in items:
                doc_idx = item.get("doc", 0)
                span_text = item.get("text", "")
                if 0 <= doc_idx < len(doc_texts) and span_text:
                    doc_text = doc_texts[doc_idx]
                    if span_text not in result[doc_text]:  # Avoid duplicates
                        result[doc_text].append(span_text)

        return result

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
