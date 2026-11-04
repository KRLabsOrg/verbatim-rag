"""Core implementation of the Verbatim RAG system.

Enhancement: Supports both legacy aggregate placeholders ([DISPLAY_SPANS], [CITATION_REFS])
and new per-fact placeholders of the form [FACT_1], [FACT_2], ... allowing templates
to interleave verbatim facts contextually. Citation-only facts (beyond the display
limit) expand to a numbered reference token (e.g. "[6]") without verbatim text.
"""

import asyncio
import copy
import logging
from typing import Any, Optional

from verbatim_core import Answer, Chunk, ResultWarning, TransformConfig, Transformer
from verbatim_core._deprecation import _UNSET, warn_deprecated
from verbatim_core.errors import ConfigurationError, UpstreamError
from verbatim_core.templates import TemplateManager

from verbatim_rag.extractors import LLMExtractConfig, LLMSpanExtractor, SpanExtractor
from verbatim_rag.index import VerbatimIndex
from verbatim_rag.ingestion import schema_to_document
from verbatim_rag.llm_client import LLMClient
from verbatim_rag.models import QueryResponse
from verbatim_rag.response_builder import ResponseBuilder
from verbatim_rag.schema import DocumentSchema

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


logger = logging.getLogger(__name__)


_USE_TRANSFORMER = "VerbatimRAG(index, transformer=Transformer(extractor, templates))"


def _argument_or(value: Any, default: Any = None) -> Any:
    """A deprecated argument's value, or the default when the caller left it out."""
    return default if value is _UNSET or value is None else value


def _after_transformer(value: Any, keyword_value: Any, name: str) -> Any:
    """An argument that followed a transformer passed by position, which arrives in the place of
    an old argument.

    :raises ConfigurationError: If the caller also passed it by keyword
    """
    if value is _UNSET:
        return keyword_value
    if keyword_value is not None:
        raise ConfigurationError(
            f"VerbatimRAG got {name} twice, by position and by keyword. Pass it once.",
            code="invalid_value",
        )
    return value


def _reranker_fallback(error: UpstreamError) -> ResultWarning:
    """Log that the reranker's service failed, and the warning of the response."""
    logger.warning(
        "Keeping the retrieval order, because the reranker failed (%s)",
        error.code,
        extra={"event": "fallback_used", "warning_code": "reranker_fallback"},
    )
    return ResultWarning(
        code="reranker_fallback",
        message="The reranker failed, so the results keep the order of the retrieval.",
    )


class VerbatimRAG:
    """Retrieve documents and compose cited source excerpts.

    Built-in verified extractors preserve evidence text provenance. Retrieval
    quality, source truth, extraction relevance, and generated contextual
    framing remain outside that guarantee.
    """

    def __init__(
        self,
        index: VerbatimIndex,
        model: str | Transformer = _UNSET,
        k: int = _UNSET,
        template_manager: TemplateManager = _UNSET,
        extractor: SpanExtractor = _UNSET,
        max_display_spans: int = _UNSET,
        template_mode: str = _UNSET,
        extraction_mode: str = _UNSET,
        llm_client: LLMClient | None = None,
        intent_detector=None,
        reranker=None,
        *,
        transformer: Transformer | None = None,
    ):
        """
        Initialize the Verbatim RAG system.

        :param index: The index to search for relevant documents
        :param transformer: Writes the answers from the retrieved chunks. It may also be passed
            second, by position.
        :param llm_client: The LLM client of structured template mode, which answers without the
            transformer (ADR 0004)
        :param intent_detector: Optional detector that answers some questions without retrieval
        :param reranker: Optional reranker of the retrieved results
        :param model: Deprecated. Without a transformer, the LLM model of the client it builds.
        :param k: Deprecated. Pass k to query.
        :param template_manager: Deprecated, with extractor, max_display_spans, template_mode and
            extraction_mode: without a transformer, VerbatimRAG builds one from them.
        :raises ConfigurationError: If a transformer comes with the arguments that build one
        """
        if isinstance(model, Transformer):
            # VerbatimRAG(index, transformer, llm_client, intent_detector, reranker), by position:
            # the arguments after the transformer arrive in the old ones' places.
            transformer, model = model, _UNSET
            llm_client = _after_transformer(k, llm_client, "llm_client")
            intent_detector = _after_transformer(
                template_manager, intent_detector, "intent_detector"
            )
            reranker = _after_transformer(extractor, reranker, "reranker")
            k, template_manager, extractor = _UNSET, _UNSET, _UNSET
        if k is not _UNSET:
            warn_deprecated("VerbatimRAG(k=...)", use="query(question, k=...)")
        self.index = index
        self.k = 5 if k is _UNSET else k
        self.intent_detector = intent_detector
        self.reranker = reranker
        self.response_builder = ResponseBuilder()

        # The old arguments that build a transformer
        builders = {
            "model": model,
            "template_manager": template_manager,
            "extractor": extractor,
            "max_display_spans": max_display_spans,
            "template_mode": template_mode,
            "extraction_mode": extraction_mode,
        }
        builders_given = [name for name, value in builders.items() if value not in (_UNSET, None)]
        if transformer is not None:
            if builders_given:
                raise ConfigurationError(
                    f"VerbatimRAG got a transformer and {', '.join(builders_given)}, which build "
                    "one. Pass the transformer alone.",
                    code="invalid_value",
                )
            self.llm_client = llm_client
            self.transformer = transformer
            # The transformer's settings apply
            self._max_display_spans = None
        else:
            warn_deprecated("VerbatimRAG without a transformer", use=_USE_TRANSFORMER)
            self.llm_client = llm_client or LLMClient(_argument_or(model, "gpt-4o-mini"))
            self.transformer = Transformer(
                _argument_or(extractor)
                or LLMSpanExtractor(
                    self.llm_client,
                    config=LLMExtractConfig(extraction_mode=_argument_or(extraction_mode, "auto")),
                ),
                _argument_or(template_manager)
                or TemplateManager(
                    llm_client=self.llm_client,
                    default_mode=_argument_or(template_mode, "contextual"),
                ),
            )
            self._max_display_spans = _argument_or(max_display_spans, 5)

        # Structured mode's helpers call back into this object, until #74 replaces them (ADR 0004)
        self.template_manager._attach_rag_system(self)

    @property
    def extractor(self) -> SpanExtractor:
        """The extractor of the transformer."""
        return self.transformer.extractor

    @extractor.setter
    def extractor(self, extractor: SpanExtractor) -> None:
        warn_deprecated("Assigning VerbatimRAG.extractor", use=_USE_TRANSFORMER)
        # A copy, so a transformer shared with other code keeps its extractor
        self.transformer = copy.copy(self.transformer)
        self.transformer.extractor = extractor

    @property
    def template_manager(self) -> TemplateManager:
        """The templates of the transformer."""
        return self.transformer.templates

    @template_manager.setter
    def template_manager(self, templates: TemplateManager) -> None:
        warn_deprecated("Assigning VerbatimRAG.template_manager", use=_USE_TRANSFORMER)
        self.transformer = copy.copy(self.transformer)
        self.transformer.templates = templates
        templates._attach_rag_system(self)

    @property
    def max_display_spans(self) -> int:
        """The number of spans an answer shows."""
        if self._max_display_spans is not None:
            return self._max_display_spans
        return self.transformer.config.max_display_spans

    @max_display_spans.setter
    def max_display_spans(self, max_display_spans: int) -> None:
        warn_deprecated(
            "Assigning VerbatimRAG.max_display_spans",
            use="Transformer(..., config=TransformConfig(max_display_spans=...))",
        )
        self._max_display_spans = max_display_spans

    def _transform_config(self) -> TransformConfig | None:
        """The config of a transformer call: the transformer's settings with the old
        max_display_spans, or None for the transformer's settings alone."""
        if self._max_display_spans is None:
            return None
        return self.transformer.config.model_copy(
            update={"max_display_spans": self._max_display_spans}
        )

    def _build_short_circuit_response(self, question: str, answer: str) -> QueryResponse:
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

    def _apply_reranker(self, question: str, results: list) -> tuple[list, list[ResultWarning]]:
        """The reranked results, or the retrieval order with a warning when the reranker's
        service fails."""
        if not self.reranker:
            return results, []
        try:
            return self.reranker.rerank(question, results), []
        except UpstreamError as error:
            return results, [_reranker_fallback(error)]

    async def _apply_reranker_async(
        self, question: str, results: list
    ) -> tuple[list, list[ResultWarning]]:
        """Async version of `_apply_reranker`."""
        if not self.reranker:
            return results, []
        try:
            if hasattr(self.reranker, "rerank_async"):
                return await self.reranker.rerank_async(question, results), []
            return await asyncio.to_thread(self.reranker.rerank, question, results), []
        except UpstreamError as error:
            return results, [_reranker_fallback(error)]

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

    def _rank_and_split_spans(
        self, relevant_spans: dict[str, list[str]]
    ) -> tuple[list[dict], list[dict]]:
        """
        Split spans into display vs citation-only, trusting the extractor's ordering. StreamingRAG
        uses it until query_stream replaces it (#97).

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
        logger.debug("Retrieved %d results", len(search_results))
        search_results, warnings = self._apply_reranker(question, search_results)

        # Step 2: Structured mode answers by itself (ADR 0004), the other modes with the transformer
        if self.template_manager.current_mode == "structured":
            response = self._structured_response(
                question,
                search_results,
                *self._process_structured(question, search_results),
                warnings,
            )
        else:
            chunks = [result.to_chunk() for result in search_results]
            logger.debug("Answering from %d chunks", len(chunks))
            answer = self.transformer.transform(question, chunks, config=self._transform_config())
            response = self._response(question, answer, chunks, warnings)

        if return_search_results:
            return response, search_results

        return response

    def _process_structured(self, question: str, search_results: list) -> tuple[str, dict]:
        """
        Process query in structured mode - template controls extraction.

        :param question: The user's question
        :param search_results: Retrieved documents
        :return: Tuple of (filled_answer, spans_dict in response_builder format)
        """
        self._check_structured_client()
        strategy = self.template_manager.strategies["structured"]
        template = strategy.template
        placeholders = strategy.get_placeholder_hints()

        # Get document texts
        doc_texts = [getattr(r, "text", str(r)) for r in search_results]

        # Structured extraction via LLM - returns {PLACEHOLDER: [{text, doc}, ...]}
        span_map = self.llm_client.extract_structured(question, template, placeholders, doc_texts)

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
        logger.debug("Retrieved %d results", len(search_results))
        search_results, warnings = await self._apply_reranker_async(question, search_results)

        # Step 2: Structured mode answers by itself (ADR 0004), the other modes with the transformer
        if self.template_manager.current_mode == "structured":
            return self._structured_response(
                question,
                search_results,
                *await self._process_structured_async(question, search_results),
                warnings,
            )
        chunks = [result.to_chunk() for result in search_results]
        logger.debug("Answering from %d chunks", len(chunks))
        answer = await self.transformer.atransform(
            question, chunks, config=self._transform_config()
        )
        return self._response(question, answer, chunks, warnings)

    def _response(
        self, question: str, answer: Answer, chunks: list[Chunk], warnings: list[ResultWarning]
    ) -> QueryResponse:
        """The QueryResponse of the transformer's answer. The warnings of the retrieval join the
        answer's."""
        answer = answer.model_copy(update={"warnings": [*warnings, *answer.warnings]})
        logger.debug("Answered with %d citations", len(answer.citations))
        return self.response_builder.build_from_answer(question, answer, chunks)

    def _structured_response(
        self,
        question: str,
        search_results: list,
        answer: str,
        relevant_spans: dict,
        warnings: list[ResultWarning],
    ) -> QueryResponse:
        """The QueryResponse of a structured answer, with the warnings of the retrieval."""
        response = self.response_builder.build_response(
            question=question,
            answer=self.response_builder.clean_answer(answer),
            search_results=search_results,
            relevant_spans=relevant_spans,
            # Structured mode keeps its count until structured answers are redesigned (#74)
            display_span_count=len(relevant_spans),
        )
        return response.model_copy(update={"warnings": warnings})

    async def _process_structured_async(
        self, question: str, search_results: list
    ) -> tuple[str, dict]:
        """
        Async structured mode processing.

        :param question: The user's question
        :param search_results: Retrieved documents
        :return: Tuple of (filled_answer, spans_dict in response_builder format)
        """
        self._check_structured_client()
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

    def _check_structured_client(self) -> None:
        """Structured mode answers with VerbatimRAG's own client (ADR 0004)."""
        if self.llm_client is None:
            raise ConfigurationError(
                "Structured template mode needs an LLM client: "
                "VerbatimRAG(index, transformer=..., llm_client=...).",
                code="missing_llm_client",
            )

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
