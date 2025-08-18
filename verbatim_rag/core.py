"""Core implementation of the Verbatim RAG system.

Enhancement: Supports both legacy aggregate placeholders ([DISPLAY_SPANS], [CITATION_REFS])
and new per-fact placeholders of the form [FACT_1], [FACT_2], ... allowing templates
to interleave verbatim facts contextually. Citation-only facts (beyond the display
limit) expand to a numbered reference token (e.g. "[6]") without verbatim text.
"""

import re
from verbatim_rag.extractors import LLMSpanExtractor, SpanExtractor
from verbatim_rag.index import VerbatimIndex
from verbatim_rag.models import (
    QueryResponse,
)
from verbatim_rag.template_manager import TemplateManager
from verbatim_rag.response_builder import ResponseBuilder

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
        template_manager: TemplateManager | None = None,
        extractor: SpanExtractor | None = None,
        max_display_spans: int = 5,
        use_contextual_templates: bool = False,
        extraction_mode: str = "batch",
    ):
        """
        Initialize the Verbatim RAG system.

        :param index: The index to search for relevant documents
        :param model: The LLM model to use for generation
        :param k: The number of documents to retrieve
        :param template_manager: Optional template manager for response templates
        :param extractor: Optional custom extractor for relevant spans
        :param max_display_spans: Maximum number of spans to display verbatim
        :param use_contextual_templates: Whether to generate templates based on content
        :param extraction_mode: "batch" or "individual" span extraction
        """
        self.index = index
        self.model = model
        self.k = k
        self.max_display_spans = max_display_spans
        self.use_contextual_templates = use_contextual_templates
        self.extraction_mode = extraction_mode

        # Use provided components or create defaults
        self.template_manager = template_manager or TemplateManager(model=model)
        self.extractor = extractor or LLMSpanExtractor(
            model=model,
            extraction_mode=extraction_mode,
            max_display_spans=max_display_spans
        )
        self.response_builder = ResponseBuilder()

    def _generate_template(self, question: str, display_spans: list[str] = None, citation_count: int = 0) -> str:
        """
        Generate or select a template for the response.

        :param question: The user's question
        :param display_spans: Spans that will be displayed (for contextual templates)
        :param citation_count: Number of additional citations
        :return: A template string with placeholders
        """
        if self.use_contextual_templates and display_spans:
            return self.template_manager.generate_contextual_template(
                question, display_spans, citation_count
            )
        else:
            return self.template_manager.get_template(question)

    def _rank_and_split_spans(self, relevant_spans: dict[str, list[str]]) -> tuple[list[dict], list[dict]]:
        """
        Split spans into display vs citation-only, trusting the extractor's ordering.

        :param relevant_spans: Dictionary mapping doc text to span lists (already ordered by relevance)
        :return: Tuple of (display_spans, citation_spans) with minimal metadata
        """
        # Flatten spans, preserving the order from the extractors
        all_spans = []
        for doc_text, spans in relevant_spans.items():
            for span in spans:
                all_spans.append({
                    'text': span,
                    'doc_text': doc_text
                })

        # Split into display and citation-only (trust the extractor's relevance ordering)
        display_spans = all_spans[:self.max_display_spans]
        citation_spans = all_spans[self.max_display_spans:]

        return display_spans, citation_spans

    def _fill_template_enhanced(self, template: str, display_spans: list[dict], citation_spans: list[dict]) -> str:
        """
        Fill the template with display spans and citation references.

        :param template: The template string with placeholders
        :param display_spans: Spans to display verbatim with metadata
        :param citation_spans: Spans for citation reference only
        :return: The filled template
        """
        def _is_table(text: str) -> bool:
            """Heuristic to detect markdown tables in a span."""
            lines = [l for l in text.strip().splitlines() if l.strip()]
            if len(lines) < 2:
                return False
            pipe_lines = sum(1 for l in lines if '|' in l)
            if pipe_lines < 2:
                return False
            # Header separator detection (--- or :--- style) optional
            return True

        if display_spans:
            formatted_content = []
            for i, span_data in enumerate(display_spans, 1):
                span_text = span_data['text']
                if _is_table(span_text):
                    # Place citation number on its own line, blank line, then table so parser renders table
                    block = f"[{i}]\n\n{span_text.strip()}"
                else:
                    block = f"[{i}] {span_text}"
                formatted_content.append(block)
            display_content = "\n\n".join(formatted_content)
        else:
            display_content = "No relevant information found in the provided documents."

        # Handle citation references
        if citation_spans:
            start_num = len(display_spans) + 1
            end_num = len(display_spans) + len(citation_spans)
            citation_refs = " ".join(f"[{i}]" for i in range(start_num, end_num + 1))
        else:
            citation_refs = ""

        # ------------------------------------------------------------------
        # Per-fact placeholder handling
        # If the template contains any [FACT_n] placeholders, we expand them
        # individually. Otherwise we fall back to legacy aggregate placeholders.
        # ------------------------------------------------------------------
        fact_pattern = re.compile(r"\[FACT_(\d+)\]")
        total_spans = display_spans + citation_spans

        if fact_pattern.search(template):
            def replace_fact(m):
                idx = int(m.group(1))
                if 1 <= idx <= len(total_spans):
                    if idx <= len(display_spans):
                        span_text = display_spans[idx-1]['text']
                        if _is_table(span_text):
                            return f"[{idx}]\n\n{span_text.strip()}"
                        return f"[{idx}] {span_text}"
                    else:
                        return f"[{idx}]"  # citation-only
                return ""  # out of range

            filled_template = fact_pattern.sub(replace_fact, template)

            # Still support [CITATION_REFS] if present (will list citation-only numbers)
            if "[CITATION_REFS]" in filled_template and citation_spans:
                start_num = len(display_spans) + 1
                end_num = len(display_spans) + len(citation_spans)
                refs = " ".join(f"[{i}]" for i in range(start_num, end_num + 1))
                filled_template = filled_template.replace("[CITATION_REFS]", refs)
            elif "[CITATION_REFS]" in filled_template:
                filled_template = filled_template.replace("[CITATION_REFS]", "")
        else:
            # Legacy aggregate mode
            filled_template = template.replace("[DISPLAY_SPANS]", display_content)
            if "[RELEVANT_SENTENCES]" in template:
                filled_template = filled_template.replace("[RELEVANT_SENTENCES]", display_content)
            if "[CITATION_REFS]" in filled_template:
                filled_template = filled_template.replace("[CITATION_REFS]", citation_refs)

        return filled_template

    def query(self, question: str) -> QueryResponse:
        """
        Process a query through the Verbatim RAG system with enhanced span management.

        :param question: The user's question
        :return: A QueryResponse object containing the structured response
        """
        # Step 1: Retrieve relevant search results from the index
        search_results = self.index.search(question, k=self.k)

        # Step 2: Extract ALL relevant spans using the extractor
        print("Extracting relevant spans...")
        all_relevant_spans = self.extractor.extract_spans(question, search_results)

        print("Ranking and splitting spans...")
        # Step 3: Rank spans and split into display vs citation-only
        display_spans, citation_spans = self._rank_and_split_spans(all_relevant_spans)

        # Step 4: Generate template with context of ALL facts (display + citation-only)
        all_ordered_spans = display_spans + citation_spans
        all_texts = [span['text'] for span in all_ordered_spans]
        citation_count = len(citation_spans) if citation_spans else 0
        print("Generating template...")
        template = self._generate_template(question, all_texts, citation_count)

        # Step 5: Fill template with enhanced formatting
        answer = self._fill_template_enhanced(template, display_spans, citation_spans)

        # Step 6: Clean up the answer
        answer = self.response_builder.clean_answer(answer)

        # Step 7: Build the complete response using ALL spans for UI highlighting
        # The response builder needs all spans for document highlighting
        return self.response_builder.build_response(
            question=question,
            answer=answer,
            search_results=search_results,
            relevant_spans=all_relevant_spans,  # Keep all spans for UI highlighting
            display_span_count=len(display_spans),
        )

    async def _generate_template_async(self, question: str, display_spans: list[str] = None, citation_count: int = 0) -> str:
        """
        Async version of _generate_template.

        :param question: The user's question
        :param display_spans: Spans that will be displayed (for contextual templates)
        :param citation_count: Number of additional citations
        :return: A template string with placeholders
        """
        # Note: For now this just calls the sync version
        # Could be made truly async if template_manager supported async
        return self._generate_template(question, display_spans, citation_count)

    async def query_async(self, question: str) -> QueryResponse:
        """
        Async version of query method with enhanced span management.

        :param question: The user's question
        :return: A QueryResponse object containing the structured response
        """
        # Step 1: Retrieve relevant search results from the index
        search_results = self.index.search(question, k=self.k)

        # Step 2: Extract ALL relevant spans using the extractor
        all_relevant_spans = self.extractor.extract_spans(question, search_results)

        # Step 3: Rank spans and split into display vs citation-only
        display_spans, citation_spans = self._rank_and_split_spans(all_relevant_spans)

        # Step 4: Generate template with context of ALL facts (display + citation-only)
        all_ordered_spans = display_spans + citation_spans
        all_texts = [span['text'] for span in all_ordered_spans]
        citation_count = len(citation_spans) if citation_spans else 0
        template = await self._generate_template_async(question, all_texts, citation_count)

        # Step 5: Fill template with enhanced formatting
        answer = self._fill_template_enhanced(template, display_spans, citation_spans)

        # Step 6: Clean up the answer
        answer = self.response_builder.clean_answer(answer)

        # Step 7: Build the complete response using ALL spans for UI highlighting
        return self.response_builder.build_response(
            question=question,
            answer=answer,
            search_results=search_results,
            relevant_spans=all_relevant_spans,  # Keep all spans for UI highlighting
            display_span_count=len(display_spans),
        )
