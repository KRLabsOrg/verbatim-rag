"""
Core implementation of the Verbatim RAG system.
"""

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
    ):
        """
        Initialize the Verbatim RAG system.

        :param index: The index to search for relevant documents
        :param model: The LLM model to use for generation
        :param k: The number of documents to retrieve
        :param template_manager: Optional template manager for response templates
        :param extractor: Optional custom extractor for relevant spans
        """
        self.index = index
        self.model = model
        self.k = k

        # Use provided components or create defaults
        self.template_manager = template_manager or TemplateManager()
        self.extractor = extractor or LLMSpanExtractor(model=model)
        self.response_builder = ResponseBuilder()

    def _generate_template(self, question: str) -> str:
        """
        Generate or select a template for the response.

        :param question: The user's question
        :return: A template string with placeholders for facts
        """
        return self.template_manager.get_template(question)

    def _fill_template(self, template: str, facts: list[list[str]]) -> str:
        """
        Fill the template with the extracted facts.

        :param template: The template string with placeholders
        :param facts: Lists of facts extracted from documents
        :return: The filled template
        """
        # Flatten the list of facts
        all_facts = [fact for doc_facts in facts for fact in doc_facts]

        if all_facts:
            # Format the facts
            formatted_content = []
            for fact in all_facts:
                formatted_content.append(f"• {fact}")

            formatted_content = "\n".join(formatted_content)
        else:
            formatted_content = (
                "No relevant information found in the provided documents."
            )

        filled_template = template.replace("[RELEVANT_SENTENCES]", formatted_content)

        return filled_template

    def query(self, question: str) -> QueryResponse:
        """
        Process a query through the Verbatim RAG system.

        :param question: The user's question
        :return: A QueryResponse object containing the structured response
        """
        # Step 1: Generate a template
        template = self._generate_template(question)

        # Step 2: Retrieve relevant search results from the index
        search_results = self.index.search(question, k=self.k)

        # Step 3: Extract relevant spans using the extractor
        relevant_spans = self.extractor.extract_spans(question, search_results)

        # Step 5: Fill the template with the marked context
        answer = self._fill_template(template, relevant_spans.values())

        # Step 6: Clean up the answer
        answer = self.response_builder.clean_answer(answer)

        # Step 7: Build the complete response using the response builder
        return self.response_builder.build_response(
            question=question,
            answer=answer,
            search_results=search_results,
            relevant_spans=relevant_spans,
        )

    async def _generate_template_async(self, question: str) -> str:
        """
        Async version of _generate_template.

        :param question: The user's question
        :return: A template string with placeholders for facts
        """
        return self.template_manager.get_template(question)

    async def query_async(self, question: str) -> QueryResponse:
        """
        Async version of query method.

        :param question: The user's question
        :return: A QueryResponse object containing the structured response
        """
        # Step 1: Generate a template
        template = await self._generate_template_async(question)

        # Step 2: Retrieve relevant search results from the index
        search_results = self.index.search(question, k=self.k)

        # Step 3: Extract relevant spans using the extractor
        relevant_spans = self.extractor.extract_spans(question, search_results)

        # Step 5: Fill the template with the marked context
        answer = self._fill_template(template, relevant_spans.values())

        # Step 6: Clean up the answer
        answer = self.response_builder.clean_answer(answer)

        # Step 7: Build the complete response using the response builder
        return self.response_builder.build_response(
            question=question,
            answer=answer,
            search_results=search_results,
            relevant_spans=relevant_spans,
        )
