"""
Centralized LLM client (RAG-agnostic) reused by verbatim_core components.
"""

from __future__ import annotations

import json
from typing import Optional, Dict, List

try:
    import openai
except ImportError:
    raise ImportError("OpenAI package required: pip install openai")


class LLMClient:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
        self.client = openai.OpenAI()
        self.async_client = openai.AsyncOpenAI()

    def complete(
        self, prompt: str, json_mode: bool = False, temperature: Optional[float] = None
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    async def complete_async(
        self, prompt: str, json_mode: bool = False, temperature: Optional[float] = None
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = await self.async_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    # Batch span extraction API
    def extract_relevant_spans_batch(
        self, question: str, documents: Dict[str, str]
    ) -> Dict[str, List[str]]:
        prompt = self._build_extraction_prompt(question, documents)
        try:
            response = self.complete(prompt, json_mode=True)
            return json.loads(response)
        except (json.JSONDecodeError, KeyError):
            return {doc_id: [] for doc_id in documents.keys()}

    async def extract_relevant_spans_batch_async(
        self, question: str, documents: Dict[str, str]
    ) -> Dict[str, List[str]]:
        prompt = self._build_extraction_prompt(question, documents)
        try:
            response = await self.complete_async(prompt, json_mode=True)
            return json.loads(response)
        except (json.JSONDecodeError, KeyError):
            return {doc_id: [] for doc_id in documents.keys()}

    # Single-doc convenience
    def extract_relevant_spans(self, question: str, document_text: str) -> List[str]:
        result = self.extract_relevant_spans_batch(question, {"doc": document_text})
        return result.get("doc", [])

    async def extract_relevant_spans_async(
        self, question: str, document_text: str
    ) -> List[str]:
        result = await self.extract_relevant_spans_batch_async(
            question, {"doc": document_text}
        )
        return result.get("doc", [])

    # Template generation
    def simple_complete(self, prompt: str) -> str:
        return self.complete(prompt)

    async def simple_complete_async(self, prompt: str) -> str:
        return await self.complete_async(prompt)

    def _build_extraction_prompt(self, question: str, documents: Dict[str, str]) -> str:
        return f"""Extract EXACT verbatim text spans from multiple documents that answer the question.

Return a JSON object mapping doc IDs to span arrays ordered by relevance.

Question: {question}
Documents:\n{json.dumps(documents, indent=2)}
"""
