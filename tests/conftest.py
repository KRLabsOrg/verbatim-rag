from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_openai_response():
    """Factory for creating mock OpenAI completion responses."""

    def _make(content: str, model: str = "test-model-2026-01-01", usage=None):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = content
        response.model = model
        response.usage = usage
        return response

    return _make


@pytest.fixture
def mock_llm_client(mock_openai_response):
    """LLMClient with mocked OpenAI sync and async clients."""
    with patch("verbatim_core.llm_client.openai") as mock_openai:
        mock_sync = MagicMock()
        mock_async = AsyncMock()
        mock_openai.OpenAI.return_value = mock_sync
        mock_openai.AsyncOpenAI.return_value = mock_async

        from verbatim_core.llm_client import LLMClient

        client = LLMClient(model="test-model")

        yield client, mock_sync, mock_async, mock_openai_response


@pytest.fixture
def sample_spans():
    """Sample display and citation spans for template tests."""
    display = [
        {"text": "The study found that X leads to Y.", "doc_text": "doc1"},
        {"text": "Results show Z is significant.", "doc_text": "doc2"},
    ]
    citation = [
        {"text": "Additional context about the methodology.", "doc_text": "doc3"},
    ]
    return display, citation


@pytest.fixture
def make_search_result():
    """Factory for creating mock search result objects."""

    def _make(text, title="", source="", score=1.0):
        result = MagicMock()
        result.text = text
        result.metadata = {"title": title, "source": source}
        result.id = "test_id"
        result.score = score
        return result

    return _make
