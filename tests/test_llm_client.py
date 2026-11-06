"""Tests for verbatim_core.llm_client."""

import json
from types import SimpleNamespace

import openai
import pytest
from verbatim_core import ConfigurationError, UpstreamError

# Stand-ins for the HTTP request and response, whose library differs between openai releases.
REQUEST = SimpleNamespace(method="POST", url="https://llm.example/v1/chat/completions")
# Providers quote part of a rejected key in their message.
PROVIDER_MESSAGE = "Incorrect API key provided: sk-abc***xyz"


def http_response(status, headers=None):
    return SimpleNamespace(status_code=status, headers=headers or {}, request=REQUEST)


def status_error(error_class, status, code=None, headers=None):
    return error_class(
        PROVIDER_MESSAGE, response=http_response(status, headers), body={"code": code}
    )


OPENAI_ERRORS = {
    "timeout": (openai.APITimeoutError(request=REQUEST), UpstreamError, True),
    "connection_failed": (openai.APIConnectionError(request=REQUEST), UpstreamError, True),
    "authentication_failed": (
        status_error(openai.AuthenticationError, 401),
        ConfigurationError,
        None,
    ),
    "permission_denied": (
        status_error(openai.PermissionDeniedError, 403),
        ConfigurationError,
        None,
    ),
    "model_not_found": (status_error(openai.NotFoundError, 404), ConfigurationError, None),
    "quota_exceeded": (
        status_error(openai.RateLimitError, 429, code="insufficient_quota"),
        ConfigurationError,
        None,
    ),
    "rate_limited": (status_error(openai.RateLimitError, 429), UpstreamError, True),
    "server_error": (status_error(openai.InternalServerError, 503), UpstreamError, True),
    "request_rejected": (status_error(openai.BadRequestError, 400), UpstreamError, False),
    "invalid_reply": (
        openai.APIResponseValidationError(response=http_response(200), body=None),
        UpstreamError,
        True,
    ),
}


class TestCompletion:
    def test_reports_the_model_and_the_tokens_the_provider_returned(
        self, mock_llm_client, mock_openai_response
    ):
        client, mock_sync, _, make_response = mock_llm_client
        usage = SimpleNamespace(prompt_tokens=120, completion_tokens=30)
        mock_sync.chat.completions.create.return_value = make_response(
            "Hello!", model="test-model-2026-01-01", usage=usage
        )

        completion = client.completion("Say hello")

        assert completion.text == "Hello!"
        assert completion.model == "test-model-2026-01-01"
        assert (completion.input_tokens, completion.output_tokens) == (120, 30)

    async def test_async_reports_the_same(self, mock_llm_client, mock_openai_response):
        client, _, mock_async, make_response = mock_llm_client
        usage = SimpleNamespace(prompt_tokens=120, completion_tokens=30)
        mock_async.chat.completions.create.return_value = make_response("Hello!", usage=usage)

        completion = await client.completion_async("Say hello")

        assert (completion.text, completion.input_tokens) == ("Hello!", 120)

    def test_tokens_are_none_when_the_provider_sends_no_usage(
        self, mock_llm_client, mock_openai_response
    ):
        client, mock_sync, _, make_response = mock_llm_client
        mock_sync.chat.completions.create.return_value = make_response("Hello!", usage=None)

        completion = client.completion("Say hello")

        assert (completion.input_tokens, completion.output_tokens) == (None, None)

    def test_a_reply_without_text_raises(self, mock_llm_client, mock_openai_response):
        client, mock_sync, _, make_response = mock_llm_client
        mock_sync.chat.completions.create.return_value = make_response(None)

        with pytest.raises(UpstreamError) as raised:
            client.complete("Say hello")

        assert (raised.value.code, raised.value.service) == ("empty_reply", "llm")

    def test_a_reply_without_choices_raises(self, mock_llm_client, mock_openai_response):
        client, mock_sync, _, make_response = mock_llm_client
        response = make_response("unused")
        response.choices = []
        mock_sync.chat.completions.create.return_value = response

        with pytest.raises(UpstreamError) as raised:
            client.complete("Say hello")

        assert raised.value.code == "empty_reply"


class TestErrorTranslation:
    @pytest.mark.parametrize("code", OPENAI_ERRORS)
    def test_openai_errors_become_the_library_errors(self, mock_llm_client, code):
        client, mock_sync, _, _ = mock_llm_client
        error, error_class, retryable = OPENAI_ERRORS[code]
        mock_sync.chat.completions.create.side_effect = error

        with pytest.raises(error_class) as raised:
            client.complete("Say hello")

        assert raised.value.code == code
        assert raised.value.__cause__ is error
        assert PROVIDER_MESSAGE not in str(raised.value)
        if error_class is UpstreamError:
            assert (raised.value.service, raised.value.retryable) == ("llm", retryable)

    @pytest.mark.parametrize("code", ["authentication_failed", "rate_limited"])
    async def test_async_calls_translate_them_too(self, mock_llm_client, code):
        client, _, mock_async, _ = mock_llm_client
        error, error_class, _ = OPENAI_ERRORS[code]
        mock_async.chat.completions.create.side_effect = error

        with pytest.raises(error_class) as raised:
            await client.complete_async("Say hello")

        assert raised.value.code == code

    def test_a_rate_limit_passes_on_the_retry_after_header(self, mock_llm_client):
        client, mock_sync, _, _ = mock_llm_client
        error = status_error(openai.RateLimitError, 429, headers={"retry-after": "7"})
        mock_sync.chat.completions.create.side_effect = error

        with pytest.raises(UpstreamError) as raised:
            client.complete("Say hello")

        assert raised.value.retry_after == 7.0

    def test_other_errors_propagate_unchanged(self, mock_llm_client):
        client, mock_sync, _, _ = mock_llm_client
        mock_sync.chat.completions.create.side_effect = KeyError("bug")

        with pytest.raises(KeyError):
            client.complete("Say hello")


class TestComplete:
    def test_explicit_api_key_is_used(self):
        from unittest.mock import MagicMock, patch

        with patch("verbatim_core.llm_client.openai") as mock_openai:
            mock_openai.OpenAI.return_value = MagicMock()
            mock_openai.AsyncOpenAI.return_value = MagicMock()

            from verbatim_core.llm_client import LLMClient

            client = LLMClient(model="test-model", api_key="explicit-key")

        assert client.api_key == "explicit-key"
        assert mock_openai.OpenAI.call_args.kwargs["api_key"] == "explicit-key"
        assert mock_openai.AsyncOpenAI.call_args.kwargs["api_key"] == "explicit-key"

    def test_basic_complete(self, mock_llm_client, mock_openai_response):
        client, mock_sync, _, make_response = mock_llm_client
        mock_sync.chat.completions.create.return_value = make_response("Hello!")

        result = client.complete("Say hello")
        assert result == "Hello!"

        call_kwargs = mock_sync.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["messages"] == [{"role": "user", "content": "Say hello"}]

    def test_json_mode(self, mock_llm_client, mock_openai_response):
        client, mock_sync, _, make_response = mock_llm_client
        mock_sync.chat.completions.create.return_value = make_response('{"key": "value"}')

        client.complete("Give JSON", json_mode=True)

        call_kwargs = mock_sync.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_temperature_override(self, mock_llm_client, mock_openai_response):
        client, mock_sync, _, make_response = mock_llm_client
        mock_sync.chat.completions.create.return_value = make_response("ok")

        client.complete("test", temperature=0.0)

        call_kwargs = mock_sync.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0


class TestExtractSpans:
    def test_successful_extraction(self, mock_llm_client, mock_openai_response):
        client, mock_sync, _, make_response = mock_llm_client
        response_data = {"doc_0": ["span one", "span two"], "doc_1": []}
        mock_sync.chat.completions.create.return_value = make_response(json.dumps(response_data))

        result = client.extract_spans("What?", {"doc_0": "text one", "doc_1": "text two"})
        assert result == response_data

    def test_json_decode_error_returns_empty(self, mock_llm_client, mock_openai_response):
        client, mock_sync, _, make_response = mock_llm_client
        mock_sync.chat.completions.create.return_value = make_response("not valid json")

        result = client.extract_spans("What?", {"doc_0": "text"})
        assert result == {"doc_0": []}


class TestExtractRelevantSpans:
    def test_single_doc_convenience(self, mock_llm_client, mock_openai_response):
        client, mock_sync, _, make_response = mock_llm_client
        response_data = {"doc": ["found span"]}
        mock_sync.chat.completions.create.return_value = make_response(json.dumps(response_data))

        result = client.extract_relevant_spans("What?", "some document text")
        assert result == ["found span"]


class TestBuildExtractionPrompt:
    def test_prompt_contains_question_and_docs(self, mock_llm_client):
        client, _, _, _ = mock_llm_client
        prompt = client._build_extraction_prompt("What color?", {"doc_0": "The sky is blue."})
        assert "What color?" in prompt
        assert "The sky is blue." in prompt
        assert "doc_0" in prompt


class TestGenerateTemplate:
    def test_per_fact_template(self, mock_llm_client, mock_openai_response):
        client, mock_sync, _, make_response = mock_llm_client
        mock_sync.chat.completions.create.return_value = make_response(
            "Here are the findings:\n[FACT_1]\n[FACT_2]"
        )

        result = client.generate_template("What?", ["span1", "span2"], citation_count=0)
        assert "[FACT_1]" in result

    def test_fallback_on_error(self, mock_llm_client):
        client, mock_sync, _, _ = mock_llm_client
        mock_sync.chat.completions.create.side_effect = Exception("API error")

        result = client.generate_template("What?", ["span1"], citation_count=0)
        assert "[DISPLAY_SPANS]" in result


class TestTemplateCompletion:
    def test_returns_the_completion_of_the_template_prompt(self, mock_llm_client):
        client, mock_sync, _, make_response = mock_llm_client
        mock_sync.chat.completions.create.return_value = make_response("Findings:\n[FACT_1]")

        completion = client.template_completion("What?", ["span1"], citation_count=0)

        assert (completion.text, completion.model) == (
            "Findings:\n[FACT_1]",
            "test-model-2026-01-01",
        )
        prompt = mock_sync.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
        assert "span1" in prompt

    async def test_a_failed_call_raises_instead_of_falling_back(self, mock_llm_client):
        client, mock_sync, mock_async, _ = mock_llm_client
        error, _, _ = OPENAI_ERRORS["server_error"]
        mock_sync.chat.completions.create.side_effect = error
        mock_async.chat.completions.create.side_effect = error

        with pytest.raises(UpstreamError):
            client.template_completion("What?", ["span1"], citation_count=0)
        with pytest.raises(UpstreamError):
            await client.template_completion_async("What?", ["span1"], citation_count=0)
        assert "[DISPLAY_SPANS]" in client.generate_template("What?", ["span1"], citation_count=0)


class TestFallbackTemplate:
    def test_without_citations(self, mock_llm_client):
        client, _, _, _ = mock_llm_client
        result = client._fallback_template(has_citations=False)
        assert "[DISPLAY_SPANS]" in result
        assert "[CITATION_REFS]" not in result

    def test_with_citations(self, mock_llm_client):
        client, _, _, _ = mock_llm_client
        result = client._fallback_template(has_citations=True)
        assert "[CITATION_REFS]" in result
