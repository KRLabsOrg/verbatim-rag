"""Tests for verbatim_core.bedrock_llm_client."""

import json

import pytest


def make_converse_response(text: str) -> dict:
    """Build a minimal Bedrock Converse API response containing ``text``."""
    return {"output": {"message": {"role": "assistant", "content": [{"text": text}]}}}


@pytest.fixture
def bedrock_client():
    """BedrockLLMClient with an injected mock ``bedrock-runtime`` client."""
    from unittest.mock import MagicMock

    from verbatim_core.bedrock_llm_client import BedrockLLMClient

    mock_runtime = MagicMock()
    client = BedrockLLMClient(model="test-model", client=mock_runtime)
    return client, mock_runtime


class TestInit:
    def test_injected_client_skips_boto3(self):
        from unittest.mock import MagicMock, patch

        from verbatim_core.bedrock_llm_client import BedrockLLMClient

        # Even with boto3 absent, an injected client must work.
        with patch("verbatim_core.bedrock_llm_client.boto3", None):
            client = BedrockLLMClient(client=MagicMock())
        assert client.model == "anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert client.temperature == 0.7

    def test_missing_boto3_raises(self):
        from unittest.mock import patch

        from verbatim_core.bedrock_llm_client import BedrockLLMClient

        with patch("verbatim_core.bedrock_llm_client.boto3", None):
            with pytest.raises(ImportError, match="boto3"):
                BedrockLLMClient()

    def test_builds_client_with_region(self):
        from unittest.mock import MagicMock, patch

        from verbatim_core.bedrock_llm_client import BedrockLLMClient

        with patch("verbatim_core.bedrock_llm_client.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            BedrockLLMClient(region_name="eu-central-1")

        assert mock_boto3.client.call_args.args[0] == "bedrock-runtime"
        assert mock_boto3.client.call_args.kwargs["region_name"] == "eu-central-1"


class TestComplete:
    def test_basic_complete(self, bedrock_client):
        client, mock_runtime = bedrock_client
        mock_runtime.converse.return_value = make_converse_response("Hello!")

        result = client.complete("Say hello")
        assert result == "Hello!"

        kwargs = mock_runtime.converse.call_args[1]
        assert kwargs["modelId"] == "test-model"
        assert kwargs["messages"] == [{"role": "user", "content": [{"text": "Say hello"}]}]
        assert kwargs["inferenceConfig"]["temperature"] == 0.7
        assert kwargs["inferenceConfig"]["maxTokens"] == 4096
        assert "system" not in kwargs

    def test_system_prompt_included(self, bedrock_client):
        client, mock_runtime = bedrock_client
        mock_runtime.converse.return_value = make_converse_response("ok")

        client.complete("question", system_prompt="You are helpful")

        kwargs = mock_runtime.converse.call_args[1]
        assert kwargs["system"] == [{"text": "You are helpful"}]

    def test_temperature_override(self, bedrock_client):
        client, mock_runtime = bedrock_client
        mock_runtime.converse.return_value = make_converse_response("ok")

        client.complete("test", temperature=0.0)

        kwargs = mock_runtime.converse.call_args[1]
        assert kwargs["inferenceConfig"]["temperature"] == 0.0

    def test_json_mode_adds_instruction(self, bedrock_client):
        client, mock_runtime = bedrock_client
        mock_runtime.converse.return_value = make_converse_response('{"key": "value"}')

        client.complete("Give JSON", json_mode=True)

        kwargs = mock_runtime.converse.call_args[1]
        # The JSON directive is appended as a system block.
        assert any("JSON" in block["text"] for block in kwargs["system"])

    def test_json_mode_strips_code_fences(self, bedrock_client):
        client, mock_runtime = bedrock_client
        fenced = '```json\n{"key": "value"}\n```'
        mock_runtime.converse.return_value = make_converse_response(fenced)

        result = client.complete("Give JSON", json_mode=True)
        assert json.loads(result) == {"key": "value"}

    def test_multiple_content_blocks_are_concatenated(self, bedrock_client):
        client, mock_runtime = bedrock_client
        mock_runtime.converse.return_value = {
            "output": {"message": {"content": [{"text": "Hello "}, {"text": "world"}]}}
        }

        assert client.complete("hi") == "Hello world"

    def test_empty_response_raises(self, bedrock_client):
        client, mock_runtime = bedrock_client
        mock_runtime.converse.return_value = {"output": {"message": {"content": []}}}

        with pytest.raises(ValueError, match="empty or filtered"):
            client.complete("anything")


class TestCompleteAsync:
    async def test_basic_complete_async(self, bedrock_client):
        client, mock_runtime = bedrock_client
        mock_runtime.converse.return_value = make_converse_response("async hi")

        result = await client.complete_async("Say hi")
        assert result == "async hi"
        assert mock_runtime.converse.called


class TestExtractSpans:
    def test_successful_extraction(self, bedrock_client):
        client, mock_runtime = bedrock_client
        response_data = {"doc_0": ["span one", "span two"], "doc_1": []}
        mock_runtime.converse.return_value = make_converse_response(json.dumps(response_data))

        result = client.extract_spans("What?", {"doc_0": "text one", "doc_1": "text two"})
        assert result == response_data

    def test_json_decode_error_returns_empty(self, bedrock_client):
        client, mock_runtime = bedrock_client
        mock_runtime.converse.return_value = make_converse_response("not valid json")

        result = client.extract_spans("What?", {"doc_0": "text"})
        assert result == {"doc_0": []}

    def test_single_doc_convenience(self, bedrock_client):
        client, mock_runtime = bedrock_client
        mock_runtime.converse.return_value = make_converse_response(json.dumps({"doc": ["found"]}))

        result = client.extract_relevant_spans("What?", "some document text")
        assert result == ["found"]


class TestGenerateTemplate:
    def test_per_fact_template(self, bedrock_client):
        client, mock_runtime = bedrock_client
        mock_runtime.converse.return_value = make_converse_response(
            "Here are the findings:\n[FACT_1]\n[FACT_2]"
        )

        result = client.generate_template("What?", ["span1", "span2"], citation_count=0)
        assert "[FACT_1]" in result

    def test_fallback_on_error(self, bedrock_client):
        client, mock_runtime = bedrock_client
        mock_runtime.converse.side_effect = Exception("Bedrock error")

        result = client.generate_template("What?", ["span1"], citation_count=0)
        assert "[DISPLAY_SPANS]" in result


class TestStripCodeFences:
    def test_no_fence_unchanged(self):
        from verbatim_core.bedrock_llm_client import _strip_code_fences

        assert _strip_code_fences('{"a": 1}') == '{"a": 1}'

    def test_plain_fence(self):
        from verbatim_core.bedrock_llm_client import _strip_code_fences

        assert _strip_code_fences('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_language_tagged_fence(self):
        from verbatim_core.bedrock_llm_client import _strip_code_fences

        assert _strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


class TestInterface:
    def test_is_base_llm_client(self, bedrock_client):
        from verbatim_core.llm_client import BaseLLMClient

        client, _ = bedrock_client
        assert isinstance(client, BaseLLMClient)
