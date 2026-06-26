"""
AWS Bedrock-backed LLM client using the Bedrock Runtime Converse API.

:class:`BedrockLLMClient` is a drop-in alternative to :class:`LLMClient` for
deployments that route language-model calls through Amazon Bedrock. It shares
all span-extraction and template-generation behaviour with the OpenAI client via
:class:`BaseLLMClient`; only the two completion primitives differ.
"""

import asyncio
import logging
from typing import Optional

from .llm_client import BaseLLMClient

logger = logging.getLogger(__name__)

try:
    import boto3
except ImportError:
    boto3 = None

# The Converse API has no native JSON-output flag, so json_mode is emulated by
# instructing the model to emit raw JSON and stripping any markdown fences from
# the response before it reaches the JSON parser in BaseLLMClient.
_JSON_INSTRUCTION = (
    "Respond with only a single valid JSON object. "
    "Do not wrap it in markdown code fences and do not add any text before or after it."
)


class BedrockLLMClient(BaseLLMClient):
    """
    LLM interaction handler backed by the AWS Bedrock Converse API.

    boto3 is synchronous, so :meth:`complete_async` runs the blocking Converse
    call in a worker thread to avoid blocking the event loop.
    """

    def __init__(
        self,
        model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        temperature: float = 0.7,
        region_name: str | None = None,
        max_tokens: int = 4096,
        client=None,
        **client_kwargs,
    ):
        """
        Initialize the Bedrock LLM client.

        :param model: The Bedrock model (or inference-profile) identifier to use,
            e.g. ``"anthropic.claude-3-5-sonnet-20241022-v2:0"`` or a regional
            profile such as ``"us.anthropic.claude-3-5-sonnet-20241022-v2:0"``.
        :param temperature: Default temperature for completions.
        :param region_name: AWS region for the Bedrock Runtime client. Falls back
            to the standard boto3 region resolution (env/config) when unset.
        :param max_tokens: Maximum tokens to generate per completion.
        :param client: Optional pre-built ``bedrock-runtime`` client. When unset,
            one is created via boto3.
        :param client_kwargs: Extra keyword arguments forwarded to
            ``boto3.client`` (e.g. credentials, endpoint_url).
        """
        super().__init__(model=model, temperature=temperature)
        self.max_tokens = max_tokens

        if client is not None:
            self.client = client
        else:
            if boto3 is None:
                raise ImportError("boto3 package required: pip install boto3")
            self.client = boto3.client("bedrock-runtime", region_name=region_name, **client_kwargs)

    def complete(
        self,
        prompt: str,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        system_prompt: str | None = None,
    ) -> str:
        response = self.client.converse(
            **self._build_request(prompt, json_mode, temperature, system_prompt)
        )
        return self._parse_response(response, json_mode)

    async def complete_async(
        self,
        prompt: str,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        system_prompt: str | None = None,
    ) -> str:
        return await asyncio.to_thread(self.complete, prompt, json_mode, temperature, system_prompt)

    def _build_request(
        self,
        prompt: str,
        json_mode: bool,
        temperature: Optional[float],
        system_prompt: str | None,
    ) -> dict:
        """Build the keyword arguments for a ``converse`` call."""
        request = {
            "modelId": self.model,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "temperature": temperature if temperature is not None else self.temperature,
                "maxTokens": self.max_tokens,
            },
        }

        system_blocks = []
        if system_prompt:
            system_blocks.append({"text": system_prompt})
        if json_mode:
            system_blocks.append({"text": _JSON_INSTRUCTION})
        if system_blocks:
            request["system"] = system_blocks

        return request

    def _parse_response(self, response: dict, json_mode: bool) -> str:
        """Extract the assistant text from a Converse response."""
        content = response.get("output", {}).get("message", {}).get("content", [])
        text = "".join(block.get("text", "") for block in content)
        if not text:
            raise ValueError("LLM returned empty or filtered response")
        if json_mode:
            text = _strip_code_fences(text)
        return text


def _strip_code_fences(text: str) -> str:
    """Remove a surrounding markdown code fence (```json ... ```), if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    # Drop the opening fence (``` or ```json) and a trailing closing fence.
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
