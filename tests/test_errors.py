"""Tests for the error classes exported by verbatim_core."""

import pickle

import pytest
from verbatim_core import ConfigurationError, InvalidInputError, UpstreamError, VerbatimError


class TestInvalidInputError:
    def test_is_caught_as_verbatim_error_and_value_error(self):
        with pytest.raises(VerbatimError):
            raise InvalidInputError("chunk ids must be unique", param="chunks")
        with pytest.raises(ValueError):
            raise InvalidInputError("chunk ids must be unique", param="chunks")

    def test_carries_code_and_param(self):
        error = InvalidInputError("chunk ids must be unique", param="chunks")
        assert str(error) == "chunk ids must be unique"
        assert error.code == "invalid_input"
        assert error.param == "chunks"

    def test_code_can_be_narrowed(self):
        error = InvalidInputError("two chunks share an id", code="duplicate_chunk_id")
        assert error.code == "duplicate_chunk_id"


class TestConfigurationError:
    def test_is_a_verbatim_error_but_not_a_value_error(self):
        error = ConfigurationError("contextual templates need an LLM client")
        assert isinstance(error, VerbatimError)
        assert not isinstance(error, ValueError)
        assert error.code == "configuration_error"


class TestUpstreamError:
    def test_carries_service_and_retry_hints(self):
        error = UpstreamError(
            "The LLM call timed out.", service="llm", retryable=True, retry_after=2.5
        )
        assert isinstance(error, VerbatimError)
        assert not isinstance(error, ValueError)
        assert error.code == "upstream_error"
        assert error.service == "llm"
        assert error.retryable is True
        assert error.retry_after == 2.5

    def test_defaults_say_nothing_about_retrying(self):
        error = UpstreamError("The LLM call failed.")
        assert error.service == "unknown"
        assert error.retryable is False
        assert error.retry_after is None


class TestPickling:
    @pytest.mark.parametrize(
        "error",
        [
            InvalidInputError("bad", code="duplicate_chunk_id", param="chunks"),
            ConfigurationError("missing client", code="missing_llm_client"),
            UpstreamError("timed out", service="llm", retryable=True, retry_after=1.0),
        ],
    )
    def test_round_trip_keeps_message_and_attributes(self, error):
        restored = pickle.loads(pickle.dumps(error))
        assert type(restored) is type(error)
        assert str(restored) == str(error)
        assert vars(restored) == vars(error)
