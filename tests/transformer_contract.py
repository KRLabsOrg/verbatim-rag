"""The contract every Transformer setup keeps, as a test suite to reuse.

Subclass TransformerContract in a test class and give it a `transformer` fixture whose extractor
and templates give the same result for the same input. Override the question and chunk fixtures
when the extractor needs other input. The chunks must yield at least one span.
"""

import logging

import pytest
from verbatim_core import Chunk, TransformConfig


class TransformerContract:
    @pytest.fixture
    def question(self):
        return "How tall is the arch of the bridge?"

    @pytest.fixture
    def chunks(self):
        return [
            Chunk(id="height", text="The arch of the bridge rises 134 metres above the water."),
            Chunk(id="tolls", text="Tolls are collected electronically in both directions."),
            Chunk(id="paint", text="Painting the arch of the bridge takes about ten years."),
        ]

    @pytest.fixture
    def other_question(self):
        return "Who designed the bridge?"

    @pytest.fixture
    def other_chunks(self):
        return [Chunk(id="design", text="A team led by John Bradfield designed the bridge.")]

    @pytest.fixture
    def other_config(self):
        """A config that changes the answer to the question and chunks."""
        return TransformConfig(max_display_spans=1, citation_mode="hidden")

    async def test_sync_async_and_streaming_calls_agree(self, transformer, question, chunks):
        answer = transformer.transform(question, chunks)

        events = list(transformer.stream(question, chunks))
        async_events = [event async for event in transformer.astream(question, chunks)]

        assert await transformer.atransform(question, chunks) == answer
        assert events == async_events
        assert [event.type for event in events] == ["spans", "answer"]
        assert events[0].spans == answer.spans
        assert events[-1].answer == answer

    def test_the_same_input_gives_the_same_result_after_other_input(
        self, transformer, question, chunks, other_question, other_chunks
    ):
        first = transformer.transform(question, chunks)
        transformer.transform(other_question, other_chunks)

        assert transformer.transform(question, chunks) == first

    def test_a_call_config_does_not_leak_into_the_next_call(
        self, transformer, question, chunks, other_config
    ):
        default = transformer.transform(question, chunks)
        other = transformer.transform(question, chunks, config=other_config)

        assert other != default
        assert transformer.transform(question, chunks) == default

    async def test_a_call_writes_nothing_to_the_working_directory_or_home(
        self, transformer, question, chunks, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))

        transformer.transform(question, chunks)
        await transformer.atransform(question, chunks)
        list(transformer.stream(question, chunks))

        assert list(tmp_path.iterdir()) == []

    def test_usage_names_the_extractor_model(self, transformer, question, chunks):
        answer = transformer.transform(question, chunks)

        assert answer.usage
        assert any(usage.model_id and usage.model_version for usage in answer.usage)

    def test_every_span_has_one_citation_at_its_offsets(self, transformer, question, chunks):
        answer = transformer.transform(question, chunks)

        assert answer.spans
        assert [(c.chunk_id, c.start, c.end) for c in answer.citations] == [
            (span.chunk_id, span.start, span.end) for span in answer.spans
        ]
        assert [citation.number for citation in answer.citations] == list(
            range(1, len(answer.spans) + 1)
        )

    async def test_a_call_prints_nothing(self, transformer, question, chunks, capsys):
        transformer.transform(question, chunks)
        await transformer.atransform(question, chunks)
        list(transformer.stream(question, chunks))
        [event async for event in transformer.astream(question, chunks)]

        assert capsys.readouterr() == ("", "")

    async def test_customer_text_is_logged_only_on_a_content_logger_at_debug(
        self, transformer, question, chunks, caplog
    ):
        caplog.set_level(logging.DEBUG)

        answer = transformer.transform(question, chunks)
        await transformer.atransform(question, chunks)

        customer_texts = [
            question,
            *(chunk.text for chunk in chunks),
            *(chunk.id for chunk in chunks),
            *(span.text for span in answer.spans),
        ]
        for record in caplog.records:
            if record.name.endswith(".content"):
                assert record.levelno == logging.DEBUG
            else:
                logged = record.getMessage() + repr(vars(record))
                assert not [text for text in customer_texts if text in logged], record.name
