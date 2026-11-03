"""The contract every span extractor keeps, as a test suite to reuse.

Subclass ExtractorContract in a test class and give it an `extractor` fixture. Override the
question and chunk fixtures when the extractor needs other input. The chunks must yield at least
one span.
"""

import logging

import pytest
from verbatim_core import Chunk, InvalidInputError


class ExtractorContract:
    @pytest.fixture
    def question(self):
        return "How tall is the arch of the bridge?"

    @pytest.fixture
    def chunks(self):
        return [
            Chunk(id="height", text="The arch of the bridge rises 134 metres above the water."),
            Chunk(id="tolls", text="Tolls are collected electronically in both directions."),
        ]

    @pytest.fixture
    def other_question(self):
        return "Who designed the bridge?"

    @pytest.fixture
    def other_chunks(self):
        return [Chunk(id="design", text="A team led by John Bradfield designed the bridge.")]

    def test_span_texts_are_the_chunk_text_at_their_offsets(self, extractor, question, chunks):
        extraction = extractor.extract(question, chunks)

        texts = {chunk.id: chunk.text for chunk in chunks}
        assert extraction.spans
        for span in extraction.spans:
            assert span.text == texts[span.chunk_id][span.start : span.end]

    async def test_extract_and_aextract_return_equal_results(self, extractor, question, chunks):
        assert await extractor.aextract(question, chunks) == extractor.extract(question, chunks)

    async def test_two_chunks_with_the_same_id_are_rejected(self, extractor, question, chunks):
        duplicated = [*chunks, Chunk(id=chunks[0].id, text="Another text.")]

        with pytest.raises(InvalidInputError) as raised:
            extractor.extract(question, duplicated)
        with pytest.raises(InvalidInputError):
            await extractor.aextract(question, duplicated)

        assert raised.value.param == f"chunks[{len(chunks)}].id"
        assert chunks[0].id not in str(raised.value)

    async def test_a_call_writes_nothing_to_the_working_directory_or_home(
        self, extractor, question, chunks, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))

        extractor.extract(question, chunks)
        await extractor.aextract(question, chunks)

        assert list(tmp_path.iterdir()) == []

    async def test_a_call_prints_nothing(self, extractor, question, chunks, capsys):
        extractor.extract(question, chunks)
        await extractor.aextract(question, chunks)

        assert capsys.readouterr() == ("", "")

    async def test_customer_text_is_logged_only_on_a_content_logger_at_debug(
        self, extractor, question, chunks, caplog
    ):
        caplog.set_level(logging.DEBUG)

        extractor.extract(question, chunks)
        await extractor.aextract(question, chunks)

        customer_texts = [question, *(chunk.text for chunk in chunks)]
        for record in caplog.records:
            if record.name.endswith(".content"):
                assert record.levelno == logging.DEBUG
            else:
                logged = record.getMessage() + repr(vars(record))
                assert not [text for text in customer_texts if text in logged], record.name

    def test_the_same_input_gives_the_same_result_after_other_input(
        self, extractor, question, chunks, other_question, other_chunks
    ):
        first = extractor.extract(question, chunks)
        extractor.extract(other_question, other_chunks)

        assert extractor.extract(question, chunks) == first

    def test_usage_names_a_model_and_its_version(self, extractor, question, chunks):
        extraction = extractor.extract(question, chunks)

        assert any(usage.model_id and usage.model_version for usage in extraction.usage)
