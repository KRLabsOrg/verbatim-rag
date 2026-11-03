"""LLMSpanExtractor, through extract and aextract with a fake LLM client."""

import json
import logging
import re
from types import SimpleNamespace

import pytest
from extractor_contract import ExtractorContract
from verbatim_core import (
    Chunk,
    ConfigurationError,
    InvalidInputError,
    UpstreamError,
    VerbatimDeprecationWarning,
)
from verbatim_core.extractors import LLMExtractConfig, LLMSpanExtractor
from verbatim_core.llm_client import Completion, LLMClient

QUESTION = "When did the bridge open?"


def sentences(text):
    return [sentence for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence]


def words(text):
    return {word for word in re.findall(r"\w+", text.lower()) if len(word) > 3}


def documents_in(prompt):
    """The documents of a prompt, as the bundled prompt or the custom one in these tests shows them."""
    bundled = re.search(r"Documents:\n(\{.*\})", prompt, re.DOTALL)
    if bundled:
        return json.loads(bundled.group(1))
    return dict(re.findall(r"\[(doc(?:_\d+)?)\]\n(.*)", prompt))


class FakeLLMClient:
    """Quotes every sentence of a chunk that shares a word with the question.

    :param quotes: The quotes to return for a chunk text instead
    :param reply: A reply text to return for every call instead
    :param fail_when: Takes the documents of a call and returns an error for the call to raise
    """

    model = "fake-llm"

    def __init__(self, quotes=None, reply=None, fail_when=None):
        self.quotes = quotes or {}
        self.reply = reply
        self.fail_when = fail_when
        self.calls = []

    def completion(self, prompt, *, json_mode=False, temperature=None, system_prompt=None):
        documents = documents_in(prompt)
        self.calls.append(
            SimpleNamespace(prompt=prompt, system_prompt=system_prompt, documents=documents)
        )
        error = self.fail_when and self.fail_when(documents)
        if error:
            raise error
        question = re.search(r"Question: (.*)", prompt).group(1)
        text = self.reply or json.dumps(
            {key: self._quotes(question, text) for key, text in documents.items()}
        )
        return Completion(
            text=text,
            model="fake-llm-2026-01-01",
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
        )

    async def completion_async(self, prompt, **settings):
        return self.completion(prompt, **settings)

    def _quotes(self, question, text):
        if text in self.quotes:
            return self.quotes[text]
        return [sentence for sentence in sentences(text) if words(sentence) & words(question)]


def chunk(text, id="chunk"):
    return Chunk(id=id, text=text)


def upstream_error():
    return UpstreamError("The LLM service failed.", code="server_error", service="llm")


OPENED = chunk("The bridge opened in March 1932. It carries eight lanes.", id="opened")
TOLLS = chunk("Tolls are collected electronically.", id="tolls")
DESIGN = chunk("The bridge opened after Bradfield designed it.", id="design")


class TestContract(ExtractorContract):
    @pytest.fixture
    def extractor(self):
        return LLMSpanExtractor(FakeLLMClient())


class TestLocatingQuotes:
    def test_a_fuzzy_quote_yields_the_chunk_text(self):
        spaced = chunk("Crews repaint the arch ; one coat takes ten years . It is grey.")
        quote = "Crews repaint the arch; one coat takes ten years."
        extractor = LLMSpanExtractor(
            FakeLLMClient(quotes={spaced.text: [quote]}),
            config=LLMExtractConfig(span_match_mode="fuzzy"),
        )

        extraction = extractor.extract(QUESTION, [spaced])

        assert [span.text for span in extraction.spans] == [
            "Crews repaint the arch ; one coat takes ten years ."
        ]
        assert (extraction.spans[0].start, extraction.spans[0].end) == (0, 51)

    def test_exact_mode_drops_a_quote_that_differs_from_the_chunk(self):
        spaced = chunk("Crews repaint the arch ; one coat takes ten years .")
        extractor = LLMSpanExtractor(
            FakeLLMClient(
                quotes={spaced.text: ["Crews repaint the arch; one coat takes ten years."]}
            )
        )

        extraction = extractor.extract(QUESTION, [spaced])

        assert extraction.spans == []
        assert [warning.code for warning in extraction.warnings] == ["quote_not_found"]

    def test_the_fuzzy_threshold_decides_which_quotes_are_located(self):
        text = chunk("The arch was painted grey in 1932.")
        llm = FakeLLMClient(quotes={text.text: ["The arch was painted gray in 1932."]})
        extractor = LLMSpanExtractor(llm, config=LLMExtractConfig(span_match_mode="fuzzy"))

        lenient = extractor.extract(QUESTION, [text])
        strict = extractor.extract(
            QUESTION, [text], config=LLMExtractConfig(span_match_mode="fuzzy", fuzzy_threshold=1.0)
        )

        assert len(lenient.spans) == 1
        assert strict.spans == []

    def test_a_repeated_quote_yields_its_first_occurrence(self):
        repeated = chunk("Tolls apply. The arch is grey. Tolls apply.")
        extractor = LLMSpanExtractor(FakeLLMClient(quotes={repeated.text: ["Tolls apply."]}))

        extraction = extractor.extract(QUESTION, [repeated])

        assert [(span.start, span.end) for span in extraction.spans] == [(0, 12)]

    def test_quotes_that_locate_the_same_span_collapse(self):
        extractor = LLMSpanExtractor(
            FakeLLMClient(quotes={TOLLS.text: ["Tolls are collected", "  Tolls are collected "]})
        )

        extraction = extractor.extract(QUESTION, [TOLLS])

        assert len(extraction.spans) == 1
        assert extraction.warnings == []

    def test_every_span_scores_one(self):
        spaced = chunk("Crews repaint the arch ; one coat takes ten years .")
        extractor = LLMSpanExtractor(
            FakeLLMClient(
                quotes={spaced.text: ["Crews repaint the arch; one coat takes ten years."]}
            ),
            config=LLMExtractConfig(span_match_mode="fuzzy"),
        )

        extraction = extractor.extract(QUESTION, [OPENED, spaced])

        assert len(extraction.spans) == 2
        assert {span.score for span in extraction.spans} == {1.0}

    def test_spans_keep_the_chunk_ids_and_the_order_of_the_chunks(self):
        extraction = LLMSpanExtractor(FakeLLMClient()).extract(QUESTION, [DESIGN, TOLLS, OPENED])

        assert [span.chunk_id for span in extraction.spans] == ["design", "opened"]


class TestUnlocatedQuotes:
    def extract(self):
        llm = FakeLLMClient(
            quotes={
                OPENED.text: ["Ferries run every ten minutes.", "The bridge opened in March 1932."],
                TOLLS.text: ["Tolls are free on Sundays."],
            }
        )
        return LLMSpanExtractor(llm).extract(QUESTION, [OPENED, TOLLS])

    def test_each_becomes_a_warning_that_names_its_chunk_by_position(self):
        extraction = self.extract()

        assert [(warning.code, warning.message) for warning in extraction.warnings] == [
            ("quote_not_found", "A quote for chunks[0] is not in the chunk."),
            ("quote_not_found", "A quote for chunks[1] is not in the chunk."),
        ]
        assert [span.text for span in extraction.spans] == ["The bridge opened in March 1932."]

    def test_they_are_counted_in_one_warning_record(self, caplog):
        caplog.set_level(logging.DEBUG)
        self.extract()

        warnings = [record for record in caplog.records if record.levelno == logging.WARNING]

        assert len(warnings) == 1
        assert warnings[0].getMessage() == "Dropped 2 quotes that are not in their chunks"
        assert (warnings[0].event, warnings[0].warning_code) == ("fallback_used", "quote_not_found")

    def test_the_quotes_appear_only_on_the_content_logger(self, caplog):
        caplog.set_level(logging.DEBUG)
        self.extract()

        logged = [
            record
            for record in caplog.records
            if "Ferries run every ten minutes." in record.getMessage()
        ]

        assert [(record.name, record.levelno) for record in logged] == [
            ("verbatim_core.content", logging.DEBUG)
        ]


class TestFailedCalls:
    async def test_a_failed_llm_call_fails_the_whole_extraction(self):
        llm = FakeLLMClient(
            fail_when=lambda documents: TOLLS.text in documents.values() and upstream_error()
        )
        extractor = LLMSpanExtractor(llm, config=LLMExtractConfig(extraction_mode="individual"))

        with pytest.raises(UpstreamError):
            extractor.extract(QUESTION, [OPENED, TOLLS])
        with pytest.raises(UpstreamError):
            await extractor.aextract(QUESTION, [OPENED, TOLLS])

    def test_a_rejected_key_propagates(self):
        rejected = ConfigurationError("The key was rejected.", code="authentication_failed")
        extractor = LLMSpanExtractor(FakeLLMClient(fail_when=lambda documents: rejected))

        with pytest.raises(ConfigurationError):
            extractor.extract(QUESTION, [OPENED])

    @pytest.mark.parametrize(
        "reply",
        [
            "The bridge opened in 1932.",
            '["The bridge opened in March 1932."]',
            '{"doc_0": "The bridge opened in March 1932."}',
            '{"doc_0": [1932]}',
        ],
        ids=["not json", "not an object", "not a list", "not text"],
    )
    async def test_a_reply_that_cant_be_read_raises(self, reply):
        extractor = LLMSpanExtractor(FakeLLMClient(reply=reply))

        with pytest.raises(UpstreamError) as raised:
            extractor.extract(QUESTION, [OPENED])
        with pytest.raises(UpstreamError):
            await extractor.aextract(QUESTION, [OPENED])

        assert (raised.value.code, raised.value.service, raised.value.retryable) == (
            "invalid_reply",
            "llm",
            True,
        )
        assert "1932" not in str(raised.value)

    def test_empty_quotes_are_skipped_without_a_warning(self):
        extractor = LLMSpanExtractor(FakeLLMClient(quotes={TOLLS.text: ["", "  "]}))

        extraction = extractor.extract(QUESTION, [TOLLS])

        assert (extraction.spans, extraction.warnings) == ([], [])

    def test_a_reply_that_leaves_a_chunk_out_gives_it_no_spans(self):
        extractor = LLMSpanExtractor(FakeLLMClient(reply='{"doc_1": []}'))

        extraction = extractor.extract(QUESTION, [OPENED, TOLLS])

        assert extraction.spans == []
        assert extraction.warnings == []


class TestCalls:
    def chunks(self, count):
        return [chunk(f"The bridge opened on day {i}.", id=f"day-{i}") for i in range(count)]

    @pytest.mark.parametrize(
        "mode, batch_size, chunk_count, calls",
        [
            ("batch", 2, 5, [2, 2, 1]),
            ("individual", 5, 3, [1, 1, 1]),
            ("auto", 3, 3, [3]),
            ("auto", 3, 4, [1, 1, 1, 1]),
        ],
    )
    async def test_the_extraction_mode_decides_the_chunks_of_each_call(
        self, mode, batch_size, chunk_count, calls
    ):
        llm = FakeLLMClient()
        config = LLMExtractConfig(extraction_mode=mode, batch_size=batch_size)
        extractor = LLMSpanExtractor(llm, config=config)

        extraction = extractor.extract(QUESTION, self.chunks(chunk_count))
        sync_calls = [len(call.documents) for call in llm.calls]
        llm.calls.clear()
        await extractor.aextract(QUESTION, self.chunks(chunk_count))

        assert sync_calls == calls
        assert sorted(len(call.documents) for call in llm.calls) == sorted(calls)
        assert len(extraction.spans) == chunk_count

    def test_a_call_config_replaces_the_default_for_that_call_only(self):
        llm = FakeLLMClient()
        extractor = LLMSpanExtractor(llm, config=LLMExtractConfig(extraction_mode="batch"))

        extractor.extract(
            QUESTION, self.chunks(3), config=LLMExtractConfig(extraction_mode="individual")
        )
        individual_calls = len(llm.calls)
        llm.calls.clear()
        extractor.extract(QUESTION, self.chunks(3))

        assert (individual_calls, len(llm.calls)) == (3, 1)

    def test_a_config_of_another_type_is_rejected(self):
        class OtherConfig:
            extraction_mode = "batch"

        with pytest.raises(InvalidInputError) as raised:
            LLMSpanExtractor(FakeLLMClient()).extract(QUESTION, [OPENED], config=OtherConfig())

        assert raised.value.param == "config"

    def test_no_chunks_need_no_call(self):
        llm = FakeLLMClient()

        extraction = LLMSpanExtractor(llm).extract(QUESTION, [])

        assert llm.calls == []
        assert extraction.spans == []
        assert extraction.usage[0].units == {"chunks": 0, "characters": 0, "llm_calls": 0}

    @pytest.mark.parametrize("mode", ["batch", "individual"])
    def test_a_custom_prompt_is_sent_with_its_system_prompt(self, mode):
        llm = FakeLLMClient()
        config = LLMExtractConfig(
            extraction_mode=mode,
            extraction_prompt="Quote the text.\nQuestion: {{ question }}\n\n{{ documents }}",
            system_prompt="You quote text.",
        )

        extraction = LLMSpanExtractor(llm, config=config).extract(QUESTION, [OPENED, TOLLS])

        assert llm.calls[0].prompt.startswith(f"Quote the text.\nQuestion: {QUESTION}\n\n[doc_0]\n")
        assert {call.system_prompt for call in llm.calls} == {"You quote text."}
        assert [span.chunk_id for span in extraction.spans] == ["opened"]

    def test_the_system_prompt_needs_a_custom_prompt(self):
        llm = FakeLLMClient()
        config = LLMExtractConfig(system_prompt="You quote text.")

        LLMSpanExtractor(llm, config=config).extract(QUESTION, [OPENED])

        assert llm.calls[0].system_prompt is None


class TestClientsWrittenBefore030:
    """Clients that replace LLMClient.complete, for example to call another provider."""

    class OwnProvider(LLMClient):
        def __init__(self):
            self.model = "own-model"
            self.fake = FakeLLMClient()

        def complete(self, prompt, json_mode=False, temperature=None, system_prompt=None):
            return self.fake.completion(prompt).text

        async def complete_async(
            self, prompt, json_mode=False, temperature=None, system_prompt=None
        ):
            return self.complete(prompt)

    async def test_their_complete_is_called(self):
        extractor = LLMSpanExtractor(self.OwnProvider())

        extraction = extractor.extract(QUESTION, [OPENED, TOLLS])

        assert [span.text for span in extraction.spans] == ["The bridge opened in March 1932."]
        assert await extractor.aextract(QUESTION, [OPENED, TOLLS]) == extraction

    def test_their_usage_has_no_model_version_or_tokens(self):
        extraction = LLMSpanExtractor(self.OwnProvider()).extract(QUESTION, [OPENED])

        [usage] = extraction.usage
        assert (usage.model_id, usage.model_version) == ("own-model", None)
        assert set(usage.units) == {"chunks", "characters", "llm_calls"}

    def test_a_client_with_only_complete_is_called(self):
        class OnlyComplete:
            model = "own-model"

            def complete(self, prompt, json_mode=False, temperature=None, system_prompt=None):
                return FakeLLMClient().completion(prompt).text

        extraction = LLMSpanExtractor(OnlyComplete()).extract(QUESTION, [OPENED])

        assert len(extraction.spans) == 1


class TestUsage:
    def test_names_the_client_model_and_the_version_the_service_returned(self):
        llm = FakeLLMClient()
        config = LLMExtractConfig(extraction_mode="batch", batch_size=2)

        extraction = LLMSpanExtractor(llm, config=config).extract(QUESTION, [OPENED, TOLLS, DESIGN])

        [usage] = extraction.usage
        assert (usage.model_id, usage.model_version) == ("fake-llm", "fake-llm-2026-01-01")
        assert usage.units == {
            "chunks": 3,
            "characters": len(OPENED.text) + len(TOLLS.text) + len(DESIGN.text),
            "llm_calls": 2,
            "input_tokens": sum(len(call.prompt.split()) for call in llm.calls),
            "output_tokens": usage.units["output_tokens"],
        }
        assert usage.units["output_tokens"] > 0

    def test_leaves_out_tokens_a_call_did_not_report(self):
        class NoTokens(FakeLLMClient):
            def completion(self, prompt, **settings):
                completion = super().completion(prompt, **settings)
                return Completion(text=completion.text, model=completion.model)

        extraction = LLMSpanExtractor(NoTokens()).extract(QUESTION, [OPENED])

        assert set(extraction.usage[0].units) == {"chunks", "characters", "llm_calls"}


class TestConstructor:
    def test_without_a_client_it_builds_one_and_warns(self):
        with pytest.warns(VerbatimDeprecationWarning, match="without an llm_client") as warned:
            extractor = LLMSpanExtractor(model="gpt-4o")

        assert isinstance(extractor.llm_client, LLMClient)
        assert extractor.llm_client.model == "gpt-4o"
        assert warned[0].filename == __file__

    def test_the_old_settings_go_into_the_config_and_warn(self):
        with pytest.warns(
            VerbatimDeprecationWarning,
            match="Passing batch_size, span_match_mode to LLMSpanExtractor",
        ):
            extractor = LLMSpanExtractor(FakeLLMClient(), batch_size=2, span_match_mode="fuzzy")

        assert extractor.config == LLMExtractConfig(batch_size=2, span_match_mode="fuzzy")

    def test_old_settings_can_be_passed_by_position(self):
        with pytest.warns(VerbatimDeprecationWarning):
            extractor = LLMSpanExtractor(FakeLLMClient(), "gpt-4o", "individual")

        assert extractor.config.extraction_mode == "individual"

    @pytest.mark.parametrize("argument", ["model", "max_display_spans"])
    def test_arguments_without_effect_warn(self, argument):
        value = {"model": "gpt-4o", "max_display_spans": 3}[argument]

        with pytest.warns(VerbatimDeprecationWarning, match=f"Passing {argument}"):
            extractor = LLMSpanExtractor(FakeLLMClient(), **{argument: value})

        assert extractor.config == LLMExtractConfig()

    @pytest.mark.parametrize(
        "setting", [{"extraction_mode": "sometimes"}, {"batch_size": 0}, {"fuzzy_threshold": 2}]
    )
    def test_an_invalid_old_setting_is_a_configuration_error(self, setting):
        with pytest.warns(VerbatimDeprecationWarning), pytest.raises(ConfigurationError) as raised:
            LLMSpanExtractor(FakeLLMClient(), **setting)

        assert raised.value.code == "invalid_value"
        assert next(iter(setting)) in str(raised.value)

    def test_old_settings_and_a_config_together_are_rejected(self):
        with pytest.raises(ConfigurationError):
            LLMSpanExtractor(FakeLLMClient(), batch_size=2, config=LLMExtractConfig())

    def test_a_config_of_another_type_is_rejected(self):
        with pytest.raises(ConfigurationError):
            LLMSpanExtractor(FakeLLMClient(), config={"batch_size": 2})


class TestExtractSpans:
    """The deprecated extract_spans and extract_spans_async."""

    def results(self, *chunks):
        return [SimpleNamespace(text=chunk.text) for chunk in chunks]

    async def test_they_return_the_span_texts_by_chunk_text_and_warn(self):
        extractor = LLMSpanExtractor(FakeLLMClient())

        with pytest.warns(VerbatimDeprecationWarning, match="LLMSpanExtractor.extract_spans"):
            spans = extractor.extract_spans(QUESTION, self.results(OPENED, TOLLS))
        with pytest.warns(VerbatimDeprecationWarning, match="extract_spans_async"):
            async_spans = await extractor.extract_spans_async(QUESTION, self.results(OPENED, TOLLS))

        assert (
            spans
            == async_spans
            == {
                OPENED.text: ["The bridge opened in March 1932."],
                TOLLS.text: [],
            }
        )

    @pytest.mark.parametrize("asynchronous", [False, True])
    async def test_a_failed_batch_call_is_retried_chunk_by_chunk(self, asynchronous):
        llm = FakeLLMClient(fail_when=lambda documents: len(documents) > 1 and upstream_error())
        extractor = LLMSpanExtractor(llm, config=LLMExtractConfig(extraction_mode="batch"))

        with pytest.warns(VerbatimDeprecationWarning):
            spans = await extract_spans(extractor, self.results(OPENED, DESIGN), asynchronous)

        assert spans == {
            OPENED.text: ["The bridge opened in March 1932."],
            DESIGN.text: ["The bridge opened after Bradfield designed it."],
        }
        assert [len(call.documents) for call in llm.calls] == [2, 1, 1]

    @pytest.mark.parametrize("mode", ["batch", "individual"])
    @pytest.mark.parametrize("asynchronous", [False, True])
    async def test_a_chunk_whose_call_fails_yields_no_spans(self, mode, asynchronous, caplog):
        llm = FakeLLMClient(
            fail_when=lambda documents: OPENED.text in documents.values() and upstream_error()
        )
        extractor = LLMSpanExtractor(llm, config=LLMExtractConfig(extraction_mode=mode))

        with pytest.warns(VerbatimDeprecationWarning):
            spans = await extract_spans(extractor, self.results(DESIGN, OPENED), asynchronous)

        assert spans == {
            DESIGN.text: ["The bridge opened after Bradfield designed it."],
            OPENED.text: [],
        }
        [failed] = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert failed.getMessage() == "No spans for chunks[1]: its LLM call failed (server_error)"
        assert (failed.event, failed.error_code) == ("item_failed", "server_error")
        assert failed.exc_info is not None

    @pytest.mark.parametrize("mode", ["batch", "individual"])
    @pytest.mark.parametrize("asynchronous", [False, True])
    async def test_a_rejected_key_still_raises(self, mode, asynchronous):
        rejected = ConfigurationError("The key was rejected.", code="authentication_failed")
        extractor = LLMSpanExtractor(
            FakeLLMClient(fail_when=lambda documents: rejected),
            config=LLMExtractConfig(extraction_mode=mode),
        )

        with pytest.warns(VerbatimDeprecationWarning), pytest.raises(ConfigurationError):
            await extract_spans(extractor, self.results(OPENED), asynchronous)

    async def test_no_results_give_an_empty_dict(self):
        extractor = LLMSpanExtractor(FakeLLMClient())

        with pytest.warns(VerbatimDeprecationWarning):
            assert extractor.extract_spans(QUESTION, []) == {}


async def extract_spans(extractor, results, asynchronous):
    if asynchronous:
        return await extractor.extract_spans_async(QUESTION, results)
    return extractor.extract_spans(QUESTION, results)
