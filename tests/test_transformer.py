"""The Transformer, with a stub extractor and real templates."""

import logging
import re

import pytest
from pydantic import ValidationError
from test_question_specific_templates import WordEmbedder
from transformer_contract import TransformerContract
from verbatim_core import (
    Chunk,
    ConfigurationError,
    ExtractedSpan,
    Extraction,
    InvalidInputError,
    ResultWarning,
    TransformConfig,
    Transformer,
    UpstreamError,
    Usage,
    VerbatimDeprecationWarning,
)
from verbatim_core.extractors import SpanExtractor
from verbatim_core.llm_client import Completion
from verbatim_core.templates import TemplateManager

QUESTION = "When did the bridge open?"
CHUNKS = [
    Chunk(id="opened", text="The bridge opened in March 1932. It carries eight lanes."),
    Chunk(id="tolls", text="Tolls are collected electronically."),
    Chunk(id="design", text="Bradfield designed the bridge. Trains crossed the bridge first."),
]


def words(text):
    return {word for word in re.findall(r"\w+", text.lower()) if len(word) > 3}


class StubExtractor(SpanExtractor):
    """Extracts every sentence that shares a word with the question."""

    def __init__(self, warnings=()):
        self.warnings = list(warnings)

    def extract(self, question, chunks, *, config=None):
        spans = []
        for chunk in chunks:
            for match in re.finditer(r"[^.!?\s][^.!?]*[.!?]", chunk.text):
                if words(match.group()) & words(question):
                    spans.append(
                        ExtractedSpan(
                            chunk_id=chunk.id,
                            start=match.start(),
                            end=match.end(),
                            text=match.group(),
                            score=0.9,
                        )
                    )
        usage = Usage(model_id="stub-extractor", model_version="1", units={"chunks": len(chunks)})
        return Extraction(spans=spans, usage=[usage], warnings=self.warnings)


class FakeTemplateLLM:
    """Writes a template that lists the shown spans and refers to the others."""

    model = "fake-llm"
    temperature = 0.0

    def __init__(self, error=None, template=None):
        self.error = error
        self.template = template
        self.calls = 0

    def template_completion(self, question, spans, citation_count, **settings):
        self.calls += 1
        if self.error:
            raise self.error
        if self.template is not None:
            return Completion(text=self.template, model="fake-llm-2026-01-01")
        shown = len(spans) - citation_count
        text = "The sources say:\n" + "\n".join(f"- [SPAN_{i}]" for i in range(1, shown + 1))
        if citation_count:
            text += "\nSee also [CITATION_REFS]."
        return Completion(text=text, model="fake-llm-2026-01-01", input_tokens=40, output_tokens=12)

    async def template_completion_async(self, *args, **settings):
        return self.template_completion(*args, **settings)


def static_transformer(**settings):
    return Transformer(StubExtractor(), TemplateManager(), **settings)


def contextual_transformer(llm=None, **settings):
    templates = TemplateManager(llm_client=llm or FakeTemplateLLM())
    return Transformer(
        StubExtractor(), templates, config=TransformConfig(template_mode="contextual", **settings)
    )


class TestStaticContract(TransformerContract):
    @pytest.fixture
    def transformer(self):
        return static_transformer()


class TestContextualContract(TransformerContract):
    @pytest.fixture
    def transformer(self):
        return contextual_transformer()


class TestRandomContract(TransformerContract):
    @pytest.fixture
    def transformer(self):
        return static_transformer(config=TransformConfig(template_mode="random"))


class ContractEmbedder(WordEmbedder):
    WORDS = ["tall", "arch", "who", "design"]


class TestQuestionSpecificContract(TransformerContract):
    @pytest.fixture
    def transformer(self):
        # The contract's two questions each match an example of their own.
        templates = TemplateManager()
        templates.use_question_specific_mode(
            [
                {"template": "Design: [RELEVANT_SENTENCES]", "examples": ["Who designed it?"]},
                {"template": "Height: [RELEVANT_SENTENCES]", "examples": ["How tall is the arch?"]},
            ],
            embedder=ContractEmbedder(),
        )
        return Transformer(
            StubExtractor(), templates, config=TransformConfig(template_mode="question_specific")
        )


class TestRandomTemplates:
    def heading(self, answer):
        return answer.text.splitlines()[0]

    def test_the_question_chooses_the_template(self):
        transformer = static_transformer(config=TransformConfig(template_mode="random"))

        headings = {self.heading(transformer.transform(QUESTION, CHUNKS)) for _ in range(5)}

        assert len(headings) == 1

    def test_the_choice_is_the_same_in_every_process(self):
        # Fixed by a hash of the question that doesn't depend on the process (ADR 0005).
        config = TransformConfig(template_mode="random")
        seeded = TransformConfig(template_mode="random", seed=7)
        transformer = static_transformer()

        assert self.heading(transformer.transform(QUESTION, CHUNKS, config=config)) == (
            "### Response"
        )
        assert self.heading(transformer.transform(QUESTION, CHUNKS, config=seeded)) == (
            "## Key Findings"
        )

    def test_the_seed_changes_the_choice(self):
        transformer = static_transformer()

        headings = {
            self.heading(
                transformer.transform(
                    QUESTION, CHUNKS, config=TransformConfig(template_mode="random", seed=seed)
                )
            )
            for seed in range(10)
        }

        assert len(headings) > 1


class TestAnswer:
    def test_the_displayed_citations_are_the_spans_the_answer_shows(self):
        transformer = static_transformer(config=TransformConfig(max_display_spans=2))

        answer = transformer.transform(QUESTION, CHUNKS)

        assert [citation.type for citation in answer.citations] == [
            "display",
            "display",
            "reference",
        ]
        shown, _, cited = (span.text for span in answer.spans)
        assert f"[1] {shown}" in answer.text
        assert cited not in answer.text

    def test_a_displayed_span_the_template_leaves_out_is_cited_for_reference(self):
        llm = FakeTemplateLLM(template="Answer: [SPAN_1] and [SPAN_3]")

        answer = contextual_transformer(llm).transform(QUESTION, CHUNKS)

        first, _, third = (span.text for span in answer.spans)
        assert answer.text == f"Answer: [1] {first} and [3] {third}"
        assert [citation.type for citation in answer.citations] == [
            "display",
            "reference",
            "display",
        ]

    def test_references_are_cited_in_a_contextual_answer(self):
        transformer = contextual_transformer(max_display_spans=1)

        answer = transformer.transform(QUESTION, CHUNKS)

        assert answer.text == (
            "The sources say:\n- [1] The bridge opened in March 1932.\nSee also [2] [3]."
        )

    def test_hidden_citations_leave_out_the_markers(self):
        config = TransformConfig(citation_mode="hidden")

        answer = static_transformer().transform(QUESTION, CHUNKS, config=config)

        assert "[1]" not in answer.text
        assert "The bridge opened in March 1932." in answer.text

    def test_the_citation_format_shapes_the_markers(self):
        config = TransformConfig(citation_format="<{number}>")

        answer = static_transformer().transform(QUESTION, CHUNKS, config=config)

        assert "<1> The bridge opened in March 1932." in answer.text

    def test_an_invalid_citation_format_is_rejected(self):
        with pytest.raises(ValidationError):
            TransformConfig(citation_format="[{page}]")

    def test_the_usage_lists_the_extractor_and_the_template_llm(self):
        answer = contextual_transformer().transform(QUESTION, CHUNKS)

        assert answer.usage == [
            Usage(model_id="stub-extractor", model_version="1", units={"chunks": 3}),
            Usage(
                model_id="fake-llm",
                model_version="fake-llm-2026-01-01",
                units={
                    "llm_calls": 1,
                    "spans": 3,
                    "characters": sum(len(span.text) for span in answer.spans),
                    "input_tokens": 40,
                    "output_tokens": 12,
                },
            ),
        ]

    def test_static_templates_add_no_usage(self):
        answer = static_transformer().transform(QUESTION, CHUNKS)

        assert [usage.model_id for usage in answer.usage] == ["stub-extractor"]

    def test_the_warnings_of_the_extraction_are_kept(self):
        warning = ResultWarning(code="quote_not_found", message="A quote for chunks[0] ...")
        transformer = Transformer(StubExtractor(warnings=[warning]), TemplateManager())

        assert transformer.transform(QUESTION, CHUNKS).warnings == [warning]


class TestConfig:
    async def test_structured_mode_in_a_call_config_is_an_input_error(self):
        transformer = static_transformer()
        config = TransformConfig(template_mode="structured")

        with pytest.raises(InvalidInputError) as raised:
            transformer.transform(QUESTION, CHUNKS, config=config)
        with pytest.raises(InvalidInputError):
            await transformer.atransform(QUESTION, CHUNKS, config=config)
        with pytest.raises(InvalidInputError):
            transformer.stream(QUESTION, CHUNKS, config=config)
        with pytest.raises(InvalidInputError):
            transformer.astream(QUESTION, CHUNKS, config=config)

        assert raised.value.param == "config.template_mode"

    def test_structured_mode_in_the_default_config_is_a_configuration_error(self):
        with pytest.raises(ConfigurationError):
            static_transformer(config=TransformConfig(template_mode="structured"))

    def test_a_config_of_another_type_is_rejected(self):
        with pytest.raises(InvalidInputError) as raised:
            static_transformer().transform(QUESTION, CHUNKS, config={"max_display_spans": 1})
        with pytest.raises(ConfigurationError):
            static_transformer(config={"max_display_spans": 1})

        assert raised.value.param == "config"


class TestDefaultConfigFromTheTemplates:
    def test_without_a_config_the_templates_settings_are_the_default(self):
        templates = TemplateManager(llm_client=FakeTemplateLLM(), default_mode="contextual")

        answer = Transformer(StubExtractor(), templates).transform(QUESTION, CHUNKS)

        assert answer.text.startswith("The sources say:")

    @pytest.mark.parametrize(
        "setter, value, check",
        [
            ("set_mode", "random", lambda text: text.startswith("### Response")),
            ("set_citation_mode", "hidden", lambda text: "[1]" not in text),
            ("set_citation_format", "<{number}>", lambda text: "<1>" in text),
        ],
    )
    def test_a_deprecated_setter_replaces_the_default(self, setter, value, check):
        templates = TemplateManager()
        transformer = Transformer(StubExtractor(), templates)

        with pytest.warns(VerbatimDeprecationWarning, match=f"TemplateManager.{setter}") as warned:
            getattr(templates, setter)(value)

        assert check(transformer.transform(QUESTION, CHUNKS).text)
        assert warned[0].filename == __file__

    def test_assigning_the_current_mode_replaces_the_default(self):
        templates = TemplateManager()
        transformer = Transformer(StubExtractor(), templates)

        with pytest.warns(VerbatimDeprecationWarning, match="TemplateManager.current_mode"):
            templates.current_mode = "random"

        assert transformer.transform(QUESTION, CHUNKS).text.startswith("### Response")

    def test_a_config_of_the_transformer_wins_over_the_templates_settings(self):
        templates = TemplateManager()
        transformer = Transformer(StubExtractor(), templates, config=TransformConfig())

        with pytest.warns(VerbatimDeprecationWarning):
            templates.set_mode("random")

        assert transformer.transform(QUESTION, CHUNKS).text.startswith("## Response")

    async def test_structured_mode_of_the_templates_is_a_configuration_error_at_the_call(self):
        # VerbatimRAG answers structured questions itself, with the same templates (ADR 0004).
        structured = TemplateManager()
        structured.use_structured_mode(template="## Results\n[RESULTS]")
        built_with_structured = Transformer(StubExtractor(), structured)
        templates = TemplateManager()
        transformer = Transformer(StubExtractor(), templates)
        templates.use_structured_mode(template="## Results\n[RESULTS]")

        with pytest.raises(ConfigurationError):
            built_with_structured.transform(QUESTION, CHUNKS)
        with pytest.raises(ConfigurationError):
            transformer.transform(QUESTION, CHUNKS)
        with pytest.raises(ConfigurationError):
            await transformer.atransform(QUESTION, CHUNKS)
        with pytest.raises(ConfigurationError):
            transformer.stream(QUESTION, CHUNKS)

    def test_a_default_that_needs_a_client_warns_at_construction(self):
        config = TransformConfig(template_mode="contextual")

        with pytest.warns(VerbatimDeprecationWarning, match="without an LLM client") as warned:
            Transformer(StubExtractor(), TemplateManager(), config=config)

        assert warned[0].filename == __file__


class TestTemplateGeneration:
    @pytest.mark.parametrize("asynchronous", [False, True])
    async def test_a_failed_llm_call_falls_back_to_the_static_template(self, asynchronous, caplog):
        error = UpstreamError("The LLM service failed.", code="server_error", service="llm")
        transformer = contextual_transformer(FakeTemplateLLM(error=error))

        if asynchronous:
            answer = await transformer.atransform(QUESTION, CHUNKS)
        else:
            answer = transformer.transform(QUESTION, CHUNKS)

        assert answer.text.startswith("## Response\n\nBased on the available documents:")
        assert [warning.code for warning in answer.warnings] == ["template_fallback"]
        [record] = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert (record.event, record.warning_code) == ("fallback_used", "template_fallback")
        assert [usage.model_id for usage in answer.usage] == ["stub-extractor"]

    @pytest.mark.parametrize("asynchronous", [False, True])
    async def test_a_rejected_key_propagates(self, asynchronous):
        rejected = ConfigurationError("The key was rejected.", code="authentication_failed")
        transformer = contextual_transformer(FakeTemplateLLM(error=rejected))

        with pytest.raises(ConfigurationError):
            if asynchronous:
                await transformer.atransform(QUESTION, CHUNKS)
            else:
                transformer.transform(QUESTION, CHUNKS)

    def test_an_empty_template_falls_back_with_a_warning(self):
        transformer = contextual_transformer(FakeTemplateLLM(template="  \n"))

        answer = transformer.transform(QUESTION, CHUNKS)

        assert answer.text.startswith("## Response\n\nBased on the available documents:")
        assert [warning.code for warning in answer.warnings] == ["template_fallback"]
        assert [usage.model_id for usage in answer.usage] == ["stub-extractor", "fake-llm"]

    async def test_a_client_written_before_030_writes_the_template_its_own_way(self):
        class OldClient:
            model = "old-llm"

            def generate_template(self, question, spans, citation_count, **settings):
                return "From my provider:\n[DISPLAY_SPANS]"

            async def generate_template_async(self, *args, **settings):
                return self.generate_template(*args, **settings)

        transformer = contextual_transformer(OldClient())

        answer = transformer.transform(QUESTION, CHUNKS)

        assert answer.text.startswith("From my provider:\n[1] The bridge opened in March 1932.")
        assert await transformer.atransform(QUESTION, CHUNKS) == answer
        assert [usage.model_id for usage in answer.usage] == ["stub-extractor"]

    def test_the_template_is_logged_only_on_the_content_logger(self, caplog):
        caplog.set_level(logging.DEBUG)

        contextual_transformer().transform(QUESTION, CHUNKS)

        logged = [record for record in caplog.records if "The sources say:" in record.getMessage()]
        assert [(record.name, record.levelno) for record in logged] == [
            ("verbatim_core.content", logging.DEBUG)
        ]

    def test_contextual_mode_without_a_client_uses_the_static_template_and_warns(self):
        config = TransformConfig(template_mode="contextual")

        with pytest.warns(VerbatimDeprecationWarning, match="without an LLM client") as warned:
            answer = static_transformer().transform(QUESTION, CHUNKS, config=config)
        with pytest.warns(VerbatimDeprecationWarning) as warned_by_stream:
            static_transformer().stream(QUESTION, CHUNKS, config=config)

        assert answer.text.startswith("## Response\n\nThe following is an unordered list")
        assert warned[0].filename == warned_by_stream[0].filename == __file__


class TestStreaming:
    def test_the_spans_arrive_before_the_template_is_written(self):
        llm = FakeTemplateLLM()
        events = contextual_transformer(llm).stream(QUESTION, CHUNKS)

        spans_event = next(events)
        calls_before_answer = llm.calls
        answer_event = next(events)

        assert (spans_event.type, calls_before_answer) == ("spans", 0)
        assert (answer_event.type, llm.calls) == ("answer", 1)

    async def test_a_failed_extraction_raises_from_the_stream(self):
        class FailingExtractor(StubExtractor):
            def extract(self, question, chunks, *, config=None):
                raise UpstreamError("The LLM service failed.", code="timeout", service="llm")

        transformer = Transformer(FailingExtractor(), TemplateManager())

        with pytest.raises(UpstreamError):
            list(transformer.stream(QUESTION, CHUNKS))
        with pytest.raises(UpstreamError):
            [event async for event in transformer.astream(QUESTION, CHUNKS)]
