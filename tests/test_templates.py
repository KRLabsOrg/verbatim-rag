"""Tests for verbatim_core.templates (static mode + filler)."""

import json
import logging

import pytest
from verbatim_core import ConfigurationError, UpstreamError, VerbatimDeprecationWarning
from verbatim_core.templates.contextual import ContextualTemplate
from verbatim_core.templates.filler import TemplateFiller
from verbatim_core.templates.manager import TemplateManager
from verbatim_core.templates.random import RandomTemplate
from verbatim_core.templates.static import StaticTemplate
from verbatim_core.templates.structured import StructuredTemplate


class TestTemplateFiller:
    def setup_method(self):
        self.filler = TemplateFiller(citation_mode="inline")

    def test_aggregate_placeholder(self):
        template = "Here are findings:\n\n[DISPLAY_SPANS]"
        spans = [{"text": "Span one."}, {"text": "Span two."}]
        result = self.filler.fill(template, spans, [])
        assert "[1] Span one." in result
        assert "[2] Span two." in result

    def test_per_fact_placeholders(self):
        template = "Finding: [FACT_1]\nAlso: [FACT_2]"
        spans = [{"text": "Alpha."}, {"text": "Beta."}]
        result = self.filler.fill(template, spans, [])
        assert "[1] Alpha." in result
        assert "[2] Beta." in result

    def test_citation_refs(self):
        template = "[DISPLAY_SPANS]\n\nRefs: [CITATION_REFS]"
        display = [{"text": "Main point."}]
        citation = [{"text": "Extra ref."}]
        result = self.filler.fill(template, display, citation)
        assert "[2]" in result

    def test_hidden_citation_mode(self):
        filler = TemplateFiller(citation_mode="hidden")
        template = "[DISPLAY_SPANS]"
        spans = [{"text": "No numbers here."}]
        result = filler.fill(template, spans, [])
        assert "[1]" not in result
        assert "No numbers here." in result

    def test_linked_citation_refs_are_appended_to_display_spans(self):
        template = "[DISPLAY_SPANS]"
        display = [{"text": "Synthesized point.", "citation_ids": ["v1", "v3"]}]
        citation = [
            {"text": "Source one.", "citation_id": "v1"},
            {"text": "Source two.", "citation_id": "v2"},
            {"text": "Source three.", "citation_id": "v3"},
        ]
        result = self.filler.fill(template, display, citation)
        assert "[1] Synthesized point. [2] [4]" in result

    def test_linked_citations_suppress_flat_citation_refs(self):
        template = "[DISPLAY_SPANS]\n\nRefs: [CITATION_REFS]"
        display = [{"text": "Synthesized point.", "citation_ids": ["v1"]}]
        citation = [{"text": "Source one.", "citation_id": "v1"}]
        result = self.filler.fill(template, display, citation)
        assert "Refs:" in result
        assert "Refs: [2]" not in result

    def test_empty_template(self):
        assert self.filler.fill("", [], []) == ""

    def test_no_spans(self):
        result = self.filler.fill("[DISPLAY_SPANS]", [], [])
        assert "No relevant information" in result

    def test_invalid_citation_mode(self):
        with pytest.raises(ValueError):
            TemplateFiller(citation_mode="bogus")


class TestIsTable:
    def test_table_detected(self):
        text = "| Col A | Col B |\n|-------|-------|\n| val1 | val2 |"
        assert TemplateFiller._is_table(text) is True

    def test_non_table(self):
        assert TemplateFiller._is_table("Just a plain sentence.") is False

    def test_single_line_with_pipe(self):
        assert TemplateFiller._is_table("one | two") is False


class TestEnsurePlaceholder:
    def test_already_has_placeholder(self):
        t = "Hello [DISPLAY_SPANS]"
        assert TemplateFiller.ensure_placeholder(t) == t

    def test_adds_placeholder(self):
        t = "Hello world"
        result = TemplateFiller.ensure_placeholder(t)
        assert "[DISPLAY_SPANS]" in result

    def test_respects_fact_placeholder(self):
        t = "Hello [FACT_1]"
        assert TemplateFiller.ensure_placeholder(t) == t


class TestStaticTemplate:
    def test_default_template(self):
        st = StaticTemplate()
        template = st.generate("Any question", ["span1"])
        assert "[DISPLAY_SPANS]" in template

    def test_custom_template(self):
        st = StaticTemplate(template="Custom: [DISPLAY_SPANS]")
        assert st.generate("Q", []) == "Custom: [DISPLAY_SPANS]"

    def test_fill_delegates_to_filler(self):
        st = StaticTemplate()
        result = st.fill(
            "Findings: [DISPLAY_SPANS]",
            [{"text": "A fact."}],
            [],
        )
        assert "A fact." in result

    def test_save_load_state(self):
        st = StaticTemplate(template="Custom [DISPLAY_SPANS]")
        state = st.save_state()
        assert state["type"] == "static"

        st2 = StaticTemplate()
        st2.load_state(state)
        assert st2.template == "Custom [DISPLAY_SPANS]"

    def test_create_academic(self):
        st = StaticTemplate.create_academic()
        assert "Literature" in st.template

    def test_create_brief(self):
        st = StaticTemplate.create_brief()
        assert "[DISPLAY_SPANS]" in st.template


class TestTemplateManager:
    def test_default_mode_is_static(self):
        tm = TemplateManager(llm_client=None, default_mode="static")
        assert tm.get_current_mode() == "static"

    def test_available_modes_without_llm(self):
        tm = TemplateManager(llm_client=None)
        modes = tm.get_available_modes()
        assert "static" in modes
        assert "contextual" not in modes

    def test_set_mode(self):
        tm = TemplateManager(llm_client=None)
        with pytest.warns(VerbatimDeprecationWarning, match="TransformConfig"):
            assert tm.set_mode("random") is True
        assert tm.get_current_mode() == "random"

    def test_set_unknown_mode(self, caplog, capsys):
        tm = TemplateManager(llm_client=None)
        with pytest.warns(VerbatimDeprecationWarning):
            assert tm.set_mode("nonexistent") is False

        assert tm.get_current_mode() == "static"
        assert [record.levelno for record in caplog.records] == [logging.WARNING]
        assert capsys.readouterr() == ("", "")

    def test_process_static(self):
        tm = TemplateManager(llm_client=None, default_mode="static")
        result = tm.process(
            question="What?",
            display_spans=[{"text": "Answer here."}],
            citation_spans=[],
        )
        assert "Answer here." in result

    def test_linked_citation_inputs_only_use_display_spans_for_template_generation(self):
        tm = TemplateManager(llm_client=None, default_mode="static")
        all_spans, citation_count = tm._get_template_inputs(
            display_spans=[{"text": "Synthesized point.", "citation_ids": ["v1"]}],
            citation_spans=[{"text": "Source one.", "citation_id": "v1"}],
        )
        assert all_spans == ["Synthesized point."]
        assert citation_count == 0

    def test_use_static_mode_custom_template(self):
        tm = TemplateManager(llm_client=None)
        tm.use_static_mode(template="Custom: [DISPLAY_SPANS]")
        result = tm.process("Q", [{"text": "Fact."}], [])
        assert "Custom:" in result
        assert "Fact." in result


class TestTemplateManagerSettings:
    def test_structured_mode_is_switched_to_with_use_structured_mode(self):
        tm = TemplateManager(llm_client=None)

        with pytest.warns(VerbatimDeprecationWarning, match="use_structured_mode"):
            tm.set_mode("structured")

        assert tm.current_mode == "structured"

    def test_use_structured_mode_does_not_warn(self):
        # Structured mode is deprecated in one step when structured answers replace it (ADR 0004).
        tm = TemplateManager(llm_client=None)

        assert tm.use_structured_mode(template="## Results\n[RESULTS]") is True
        assert tm.current_mode == "structured"

    def test_switching_mode_with_its_setup_method_does_not_warn(self):
        tm = TemplateManager(llm_client=None)

        assert tm.use_random_mode() is True
        tm.use_static_mode(template="Facts: [DISPLAY_SPANS]")

        assert tm.current_mode == "static"

    def test_contextual_mode_without_a_client_falls_back_to_static_and_warns(self, capsys):
        with pytest.warns(VerbatimDeprecationWarning, match="without an LLM client") as warned:
            tm = TemplateManager(llm_client=None, default_mode="contextual")
        with pytest.warns(VerbatimDeprecationWarning, match="without an LLM client"):
            assert tm.use_contextual_mode() is False

        assert tm.current_mode == "static"
        assert warned[0].filename == __file__
        assert capsys.readouterr() == ("", "")

    def test_set_mode_to_contextual_without_a_client_keeps_the_mode(self):
        tm = TemplateManager(llm_client=None)

        with pytest.warns(VerbatimDeprecationWarning) as warned:
            assert tm.set_mode("contextual") is False

        assert tm.current_mode == "static"
        assert any("without an LLM client" in str(warning.message) for warning in warned)

    def test_an_invalid_citation_format_is_a_configuration_error(self):
        tm = TemplateManager(llm_client=None)

        with pytest.warns(VerbatimDeprecationWarning):
            with pytest.raises(ConfigurationError) as raised:
                tm.set_citation_format("[{page}]")

        assert raised.value.code == "invalid_value"
        assert tm.config.citation_format == "[{number}]"

    def test_assigning_an_invalid_citation_mode_is_a_configuration_error(self):
        tm = TemplateManager(llm_client=None)

        with pytest.warns(VerbatimDeprecationWarning):
            with pytest.raises(ConfigurationError):
                tm.citation_mode = "footnotes"

        assert tm.config.citation_mode == "inline"

    def test_assigning_the_citation_mode_warns(self):
        tm = TemplateManager(llm_client=None)

        with pytest.warns(VerbatimDeprecationWarning, match="TemplateManager.citation_mode"):
            tm.citation_mode = "hidden"

        assert tm.config.citation_mode == "hidden"

    def test_random_mode_chooses_by_the_question(self):
        # The same hash of the question as the Transformer's random mode (ADR 0005).
        tm = TemplateManager(llm_client=None)
        tm.use_random_mode()

        answers = {
            tm.process("When did the bridge open?", [{"text": "In 1932."}], []) for _ in "abc"
        }

        assert [answer.splitlines()[0] for answer in answers] == ["### Response"]

    def test_a_rejected_key_propagates_from_generating_templates(self):
        tm = TemplateManager(llm_client=FakeLLM(error=REJECTED_KEY))
        tm.use_random_mode()

        with pytest.raises(ConfigurationError):
            tm.generate_random_templates()

    def test_generating_templates_outside_random_mode_is_logged(self, caplog, capsys):
        tm = TemplateManager(llm_client=None)

        assert tm.generate_random_templates() is False
        assert [record.levelno for record in caplog.records] == [logging.WARNING]
        assert capsys.readouterr() == ("", "")


class TestTemplateManagerFiles:
    def test_a_saved_setup_loads_without_warnings(self, tmp_path):
        saved = TemplateManager(llm_client=None)
        saved.use_random_mode(templates=["One: [DISPLAY_SPANS]", "Two: [DISPLAY_SPANS]"])
        saved.save(str(tmp_path / "templates.json"))

        loaded = TemplateManager(llm_client=None)

        assert loaded.load(str(tmp_path / "templates.json")) is True
        assert loaded.current_mode == "random"
        assert loaded.strategies["random"].templates == [
            "One: [DISPLAY_SPANS]",
            "Two: [DISPLAY_SPANS]",
        ]

    def test_the_citation_settings_are_saved_and_loaded(self, tmp_path):
        saved = TemplateManager(llm_client=FakeLLM())
        with pytest.warns(VerbatimDeprecationWarning):
            saved.set_citation_format("<{number}>")
            saved.set_citation_mode("hidden")
        saved.save(str(tmp_path / "templates.json"))

        loaded = TemplateManager(llm_client=FakeLLM())
        loaded.load(str(tmp_path / "templates.json"))

        assert (loaded.config.citation_mode, loaded.config.citation_format) == (
            "hidden",
            "<{number}>",
        )
        assert "<1>" not in loaded.process("Q?", [{"text": "A fact."}], [])

    def test_the_citation_format_of_a_setup_saved_before_030_is_loaded(self, tmp_path):
        state = {
            "current_mode": "static",
            "strategies": {"contextual": {"citation_format": "<{number}>"}},
        }
        (tmp_path / "templates.json").write_text(json.dumps(state))

        loaded = TemplateManager(llm_client=FakeLLM())
        loaded.load(str(tmp_path / "templates.json"))

        assert loaded.config.citation_format == "<{number}>"
        assert "<1>" in loaded.process("Q?", [{"text": "A fact."}], [])

    def test_a_missing_file_is_logged(self, tmp_path, caplog, capsys):
        assert TemplateManager(llm_client=None).load(str(tmp_path / "missing.json")) is False

        assert [record.levelno for record in caplog.records] == [logging.WARNING]
        assert capsys.readouterr() == ("", "")

    def test_a_file_that_cannot_be_read_is_logged(self, tmp_path, caplog, capsys):
        (tmp_path / "broken.json").write_text("{not json")

        assert TemplateManager(llm_client=None).load(str(tmp_path / "broken.json")) is False

        assert [record.levelno for record in caplog.records] == [logging.WARNING]
        assert capsys.readouterr() == ("", "")

    def test_a_strategy_state_that_cannot_be_read_is_logged(self, tmp_path, caplog, capsys):
        state = {"current_mode": "random", "strategies": {"random": {"templates": ["no slot"]}}}
        (tmp_path / "templates.json").write_text(json.dumps(state))
        tm = TemplateManager(llm_client=None)

        assert tm.load(str(tmp_path / "templates.json")) is True

        assert tm.strategies["random"].templates == []
        assert [record.levelno for record in caplog.records] == [logging.WARNING]
        assert capsys.readouterr() == ("", "")


class FakeLLM:
    """Replies with a fixed text, or raises."""

    model = "fake-llm"
    temperature = 0.0

    def __init__(self, reply="", error=None):
        self.reply = reply
        self.error = error

    def complete(self, prompt, **settings):
        if self.error:
            raise self.error
        return self.reply

    def generate_template(self, **request):
        return self.complete("")

    async def generate_template_async(self, **request):
        return self.complete("")


SERVER_ERROR = UpstreamError("The LLM service failed.", code="server_error", service="llm")
REJECTED_KEY = ConfigurationError("The LLM service rejected the key.", code="authentication_failed")


class TestRandomTemplatePool:
    def test_generated_templates_are_added_and_logged(self, caplog, capsys):
        caplog.set_level(logging.INFO)
        llm = FakeLLM(reply="Facts:\n[DISPLAY_SPANS] and more\nno slot here")
        strategy = RandomTemplate(templates=["A: [DISPLAY_SPANS]"], llm_client=llm)

        strategy.generate_diverse_templates(count=2)

        assert strategy.templates == ["A: [DISPLAY_SPANS]", "[DISPLAY_SPANS] and more"]
        assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
            (logging.INFO, "Generated and added 1 new templates to the pool")
        ]
        assert capsys.readouterr() == ("", "")

    def test_a_failed_llm_call_adds_the_fallback_templates(self, caplog, capsys):
        strategy = RandomTemplate(
            templates=["A: [DISPLAY_SPANS]"], llm_client=FakeLLM(error=SERVER_ERROR)
        )

        strategy.generate_diverse_templates(count=2)

        assert len(strategy.templates) == 4
        [record] = caplog.records
        assert (record.levelno, record.event, record.warning_code) == (
            logging.WARNING,
            "fallback_used",
            "template_fallback",
        )
        assert capsys.readouterr() == ("", "")

    def test_a_rejected_key_propagates(self):
        strategy = RandomTemplate(llm_client=FakeLLM(error=REJECTED_KEY))

        with pytest.raises(ConfigurationError):
            strategy.generate_diverse_templates(count=2)


class TestContextualTemplateWithoutAResult:
    """`generate` and `generate_async`, which return the template text alone."""

    @pytest.mark.parametrize("asynchronous", [False, True])
    async def test_a_failed_llm_call_falls_back_to_the_fallback_template(
        self, asynchronous, caplog, capsys
    ):
        strategy = ContextualTemplate(FakeLLM(error=SERVER_ERROR))

        if asynchronous:
            template = await strategy.generate_async("Q?", ["A span."], 0)
        else:
            template = strategy.generate("Q?", ["A span."], 0)

        assert template.startswith("## Response\n\nBased on the available documents:")
        [record] = caplog.records
        assert (record.levelno, record.event, record.warning_code) == (
            logging.WARNING,
            "fallback_used",
            "template_fallback",
        )
        assert capsys.readouterr() == ("", "")

    @pytest.mark.parametrize("asynchronous", [False, True])
    async def test_a_rejected_key_propagates(self, asynchronous):
        strategy = ContextualTemplate(FakeLLM(error=REJECTED_KEY))

        with pytest.raises(ConfigurationError):
            if asynchronous:
                await strategy.generate_async("Q?", ["A span."], 0)
            else:
                strategy.generate("Q?", ["A span."], 0)

    def test_llm_settings_in_a_saved_state_are_logged(self, caplog, capsys):
        caplog.set_level(logging.DEBUG)

        ContextualTemplate(FakeLLM()).load_state({"model": "gpt-4o-mini", "temperature": 0.3})

        assert [record.levelno for record in caplog.records] == [logging.DEBUG]
        assert capsys.readouterr() == ("", "")


class FakeRAG:
    async def query_async(self, question):
        return type("Response", (), {"answer": "The structured answer."})()


class TestStructuredModeCallbacks:
    """The templates call back into VerbatimRAG through these until #74 (ADR 0004)."""

    def test_set_rag_system_warns(self):
        with pytest.warns(VerbatimDeprecationWarning, match="VerbatimRAG.query_async") as warned:
            TemplateManager(llm_client=None).set_rag_system(FakeRAG())
        with pytest.warns(VerbatimDeprecationWarning, match="VerbatimRAG.query_async"):
            StructuredTemplate().set_rag_system(FakeRAG())

        assert warned[0].filename == __file__

    async def test_fill_async_warns_and_still_answers(self):
        strategy = StructuredTemplate(rag_system=FakeRAG())

        with pytest.warns(VerbatimDeprecationWarning, match="VerbatimRAG.query_async"):
            answer = await strategy.fill_async("Q?")

        assert answer == "The structured answer."

    def test_a_rag_system_given_to_the_constructor_does_not_warn(self):
        rag = FakeRAG()
        tm = TemplateManager(llm_client=None, rag_system=rag)

        tm.use_structured_mode(template="## Results\n[RESULTS]")

        assert tm.strategies["structured"].rag_system is rag


class TestContextualTemplateWithAClientWrittenBefore030:
    """A client that replaces generate_template, for example to call another provider."""

    @pytest.mark.parametrize("asynchronous", [False, True])
    async def test_an_exception_of_its_own_falls_back_to_the_fallback_template(
        self, asynchronous, caplog
    ):
        strategy = ContextualTemplate(FakeLLM(error=RuntimeError("Provider down for: Q?")))

        if asynchronous:
            result = await strategy.generate_result_async("Q?", ["A span."], 0)
        else:
            result = strategy.generate_result("Q?", ["A span."], 0)

        assert result.text.startswith("## Response\n\nBased on the available documents:")
        assert [warning.code for warning in result.warnings] == ["template_fallback"]
        [record] = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert "Provider down" not in record.getMessage()

    @pytest.mark.parametrize("asynchronous", [False, True])
    async def test_a_rejected_key_still_propagates(self, asynchronous):
        strategy = ContextualTemplate(FakeLLM(error=REJECTED_KEY))

        with pytest.raises(ConfigurationError):
            if asynchronous:
                await strategy.generate_result_async("Q?", ["A span."], 0)
            else:
                strategy.generate_result("Q?", ["A span."], 0)
