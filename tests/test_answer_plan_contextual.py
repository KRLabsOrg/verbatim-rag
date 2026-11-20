"""Contextual mode on answer plans, with TransformConfig(use_answer_plan=True) (ADR 0007)."""

import json

import pytest
from verbatim_core import (
    Chunk,
    ConfigurationError,
    ExtractedSpan,
    Extraction,
    TransformConfig,
    Transformer,
    UpstreamError,
)
from verbatim_core.extractors import SpanExtractor
from verbatim_core.llm_client import Completion
from verbatim_core.plan import LLMPlan, PlanText, QuoteItem
from verbatim_core.templates import TemplateManager
from verbatim_core.templates.contextual import ContextualTemplate

CHUNKS = [
    Chunk(id="monograph", text="Dose: 20 mg once daily.\n\nCrCl 15-49 ml/min: 15 mg once daily."),
    Chunk(id="textbook", text="Contraindicated below 15 ml/min. Take with food."),
    Chunk(id="guideline", text="Dose: 20 mg once daily."),
]
GERMAN = PlanText(
    fallback_intro="Die Quellen enthalten dazu folgende Aussagen:",
    references_label="Weitere Belege:",
    no_spans="Keine passenden Aussagen gefunden.",
    no_evidence="Keine Belegstelle gefunden.",
)


class SentenceExtractor(SpanExtractor):
    """Extracts every paragraph or sentence of every chunk."""

    def extract(self, question, chunks, *, config=None):
        spans = []
        for chunk in chunks:
            start = 0
            for part in chunk.text.replace(". ", ".|").replace("\n\n", "|").split("|"):
                begin = chunk.text.index(part, start)
                spans.append(
                    ExtractedSpan(
                        chunk_id=chunk.id, start=begin, end=begin + len(part), text=part, score=1.0
                    )
                )
                start = begin + len(part)
        return Extraction(spans=spans)


class PlanLLM:
    """Replies with a fixed plan or template, and records how it was asked."""

    model = "fake-llm"
    temperature = 0.0

    def __init__(self, reply="", template="", error=None):
        self.reply = reply
        self.template = template
        self.error = error
        self.calls = []

    def completion(self, prompt, **settings):
        self.calls.append({"prompt": prompt, **settings})
        if self.error:
            raise self.error
        return Completion(text=self.reply, model="fake-llm-1", input_tokens=10, output_tokens=5)

    async def completion_async(self, prompt, **settings):
        return self.completion(prompt, **settings)

    def template_completion(self, question, spans, citation_count, **settings):
        self.calls.append({"spans": spans, **settings})
        return Completion(text=self.template)

    async def template_completion_async(self, *args, **settings):
        return self.template_completion(*args, **settings)


def transformer(llm, **strategy_settings):
    templates = TemplateManager(llm_client=llm, default_mode="contextual")
    templates.strategies["contextual"] = ContextualTemplate(
        llm, plan_text=GERMAN, **strategy_settings
    )
    config = TransformConfig(template_mode="contextual", use_answer_plan=True, max_display_spans=3)
    return Transformer(SentenceExtractor(), templates, config=config)


def plan_reply(**plan):
    return json.dumps({"intro": None, "sections": [], **plan})


class TestLLMPlan:
    REPLY = plan_reply(
        intro="Zur Dosierung von Rivaroxaban:",
        sections=[
            {"label": "Dosierung", "items": [{"spans": [2, 99]}, {"spans": [1]}]},
            {"label": "Kontraindikation", "items": [{"spans": [3, 1]}]},
        ],
    )

    def test_the_llm_sees_every_candidate_in_full_and_gets_the_plan_schema(self):
        llm = PlanLLM(self.REPLY)

        transformer(llm).transform("Dosis?", CHUNKS)

        [call] = llm.calls
        assert call["json_schema"] == {"name": "answer_plan", "schema": LLMPlan.model_json_schema()}
        assert "[1] Dose: 20 mg once daily." in call["prompt"]
        assert "[2] Contraindicated below 15 ml/min." in call["prompt"]
        assert "[3] CrCl 15-49 ml/min: 15 mg once daily." in call["prompt"]
        assert "at most 3 items" in call["prompt"]

    def test_the_answer_is_the_validated_plan_rendered_with_citations_in_order(self):
        answer = transformer(PlanLLM(self.REPLY)).transform("Dosis?", CHUNKS)

        assert answer.text == (
            "Zur Dosierung von Rivaroxaban:\n\n"
            "**Dosierung**\n- [1] Contraindicated below 15 ml/min.\n- [2] Dose: 20 mg once daily.\n\n"
            "**Kontraindikation**\n- [3] CrCl 15-49 ml/min: 15 mg once daily."
        )
        assert [(c.number, c.type, c.chunk_id) for c in answer.citations] == [
            (1, "display", "textbook"),
            (2, "display", "monograph"),
            (3, "display", "monograph"),
            (4, "reference", "textbook"),
            (5, "reference", "guideline"),
        ]
        assert answer.plan.sections[0].items == [QuoteItem(spans=[2]), QuoteItem(spans=[1])]
        assert answer.usage[-1].units["llm_calls"] == 1

    @pytest.mark.parametrize(
        "reply", ["not json", plan_reply(sections=[{"label": None, "items": [{"spans": [42]}]}])]
    )
    def test_an_unusable_plan_falls_back_to_the_first_candidates(self, reply):
        answer = transformer(PlanLLM(reply)).transform("Dosis?", CHUNKS)

        assert answer.text.startswith(
            "Die Quellen enthalten dazu folgende Aussagen:\n\n- [1] Dose: 20 mg once daily."
        )
        assert [warning.code for warning in answer.warnings] == ["template_fallback"]

    def test_a_failed_call_falls_back_and_a_rejected_key_propagates(self):
        failed = UpstreamError("The LLM service failed.", code="server_error", service="llm")
        answer = transformer(PlanLLM(error=failed)).transform("Dosis?", CHUNKS)
        assert [warning.code for warning in answer.warnings] == ["template_fallback"]

        rejected = ConfigurationError(
            "The LLM service rejected the key.", code="authentication_failed"
        )
        with pytest.raises(ConfigurationError):
            transformer(PlanLLM(error=rejected)).transform("Dosis?", CHUNKS)

    async def test_the_async_path_writes_the_same_answer(self):
        sync = transformer(PlanLLM(self.REPLY)).transform("Dosis?", CHUNKS)
        events = [
            event async for event in transformer(PlanLLM(self.REPLY)).astream("Dosis?", CHUNKS)
        ]

        assert [event.type for event in events] == ["spans", "answer"]
        assert events[-1].answer == sync

    def test_unused_spans_are_listed_when_the_plan_text_asks(self):
        llm = PlanLLM(plan_reply(sections=[{"label": None, "items": [{"spans": [1]}]}]))
        templates = TemplateManager(llm_client=llm, default_mode="contextual")
        templates.strategies["contextual"] = ContextualTemplate(
            llm, plan_text=GERMAN.model_copy(update={"show_references": True})
        )
        config = TransformConfig(template_mode="contextual", use_answer_plan=True)

        answer = Transformer(SentenceExtractor(), templates, config=config).transform(
            "Dosis?", CHUNKS
        )

        assert answer.text == "- [1] Dose: 20 mg once daily.\n\nWeitere Belege: [2] [3] [4] [5]"


class TestCustomTemplatePrompt:
    def test_a_placeholder_template_is_parsed_and_validated(self):
        llm = PlanLLM(template="Zur Dosierung:\n- [SPAN_3]\n- [SPAN_3]\n- [SPAN_7]\n- [SPAN_1]")

        answer = transformer(llm, template_prompt="{{ question }}\n{{ spans_block }}").transform(
            "Dosis?", CHUNKS
        )

        assert answer.text == (
            "Zur Dosierung:\n- [1] CrCl 15-49 ml/min: 15 mg once daily.\n- [2] Dose: 20 mg once daily."
        )
        assert [warning.code for warning in answer.warnings] == []
        assert llm.calls[0]["spans"][2] == "CrCl 15-49 ml/min: 15 mg once daily."


class TestCompatibility:
    def test_without_the_switch_contextual_answers_are_as_before(self):
        llm = PlanLLM(template="Antwort: [SPAN_1]")
        templates = TemplateManager(llm_client=llm, default_mode="contextual")
        config = TransformConfig(template_mode="contextual")

        answer = Transformer(SentenceExtractor(), templates, config=config).transform(
            "Dosis?", CHUNKS
        )

        assert answer.text == "Antwort: [1] Dose: 20 mg once daily."
        assert answer.plan is None
        assert "plan" not in answer.model_dump()

    def test_the_switch_leaves_other_modes_alone(self):
        config = TransformConfig(template_mode="static", use_answer_plan=True)
        answer = Transformer(SentenceExtractor(), TemplateManager(), config=config).transform(
            "Dosis?", CHUNKS
        )
        before = Transformer(SentenceExtractor(), TemplateManager()).transform("Dosis?", CHUNKS)

        assert answer == before

    def test_an_answer_with_a_plan_serialises_it(self):
        answer = transformer(PlanLLM(TestLLMPlan.REPLY)).transform("Dosis?", CHUNKS)

        assert answer.model_dump()["plan"]["sections"][0]["label"] == "Dosierung"
