"""Plans the developer writes, and field items filled by the extractor (ADR 0010)."""

import re

import pytest
from transformer_contract import TransformerContract
from verbatim_core import (
    Chunk,
    ExtractedSpan,
    Extraction,
    InvalidInputError,
    TransformConfig,
    Transformer,
    Usage,
)
from verbatim_core.decision_models import DecisionModelExtractor, LocalDecisionModel
from verbatim_core.extractors import SpanExtractor
from verbatim_core.plan import AnswerPlan, FieldItem, PlanConfig, PlanSection, PlanText, RestItem
from verbatim_core.render import layout_plan, render
from verbatim_core.templates import TemplateManager

CHUNKS = [
    Chunk(
        id="label",
        text="Metformin is indicated for type 2 diabetes. The usual dose is 500 mg twice daily.",
    ),
    Chunk(
        id="warn",
        text="Metformin is contraindicated in severe renal failure. Stop it before contrast media.",
    ),
]


def words(text):
    return {word for word in re.findall(r"\w+", text.lower()) if len(word) > 3}


class SentenceExtractor(SpanExtractor):
    """Extracts every sentence that shares a word with the question, and records the questions."""

    def __init__(self):
        self.questions = []

    def extract(self, question, chunks, *, config=None):
        self.questions.append(question)
        spans = [
            ExtractedSpan(
                chunk_id=chunk.id, start=m.start(), end=m.end(), text=m.group(), score=0.9
            )
            for chunk in chunks
            for m in re.finditer(r"[^.\s][^.]*\.", chunk.text)
            if words(m.group()) & words(question)
        ]
        return Extraction(spans=spans, usage=[Usage(model_id="stub", model_version="1", units={})])


CHECKLIST = AnswerPlan(
    intro="Drug check:",
    sections=[
        PlanSection(
            items=[
                FieldItem(name="Indication", question="indicated"),
                FieldItem(name="Contraindications", question="contraindicated", required=True),
                FieldItem(name="Pregnancy", question="pregnancy", required=True),
                FieldItem(name="Interactions", question="interactions"),
            ]
        )
    ],
)


def checklist(plan=CHECKLIST, extractor=None, **settings):
    config = TransformConfig(answer_plan=PlanConfig(plan=plan, **settings))
    return Transformer(extractor or SentenceExtractor(), TemplateManager(), config=config)


def test_a_field_plan_answers_in_static_mode_with_one_span_per_field():
    answer = checklist().transform("Metformin check", CHUNKS)

    assert answer.text == (
        "Drug check:\n\n"
        "- **Indication:** [1] Metformin is indicated for type 2 diabetes.\n"
        "- **Contraindications:** [2] Metformin is contraindicated in severe renal failure.\n"
        "- **Pregnancy:** No supporting passage found."
    )
    assert {name: [s.text for s in spans] for name, spans in answer.field_spans().items()} == {
        "Indication": ["Metformin is indicated for type 2 diabetes."],
        "Contraindications": ["Metformin is contraindicated in severe renal failure."],
        "Pregnancy": [],
        "Interactions": [],
    }
    assert [(c.number, c.chunk_id, c.type) for c in answer.citations] == [
        (1, "label", "display"),
        (2, "warn", "display"),
    ]


def test_a_plan_of_fields_only_doesnt_extract_the_question():
    extractor = SentenceExtractor()
    checklist(extractor=extractor).transform("Metformin check", CHUNKS)

    assert extractor.questions == ["indicated", "contraindicated", "pregnancy", "interactions"]


async def test_fields_are_extracted_concurrently_and_agree_with_the_sync_call():
    transformer = checklist()

    assert await transformer.atransform("Metformin check", CHUNKS) == transformer.transform(
        "Metformin check", CHUNKS
    )


def test_a_field_with_several_spans_is_a_nested_list_and_fields_mix_with_rest():
    plan = AnswerPlan(
        sections=[
            PlanSection(
                label="Checked",
                items=[FieldItem(name="Metformin", question="metformin", max_spans=2)],
            ),
            PlanSection(label="Also in the sources", items=[RestItem()]),
        ]
    )
    answer = checklist(plan).transform("dose daily", CHUNKS)

    assert answer.text == (
        "**Checked**\n"
        "- **Metformin:**\n"
        "  - [1] Metformin is indicated for type 2 diabetes.\n"
        "  - [2] Metformin is contraindicated in severe renal failure.\n\n"
        "**Also in the sources**\n"
        "- [3] The usual dose is 500 mg twice daily."
    )
    # The question's span comes first in Answer.spans, the field spans after it
    assert [span.text for span in answer.spans][0] == "The usual dose is 500 mg twice daily."
    assert answer.plan.sections[0].items[0].spans == [2, 3]


def test_a_field_span_the_question_also_found_is_one_span_with_one_citation():
    plan = AnswerPlan(
        sections=[PlanSection(items=[FieldItem(name="Dose", question="dose"), RestItem()])]
    )
    answer = checklist(plan).transform("usual dose", CHUNKS)

    assert answer.text == "- **Dose:** [1] The usual dose is 500 mg twice daily."
    assert len(answer.spans) == 1
    assert len(answer.citations) == 1


def test_the_plan_text_sets_the_fixed_texts():
    answer = checklist(text=PlanText(no_evidence="Keine Belegstelle gefunden.")).transform(
        "Metformin check", CHUNKS
    )

    assert "- **Pregnancy:** Keine Belegstelle gefunden." in answer.text


def test_a_field_without_filled_spans_cant_be_rendered_outside_the_transformer():
    with pytest.raises(InvalidInputError):
        layout_plan(CHECKLIST, [])


def test_rendering_a_filled_field_plan():
    plan = AnswerPlan(
        sections=[PlanSection(items=[FieldItem(name="Dose", question="dose", spans=[1])])]
    )

    assert render(layout_plan(plan, ["500 mg."]), ["500 mg."]).text == "- **Dose:** [1] 500 mg."


class TestFieldPlanContract(TransformerContract):
    @pytest.fixture
    def transformer(self):
        plan = AnswerPlan(
            sections=[
                PlanSection(items=[FieldItem(name="Height", question="tall arch"), RestItem()])
            ]
        )
        return checklist(plan)


class FieldModel:
    """A stand-in for Vela: each span question finds the sentences with its label, and calls are recorded."""

    def __init__(self):
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        context = state["context"]
        spans = {}
        for qid, question in questions.items():
            (label,) = question["criteria"]
            spans[qid] = [
                {"start": m.start(), "end": m.end(), "text": m.group(), "probability": 0.8}
                for m in re.finditer(r"[^.\s][^.]*\.", context)
                if label.lower() in m.group().lower()
            ]
        return {"spans": spans}


def test_a_decision_model_asks_every_field_in_one_call_per_chunk():
    model = FieldModel()
    extractor = DecisionModelExtractor(
        LocalDecisionModel(model, model_id="stub/vela", model_version="1")
    )
    plan = AnswerPlan(
        sections=[
            PlanSection(
                items=[
                    FieldItem(name="dose", question="What is the usual dose?"),
                    FieldItem(name="renal", question="What about the kidneys?"),
                ]
            )
        ]
    )

    answer = checklist(plan, extractor=extractor).transform("Metformin check", CHUNKS)

    assert len(model.calls) == len(CHUNKS)
    state, questions = model.calls[0]
    assert state == {"request": "Metformin check", "context": CHUNKS[0].text}
    assert questions["field_0"]["instructions"] == "What is the usual dose?"
    assert questions["field_0"]["criteria"] == {"dose": "What is the usual dose?"}
    assert {name: [s.text for s in spans] for name, spans in answer.field_spans().items()} == {
        "dose": ["The usual dose is 500 mg twice daily."],
        "renal": ["Metformin is contraindicated in severe renal failure."],
    }
