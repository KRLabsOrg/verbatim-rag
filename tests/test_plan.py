"""Answer plans: models, parsing, validation, fallback, candidates and rendering (ADR 0007)."""

import pytest
from pydantic import ValidationError
from verbatim_core import InvalidInputError
from verbatim_core.plan import (
    AnswerPlan,
    FieldItem,
    LLMPlan,
    PlanSection,
    PlanText,
    QuoteItem,
    ReferencesItem,
    RestItem,
    TextItem,
    fallback_plan,
    parse_template,
    plan_candidates,
    plan_from_llm,
    render_plan,
    validate_plan,
)

SPANS = [
    "Reduce the dose to 15 mg.",
    "Contraindicated below 15 ml/min.",
    "Take with food.",
    "Monitor renal function.",
]


def plan(*sections, intro=None, layout="list"):
    return AnswerPlan(intro=intro, sections=list(sections), layout=layout)


class TestModels:
    def test_items_are_read_by_kind_from_json(self):
        data = {
            "intro": "Zur Dosierung:",
            "sections": [
                {
                    "label": "Dosis",
                    "items": [
                        {"kind": "quote", "spans": [2, 1]},
                        {"kind": "rest", "max": 2},
                        {"kind": "references"},
                        {"kind": "text", "text": "Ende."},
                        {
                            "kind": "field",
                            "name": "Niere",
                            "question": "Dosis bei Niereninsuffizienz?",
                        },
                    ],
                }
            ],
        }

        parsed = AnswerPlan.model_validate_json(AnswerPlan.model_validate(data).model_dump_json())

        assert [type(item) for item in parsed.sections[0].items] == [
            QuoteItem,
            RestItem,
            ReferencesItem,
            TextItem,
            FieldItem,
        ]
        assert parsed.sections[0].items[4] == FieldItem(
            name="Niere", question="Dosis bei Niereninsuffizienz?"
        )

    @pytest.mark.parametrize(
        "item",
        [
            {"kind": "quote", "spans": []},
            {"kind": "quote", "spans": [0]},
            {"kind": "poem"},
            {"kind": "quote", "spans": [1], "x": 1},
        ],
    )
    def test_invalid_items_are_rejected(self, item):
        with pytest.raises(ValidationError):
            AnswerPlan.model_validate({"sections": [{"items": [item]}]})


class TestParseTemplate:
    def test_a_per_span_template_keeps_every_character(self):
        template = "Intro:\n- [SPAN_2]\n- [FACT_1] [DISPLAY_SPANS]\n\nSee [CITATION_REFS]."

        parsed = parse_template(template)

        assert parsed.layout == "inline"
        assert parsed.sections[0].items == [
            TextItem(text="Intro:\n- "),
            QuoteItem(spans=[2]),
            TextItem(text="\n- "),
            QuoteItem(spans=[1]),
            TextItem(text=" [DISPLAY_SPANS]\n\nSee "),
            ReferencesItem(),
            TextItem(text="."),
        ]

    def test_an_aggregate_template_gets_rest_items(self):
        parsed = parse_template("## Findings\n\n[DISPLAY_SPANS]\n[RELEVANT_SENTENCES]")

        assert parsed.sections[0].items == [
            TextItem(text="## Findings\n\n"),
            RestItem(),
            TextItem(text="\n"),
            RestItem(),
        ]


class TestValidatePlan:
    def test_unknown_and_repeated_spans_are_dropped_and_empty_parts_too(self):
        raw = plan(
            PlanSection(label="A", items=[QuoteItem(spans=[1, 9, 1]), QuoteItem(spans=[9])]),
            PlanSection(label="B", items=[QuoteItem(spans=[1])]),
            PlanSection(label="C", items=[QuoteItem(spans=[3, 2])]),
        )

        valid = validate_plan(raw, span_count=4)

        assert valid.sections == [
            PlanSection(label="A", items=[QuoteItem(spans=[1])]),
            PlanSection(label="C", items=[QuoteItem(spans=[3, 2])]),
        ]

    def test_the_shown_items_are_capped(self):
        raw = plan(PlanSection(items=[QuoteItem(spans=[n]) for n in range(1, 5)] + [RestItem()]))

        valid = validate_plan(raw, span_count=4, max_items=2)

        assert valid.sections[0].items == [
            QuoteItem(spans=[1]),
            QuoteItem(spans=[2]),
            RestItem(max=2),
        ]

    def test_the_fallback_plan_shows_the_first_spans(self):
        fallback = fallback_plan(
            span_count=4, max_items=2, text=PlanText(fallback_intro="Aus den Quellen:")
        )

        assert fallback == plan(
            PlanSection(items=[QuoteItem(spans=[1]), QuoteItem(spans=[2])]),
            intro="Aus den Quellen:",
        )


class TestPlanFromLLM:
    def test_blank_labels_and_numbers_below_one_are_dropped(self):
        llm = LLMPlan.model_validate(
            {
                "intro": " ",
                "sections": [{"label": "", "items": [{"spans": [0, 2]}, {"spans": [-1]}]}],
            }
        )

        assert plan_from_llm(llm) == plan(PlanSection(items=[QuoteItem(spans=[2])]))


class TestCandidates:
    def test_repeats_are_dropped_and_chunks_take_turns(self):
        texts = ["Reduce to 15 mg.", "reduce to 15  mg", "a1", "a2", "b1", "to 15 mg"]
        chunk_ids = ["a", "b", "a", "a", "c", "c"]

        assert plan_candidates(texts, chunk_ids, max_candidates=10) == [0, 4, 2, 3]
        assert plan_candidates(texts, chunk_ids, max_candidates=2) == [0, 4]


class TestRenderPlan:
    def test_citations_are_numbered_as_they_appear(self):
        valid = plan(
            PlanSection(label="Dosis", items=[QuoteItem(spans=[3]), QuoteItem(spans=[1, 4])]),
            intro="Zur Dosierung:",
        )

        rendered = render_plan(valid, SPANS)

        assert rendered.text == (
            "Zur Dosierung:\n\n**Dosis**\n- [1] Take with food.\n- [2] Reduce the dose to 15 mg. [3]"
        )
        assert [(c.number, c.span, c.type) for c in rendered.citations] == [
            (1, 2, "display"),
            (2, 0, "display"),
            (3, 3, "reference"),
            (4, 1, "reference"),
        ]

    def test_unused_spans_are_listed_after_the_answer_when_asked(self):
        text = PlanText(references_label="Weitere Belege:", show_references=True)

        rendered = render_plan(plan(PlanSection(items=[QuoteItem(spans=[2])])), SPANS, text=text)

        assert (
            rendered.text == "- [1] Contraindicated below 15 ml/min.\n\nWeitere Belege: [2] [3] [4]"
        )

    def test_hidden_citations_leave_no_markers(self):
        rendered = render_plan(
            plan(PlanSection(items=[QuoteItem(spans=[2, 1])])), SPANS, citation_mode="hidden"
        )

        assert rendered.text == "- Contraindicated below 15 ml/min."
        assert len(rendered.citations) == 4

    def test_a_quote_with_blank_lines_stays_in_its_item_and_its_words_stay(self):
        spans = ["Symptoms:\n\nSyncope\n\n   Dyspnoea"]

        rendered = render_plan(plan(PlanSection(items=[QuoteItem(spans=[1])])), spans)

        assert rendered.text == "- [1] Symptoms:\\\n  Syncope\\\n  Dyspnoea"

    def test_a_rest_item_shows_the_unused_spans_up_to_the_limit(self):
        valid = plan(PlanSection(items=[QuoteItem(spans=[2]), RestItem()]))

        rendered = render_plan(valid, SPANS, shown_limit=3)

        assert (
            rendered.text
            == "- [1] Contraindicated below 15 ml/min.\n- [2] Reduce the dose to 15 mg.\n- [3] Take with food."
        )
        assert rendered.citations[3].type == "reference"

    def test_an_inline_plan_keeps_its_text(self):
        parsed = validate_plan(
            parse_template("Intro: [SPAN_2], see [CITATION_REFS]."), span_count=4
        )

        assert (
            render_plan(parsed, SPANS).text
            == "Intro: [1] Contraindicated below 15 ml/min., see [2] [3] [4]."
        )

    def test_field_items_are_not_rendered_yet(self):
        with_field = plan(PlanSection(items=[FieldItem(name="Niere", question="Dosis?")]))

        with pytest.raises(InvalidInputError) as raised:
            render_plan(with_field, SPANS)

        assert raised.value.code == "invalid_value"
