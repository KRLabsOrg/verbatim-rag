"""Answer plans and their rendering: models, validation, fallback, candidates (ADR 0007, 0008)."""

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
    distinct_spans,
    fallback_plan,
    plan_candidates,
    plan_from_llm,
    renumber_plan,
    validate_plan,
)
from verbatim_core.render import Markers, Quote, RenderOptions, layout_plan, render


def render_plan(plan, spans, shown_limit=None, **options):
    """Render a plan as the Transformer does, with compact quotes unless the test says otherwise."""
    options.setdefault("compact_quotes", True)
    return render(layout_plan(plan, spans, shown_limit), spans, RenderOptions(**options))


SPANS = [
    "Reduce the dose to 15 mg.",
    "Contraindicated below 15 ml/min.",
    "Take with food.",
    "Monitor renal function.",
]


def plan(*sections, intro=None):
    return AnswerPlan(intro=intro, sections=list(sections))


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
                        {"kind": "references", "spans": [3]},
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
            ReferencesItem,
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
            {"kind": "text", "text": "Template text is not part of a plan."},
        ],
    )
    def test_invalid_items_are_rejected(self, item):
        with pytest.raises(ValidationError):
            AnswerPlan.model_validate({"sections": [{"items": [item]}]})


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

        assert valid.sections[0].items == [QuoteItem(spans=[1]), QuoteItem(spans=[2])]

    def test_a_rest_item_gets_the_places_left(self):
        raw = plan(PlanSection(items=[QuoteItem(spans=[1]), RestItem()]))

        valid = validate_plan(raw, span_count=4, max_items=3)

        assert valid.sections[0].items == [QuoteItem(spans=[1]), RestItem(max=2)]

    def test_references_keep_known_spans_and_go_when_none_is_left(self):
        raw = plan(PlanSection(items=[ReferencesItem(spans=[4, 9, 4]), ReferencesItem(spans=[9])]))

        assert validate_plan(raw, span_count=4).sections[0].items == [ReferencesItem(spans=[4])]

    def test_a_section_of_field_items_stays(self):
        field = FieldItem(name="Niere", question="Dosis?")

        assert validate_plan(plan(PlanSection(items=[field])), span_count=0).sections[0].items == [
            field
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


class TestRenumber:
    def test_span_numbers_are_mapped_and_the_rest_stays(self):
        raw = plan(
            PlanSection(label="A", items=[QuoteItem(spans=[2, 1]), RestItem(max=1)]),
            PlanSection(items=[ReferencesItem(spans=[3]), ReferencesItem()]),
            intro="Intro",
        )

        assert renumber_plan(raw, [10, 20, 30]) == plan(
            PlanSection(label="A", items=[QuoteItem(spans=[20, 10]), RestItem(max=1)]),
            PlanSection(items=[ReferencesItem(spans=[30]), ReferencesItem()]),
            intro="Intro",
        )


class TestCandidates:
    def test_distinct_spans_leave_out_words_an_earlier_span_has(self):
        assert distinct_spans(
            ["Reduce to 15 mg.", "reduce to 15  mg", "", "...", "to 15 mg", "Other"]
        ) == [0, 5]

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
        rendered = render_plan(
            plan(PlanSection(items=[QuoteItem(spans=[2])])),
            SPANS,
            references_label="Weitere Belege:",
        )

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

    def test_a_references_item_cites_what_the_plan_does_not_use(self):
        valid = plan(
            PlanSection(items=[ReferencesItem(), QuoteItem(spans=[2]), ReferencesItem(spans=[1])])
        )

        layout = layout_plan(valid, SPANS)

        assert layout == [Markers((1, 3, 4)), "\n", "- ", Quote((2,)), "\n", Markers((1,))]
        assert (
            render(layout, SPANS).text == "[1] [2] [3]\n- [4] Contraindicated below 15 ml/min.\n[1]"
        )

    def test_spans_without_text_are_left_out(self):
        valid = plan(
            PlanSection(label="A", items=[QuoteItem(spans=[1])]), PlanSection(items=[RestItem()])
        )

        assert render_plan(valid, ["  ", "B."]).text == "- [1] B."

    def test_field_items_are_not_rendered_yet(self):
        with_field = plan(PlanSection(items=[FieldItem(name="Niere", question="Dosis?")]))

        with pytest.raises(InvalidInputError) as raised:
            render_plan(with_field, SPANS)

        assert raised.value.code == "invalid_value"


class TestCompactQuotes:
    PDF = "Der Begriff „Herzinsuffizienz“\n                            bezeichnet ein Syndrom.\n\n  Ursachen:\n- KHK\n     - Hypertonie"

    def test_pdf_layout_is_removed_and_paragraphs_stay_in_the_item(self):
        rendered = render_plan(plan(PlanSection(items=[QuoteItem(spans=[1])])), [self.PDF])

        assert rendered.text == (
            "- [1] Der Begriff „Herzinsuffizienz“ bezeichnet ein Syndrom.\\\n  Ursachen:\\\n  - KHK\\\n  - Hypertonie"
        )

    def test_the_source_layout_can_be_kept(self):
        rendered = render_plan(
            plan(PlanSection(items=[QuoteItem(spans=[1])])),
            ["A\n     b"],
            compact_quotes=False,
        )

        assert rendered.text == "- [1] A\n     b"


class TestMaxSupport:
    def test_an_item_keeps_at_most_max_support_supporting_spans(self):
        raw = plan(PlanSection(items=[QuoteItem(spans=[1, 2, 3, 4])]))

        valid = validate_plan(raw, span_count=4, max_support=2)

        assert valid.sections[0].items == [QuoteItem(spans=[1, 2, 3])]
        assert render_plan(valid, SPANS).text == "- [1] Reduce the dose to 15 mg. [2] [3]"
