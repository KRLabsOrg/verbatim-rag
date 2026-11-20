"""The filler runs on answer plans (ADR 0007): its output is the same as the old filler's."""

import itertools
import re
from pathlib import Path

import pytest
from legacy_filler import LegacyTemplateFiller
from verbatim_core.templates.filler import TemplateFiller
from verbatim_core.templates.question_specific import QuestionSpecificTemplate
from verbatim_core.templates.random import RandomTemplate
from verbatim_core.templates.static import StaticTemplate

PROMPTS = Path(__file__).parent.parent / "packages/core/verbatim_core/prompts/template"
PLACEHOLDER = re.compile(
    r"\[(?:SPAN_\d+|FACT_\d+|DISPLAY_SPANS|RELEVANT_SENTENCES|CITATION_REFS)\]"
)


def suite_templates():
    """Every template string with a placeholder in the test suite, as it is written there."""
    found = set()
    for path in Path(__file__).parent.glob("test_*.py"):
        for literal in re.findall(r'"((?:[^"\\\n]|\\.)*)"', path.read_text()):
            if PLACEHOLDER.search(literal):
                found.add(literal.encode().decode("unicode_escape"))
    return sorted(found)


LIBRARY_TEMPLATES = [
    StaticTemplate.DEFAULT_TEMPLATE,
    QuestionSpecificTemplate.DEFAULT_TEMPLATE,
    *RandomTemplate.DEFAULT_TEMPLATES,
    (PROMPTS / "fallback.txt").read_text(),
]

EDGE_TEMPLATES = [
    "",
    "No placeholder at all.",
    "[SPAN_1]",
    "Intro:\n\n- [SPAN_1]\n- [SPAN_2]\n- [SPAN_3]\n\nSee also [CITATION_REFS].",
    "1. [FACT_2]\n2. [FACT_1]\n   - nested [SPAN_3]",
    "- [SPAN_2] and [SPAN_1]\n* [SPAN_9]\n+ [SPAN_0]",
    "Mixed: [SPAN_1] [DISPLAY_SPANS] [CITATION_REFS] [CITATION_REFS]",
    "## Findings\n\n[DISPLAY_SPANS]\n\n[RELEVANT_SENTENCES]\n\nRefs: [CITATION_REFS]",
    "[RELEVANT_SENTENCES]",
    "  [DISPLAY_SPANS]  \n\n",
    "- [DISPLAY_SPANS]\n- [CITATION_REFS]",
    "| a | [SPAN_1] |\n|---|---|",
]

SPAN_SETS = [
    ([], []),
    ([{"text": "One."}], []),
    ([{"text": "One."}, {"text": "  Two.  "}], [{"text": "Three."}]),
    (
        [
            {"text": "Symptoms:\n\nSyncope\n\n   Dyspnoea,\n  tachypnoea"},
            {"text": "Line\n   wrapped"},
        ],
        [],
    ),
    (
        [
            {"text": "| a | b |\n|---|---|\n| 1 | 2 |"},
            {"text": "Dose:\n\n- *Adults:*\n  - 2000 mg"},
        ],
        [{"text": "x"}],
    ),
    ([{"text": ""}, {"text": "   "}], [{"text": "ref"}]),
    (
        [
            {"text": "Claim A.", "citation_ids": ["v1", "v3"], "span_id": "cite1"},
            {"text": "Claim B.", "span_id": "cite2"},
        ],
        [
            {"text": "Table 2.", "citation_id": "v1", "span_id": "cite3"},
            {"text": "Other.", "citation_id": "v2", "span_id": "cite4"},
            {"text": "p<0.01.", "citation_id": "v3"},
        ],
    ),
    ([{"text": "A."}, {"text": "B."}, {"text": "C."}], [{"text": "D."}, {"text": "E."}]),
]

SETTINGS = list(
    itertools.product(["inline", "hidden"], ["[{number}]", "[{span_id}]", "<{number}>"])
)


@pytest.mark.parametrize("template", LIBRARY_TEMPLATES + EDGE_TEMPLATES + suite_templates())
def test_the_filler_on_plans_gives_the_old_output(template):
    for (display, citations), (mode, citation_format) in itertools.product(SPAN_SETS, SETTINGS):
        new = TemplateFiller(citation_mode=mode, citation_format=citation_format)
        old = LegacyTemplateFiller(citation_mode=mode, citation_format=citation_format)
        assert new.fill(template, display, citations) == old.fill(template, display, citations), (
            template,
            display,
            citations,
            mode,
            citation_format,
        )


def test_the_test_suite_has_templates_to_compare():
    assert len(suite_templates()) > 20
