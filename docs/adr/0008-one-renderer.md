# One renderer for every answer

ADR 0007 introduced answer plans. Its first step left two renderers side by side: `TemplateFiller` parsed a placeholder template into a plan and then filled it with its own code, and `render_plan` rendered LLM plans. The plan model also carried the template's text (`TextItem`, `layout="inline"`), so an application reading `Answer.plan` had to skip text it can't use. The settings were spread out: `use_answer_plan`, `max_candidates` and `max_support` on `TransformConfig`, `max_display_spans` reused as the number of plan items, `compact_quotes` and `show_references` in `PlanText` next to the translated strings, and `answer_language` on `ContextualTemplate`. `Answer.plan` numbered spans by their place in the candidate list, so an application had to compute the candidates again to find a plan's spans. None of this is released, so it can change without a deprecation.

## Decision

An answer passes through three layers, and every template mode uses the same last two.

```text
plan     what the answer shows         AnswerPlan: intro, sections (label, items)
                                       items: quote (spans), rest (max), references (spans), field
layout   how it looks in markdown      text pieces and slots (the plan's items)
render   the slots filled with spans   text + citations, numbered
```

**Plan** (`verbatim_core.plan`) holds meaning only: no template text and no layout. A plan comes from the LLM (contextual mode with an answer plan, as JSON through structured output); plans written by the developer are the next step (ADR 0007, step 4). `validate_plan` checks every plan against the spans of the call.

**Layout** (`verbatim_core.render`) is a list of text pieces and slots. It has two producers:

- `layout_plan(plan)`: the markdown list layout of a plan (the intro, a bold line per label, a list item per quote).
- `layout_template(text, display_spans, citation_spans)` in `verbatim_core.templates.filler`: a placeholder template (`[SPAN_n]`, `[FACT_n]`, `[DISPLAY_SPANS]`, `[RELEVANT_SENTENCES]`, `[CITATION_REFS]`) keeps every character of its text and has a slot for each placeholder.

**Render** (`render(layout, spans, options)`) is the only code that writes span text and citation markers. Its options: the citation mode and format, the numbering (`appearance`, as plans number, or `position`, as placeholder templates always numbered), whether quotes lose their source layout, and the label of a references line after the answer.

**Placeholder templates.** `TemplateFiller` keeps its interface and its output. It lays out the template with the call's display and citation spans, as the old filler read them: a rest slot becomes the display spans in order, a `[SPAN_n]` past the display spans becomes a reference marker, `[CITATION_REFS]` becomes the citation spans' markers, and a display span's linked citations (`citation_ids`) become its supporting spans. It then renders with `position` numbering and the source layout kept. `tests/test_plan_filler_identity.py` compares this with the old filler over every template of the library and the test suite, and over 3000 templates built at random from placeholders and markdown pieces. The Transformer takes the citations of a placeholder answer from the same render, so the text and `Answer.citations` can't disagree; a strategy with its own `fill` still writes the text.

**Settings.** `TransformConfig(answer_plan=PlanConfig(...))` turns plans on in contextual mode and holds their settings: `max_candidates`, `max_items`, `max_support`, `compact_quotes` and `show_references`. It replaces `use_answer_plan`, `max_candidates` and `max_support`, and `max_display_spans` keeps its one meaning (the shown spans of a placeholder template). `PlanText` holds the strings of an answer language and the language's name (`language`, which the bundled plan prompt asks the LLM to write in); it replaces `answer_language`. `PlanText.no_spans` is removed: only placeholder templates showed it, with their own English text.

**Contextual mode with plans always asks for JSON.** With `answer_plan` set, the LLM writes a plan with the bundled prompt or `plan_prompt`. A `template_prompt` is used only for placeholder templates. The path that parsed an LLM's placeholder text into a plan, and the bullet repair it needed, is removed.

**`Answer.plan` numbers spans by their place in `Answer.spans`**, from 1. The LLM still sees the candidates numbered 1 to n; the Transformer maps the numbers back before it returns the plan.

Field items stay validated but not rendered until #74 (filled and rendered since ADR 0010). The five mode names stay; the two settings that replace them (plan source and selection, ADR 0007) are a later change.

## Consequences

- One code path writes span text and markers. The old filler's own filling code, `render_plan`, `parse_template`, `validate_plan`'s handling of inline plans and the trailing-bullet repair are deleted. Repeated spans are found by one function, `distinct_spans`, for `drop_repeated_spans` and the plan candidates alike.
- `TextItem` and `AnswerPlan.layout` are removed. `ReferencesItem(spans=[...])` names the spans it cites; without `spans` it cites the spans the plan doesn't use.
- `TransformConfig.use_answer_plan`, `max_candidates`, `max_support`, `PlanText.compact_quotes`, `PlanText.show_references` and `ContextualTemplate(answer_language=...)` are replaced, as above. Code written against the unreleased ADR 0007 API, such as the Thieme deployment, changes its settings.
- An empty span is no longer counted as shown. The library's extractors never return one.
