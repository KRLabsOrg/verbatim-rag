# Answers are rendered from validated plans

*Changed by [ADR 0008](0008-one-renderer.md): one renderer for every mode, plans without template text (`TextItem` and `layout` removed), and the settings in `PlanConfig`.* *Field items are filled since [ADR 0010](0010-field-plans.md), and plans can come from the developer.*

Every template mode turns placeholder text (`[SPAN_n]`, `[FACT_n]`, `[DISPLAY_SPANS]`, `[RELEVANT_SENTENCES]`, `[CITATION_REFS]`, structured `[NAME]`) into an answer by string substitution. When an LLM writes that text (contextual mode), the library repaired it afterwards with string rules, and on a German medical deployment (Thieme sources, Mistral) that failed in ways users saw: the LLM ordered and labelled quotes it never saw (an empty quote list above 8 spans), a raw `[DISPLAY_SPANS]` reached the answer, German answers ended in an English sentence, a span could be shown twice or not at all, and with 20 to 60 spans the first ones in extraction order were shown and the rest became a list of numbers. Applications added their own regular expressions on top. The bugs are fixed one by one in 0.3.1; this ADR removes their cause.

## Decision

One engine for every mode: an **answer plan**, validated in code, and one **renderer**. A mode only decides who writes the plan: the developer (static, random, question_specific, structured) or the LLM (contextual). The models live in `verbatim_core.plan` and are Pydantic v2, so they validate, give the JSON schema for LLM structured output, and serialise.

```text
AnswerPlan      intro, sections, layout ("list" or "inline")
PlanSection     label, items
QuoteItem       spans: list[int]           the first span is shown, the others are cited next to it (same claim, other sources)
FieldItem       name, question, required, max_spans
RestItem        max                        the shown spans no other item uses: the plan form of [DISPLAY_SPANS] and [RELEVANT_SENTENCES]
ReferencesItem                             the cited spans the text doesn't show, as markers: the plan form of [CITATION_REFS]
TextItem        text                       template text around the spans, kept as it is
```

`TextItem` and the `inline` layout keep every character of a developer's template, so parsed templates render exactly as before; `list` layout is what the renderer lays out itself (a bullet per quote, a bold line per label).

| Mode | Who writes the plan | How |
|---|---|---|
| static | developer | one placeholder template, parsed into a plan, or later a plan given directly |
| random | developer | a list of templates or plans; the choice by question hash stays (ADR 0005) |
| question_specific | developer | templates or plans with example questions; the choice by similarity stays |
| structured | developer | `[NAME]` placeholders parse into field items, or a plan with field items is given |
| contextual | LLM | a JSON plan through structured output, or placeholder text from a custom `template_prompt`, parsed |

**LLM plans.** The LLM gets the question and up to `max_candidates` spans (default 20), numbered, with full text. The candidates leave out spans whose words an earlier span already has, and take turns between chunks, so one chunk can't fill the list. The LLM returns `{"intro", "sections": [{"label", "items": [{"spans": [...]}]}]}`; it selects, orders, groups and labels, and writes no quote text. The intro and labels are its only words.

**Validation**, in code, for every plan: span numbers outside the candidates are dropped, each span is used once, items and sections left empty are dropped (in an inline plan with the list bullet that led to them), and at most `max_items` items show spans. If nothing valid remains, or the call fails with an `UpstreamError`, a deterministic fallback plan shows the first candidates in order with the warning `template_fallback`.

**Rendering.** The renderer numbers citations in the order they appear, so the markers and `Answer.citations` agree; a shown span is a `display` citation, a span cited next to it or unused a `reference` citation, unused spans numbered last. A quote with blank lines stays in its list item; span text never changes. Fixed strings (fallback intro, references label, "no spans", "no evidence") come from `PlanText`, English by default. Unused spans appear in the text only through a references item, or after the answer when `PlanText.show_references` is set. The placeholder modes keep their own numbering (display spans first, by position), which the filler applies to the parsed plan. The validated plan is returned on the answer (`Answer.plan`), so an application can lay it out itself.

**Field items** are span questions. For each field the transformer extracts `question` over the call's chunks, with the user's question as context, all fields concurrently, keeping up to `max_spans` located and verified spans. A field's spans are shown under its `name`; an empty field shows `PlanText.no_evidence` when `required`, and is left out otherwise. A plan can mix fields with rest and references items, for example a drug approval checklist (indication, contraindications, dosage, renal function) followed by the remaining evidence. The filled plan, with each field's spans, is the JSON answer of #74; with a decision-model extractor (#71) each field is one span question with its own probability.

## Compatibility and deprecation

Nothing changes for existing users. The placeholder modes render through the engine with byte-identical output: the old filler is kept in the tests and compared with the new one over every template of the library and the test suite. Contextual mode uses plans only with `TransformConfig(use_answer_plan=True)`; without it, prompts, templates and answers are as before. Saved states, the CLI and `Answer` serialisation (no `plan` key without a plan) don't change, and no deprecation warnings are added.

Later, old names become adapters that produce plans with identical output and warn with `VerbatimDeprecationWarning` for one window before they are removed, as the 0.2 entry points (deprecated in 0.3.0, removed in 0.5.0).

## Implementation steps

1. **This change:** the models (all item kinds; field items are validated but not filled, and the renderer refuses them), the placeholder parser, validation, fallback and markdown renderer; static, random, question_specific and contextual's placeholder templates fill through the engine with byte-identical output; contextual writes JSON plans behind `use_answer_plan`. Structured mode stays on its own path.
2. After measuring contextual plans on real questions: `use_answer_plan` becomes the default, and the placeholder path stays as an adapter for one deprecation window.
3. Field items, with #74 and #71: extraction per field, `[NAME]` parsed into fields, and `VerbatimRAG`'s separate structured path (ADR 0004) replaced by field plans. Checklists per request type are then field plans chosen by question_specific.
4. Plans given directly to static, random and question_specific (objects, JSON; YAML as an optional extra), a friendlier template syntax and further output formats, when users ask for them.

The target API once every mode runs on the engine has two settings instead of five modes: `plan_source`, the developer (templates or plans) or the LLM, and `selection` among several developer plans: fixed, by question hash (today's random) or by similarity to example questions (today's question_specific). Structured answers are developer plans with field items. The five mode names stay as aliases for one deprecation window.

## Consequences

- New: `verbatim_core.plan`, `TemplateResult.plan`, `Answer.plan`, `TransformConfig.use_answer_plan` and `max_candidates`, `ContextualTemplate(plan_text=...)`, `LLMClient.completion(json_schema=...)`.
- `TemplateFiller` parses each template into a plan; its output doesn't change. One difference remains on purpose: the old filler also replaced a placeholder that appeared inside a filled span's text (source text that contains "[CITATION_REFS]"); the plan fills each placeholder once.
- Prior work: the NLG pipeline of content selection, document planning and surface realisation (Reiter and Dale), which contextual mode had put into one free-text LLM call; "Attribute First, then Generate" (ACL 2024); structured citations with offsets in Anthropic Citations, Cohere RAG and LangExtract; SEMQA and "The Extractive-Abstractive Spectrum" (TACL 2026) on how much easier quoted answers are to verify.
