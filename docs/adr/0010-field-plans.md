# Field items and plans written by the developer

ADR 0007 defined field items, a named question whose answer is extracted for the item, such as one row of a drug approval checklist, and planned to fill them with #74 and #71. ADR 0008 kept them validated but not rendered. Plans could come only from the LLM, in contextual mode. Applications want answers of a fixed shape from their documents: a checklist per request type, or a JSON object whose every field is a verbatim span (#74). Decision models such as Vela 2.0 (ADR 0009) answer several span questions in one call, which suits fields well.

## Decision

**Plans written by the developer.** `PlanConfig(plan=AnswerPlan(...))` gives the plan of every answer. The LLM writes no plan, and the answer is rendered from this one in any template mode, contextual included. The plan is validated against the call's candidates, as an LLM plan is. Its rest and references items take the spans of the user's question; quote items, which name candidates by number, are allowed but rarely useful in a fixed plan. `PlanConfig(text=PlanText(...))` sets the fixed texts; without it a contextual strategy's `plan_text` applies, otherwise English.

**Field items are filled by the extractor.** For each field item the Transformer asks the extractor for the field's spans with `extract_fields(question, fields, chunks)`, one `Extraction` per field (`aextract_fields` asks them concurrently, together with the question's extraction). The base `SpanExtractor` extracts each field's `question` as a question of its own, so every extractor fills fields. `DecisionModelExtractor` asks all fields in one call per chunk: the user's question is the request, and each field is a span question with the field's question as instructions and its name as the label.

A field keeps its `max_spans` spans with the highest scores. The Transformer writes their numbers into `FieldItem.spans`, which a written plan leaves out. A plan made of field items only skips the extraction of the user's question.

**Spans and numbering.** `Answer.spans` holds the question's spans, then the field spans they don't already have; a field span that the question also found is the same span with one citation. `Answer.plan` numbers every span by its place in `Answer.spans`, fields included. `Answer.field_spans()` returns each field's spans by name, an empty list for a field without spans: the JSON answer of #74 is `{name: [span.model_dump() ...]}`, with the chunk id, the offsets and the score of every value.

**Rendering.** A field is a list item that starts with its name in bold. One span follows the name; several are a nested list. An empty field shows `PlanText.no_evidence` when it is `required` and is left out otherwise. The quote text is the span's own text, and the markers are written like those of any quote. `layout_plan` refuses a field item whose spans aren't filled in.

## Consequences

- Checklists and structured answers need no LLM: a plan of field items with a decision-model extractor answers in one model call per chunk.
- `FieldItem.spans`, `PlanConfig.plan`, `PlanConfig.text`, `SpanExtractor.extract_fields`, `aextract_fields` and `Answer.field_spans` are new. Answers without plans and LLM plans render as before.
- Still open: `[NAME]` placeholders parsed into field items, the structured mode of `VerbatimRAG` (ADR 0004) replaced by field plans, several written plans chosen by question (ADR 0007, step 4), and a threshold per field.
