# Structured template mode stays outside the stateless core

In structured mode (`template_mode="structured"`), `VerbatimRAG` extracts by itself: it calls the LLM client's structured extraction directly instead of a `SpanExtractor`. It stays that way through the stateless-core rollout, as a deliberate exception to "the stateful classes only retrieve, store and call the core", because issue #74 redesigns structured answers (a JSON answer with a span per field) and would replace whatever the rollout built.

## Consequences

- `Transformer` raises when asked for structured mode.
- `VerbatimRAG` keeps an `llm_client` argument for structured mode alone, and structured mode without one raises.
- `use_structured_mode` and the structured template's setup keep working without warnings. The whole structured API is deprecated in one step when #74 provides the replacement.
- `fill_async` and `set_rag_system`, which let the templates call back into `VerbatimRAG`, are deprecated now in favour of `VerbatimRAG.query_async`.
