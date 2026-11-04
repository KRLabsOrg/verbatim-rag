# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Docker Compose development and demo stack: the FastAPI API and the Vite/React UI run
  locally behind nginx, which proxies `/api/` same-origin, published on `:8080` by default
  (`FRONTEND_PORT` overrides). Milvus Lite data persists on a named volume. Not a
  production deployment — no TLS, authentication, tenancy, or scaling.
- `docker/constraints.txt`: a checked-in dependency lock used only by the container image;
  the published ranges in `pyproject.toml` are unchanged. Pins `milvus-lite<3`, whose 3.x
  line rejects the vector-less metadata collection `LocalMilvusStore` creates.
- Groundwork for the stateless core (#89), not yet used by the rest of the library.
  `verbatim_core` exports the result types `Chunk`, `Span`, `ExtractedSpan`, `Usage`,
  `ResultWarning`, `Extraction`, `AnswerCitation` and `Answer`, whose validation errors leave
  out the input values so that customer text can't reach logs through them, and the errors
  `VerbatimError`, `InvalidInputError` (also a `ValueError`), `ConfigurationError` and
  `UpstreamError`. Both `verbatim_core` and `verbatim_rag` export `VerbatimDeprecationWarning`.
  verbatim-core now lists `typing_extensions>=4.5.0` as a direct dependency.
- `SpanExtractor.extract(question, chunks)` and `aextract`, which return an `Extraction`: spans
  with their chunk ids and offsets, plus the usage and warnings of the call (#90). A new
  extractor implements only `extract` and gets `extract_spans` from the base class. An extractor
  that implements only `extract_spans` gets `extract` and `aextract`: each span text is located
  in its chunk (the first exact occurrence, else the best fuzzy match), and texts that can't be
  located are dropped and reported in the result's warnings with the code `span_text_not_found`.
- The `verbatim_core.content` logger. Log records that hold customer text, such as questions,
  chunks and span texts, go only to this logger and only at DEBUG. Leave it at INFO to keep
  customer text out of your logs whatever level the other `verbatim_core` loggers have.
- `LLMSpanExtractor.extract` and `aextract` (#91). Each quote of the LLM is located in its
  chunk, and the span's text is the chunk's text at the located offsets. Every span has the
  score 1.0, "extracted without a score". Quotes that can't be located are dropped and reported
  in the result's warnings with the code `quote_not_found`. A failed LLM call raises, and the
  whole extraction fails. The usage names the LLM, with the model version the LLM service
  returned, and counts the chunks, characters, LLM calls and, where the service reports them,
  tokens. `aextract` makes its LLM calls concurrently. A client that replaces
  `LLMClient.complete`, for example to call another provider, is still called that way, and its
  usage then has no model version or tokens.
- `ExtractConfig`, the base class of the extractors' settings, and `LLMExtractConfig`, the
  settings of `LLMSpanExtractor`: `LLMSpanExtractor(llm_client, config=LLMExtractConfig(...))`.
  A `config` passed to `extract` replaces the default for one call (#91).
- `LLMClient.completion` and `completion_async`, which return the text of a reply with the
  model name and the token counts the LLM service sent (#91).
- `ModelSpanExtractor.from_pretrained(model_id, device=None)` and
  `SemanticHighlightExtractor.from_pretrained(...)`, which download and load a model or load it
  from a local path (#92). They record the model's Hugging Face commit as the version the usage
  of each extraction reports; for a local path, pass `model_version`. They also load the
  highlighter's tokenizer, which it used to load on the first extraction. Loading logs the event
  `model_loaded` at INFO. The constructors take a loaded model and a config, and load nothing.
- `extract` and `aextract` on `ModelSpanExtractor` and `SemanticHighlightExtractor`, whose spans
  keep the offsets and probabilities the model computed (#92). Highlighter models give their
  character spans with the span probability, the v1 sentence model the sentences of a chunk with
  the sentence probability. `SemanticHighlightExtractor` locates the sentences it highlights in
  order, scored with the sentence probability, and turns token regions into offsets, scored with
  the mean token probability. When a model examines only part of a chunk, the result has a
  warning with the code `chunk_truncated`. `aextract` runs the model in a worker thread.
- `ModelExtractConfig` and `SemanticHighlightConfig`, the settings of the model extractors (#92).
- `Transformer(extractor, templates, config=TransformConfig(...))`, which writes a cited answer
  from a question and chunks, without an index (#93). `transform` and `atransform` return an
  `Answer`: its text, a citation for each span (its number, whether the answer shows the span or
  cites it for reference only, and the span's chunk id and offsets), the spans, the usage of the
  extractor and of the template LLM when one ran, and the warnings. `stream` and `astream` yield a
  `spans` event once the spans are extracted and an `answer` event with the same `Answer`.
  `TransformConfig` holds the settings of a call: the number of displayed spans, the template
  mode, the citation mode and format, and a seed. Nothing is set on the transformer between
  calls, so one transformer can serve concurrent calls. In random template mode it chooses the
  template by a hash of the question, combined with the seed, so the same question gets the same
  template. Structured template mode raises: it stays in `VerbatimRAG`.
- In a `Transformer` call, a contextual template falls back to the fallback template only when
  the LLM call fails with an `UpstreamError` or the LLM sends an empty template, and the answer
  then has the warning `template_fallback`. A rejected API key, model or quota raises
  `ConfigurationError`. Generated templates are logged only on the `verbatim_core.content`
  logger. A client that replaces `LLMClient.generate_template` is still called that way (#93).
- `ResponseBuilder.build_from_answer`, which builds today's `QueryResponse` from an `Answer`. It
  highlights each cited span at its offsets, without searching for its text, and of two
  overlapping spans keeps the first. A citation of a chunk it isn't given, or with offsets
  outside its chunk, raises `InvalidInputError` (#93).
- `LLMClient.template_completion` and `template_completion_async`, which return the completion
  of a template prompt and raise when the call fails, and `TemplateStrategy.generate_result`,
  which returns a template with the usage and warnings of writing it (#93).
- `TemplateManager.config`, the `TransformConfig` that the template manager's constructor
  arguments describe. A `Transformer` built without a `config` uses it as its default, and raises
  `ConfigurationError` at the call while the template manager is in structured mode (#94).
  `TemplateManager.save` also saves the citation mode and format, and `load` restores them.
- `VerbatimRAG(index, transformer=..., llm_client=None, intent_detector=None, reranker=None)`
  (#96). `query` and `query_async` take `k` and the other retrieval arguments per call, turn the
  retrieved rows into chunks with `SearchResult.to_chunk()`, whose id is the row's id, and answer
  with the `Transformer`. Nothing is set on `VerbatimRAG` by a call. The `llm_client` serves
  structured template mode alone, which still answers without the `Transformer`.
- `VerbatimRAG.query_stream(question, k=..., ...)`, an async generator with the retrieval
  arguments of `query` (#97). It yields a `DocumentsEvent` with the retrieved chunks, a
  `SpansEvent` with their spans and a `ResponseEvent` with the `QueryResponse` that `query`
  returns for the same input, and raises the errors `query` raises. A question the intent
  detector answers yields the `ResponseEvent` alone. The events are exported from `verbatim_rag`.
- `DocumentChunk`, the new name of the ingestion record `verbatim_rag.document.Chunk` (#96).
- `QueryResponse.warnings`: how an answer is degraded but still correct, as `ResultWarning`s with
  a code and a message, such as `quote_not_found`, `chunk_truncated`, `template_fallback` and
  `reranker_fallback` (#96). `VerbatimRAG`, `VerbatimTransform`, `verbatim_enhance`, the CLI and
  the API server fill it; it was dropped before, so a response without warnings now has
  `"warnings": []`.
- `QuestionSpecificTemplate.from_pretrained(model_name, device=None, templates=...)`, and the
  `embedder` argument of `QuestionSpecificTemplate` and `TemplateManager.use_question_specific_mode`,
  which take a loaded embedding model (#94). The template examples are embedded when the
  templates are built, and each question with the same model, so no call loads a model.
  `from_pretrained` logs the event `model_loaded` at INFO.

### Changed
- A `SpanExtractor` subclass that implements neither `extract` nor `extract_spans` raises
  `TypeError` when it is defined, not when it is instantiated (#90).
- `LLMClient` translates the errors of the OpenAI client (#91). A rejected API key, model or
  quota raises `ConfigurationError`. A timeout, a connection error, a rate limit, a server error
  or a reply that can't be read raises `UpstreamError`, which says whether a retry may help. The
  OpenAI exception is the `__cause__`. An empty reply raises `UpstreamError` instead of
  `ValueError`, and so does a reply without text, which returned `None`.
- `LLMSpanExtractor.extract_spans` raises `ConfigurationError` when the LLM service rejects the
  API key, the model or the quota, instead of returning no spans. It still tolerates the other
  failed calls, and now also counts a reply that isn't a JSON object of quote lists as one (#91).
- `LLMSpanExtractor` logs quotes it can't locate in one WARNING record that counts them, without
  their text, and its steps at DEBUG instead of INFO. The quotes go to the
  `verbatim_core.content` logger at DEBUG (#91).
- The settings of an `LLMSpanExtractor` are in its `config`. The attributes `extraction_mode`,
  `batch_size`, `span_match_mode`, `fuzzy_threshold`, `extraction_prompt`, `system_prompt` and
  `max_display_spans` are gone (#91).
- An unknown `extraction_mode`, which acted like `individual`, raises: `ConfigurationError` from
  `LLMSpanExtractor`, and pydantic's `ValidationError`, a `ValueError`, from `VerbatimRAG` and
  `VerbatimTransform` (#91).
- An error of a model extractor's model propagates, instead of giving its chunk no spans (#92).
- `SemanticHighlightExtractor` runs its model on the device it is given, or the best available
  one, where it ignored `device`, and the model no longer draws progress bars (#92).
- The model extractors raise `ConfigurationError` instead of `ImportError` when torch,
  transformers or a package the model's code needs is missing (code `missing_dependency`),
  instead of `OSError` when a model can't be downloaded or read (code `model_not_loaded`), and
  instead of `ValueError` for an invalid `output_mode` (#92).
- The settings of the model extractors are in their `config`. The attributes `threshold`,
  `min_span_chars`, `merge_gap_chars`, `max_length`, `doc_stride`, `output_mode`, `language`,
  `min_span_tokens`, `merge_gap`, `max_tokens`, `device` and `model_path` are gone (#92).
- Random template mode chooses the template by a hash of the question in `TemplateManager.process`
  too, as in a `Transformer` call, instead of at random: the same question gets the same
  template (#94).
- The template classes log instead of printing, and `QuestionSpecificTemplate` logs its steps at
  DEBUG instead of INFO. When a template can't be written, the WARNING record has the event
  `fallback_used` and the warning code `template_fallback` (#94).
- `ContextualTemplate.generate` and `generate_async`, for a client that replaces
  `generate_template`, and `RandomTemplate.generate_diverse_templates` fall back only when the
  LLM call fails with an `UpstreamError`. A rejected API key, model or quota raises
  `ConfigurationError` (#94).
- `QuestionSpecificTemplate` raises `ConfigurationError` (code `missing_dependency`) instead of
  `ImportError` when sentence-transformers is missing, and `ConfigurationError` (code
  `model_not_loaded`) instead of `OSError` when the model can't be loaded (#94).
- `TemplateManager.load` raises these errors instead of logging them and going on, and raises
  `ConfigurationError` (code `missing_embedder`) for question-specific templates that were saved
  with an embedder of their own when no embedder is given (#94). It still returns `False` for a
  file that can't be read or isn't JSON.
- `TemplateManager.set_citation_format` raises `ConfigurationError` for a format with variables
  other than `{number}` and `{span_id}`, instead of failing when a template is filled (#94).
- `VerbatimTransform` answers with a `Transformer` (#95). verbatim-rag's copy is gone:
  `verbatim_rag.VerbatimTransform` is verbatim-core's, which also accepts context items that are
  objects with a `text` attribute. `verbatim_enhance`, the `verbatim-enhance` CLI and the API
  server's transform route use a `Transformer` directly; the route builds it once at startup and
  logs no question or chunk text. The answers are today's, except:
  - A failed LLM call raises, from `VerbatimTransform`, `verbatim_query`, `verbatim_enhance` and
    the CLI, and the API route answers 500, instead of an answer without the chunks the call
    covered (ADR 0006).
  - A displayed span that a contextual template leaves out is cited as a reference.
  - Context items with the same text each get their spans, instead of sharing one entry.
- `VerbatimRAG.query` and `query_async` answer with a `Transformer` (#96). The answers are
  today's, except:
  - A failed LLM call during extraction raises, instead of answering from the remaining chunks
    (ADR 0006).
  - Retrieved rows with the same text each get their spans, instead of sharing one entry.
  - Their steps log at DEBUG instead of printing.
  - An extractor that implements only `extract_spans` receives the chunks, with `id`, `text` and
    `metadata`, instead of the search results; one that reads `score` or `enhanced_text` needs to
    implement `extract`. The same holds for `VerbatimTransform` (#95).
- `StreamingRAG` and the API server's streaming route send the events of
  `VerbatimRAG.query_stream` as the dicts they sent before (#97). `num_docs` and the other
  retrieval arguments apply to the call: `StreamingRAG` no longer sets `k` on the `VerbatimRAG`
  it streams, so concurrent streams don't interfere. The events are today's, except:
  - The response of the `answer` event has the `warnings` that `query` reports.
  - An `error` event has the error's `code`, and its `error` is the code, or the message of an
    `InvalidInputError`, instead of the exception's text. An error that isn't a `VerbatimError`
    has the code `internal_error`. The failure is logged once on `verbatim_rag.streaming`, with
    its traceback. The traceback shows the messages of `VerbatimError`s only, since another
    exception's message may quote the question or a chunk, and the route logs no question text.
  - A failed LLM call during extraction ends the stream with an `error` event, instead of
    answering from the remaining chunks (ADR 0006).
  - The `highlights` event has the response's highlights: one per span, at its offsets
    (ADR 0001). Retrieved rows with the same text each get their spans, instead of sharing one
    entry.
  - In structured template mode the stream answers as `query` does, with `VerbatimRAG`'s
    client, and sends no `progress` or `highlights` event. Before, it filled the structured
    template from the extractor's spans.
- A reranker failure keeps the retrieval order only when its service fails (`UpstreamError`), adds
  the warning `reranker_fallback` to the response, and logs `fallback_used` on the
  `verbatim_rag.core` logger instead of the root logger. Other errors propagate. `CohereReranker`
  and `JinaReranker` raise `UpstreamError` for a failed call, and `ConfigurationError` for a
  rejected API key or permission or an unknown model (#96).
- Structured template mode without an LLM client raises `ConfigurationError` (code
  `missing_llm_client`) (#96).
- A context item of the wrong shape raises `InvalidInputError` (code `missing_field` or
  `invalid_value`, `param` such as `context[3]` or `context[3].title`) instead of `TypeError` or
  `ValueError`, before any LLM call. This includes metadata that isn't a dict with string keys,
  and a title or source that isn't a string. `InvalidInputError` is a `ValueError`, but
  `except TypeError` no longer catches a context item that isn't a dict (#95).

### Deprecated
- Implementing only `extract_spans` in a `SpanExtractor` subclass: `extract` and `aextract` warn
  when they are called on it. Implement `extract` instead (#90).
- `extract_spans` and `extract_spans_async` where an extractor inherits them from
  `SpanExtractor`. Use `extract` and `aextract` (#90).
- `LLMSpanExtractor.extract_spans` and `extract_spans_async`. Use `extract` and `aextract` (#91).
- `LLMSpanExtractor()` without an `llm_client`, which builds `LLMClient(model)`. Pass the client:
  `LLMSpanExtractor(LLMClient(model))`. The argument `model` next to a client and the argument
  `max_display_spans` have no effect (#91).
- The `LLMSpanExtractor` arguments `extraction_mode`, `batch_size`, `span_match_mode`,
  `fuzzy_threshold`, `extraction_prompt` and `system_prompt`. Pass them in
  `config=LLMExtractConfig(...)` (#91).
- `ModelSpanExtractor(model_path=...)` and `SemanticHighlightExtractor(model_name=...)`, and
  both without a loaded model or with a model name, which load the model. Use `from_pretrained`
  (#92).
- The settings arguments of the model extractors. Pass them in `config=ModelExtractConfig(...)`
  or `config=SemanticHighlightConfig(...)`. The `ModelSpanExtractor` arguments
  `extraction_mode` and `max_display_spans` have no effect (#92).
- Contextual template mode without an LLM client. It warns where the template manager printed a
  note. A `Transformer` call or its default config (#93) and
  `TemplateManager(default_mode="contextual")` use the static template; `set_mode` and
  `use_contextual_mode` return `False` and keep the template mode (#94). Give the
  `TemplateManager` an `llm_client`.
- `TemplateManager.set_mode`, `set_citation_mode` and `set_citation_format`, and assigning its
  `current_mode` or `citation_mode`. Pass the settings in a `TransformConfig` to the `Transformer`
  or to its call. Until they are removed, they change the template manager's `config`, the
  default of every `Transformer` built with it and no `config` of its own. `use_structured_mode`
  switches to structured mode without a warning (#94).
- `TemplateManager.set_rag_system`, `StructuredTemplate.set_rag_system` and
  `StructuredTemplate.fill_async`, through which the templates call back into `VerbatimRAG`. Use
  `VerbatimRAG.query_async` (#94).
- `QuestionSpecificTemplate(model_name=..., device=...)`, and question-specific templates
  without an `embedder`, which load an embedding model by name. Use
  `QuestionSpecificTemplate.from_pretrained(model_name)` or pass `embedder` (#94).
- `VerbatimRAG` without a `transformer`, and its arguments `model`, `k`, `template_manager`,
  `extractor`, `max_display_spans`, `template_mode` and `extraction_mode`. Pass a `Transformer`,
  and `k` to `query` (#96).
- Assigning `VerbatimRAG.extractor`, `template_manager` or `max_display_spans`. Pass a
  `Transformer` with the extractor, templates and config (#96).
- `verbatim_rag.document.Chunk`. Use `DocumentChunk` (#96).
- `StreamingRAG`, `StreamingRAG.stream_query` and `stream_query_sync`. Use
  `VerbatimRAG.query_stream` (#97).
- `VerbatimTransform`. Use `Transformer`, and `ResponseBuilder.build_from_answer` for a
  `QueryResponse` (#95).
- `verbatim_query` and `verbatim_query_async`. Retrieve with the provider, then call a
  `Transformer` (#95).
- The module `verbatim_rag.transform`. Its names are verbatim-core's: import them from
  `verbatim_core` (#95).

### Fixed
- `VerbatimRAG.query` and `query_async` mark as `display` exactly the citations whose spans
  the answer shows, as the streaming path already did. They counted the retrieved chunks
  instead, so with more chunks than shown spans, cited-only spans were marked as displayed,
  and with fewer, shown spans were marked as references (#86). Structured mode is unchanged.
- `LocalMilvusStore` no longer fails on Milvus Lite 3 and pymilvus 3. Its metadata collection
  (`<collection>_documents`) now has a placeholder vector field, as `CloudMilvusStore` already
  does, because the 3.x line rejects collections without a vector field (#69).

## [0.2.8] - 2026-05-31

### Added
- **FR-1** `TemplateFiller`, `StaticTemplate`, `ContextualTemplate`, and `TemplateManager` now accept a `citation_format` parameter (default `"[{number}]"`). Supports `{number}` (sequential integer) and `{span_id}` (from span data, falls back to `str(number)`). Use `citation_format="[{span_id}]"` to render markers like `[cite1]` directly from span ids. Fully backward compatible — default reproduces pre-0.2.8 output exactly.
- **FR-2** `ContextualTemplate` and `TemplateManager.use_contextual_mode` accept `template_preview_chars` (default `100`, `None` = no truncation) and `preserve_span_newlines` (default `False`). Allows multi-line structured spans (e.g. dosage blocks) to reach the template LLM without being collapsed or cut.
- **FR-3** `ContextualTemplate` and `TemplateManager.use_contextual_mode` accept `template_prompt` (custom Jinja2 string replacing the bundled per-fact/aggregate prompt) and `system_prompt` (system message for the template-generation call). Mirrors the `LLMSpanExtractor` API for custom extraction prompts.

### Fixed
- `QueryResponse.documents` field used Pydantic v1 `min_items=0` (deprecation warning under Pydantic 2); replaced with `default_factory=list` and migrated `class Config` to `model_config`.

### Changed
- `verbatim-rag` now requires `verbatim-core>=0.2.8`.

## [0.2.7] - 2026-05-22

### Changed
- Template generation prompts no longer use the word "facts": replaced with "quotes" throughout, renamed per-span placeholders from `[FACT_N]` to `[SPAN_N]`, and added explicit instructions to infer labels from content rather than placeholder names. Existing stored templates with `[FACT_N]` continue to work.
- `verbatim-rag` now requires `verbatim-core>=0.2.7`.

## [0.2.6] - 2026-05-18

### Changed
- `LLMSpanExtractor` fuzzy matching now normalizes text at token boundaries (NFKC + casefold) before alignment, preventing RapidFuzz offsets from landing mid-word. Matched slices are mapped back to original token boundaries in the source document.
- `verbatim-rag` now requires `verbatim-core>=0.2.6`.

## [0.2.5] - 2026-04-27

### Changed
- Tighten sdist packaging: `pyproject.toml` for both `verbatim-rag` and `verbatim-core` now declares an explicit `[tool.hatch.build.targets.sdist]` allowlist so source distributions only ship listed package and metadata files.
- `verbatim-rag` now requires `verbatim-core>=0.2.5`.

## [0.2.4] - 2026-04-27

### Added
- `ModelSpanExtractor` now supports the `KRLabsOrg/verbatim-rag-modern-bert-v2` highlighter (token classifier with `.process()` API loaded via `trust_remote_code=True`); legacy `QAModel` checkpoints continue to work and are auto-detected at load time
- `model_path` now defaults to `KRLabsOrg/verbatim-rag-modern-bert-v2`
- Highlighter tuning knobs: `min_span_chars`, `merge_gap_chars`, `max_length`, `doc_stride`
- MPS device auto-detection for Apple Silicon

### Changed
- Default `ModelSpanExtractor` threshold lowered from `0.5` to `0.2` to match the v2 highlighter's published headline. **If you use the legacy v1 sentence model, pass `threshold=0.5` explicitly.**
- `verbatim-rag` now requires `verbatim-core>=0.2.4`

## [0.2.3] - 2026-03-22

### Added
- `api_key` parameter on `LLMClient` for explicit API key passing (falls back to `OPENAI_API_KEY` env var)
- Test for explicit API key usage

## [0.2.2] - 2026-03-22

### Added
- Linked citations: display spans can now reference specific citation sources via `citation_ids`, enabling per-claim traceability
- When linked citations are present, flat `[CITATION_REFS]` is suppressed to avoid double-referencing
- 3 new tests for linked citation behavior

### Changed
- `TemplateFiller._format_span` now takes full span data dict instead of raw text, enabling richer formatting
- Extracted `_get_template_inputs` and `_build_citation_number_map` as clean static methods

## [0.2.1] - 2026-03-16

### Added
- Fuzzy span matching (`span_match_mode="fuzzy"`) using rapidfuzz for documents with OCR artifacts or encoding issues
- Custom extraction prompts via `extraction_prompt` and `system_prompt` parameters (Jinja2 syntax)
- Prompt bank: built-in prompt templates in `verbatim_core.prompts` with `load_prompt()` and `list_prompts()`
- All internal prompts now load from the prompt bank -- no hardcoded prompts in Python code

### Changed
- Added `rapidfuzz>=3.0.0` and `jinja2>=3.0.0` to verbatim-core base dependencies
- Replaced print statements with proper `logging` throughout extractors

## [0.2.0] - 2026-03-10

### Added
- **verbatim-core** package: lean, separately installable extraction core (openai + pydantic only)
- Optional `[model]` extra for verbatim-core: `pip install verbatim-core[model]` for ModernBERT/Zilliz extractors
- CI pipeline: linting (ruff), tests (Python 3.10-3.12), pip-audit
- MkDocs Material documentation site with API reference
- 88 unit tests covering models, response builder, LLM client, extractors, templates, and transform
- CONTRIBUTING.md

### Changed
- Moved `verbatim_core/` into `packages/core/` (standard Python monorepo layout)
- `verbatim-rag` now depends on `verbatim-core` (transitive dependency)
- Lazy imports for numpy, torch, and transformers in verbatim_core to keep lean install clean
- Import sorting and linting fixes across verbatim_core, verbatim_rag, api, and tests

### Removed
- Old test files that required heavy dependencies (torch, milvus)

## [0.1.9] - 2026-02-28

### Added
- `max_tokens` parameter to `SemanticHighlightExtractor`
- Flag to return search results along with RAG response
- `verify_spans` parameter to `LLMSpanExtractor`

## [0.1.8] - 2026-01-15

### Added
- `SemanticHighlightExtractor` using Zilliz semantic-highlight model
- Reranker support (Cohere, Jina, SentenceTransformers)
- Intent detection for query routing
- `VerbatimDOC` for document-level processing
- `UniversalDocument` for unified document representation
- Hybrid search (dense + sparse) support
- SPLADE embedding provider for CPU-only operation
