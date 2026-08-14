# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Docker Compose development and demo stack: the FastAPI API and the Vite/React UI run
  locally behind nginx, which proxies `/api/` same-origin. Milvus Lite data persists on a
  named volume. Not a production deployment — no TLS, authentication, tenancy, or scaling.
- `docker/constraints.txt`: a checked-in dependency lock used only by the container image;
  the published ranges in `pyproject.toml` are unchanged. Pins `milvus-lite<3`, whose 3.x
  line rejects the vector-less metadata collection `LocalMilvusStore` creates.

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
