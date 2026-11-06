# Stateless core: open questions for VerbatimRAG

This file collects the decisions VerbatimRAG needs before the stateless-core work ("One interface per capability: stateless core, stores passed in") can be implemented in this repository. Answers that are decided here become ADRs; terms that get sharpened go into `GLOSSARY.md`.

The questions are split in three groups:

- **Decided here**: only VerbatimRAG is affected, or VerbatimRAG can decide its part on its own.
- **Decided elsewhere**: cross-library questions. VerbatimRAG waits for the answer and lists what it needs from it.
- **Not relevant**: questions from the design doc that don't touch this repository.

Each question has a recommended answer. A recommendation is not a decision until the Decision field is filled in.

## Decided here

❓ **Q1 - Score for LLM span extraction** (design doc, open question 3)

`LLMSpanExtractor` produces no probability, but `Span.score` is a required float and `min_score` filters on it. What does `score` hold for spans the LLM extracted?

➡️ 1.0, documented as "extracted without a score", so `score` is always a number and a `min_score` filter keeps these spans. This must match what LettuceDetect decides for the LLM judge without reasoning, so confirm with the LettuceDetect side before writing the ADR.

Decision: 1.0, documented as "extracted without a score". The fuzzy matcher's match quality does not go into `score`: it says how faithfully a quote was located, not how relevant it is. VerbatimRAG has no `min_score`; nothing on this side filters or sorts by score. The ADR waits until LettuceDetect picks the same value for the judge without reasoning. (2026-11-02)

---

❓ **Q2 - Default LLM client after the deprecation window** (design doc, open question 4)

During the window, `VerbatimRAG(index)`, `LLMSpanExtractor()` and `TemplateManager()` without a client build `LLMClient(model)` as today and warn. After the window, is the client required?

➡️ Required wherever an LLM is actually called, and nowhere else. A `VerbatimRAG` with a `ModelSpanExtractor` and static templates calls no LLM and should need no client at all. The READMEs show the explicit form from 0.3.0 on.

Decision: Correction to the question: `TemplateManager` never builds a client. Without one, contextual mode silently falls back to static with a `print` (`templates/manager.py:65`).
- `LLMSpanExtractor(llm_client)`: the client is required after the window. During the window, omitting it builds `LLMClient(model)` and warns.
- `TemplateManager`: the client stays optional. During the window, contextual mode without a client still falls back to static but warns. After the window it raises a typed error, at the call and at `Transformer` construction when the default config already needs a client.
- `VerbatimRAG`: after the window it takes no `model`, and `llm_client` only for structured mode (Q8). Structured mode without a client raises.
- The READMEs show the explicit form from 0.3.0 on. (2026-11-02)

---

❓ **Q3 - Span text for fuzzy matches**

The design doc requires that a span's `text` always equals the slice at its offsets. With fuzzy matching (`verbatim_core/extractors.py:838`), the quote the LLM returned and the matched slice of the chunk can differ. Which one is `Span.text`, and what happens to a quote that matches nothing above the threshold?

➡️ `Span.text` is the slice of the chunk, never the LLM's quote. A quote without a match is dropped and reported in `Extraction.warnings`, not silently discarded, in line with "Typed errors, and no fallbacks that hide a failure".

Decision: `Span.text` is the slice of the chunk, which both match modes already return today; the matchers now return the offsets too. A quote without a match is dropped and reported in `Extraction.warnings`. "Quote" (the string the LLM returns) and "span" (its located slice) are defined in `GLOSSARY.md`. (2026-11-02)

Revised: the offsets work no longer waits for PR #79 (#34). Fixing #34 well is a larger problem than it looks, so it is done separately. The locator uses today's fuzzy matcher, changed only to return offsets with the text and kept behind one function, which the fix for #34 replaces. The fuzzy-mode fixtures record today's matcher, and that fix updates them as an intended change. (2026-11-03)

---

❓ **Q4 - Which occurrence of a repeated quote**

When a quote occurs more than once in a chunk, the extractor picks one occurrence (design doc, `verbatim.extract`). Model extractors get offsets from the model, so this only concerns the LLM extractor and subclasses written against the old interface. Which occurrence is picked?

➡️ The first exact occurrence; if there is none, the best fuzzy match. Documented in the ADR and in the changelog entry that already covers "highlights the chosen occurrence instead of every occurrence".

Decision: The first exact occurrence, else the best fuzzy match, the earliest on a tie. Identical `(start, end)` spans collapse into one. The response builder keeps today's first-wins rule for overlapping highlights. The changelog entry also says that repeated text now gets one citation instead of one per occurrence, since today every highlighted occurrence gets its own citation number. (2026-11-02)

---

❓ **Q5 - Where `Chunk.id` comes from**

`ExtractedSpan.chunk_id` points at a `Chunk.id`. `SearchResult.to_chunk()` has to fill it, and the deprecated `extract_spans` adapter builds chunks from positions (`_chunk_from_result(i, result)`). Is a chunk id meant to be stable beyond a single call, for example in citations returned by the API?

➡️ Yes. `SearchResult.to_chunk()` uses the id the index stores for the chunk, stable across queries. Positional ids are only used inside the deprecated adapters, where the id never leaves the call. Chunks sent with a request to `verbatim.transform` keep the ids the caller gave them.

Decision: `SearchResult.to_chunk()` uses `SearchResult.id`, the vector row id (`ProcessedChunk.id`, a random UUID). It is stable while the document stays indexed but not across re-indexing, and the documentation says so. A deterministic `"{document_id}:{chunk_number}"` was rejected: after a chunking change it would silently point at different text. Positional ids are used only inside the deprecated adapters; ids sent to `verbatim.transform` are kept as given. (2026-11-02)

---

❓ **Q6 - Streaming events and today's API**

`Transformer.stream` and `astream` yield a `spans` event and an `answer` event. `VerbatimRAG.query_stream` sends a documents event first. `StreamingRAG` (`verbatim_rag/streaming.py`) and the route at `api/app.py:350` emit today's event dicts. Does `query_stream` yield typed `TransformEvent` objects or today's dicts, and what exactly is in each event?

➡️ `query_stream` yields typed events. The deprecated `StreamingRAG` adapter converts them to today's dicts, and the existing streaming route keeps its current shape until a new route replaces it. Define the fields of the `documents`, `spans` and `answer` events in the ADR.

Decision: `query_stream` yields typed events: `documents` (the chunks), `spans` (the `ExtractedSpan`s), and a final event carrying the same `QueryResponse` that `query` returns, so the two are testably equal. Errors raise typed exceptions instead of arriving as events. `progress` is not a typed event; the `StreamingRAG` adapter rebuilds it from timing. The existing route keeps today's five event types (`documents`, `progress`, `highlights`, `answer`, `error`). (2026-11-02)

---

❓ **Q7 - The name `Chunk` is already taken**

`verbatim_rag/document.py:157` defines `Chunk`, the ingestion record (`document_id`, `content`, `start_index`, `end_index`, `chunk_number`, `chunk_type`, `processed_chunks`), which users pass in to index pre-chunked documents. The shared `Chunk` is the core's input. The default chunker never fills `start_index` and `end_index`, and they aren't stored.

➡️ "Chunk" means the shared core type. The ingestion dataclass becomes `DocumentChunk`, with `verbatim_rag.document.Chunk` as a deprecated alias. Indexed chunks have `source_start` and `source_end` set to `None` until the index stores them.

Decision: Accepted as recommended. (2026-11-02)

---

❓ **Q8 - Structured template mode**

The design doc doesn't cover `template_mode="structured"`. `VerbatimRAG.query` branches on it (`verbatim_rag/core.py:250`, `:344`) and calls `self.llm_client.extract_structured(...)` directly, bypassing `SpanExtractor`. `TemplateManager` and `StructuredTemplate` hold a back-reference to the RAG object (`set_rag_system`), and their `fill_async` calls `rag_system.query_async`. Issue #74 redesigns structured answers.

➡️ Structured mode stays out of the core for this rollout, as a documented exception inside `VerbatimRAG`. `Transformer` raises if asked for it. `fill_async` and `set_rag_system` are deprecated. #74 brings structured answers into the core as their own capability.

Decision: Accepted as recommended. (2026-11-02)

---

❓ **Q9 - Template settings changed between calls**

`TemplateManager` has setters that change it between calls: `set_mode`, `set_citation_mode`, `set_citation_format`, `use_structured_mode` (`templates/manager.py:69`, `:377`, `:456`, `:468`). A `Transformer` shared across requests would race on them.

➡️ `TransformConfig` carries `template_mode`, `citation_mode` and `citation_format`. The `TemplateManager` constructor arguments become the default config, the setters become deprecated adapters that replace it, and nothing in the core reads `current_mode`.

Decision: Accepted as recommended, except `use_structured_mode` (see Q12). `TemplateManager.save` and `load` stay: "the core writes nothing to disk" applies to capability calls, and these are explicit I/O the caller asks for at setup (Q14). (2026-11-02)

---

❓ **Q10 - Random and question-specific templates**

`RandomTemplate.generate` uses `random.choice` per call (`templates/random.py:101`), which fails "same input, same result". `QuestionSpecificTemplate` loads a `SentenceTransformer` by name inside a call (`templates/question_specific.py:84`) and caches embeddings on `self`.

➡️ Random mode picks its template from a stable hash of the question, with an optional `seed` in `TransformConfig`. `QuestionSpecificTemplate` takes a loaded embedder, with `from_pretrained(model_name)` as the one place a name becomes a model, and embeds its examples at construction.

Decision: Accepted as recommended. (2026-11-02)

---

❓ **Q11 - A failed LLM call silently returns no spans**

In batch mode a failed call falls back to individual calls (`verbatim_core/extractors.py:616`). If that fails too, or in individual mode, the chunk gets no spans with only a `logger.error` (`:740`, `:771`). The caller can't tell "nothing relevant" from "never examined".

➡️ A failed LLM call raises a typed error and `extract` fails as a whole. Retries belong in the LLM-call layer. The batch-to-individual fallback stays only in the deprecated path.

Decision: Accepted as recommended. (2026-11-02)

---

❓ **Q12 - Q8 and Q9 conflict on `use_structured_mode`**

Q9 deprecates every `TemplateManager` setter, including `use_structured_mode`, but Q8 keeps structured mode with no replacement until #74. A deprecation warning has to name a replacement.

➡️ `use_structured_mode` and the structured template's setup are left out of Q9's deprecation. `VerbatimRAG` keeps detecting structured mode from the template manager's mode, as today. The whole structured API is deprecated in one step when #74 provides the replacement. `fill_async` is deprecated now, in favour of `VerbatimRAG.query_async`.

Decision: Accepted as recommended. (2026-11-02)

---

❓ **Q13 - Duplicate chunks in one call**

A `verbatim.transform` request can send two chunks with the same id, which makes `ExtractedSpan.chunk_id` ambiguous. Retrieval can return two rows with identical text, the processed versions of one document chunk (`verbatim_rag/index.py:392`); today `extract_spans` merges them because it keys its result by text.

➡️ Duplicate ids in one call raise a typed input error. Chunks with the same text but different ids are extracted independently, with no merging by text.

Decision: Accepted as recommended. (2026-11-02)

---

❓ **Q14 - `TemplateManager.save` and `load`**

`TemplateManager.save(filepath)` and `load(filepath)` read and write JSON (`templates/manager.py:195`), and `TemplateManager` is passed into the core `Transformer`.

➡️ They stay: explicit I/O the caller asks for at setup, never a side effect of `transform`.

Decision: Accepted as recommended, and recorded with Q9. (2026-11-02)

---

❓ **Q15 - Which decisions get an ADR**

➡️ ADRs for Q1 (after LettuceDetect agrees), Q4, Q5, Q6, Q8 with Q12, Q10 and Q11, in `docs/adr/` numbered from 0001. Q3, Q7 and Q9 need none: `GLOSSARY.md` and the design doc explain them.

Decision: Accepted as recommended. Written: `docs/adr/0001` (Q4) to `0006` (Q11). The Q1 ADR waits for LettuceDetect. (2026-11-02)

---

❓ **Q16 - Frontend stream errors**

The frontend drops `error` events (`frontend/src/contexts/ApiContext.js:165`), so a failed streamed query stops loading without a message. This is a bug in today's code, independent of the stateless-core work.

➡️ File it once these questions are settled, as `Bug: streamed query errors never reach the user`, labelled `bug` and `needs-triage`.

Decision: Accepted as recommended. Filed as #85. (2026-11-02)

---

❓ **Q17 - `query` and `query_stream` count display citations differently**

The response builder marks citation *n* as `display` while *n* ≤ `display_span_count`. `VerbatimRAG.query` and `query_async` pass `len(all_relevant_spans)`, the number of chunks (`verbatim_rag/core.py:271`, `:369`). `StreamingRAG` and both `VerbatimTransform` copies pass `len(display_spans)`, the number of spans shown in the answer. Q6 requires the two to return equal responses.

➡️ The number of spans shown is right, and `query`'s count is a bug. File it next to Q16 as `Bug: VerbatimRAG.query marks citations as display by chunk count`, and fix it before the rollout so the regression fixtures record the fixed behaviour.

Decision: Accepted as recommended. Filed as #86. (2026-11-02)

## Decided elsewhere

| Design doc question | What VerbatimRAG needs from the answer | Blocks |
|---------------------|----------------------------------------|--------|
| 1 - shared package for `Chunk`, `Span`, `Usage`, or copies per library | where `verbatim_core` imports the types from, and whether `ExtractedSpan` subclasses a shared `Span` | rollout step 1 (shared types) |
| 2 - length of the deprecation window | the `since` and `removed_in` values in every `@deprecated` call and in the changelog | rollout step 6 (adapters) |
| "One LLM-call layer in each library" | what replaces `LLMClient` as the client passed to `LLMSpanExtractor` and `TemplateManager` | nothing yet: `LLMClient` is used until then |
| "Make the vector store easy to swap" | the `VectorStore` protocol `VerbatimIndex` is built on | `VerbatimIndex` changes only |

## Not relevant here

Design doc questions 5 (default local store for `RuleChef`), 6 (store protocol for training examples) and 7 (async methods on `RulesetStore` and `TrafficStore`) concern RuleChef only.
