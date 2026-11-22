# Decision models as span extractors, in memory or on a server

Decision models answer typed questions about a request and its context in one call. Vela 2.0 (vllm-sr/Vela-2.0, Apache-2.0, 17 languages, up to 16,384 tokens) is the first family: its decoders have a broad span head that returns spans of the context, with offsets in Unicode code points and a probability each, for labels given at request time. Issue #71 asks for a VerbatimRAG extractor on these models, and #74 and #75 build on it: structured answers whose fields are span questions, and a pipeline that runs on one model.

The models run in two ways. In memory, `AutoModel.from_pretrained(..., trust_remote_code=True)` gives a model with `system_one(state, questions)`. On a server, Vela's `vela2_serve.py` serves the same call as `POST /v1/systemone`, with the body `{"model", "state", "questions"}` and the response of `system_one`. Neither is vLLM or an OpenAI-compatible API. A deployment may also put the model behind a server of its own.

## Decision

`verbatim_core.decision_models.DecisionModelExtractor` extracts spans with a decision model. It asks one span question per chunk with text: the state is `{"request": question, "context": chunk.text}`, and the question is

```python
{"evidence": {"type": "span", "instructions": "Which spans of the context answer the request?",
              "over": "context", "criteria": {"answer": "text that answers the request"}}}
```

`DecisionModelConfig` sets the instructions, the label and its description, and the threshold (None keeps the model's calibrated threshold; Vela's broad head uses 0.55). A span is kept when its offsets lie in the chunk, hold more than whitespace and hold the text the model returned; any other span is dropped with the warning `span_offsets_mismatch`, so a model can't put text into an answer that isn't in the chunk. A span's probability is its score.

The extractor calls a **backend**: any object with `model_id`, `model_version` and `system_one(state, questions)`, and optionally an async `asystem_one` (the `DecisionModel` protocol). Two backends are built in:

- `LocalDecisionModel` wraps a loaded model, and `LocalDecisionModel.load(model_id, device, revision=...)` loads one with transformers. Its calls take turns on one lock, as `ModelSpanExtractor`'s do. torch and transformers are not a dependency: Vela needs a newer transformers than the `model` extra pins, and an extra for each would conflict in one lock file. A missing package raises `ConfigurationError` with the code `missing_dependency`.
- `RemoteDecisionModel(base_url, model, api_key=..., model_version=...)` posts to `/v1/systemone` (the path can be changed) with httpx, which becomes a direct dependency of the core (the OpenAI client already installed it). Callers can pass their own httpx clients for proxies, certificates or tests.

`DecisionModelExtractor.from_pretrained(...)` and `DecisionModelExtractor.from_url(...)` build the extractor on one of them. A server with another protocol, such as a team's own gateway, needs a small class with `system_one`; the extractor doesn't change.

`aextract` asks about all chunks at once when the backend has `asystem_one`, and otherwise runs the calls in worker threads one after another.

Server errors become the library's errors, without the server's message: 401 and 403 `ConfigurationError("authentication_failed")`; 413 `InvalidInputError("input_too_long")`, since a chunk is too long for the model; 408, 429 and 5xx a retryable `UpstreamError("server_error")`; other 4xx, such as 422 for a wrong model name, `UpstreamError("request_rejected")`; a timeout or a failed connection a retryable `UpstreamError`, and a body that isn't a JSON object `UpstreamError("invalid_reply")`. As for LLM calls (ADR 0006), a failed call fails the extraction.

## Consequences

- VerbatimRAG can use Vela as its extractor today, through `Transformer(DecisionModelExtractor.from_url(...), templates)` or in memory. Whether it should is a measurement on the evaluation harness (#29), not part of this change.
- One span question per chunk keeps every chunk's spans independent and its offsets in the chunk. Packing several chunks into one context, as the Vela 2.0 Studio does, would save calls; it can come later if the number of calls matters.
- Field items (ADR 0007, #74) are the next step: each field becomes a span question with its own instructions and label, asked in the same call as the others.
- The core depends on httpx directly.
