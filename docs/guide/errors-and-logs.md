# Errors, warnings and logs

A call of verbatim-core or verbatim-rag returns a correct result or raises. A result that is degraded but still correct says so in its `warnings`. Error codes, warning codes and log events are public API: they keep their names across releases.

## Errors

Every error the library raises on purpose is a `VerbatimError`, exported from `verbatim_core`, with a stable snake_case `code`. Its class says where the problem is:

- **`InvalidInputError`**: an argument of the call is wrong, and only a different input fixes it. `param` names the argument, such as `chunks[3].id`. It is also a `ValueError`.
- **`ConfigurationError`**: a constructor argument, a setting, an API key or a package is wrong or missing. Only the operator can fix it.
- **`UpstreamError`**: a service the library calls failed or didn't answer. `service` names it (`"llm"` or `"reranker"`), `retryable` says whether a retry may help, and `retry_after` gives the seconds to wait when the service said so.

Errors in the library's own code, such as `KeyError` or `TypeError`, propagate as they are. Messages never contain a question, chunk text, quote, span text, chunk id or metadata value; they refer to items by position, such as `chunks[3]`.

```python
from verbatim_core import ConfigurationError, InvalidInputError, UpstreamError

try:
    answer = transformer.transform(question, chunks)
except InvalidInputError as e:
    print("Fix the input:", e.param, e.code)
except UpstreamError as e:
    if e.retryable:
        ...  # retry, after e.retry_after seconds if set
    else:
        raise
```

| Code | Class | When |
|---|---|---|
| `invalid_value` | `InvalidInputError` | An argument of the call has a wrong value, such as two chunks with the same id, a config of the wrong type, structured template mode in a `Transformer` call, or a context item of the wrong shape |
| `missing_field` | `InvalidInputError` | A context item has neither `content` nor `text` |
| `invalid_value` | `ConfigurationError` | A constructor argument or setting has a wrong value, such as an unknown extraction mode, a citation format with unknown variables, or old arguments next to a `transformer` |
| `missing_dependency` | `ConfigurationError` | A package a model needs, such as torch, transformers or sentence-transformers, isn't installed |
| `model_not_loaded` | `ConfigurationError` | A model can't be downloaded or read |
| `missing_embedder` | `ConfigurationError` | Question-specific templates need an embedder and have none |
| `missing_llm_client` | `ConfigurationError` | Structured template mode has no LLM client: `VerbatimRAG(index, transformer=..., llm_client=...)` |
| `authentication_failed` | `ConfigurationError` | The LLM service or the reranker rejects the API key |
| `permission_denied` | `ConfigurationError` | The API key may not use the model or the endpoint |
| `model_not_found` | `ConfigurationError` | The service doesn't know the model |
| `quota_exceeded` | `ConfigurationError` | The LLM account's quota is used up |
| `timeout` | `UpstreamError` | The service didn't answer in time. Retryable |
| `connection_failed` | `UpstreamError` | The service couldn't be reached. Retryable |
| `rate_limited` | `UpstreamError` | The service limits the request rate. Retryable; from the LLM service, with `retry_after` when it sends one |
| `server_error` | `UpstreamError` | The service failed with a 5xx status. Retryable |
| `request_rejected` | `UpstreamError` | The service rejected the request with another 4xx status |
| `invalid_reply` | `UpstreamError` | The reply can't be read, such as an extraction reply that isn't JSON. Retryable |
| `empty_reply` | `UpstreamError` | The LLM sent an empty reply or one without text, which a content filter may have removed |
| `client_failed` | `UpstreamError` | An LLM client written before 0.3.0, which replaces a method of `LLMClient` such as `complete` or `extract_spans`, raised an exception of its own. The exception is the `__cause__` |

An error raised without a code of its own has its class's: `verbatim_error`, `invalid_input`, `configuration_error` or `upstream_error`.

The dict events of the deprecated `StreamingRAG`, which the API server's streaming route also sends, end a failed stream with `{"type": "error", "code": ..., "error": ..., "done": true}`. Its `code` is the error's code, or `internal_error` for an error that isn't a `VerbatimError`.

## Result warnings

`Extraction`, `Answer` and `QueryResponse` list in `warnings` how a result is degraded but still correct. Each `ResultWarning` has a `code` and a `message`:

| Code | Emitted by | Meaning |
|---|---|---|
| `quote_not_found` | `LLMSpanExtractor` | A quote of the LLM isn't in its chunk, so it was dropped |
| `span_text_not_found` | An extractor that implements only `extract_spans` | A span text isn't in its chunk, so it was dropped |
| `chunk_truncated` | `ModelSpanExtractor`, `SemanticHighlightExtractor` | The model examined only part of a chunk, which is longer than its window |
| `template_fallback` | Contextual templates | The LLM couldn't write a template, so the answer uses the fallback template |
| `reranker_fallback` | `VerbatimRAG` | The reranker's service failed, so the results keep the order of the retrieval |

A fallback happens only when a service fails with an `UpstreamError`. A `ConfigurationError`, an `InvalidInputError` or a bug raises instead.

## Logs

Each module logs through `logging.getLogger(__name__)`, under `verbatim_core` and `verbatim_rag`. The library's modules never configure logging: no handlers and no levels. Only the training code in `verbatim_core.extractor_models`, an application of its own, does when you run it. The levels:

- **DEBUG** for the steps of a call, with counts and positions
- **INFO** for what an operator wants in a healthy log, such as a model loaded or a retry
- **WARNING** for a degraded result
- **ERROR** where no caller can receive the exception, such as a failed stream of `StreamingRAG` or of the API server's streaming route, logged on `verbatim_rag.streaming` with its traceback

Customer text, such as questions, chunks, quotes and generated templates, is logged only on the `verbatim_core.content` logger, and only at DEBUG. Leave that logger at INFO to keep customer text out of your logs, whatever level the other loggers have:

```python
import logging

logging.getLogger("verbatim_core").setLevel(logging.DEBUG)
logging.getLogger("verbatim_core.content").setLevel(logging.INFO)
```

Key events carry `event` among the record's attributes, for structured logging:

| `event` | Level | Attributes | When |
|---|---|---|---|
| `model_loaded` | INFO | `model`, `device`, `duration_ms` | `from_pretrained` loaded a model |
| `fallback_used` | WARNING | `warning_code` | A result is degraded but still correct: one record per code of its result warnings, such as `quote_not_found`, `span_text_not_found`, `chunk_truncated`, `template_fallback` or `reranker_fallback`, with their number in the message. Random templates that the LLM couldn't write log it with `template_fallback` too |
| `item_failed` | WARNING | `error_code` | The deprecated `LLMSpanExtractor.extract_spans` gave a chunk no spans, because its LLM call failed |
