# `query_stream` yields typed events and ends with the `query` response

`VerbatimRAG.query_stream` yields typed events: `documents` with the retrieved chunks, `spans` with the extracted spans, and a final event that carries the same `QueryResponse` that `query` returns for the same input, so tests can compare the two. Failures raise typed exceptions instead of arriving as an `error` event, and there is no `progress` event.

## Consequences

The deprecated `StreamingRAG` adapter rebuilds today's dict stream (`documents`, `progress`, `highlights`, `answer`, `error`) from the typed events, the exceptions and its own timing. The API server's streaming route keeps that dict stream until a new route replaces it.
