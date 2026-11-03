# A failed LLM call fails the whole extraction

When an LLM call fails during span extraction, `extract` raises a typed error instead of returning no spans for the chunks that call covered. Until 0.3.0 the extractor logged the error and returned empty results, so a caller couldn't tell "nothing relevant in this chunk" from "this chunk was never examined", and an answer built from the remaining chunks looked complete. Retrying is the job of the LLM-call layer, not the extractor.

## Consequences

The fallback from one batch call to one call per chunk survives only in the deprecated `extract_spans` path.
