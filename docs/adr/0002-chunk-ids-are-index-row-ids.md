# Chunk ids are the index's row ids

`SearchResult.to_chunk()` gives a chunk the id of its vector row (`ProcessedChunk.id`, a random UUID), so spans and citations can point at a chunk by id. The id is stable while the document stays indexed, and changes when the document is re-indexed.

## Considered options

- A deterministic `"{document_id}:{chunk_number}"`, which survives re-indexing. Rejected: after a chunking change the same id silently points at different text, and a citation that resolves to the wrong text is worse than one that no longer resolves.
