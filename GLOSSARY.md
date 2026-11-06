# Verbatim RAG

Verbatim RAG answers questions with text copied word for word from source documents, so every part of an answer points at the exact place it came from.

## Language

### Source text

**Chunk**:
A piece of source text with an id, the unit spans are extracted from. Its id stays the same for as long as the chunk stays indexed.
_Avoid_: document, passage, context, search result

**Document chunk**:
A chunk as ingestion cut it from its document, before it is indexed.
_Avoid_: chunk (when meaning the ingestion record)

### Extraction

**Quote**:
The text a language model returns as relevant, before it is located in a chunk. It can differ from the chunk's text.
_Avoid_: span, excerpt

**Span**:
A part of a chunk, given by its start and end offsets. Its text is always exactly the chunk's text between those offsets.
_Avoid_: quote, excerpt, fact

**Score**:
The extractor's probability that a span is relevant to the question. It says nothing about how closely a quote matched the chunk.
_Avoid_: confidence, match quality

### Answers

**Highlight**:
A span as a reader sees it in the source text of an answer, after overlapping spans are resolved.
_Avoid_: span (when meaning what the reader sees)
