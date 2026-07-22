# Verbatim RAG roadmap

Verbatim RAG is evidence-driven. This roadmap separates correctness work on
the existing extraction contract from product directions that still need real
workloads and contributor feedback. It intentionally has no delivery dates.

## Vision

Verbatim is a provenance-first answer layer: given a question and source
context, it returns source excerpts with structured citations while minimizing
the amount of freely generated factual text.

The enforceable guarantee is about evidence provenance, not truth. Verbatim can
show that a cited excerpt came from supplied source text; it cannot guarantee
that the source is correct, retrieval is complete, or an extracted passage is
the best answer. Contextual template mode can also generate presentation text
around the cited excerpts. Static template mode keeps that framing fixed and
transparent.

## Now: [v0.3 — Trustworthy extraction](https://github.com/KRLabsOrg/verbatim-rag/milestone/1)

This milestone tightens behavior that matters whichever product surface proves
useful:

- [surface model-format detection failures](https://github.com/KRLabsOrg/verbatim-rag/issues/33)
  instead of silently selecting an incompatible extractor;
- [make fuzzy validation asymmetric](https://github.com/KRLabsOrg/verbatim-rag/issues/34)
  so permitted source-side markup can lower a score but changed content cannot
  pass it;
- [window the legacy sentence-model path](https://github.com/KRLabsOrg/verbatim-rag/issues/37)
  while keeping the published v2 model's trusted remote windowing contract.

The milestone is about extraction correctness and provenance. It does not
commit a framework integration, ingestion product, or supported self-hosting
surface.

## Next: [Validation — Run it on your data](https://github.com/KRLabsOrg/verbatim-rag/milestone/2)

[The evaluation harness](https://github.com/KRLabsOrg/verbatim-rag/issues/29)
will make the published span metric reproducible on another corpus, including
negative examples and per-example output. The aim is to learn where extraction
is useful and where it fails before widening the API or training story.

## Exploring

These are open design questions, not promised release features:

- a reproducible [local Docker Compose demo](https://github.com/KRLabsOrg/verbatim-rag/issues/27);
- the right [adapter contract for an existing RAG stack](https://github.com/KRLabsOrg/verbatim-rag/issues/28);
- a bounded [document-ingestion and lifecycle API](https://github.com/KRLabsOrg/verbatim-rag/issues/31);
- supported domain adaptation for the current token-level extractors;
- stable document identities and content hashes in citation records;
- whether the main product pull is a reusable transform, curated hosted
  collections, or a supported self-hosted pipeline.

Exploring items move into a milestone only after a bounded contract and evidence
from an actual workflow. Negative results and narrower designs are useful
outcomes.

## How this roadmap changes

- Correctness work can move directly into the current release milestone.
- Validation work must state its dataset, metric, split, and failure cases.
- Product-surface work needs a scoped design and a user workflow before it is
  treated as implementation-ready.
- User-facing changes land with tests, documentation, and a changelog entry.
- Ideas and early design feedback belong in
  [Discussions](https://github.com/KRLabsOrg/verbatim-rag/discussions); scoped
  implementation belongs in issues.
