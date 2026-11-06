# A span points at one occurrence of its text

When a quote occurs more than once in a chunk, the extractor locates one occurrence: the first exact one, else the best fuzzy match, the earliest on a tie. Identical spans collapse into one, and the response builder keeps the first of any overlapping highlights. Until 0.3.0 the response builder searched for each span's text afterwards and highlighted every occurrence. That made the number of citations depend on how often a phrase repeats, not on what the extractor chose, and it can't be expressed with offsets.

## Consequences

Repeated text now gets one highlight and one citation instead of one per occurrence. The 0.3.0 changelog says so under "Changed".
