# Random template mode chooses by question, not at random

Random template mode picks its template from a hash of the question that is the same in every process (so not Python's `hash()`), optionally combined with a `seed` from `TransformConfig`. The stateless core promises the same result for the same input, and a template drawn with `random.choice` on every call broke that. The mode's purpose, variety across questions, doesn't need randomness per call. The same question now always gets the same template.
