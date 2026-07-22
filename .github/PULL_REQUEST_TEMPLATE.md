## Summary

<!-- What changes, why, and what user-visible outcome should reviewers verify? -->

## Related issue

<!-- Link the scoped issue, for example: Fixes #123. -->

## Type of change

- [ ] Bug fix
- [ ] Feature
- [ ] Documentation
- [ ] Tests
- [ ] Refactor or maintenance

## Provenance and compatibility

<!-- If extraction, validation, templates, citations, schemas, or defaults change,
describe the guarantee and compatibility impact. Otherwise write "Not applicable". -->

## Verification

<!-- List the exact commands and focused cases you ran. Delete commands that are not relevant. -->

- [ ] `pytest tests/ -v`
- [ ] `ruff format --check packages/core/verbatim_core/ verbatim_rag/ api/ tests/`
- [ ] `ruff check packages/core/verbatim_core/ verbatim_rag/ api/ tests/`
- [ ] Frontend changes: `cd frontend && npm ci && npm run build`
- [ ] Packaging changes: `python -m build packages/core && python -m build .`
- [ ] Other:

## Checklist

- [ ] I kept the PR focused on one issue or decision.
- [ ] I added or updated tests for behavior changes.
- [ ] I updated public docs and the changelog for user-facing changes.
- [ ] I did not include secrets, API keys, model credentials, or private source text.

## Rights & sign-off (required)

- [ ] I certify that I have the right to submit this code and that it may be
      distributed under the repository's MIT license
      (see [CONTRIBUTING](../CONTRIBUTING.md#contribution-rights)).
