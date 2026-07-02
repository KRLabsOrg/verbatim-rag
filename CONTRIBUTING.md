# Contributing to Verbatim RAG

Thanks for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/KRLabsOrg/verbatim-rag.git
cd verbatim-rag
pip install -e packages/core/
pip install -e ".[dev]"
```

## Project Structure

```
packages/core/verbatim_core/   # Lean extraction package (verbatim-core on PyPI)
verbatim_rag/                  # Full RAG pipeline (verbatim-rag on PyPI)
api/                           # FastAPI server
frontend/                      # React UI
tests/                         # Tests (run against verbatim-core only)
docs/                          # MkDocs documentation
```

## Running Tests

```bash
pytest tests/ -v
```

Tests only depend on `verbatim-core` (openai + pydantic). All LLM calls are mocked.

## Linting

```bash
ruff check packages/core/verbatim_core/ verbatim_rag/ api/ tests/
ruff format packages/core/verbatim_core/ verbatim_rag/ api/ tests/
```

## Making Changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Run tests and linting
4. Open a pull request against `main`

CI runs lint, tests (Python 3.10-3.12), and pip-audit on all PRs.

## Releasing

Both packages share the same version number. To release:

1. Bump version in `packages/core/pyproject.toml` and `pyproject.toml`
2. Update `CHANGELOG.md`
3. Commit, tag, and push: `git tag v0.x.y && git push --tags`
4. Create a GitHub release from the tag
5. Publish to PyPI (extractor first, then rag):
   ```bash
   cd packages/core && python -m build && twine upload dist/*
   cd ../.. && python -m build && twine upload dist/*
   ```

## Contribution rights

By submitting a pull request you confirm that you wrote the contribution (or
have the right to submit it) and that it may be distributed under this
repository's MIT license (this is what the checkbox in the PR template
attests). Optionally you can also sign off commits with `git commit -s`
([DCO](https://developercertificate.org/)); appreciated, not required.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

