"""SearchResult.to_chunk id handling: vector stores may key rows by non-string ids."""

import pytest
from verbatim_core import InvalidInputError

verbatim_rag = pytest.importorskip("verbatim_rag")

from verbatim_rag.vector_stores.base import SearchResult  # noqa: E402


def _row(id):
    return SearchResult(id=id, score=1.0, metadata={"page": 1}, text="hello")


def test_an_integer_row_id_reaches_the_chunk_as_a_string():
    chunk = _row(5).to_chunk()

    assert chunk.id == "5"
    assert chunk.text == "hello"
    assert chunk.metadata == {"page": 1}


def test_a_none_row_id_is_rejected_as_invalid_input():
    with pytest.raises(InvalidInputError):
        _row(None).to_chunk()


def test_an_empty_row_id_is_rejected_as_invalid_input():
    with pytest.raises(InvalidInputError):
        _row("").to_chunk()
