import pytest
from verbatim_rag.document import Document

def test_document_defaults_id_and_content():
    doc = Document(content="hello world", doc_id="doc123")
    # .content and .id
    assert doc.content == "hello world"
    assert doc.id == "doc123"

    # metadata must have the id entry set
    assert isinstance(doc.metadata, dict)
    assert doc.metadata["id"] == "doc123"

def test_document_override_metadata_and_id():
    # if you pass in metadata with an "id" key, it should keep that one
    meta = {"id": "custom_id", "source": "unit_test"}
    doc = Document(content="foo", doc_id=None, metadata=meta)

    # id attribute should be the explicit doc_id, but metadata['id']
    # keeps the user-provided one
    assert doc.id == "custom_id"
    assert doc.metadata["id"] == "custom_id"
    assert doc.metadata["source"] == "unit_test"

def test_document_additional_metadata():
    # metadata merges extra keys
    extra = {"foo": 1050, "bar": "baz"}
    doc = Document(content="xyz", doc_id="myid", metadata=extra)

    assert doc.id == "myid"
    # extra keys stay
    assert doc.metadata["foo"] == 1050
    assert doc.metadata["bar"] == "baz"
    # 'id' was missing so got set to doc_id
    assert doc.metadata["id"] == "myid"
