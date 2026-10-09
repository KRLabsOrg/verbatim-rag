"""Tests for LocalMilvusStore against Milvus Lite."""

import pytest

pytest.importorskip("pymilvus")
pytest.importorskip("milvus_lite")


@pytest.fixture
def store(tmp_path):
    from verbatim_rag.vector_stores.milvus_local import LocalMilvusStore

    return LocalMilvusStore(
        db_path=str(tmp_path / "milvus.db"),
        collection_name="test_chunks",
        enable_dense=False,
        enable_sparse=True,
    )


def test_documents_collection_has_a_vector_field(store):
    description = store.client.describe_collection(store.documents_collection_name)
    field_names = [field["name"] for field in description["fields"]]
    assert "dummy_vector" in field_names


def test_document_round_trip(store):
    store.add_documents(
        [
            {
                "id": "doc-1",
                "title": "A title",
                "source": "https://example.org/paper.pdf",
                "doc_type": "pdf",
                "raw_content": "Some document text.",
                "metadata": {"authors": ["A. Author"]},
            }
        ]
    )

    document = store.get_document("doc-1")
    assert document["title"] == "A title"
    assert document["content_type"] == "pdf"
    assert [row["id"] for row in store.get_all_documents()] == ["doc-1"]

    store.delete_document("doc-1")
    assert store.get_document("doc-1") is None
