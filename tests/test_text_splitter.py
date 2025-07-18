import pytest
from verbatim_rag.text_splitter import ChonkieSplitter
from verbatim_rag.document import Document

# Fixture to provide a sample document with metadata
@pytest.fixture
def sample_documents():
    text = (
        "Large Language Models (LLMs) are revolutionizing natural language processing. "
        "These models have billions of parameters and are trained on massive corpora. "
        "They can perform tasks such as summarization, translation, and question answering."
    )
    metadata = {"source": "sample.txt", "section": "Intro", "title": "LLMs"}
    return [Document(content=text, doc_id="doc1", metadata=metadata)]


# Test that ChonkieSplitter returns a list of non-empty strings
def test_chonkie_split_returns_chunks(sample_documents):
    splitter = ChonkieSplitter(max_tokens=20)
    texts = [doc.content for doc in sample_documents]  # Extract text from Document
    chunks = splitter.split(texts)

    assert isinstance(chunks, list)                 # Output is a list
    assert len(chunks) > 0                          # List is non-empty
    assert all(isinstance(c, str) for c in chunks)  # Each item is a string
    assert all(c.strip() != "" for c in chunks)     # Each string is non-empty


# Test that splitting an empty input returns an empty list
def test_chonkie_split_empty_input():
    splitter = ChonkieSplitter()
    chunks = splitter.split([])

    assert chunks == []


# Test that a very short text returns one chunk identical to the input
def test_chonkie_split_short_text():
    short_text = ["Hello world!"]
    splitter = ChonkieSplitter(max_tokens=512)
    chunks = splitter.split(short_text)

    assert len(chunks) == 1                  # One chunk expected
    assert chunks[0] == "Hello world!"       # Chunk matches input
