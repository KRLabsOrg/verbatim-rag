import pytest
import json
from unittest.mock import patch, mock_open
from verbatim_rag.docling_loader import DoclingLoader
from verbatim_rag.document import Document


# Mock the subprocess.run call in run_docling
def test_run_docling_calls_subprocess():
    with patch("subprocess.run") as mock_run:
        DoclingLoader.run_docling("input", "output")
        mock_run.assert_called_once_with(
            ["docling", "input", "--output", "output", "--to", "json"],
            check=True
        )

# Mock os.listdir and open for load_docling_output
def test_load_docling_output_creates_documents():
    fake_json = json.dumps({
        "name": "test_doc",
        "texts": [
            {"text": "First chunk", "label": "section1"},
            {"text": "Second chunk", "label": "section2"},
        ]
    })

    with patch("os.listdir", return_value=["file1.json"]), \
         patch("builtins.open", mock_open(read_data=fake_json)):
        docs = DoclingLoader.load_docling_output("fake_dir")
        assert len(docs) == 2
        assert isinstance(docs[0], Document)
        assert docs[0].content == "First chunk"
        assert docs[0].id == "test_doc-chunk_0"
        assert docs[0].metadata["section"] == "section1"


# Test full DoclingLoader pipeline:
# - Mocks subprocess call to run docling
# - Mocks reading JSON output file
# - Verifies that Documents are created correctly from parsed conten
def test_load_with_docling_full_pipeline():
    fake_json = json.dumps({
        "name": "sample_doc",
        "texts": [
            {"text": "Chunk A", "label": "intro"},
            {"text": "Chunk B", "label": "body"}
        ]
    })

    with patch("subprocess.run") as mock_run, \
         patch("os.listdir", return_value=["sample.json"]), \
         patch("builtins.open", mock_open(read_data=fake_json)):

        docs = DoclingLoader.load_with_docling("input_path", "output_path")

        # subprocess.run called correctly
        mock_run.assert_called_once_with(
            ["docling", "input_path", "--output", "output_path", "--to", "json"],
            check=True
        )

        # validate documents
        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)
        assert docs[0].content == "Chunk A"
        assert docs[0].metadata["section"] == "intro"
        assert docs[0].id == "sample_doc-chunk_0"