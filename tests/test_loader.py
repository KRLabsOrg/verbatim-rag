import csv
import os
import json
import pandas as pd
import pytest

from verbatim_rag.loader import DocumentLoader
from verbatim_rag.document import Document

def test_load_text(tmp_path):
    # create a simple text file
    p = tmp_path / "foo.txt"
    text = "Hello\nWorld"
    p.write_text(text, encoding="utf-8")

    docs = DocumentLoader.load_text(str(p))
    assert isinstance(docs, Document)
    assert docs.content == text
    assert docs.metadata["source"] == str(p)
    assert docs.metadata["type"] == "text"
    # id comes from Document defaulting (if you added doc_id) — skip here

def test_load_csv_and_load_file(tmp_path):
    # prepare CSV
    p = tmp_path / "data.csv"
    header = ["col1", "col2"]
    rows = [
        {"col1": "A", "col2": "1"},
        {"col1": "B", "col2": "2"},
    ]
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    # direct load_csv
    docs = DocumentLoader.load_csv(str(p))
    assert isinstance(docs, list)
    assert len(docs) == 2
    for idx, doc in enumerate(docs):
        assert f"col1: {rows[idx]['col1']}" in doc.content
        assert f"col2: {rows[idx]['col2']}" in doc.content
        assert doc.metadata["source"] == str(p)
        assert doc.metadata["type"] == "csv"
        assert doc.metadata["row"] == idx + 1 # (row-info starts at 1 instead of 0)

    # load_file should dispatch to load_csv for .csv
    docs2 = DocumentLoader.load_file(str(p))
    assert isinstance(docs2, list)
    assert len(docs2) == len(docs)
    # and contents and metadata should match
    for d_orig, d_disp in zip(docs, docs2):
        assert d_disp.content == d_orig.content
        assert d_disp.metadata == d_orig.metadata

def test_load_dataframe():
    df = pd.DataFrame([
        {"a": 10, "b": "x"},
        {"a": 20, "b": "y"},
    ])
    docs = DocumentLoader.load_dataframe(df)
    assert len(docs) == 2
    for idx, doc in enumerate(docs):
        assert f"a: {df.iloc[idx]['a']}" in doc.content
        assert f"b: {df.iloc[idx]['b']}" in doc.content
        assert doc.metadata["source"] == "dataframe"
        assert doc.metadata["type"] == "dataframe"
        assert doc.metadata["row"] == idx

    # selecting only one column
    docs_sel = DocumentLoader.load_dataframe(df, content_columns=["b"])
    assert "a:" not in docs_sel[0].content
    assert "b: x" in docs_sel[0].content

def test_load_file_txt_and_unknown(tmp_path):
    # a .md file should go through load_text
    md = tmp_path / "readme.md"
    md.write_text("MD content", encoding="utf-8")
    docs = DocumentLoader.load_file(str(md))
    assert len(docs) == 1
    assert docs[0].content == "MD content"

    # an unknown extension defaults to text loader
    misc = tmp_path / "foo.unknown"
    misc.write_text("??", encoding="utf-8")
    docs2 = DocumentLoader.load_file(str(misc))
    assert len(docs2) == 1
    assert docs2[0].content == "??"

def test_load_directory(tmp_path):
    # create structure:
    # tmp/
    #   a.txt
    #   sub/
    #     b.csv
    txt = tmp_path / "a.txt"
    txt.write_text("AAA", encoding="utf-8")

    sub = tmp_path / "sub"
    sub.mkdir()
    csvf = sub / "b.csv"
    # write a one-row CSV
    with open(csvf, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["c"])
        writer.writeheader()
        writer.writerow({"c": "C1"})

    all_docs = DocumentLoader.load_directory(str(tmp_path), recursive=True)
    # should find both a.txt and b.csv → total 1 + 1 doc
    assert any(isinstance(d, Document) for d in all_docs)
    contents = [d.content for d in all_docs]
    assert "AAA" in contents
    assert "c: C1" in " ".join(contents)

    # non-recursive should only load a.txt
    docs_nr = DocumentLoader.load_directory(str(tmp_path), recursive=False)
    assert any("AAA" == d.content for d in docs_nr)
    assert all("C1" not in d.content for d in docs_nr)
