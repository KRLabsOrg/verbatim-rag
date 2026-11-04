"""Tests for the error output of the verbatim_core command-line interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

from verbatim_core.cli import main

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_unreadable_input_reports_error_on_stderr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An input file that cannot be read is reported on stderr and yields no records."""
    input_path = tmp_path / "input.jsonl"
    input_path.write_bytes(b"\xff\xfe\x00 not valid utf-8")
    output_path = tmp_path / "output.jsonl"

    exit_code = main(["--input", str(input_path), "--output", str(output_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err.startswith("Error reading input: ")
    assert "utf-8" in captured.err
    assert output_path.read_text(encoding="utf-8") == ""


def test_malformed_jsonl_line_is_skipped_with_stderr_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each malformed JSONL line is reported on stderr and skipped."""
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"question": \nnot json at all\n', encoding="utf-8")
    output_path = tmp_path / "output.jsonl"

    exit_code = main(["--input", str(input_path), "--output", str(output_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    error_lines = captured.err.splitlines()
    assert len(error_lines) == 2  # noqa: PLR2004
    assert all(line.startswith("Skipping malformed JSONL line: ") for line in error_lines)
    assert output_path.read_text(encoding="utf-8") == ""
