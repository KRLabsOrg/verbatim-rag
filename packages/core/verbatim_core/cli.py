"""Command-line interface for enhancing JSON(L) records with verbatim transforms."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from .transform import VerbatimTransform

if TYPE_CHECKING:
    from collections.abc import Iterable


def _iter_records(fp: IO[str]) -> Iterable[dict[str, Any]]:
    """Yield JSON objects from a file path or stdin. Supports JSONL or JSON array."""
    try:
        data = fp.read()
    except Exception:  # noqa: BLE001  # unreadable input yields no records
        return
    data = (data or "").strip()
    if not data:
        return
    # Try JSONL first
    if "\n" in data and not data.startswith("["):
        for line in data.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            with contextlib.suppress(Exception):
                yield json.loads(stripped)
    else:
        try:
            obj = json.loads(data)
        except Exception:  # noqa: BLE001  # invalid JSON yields no records
            return
        if isinstance(obj, list):
            yield from obj
        else:
            yield obj


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    p = argparse.ArgumentParser(description="Verbatim enhance JSON(L) records")
    p.add_argument("--input", "-i", help="Input file (JSON or JSONL). Defaults to stdin.")
    p.add_argument("--output", "-o", help="Output file (JSONL). Defaults to stdout.")
    p.add_argument("--max-spans", type=int, default=5, help="Max display spans")
    args = p.parse_args(argv)

    vt = VerbatimTransform(max_display_spans=args.max_spans)

    with contextlib.ExitStack() as stack:
        fin = stack.enter_context(Path(args.input).open(encoding="utf-8")) if args.input else sys.stdin
        fout = stack.enter_context(Path(args.output).open("w", encoding="utf-8")) if args.output else sys.stdout

        for rec in _iter_records(fin) or []:
            question = rec.get("question") or ""
            context = rec.get("context") or rec.get("sources") or []
            answer = rec.get("answer")
            resp = vt.transform(question=question, context=context, answer=answer)
            fout.write(json.dumps(resp.model_dump()) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
