from __future__ import annotations

import argparse
import contextlib
import json
import sys
from typing import TYPE_CHECKING, Any

from .transform import VerbatimTransform

if TYPE_CHECKING:
    from collections.abc import Iterable


def _iter_records(fp) -> Iterable[dict[str, Any]]:
    """Yield JSON objects from a file path or stdin. Supports JSONL or JSON array."""
    try:
        data = fp.read()
    except Exception:
        return
    data = (data or "").strip()
    if not data:
        return
    # Try JSONL first
    if "\n" in data and not data.startswith("["):
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            with contextlib.suppress(Exception):
                yield json.loads(line)
    else:
        try:
            obj = json.loads(data)
        except Exception:
            return
        if isinstance(obj, list):
            yield from obj
        else:
            yield obj


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Verbatim enhance JSON(L) records")
    p.add_argument("--input", "-i", help="Input file (JSON or JSONL). Defaults to stdin.")
    p.add_argument("--output", "-o", help="Output file (JSONL). Defaults to stdout.")
    p.add_argument("--max-spans", type=int, default=5, help="Max display spans")
    args = p.parse_args(argv)

    fin = open(args.input, encoding="utf-8") if args.input else sys.stdin
    fout = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout

    vt = VerbatimTransform(max_display_spans=args.max_spans)

    for rec in _iter_records(fin) or []:
        question = rec.get("question") or ""
        context = rec.get("context") or rec.get("sources") or []
        answer = rec.get("answer")
        resp = vt.transform(question=question, context=context, answer=answer)
        fout.write(json.dumps(resp.model_dump()) + "\n")

    if fin is not sys.stdin:
        fin.close()
    if fout is not sys.stdout:
        fout.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
