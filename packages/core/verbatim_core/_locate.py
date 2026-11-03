"""Locating a text in its chunk, for extractors whose texts come without offsets."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List

from rapidfuzz.fuzz import partial_ratio_alignment

from .types import Chunk, ExtractedSpan

# The score of a span located from its text alone: "extracted without a score".
UNSCORED = 1.0

_FUZZY_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def locate(text: str, chunk: Chunk, *, fuzzy_threshold: float | None = 0.8) -> ExtractedSpan | None:
    """Locate a text in its chunk, ignoring its leading and trailing whitespace.

    The span is the first exact occurrence of the text, else the best fuzzy match that scores at
    least ``fuzzy_threshold``, the earliest on a tie. Its text is the chunk's text at its offsets,
    never the text that was located.

    :param fuzzy_threshold: the minimum fuzzy score from 0 to 1, or None to match exactly only
    :return: the span, or None if the text is empty or can't be located
    """
    text = text.strip()
    if not text:
        return None

    start = chunk.text.find(text)
    if start >= 0:
        end = start + len(text)
    elif fuzzy_threshold is None:
        return None
    else:
        match = find_fuzzy_match(text, chunk.text)
        if match is None or match[0] < fuzzy_threshold:
            return None
        _, start, end = match

    return ExtractedSpan(
        chunk_id=chunk.id, start=start, end=end, text=chunk.text[start:end], score=UNSCORED
    )


@dataclass(frozen=True)
class _NormalizedTokenText:
    text: str
    original_token_offsets: List[tuple[int, int]]
    normalized_token_offsets: List[tuple[int, int]]


def find_fuzzy_match(quote: str, text: str) -> tuple[float, int, int] | None:
    """Find the best fuzzy match of a quote in a text, the earliest on a tie.

    RapidFuzz alignment offsets are character-level and can land in the middle of words.
    Matching normalized token text keeps OCR/punctuation tolerance while mapping the match back
    to the original token offsets.

    :return: the score from 0 to 1 and the offsets of the match in ``text``, or None if
        nothing aligns
    """
    normalized_quote = _normalize(quote)
    normalized_text = _normalize(text)

    if not normalized_quote.text or not normalized_text.text:
        return None

    result = partial_ratio_alignment(normalized_quote.text, normalized_text.text)
    offsets = _original_offsets(normalized_text, result.dest_start, result.dest_end)
    if offsets is None:
        return None
    return result.score / 100.0, *offsets


def _normalize(text: str) -> _NormalizedTokenText:
    parts: List[str] = []
    original_token_offsets: List[tuple[int, int]] = []
    normalized_token_offsets: List[tuple[int, int]] = []
    normalized_pos = 0

    for match in _FUZZY_TOKEN_RE.finditer(text):
        token = unicodedata.normalize("NFKC", match.group(0)).casefold()
        if not token:
            continue

        if parts:
            parts.append(" ")
            normalized_pos += 1

        token_start = normalized_pos
        parts.append(token)
        normalized_pos += len(token)

        original_token_offsets.append((match.start(), match.end()))
        normalized_token_offsets.append((token_start, normalized_pos))

    return _NormalizedTokenText(
        text="".join(parts),
        original_token_offsets=original_token_offsets,
        normalized_token_offsets=normalized_token_offsets,
    )


def _original_offsets(
    normalized_text: _NormalizedTokenText,
    normalized_start: int,
    normalized_end: int,
) -> tuple[int, int] | None:
    first_token_idx = None
    last_token_idx = None

    for idx, (token_start, token_end) in enumerate(normalized_text.normalized_token_offsets):
        if token_end <= normalized_start:
            continue
        if token_start >= normalized_end:
            break

        if first_token_idx is None:
            first_token_idx = idx
        last_token_idx = idx

    if first_token_idx is None or last_token_idx is None:
        return None

    original_start = normalized_text.original_token_offsets[first_token_idx][0]
    original_end = normalized_text.original_token_offsets[last_token_idx][1]
    return original_start, original_end
