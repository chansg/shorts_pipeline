"""Pure transcript-grid editing operations for the gameplay transcript gate.

These power the editor's bulk-edit buttons (multi-row speaker assign, find/replace,
merge, split). They operate on the editable grid's raw rows — `[text, speaker,
start, end]` — and return new rows, preserving the `(text, start, end, speaker)`
tuple contract that `karaoke_captions.build_ass` consumes. No Gradio import, so
they're trivially unit-testable.

Row indices in the public API are 1-based (what the user sees in the grid).
"""
from __future__ import annotations

import re

COLS = 4   # text, speaker, start, end


def _to_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _norm(rows) -> list[list]:
    """Coerce incoming rows (possibly ragged / numpy / tuples) to [text,spk,start,end]."""
    out = []
    for row in rows or []:
        row = list(row) + ["", "", 0.0, 0.0]
        out.append([str(row[0] or ""), str(row[1] or "").strip(),
                    _to_float(row[2]), _to_float(row[3])])
    return out


def parse_row_span(span: str, n_rows: int) -> list[int]:
    """'1-3,7 9' -> sorted unique 0-based indices in range. Tolerant of spaces,
    commas, and 'a-b' ranges; silently drops out-of-range / malformed parts."""
    if not span:
        return []
    idx: set[int] = set()
    for part in re.split(r"[,\s]+", str(span).strip()):
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            try:
                lo, hi = int(a), int(b)
            except ValueError:
                continue
            for i in range(min(lo, hi), max(lo, hi) + 1):
                if 1 <= i <= n_rows:
                    idx.add(i - 1)
        else:
            try:
                i = int(part)
            except ValueError:
                continue
            if 1 <= i <= n_rows:
                idx.add(i - 1)
    return sorted(idx)


def assign_speaker(rows, span: str, speaker: str) -> list[list]:
    """Set the speaker on every row in `span` (1-based) at once — the fix for a
    stretch the diariser mislabelled."""
    rows = _norm(rows)
    speaker = (speaker or "").strip()
    for i in parse_row_span(span, len(rows)):
        rows[i][1] = speaker
    return rows


def find_replace(rows, find: str, repl: str, case_sensitive: bool = False,
                 whole_word: bool = False) -> tuple[list[list], int]:
    """Replace `find` with `repl` in the text column across all rows. Returns
    (rows, n_replacements). For a repeated ASR mishearing — fix once."""
    rows = _norm(rows)
    if not find:
        return rows, 0
    flags = 0 if case_sensitive else re.IGNORECASE
    pat = re.escape(find)
    if whole_word:
        pat = rf"\b{pat}\b"
    rx = re.compile(pat, flags)
    n = 0
    for row in rows:
        row[0], k = rx.subn(repl, row[0])
        n += k
    return rows, n


def merge_rows(rows, span: str) -> list[list]:
    """Merge the rows in `span` (1-based) into one: text joined, start=min,
    end=max, speaker = first non-empty in the span. Fixes a mis-segmented phrase."""
    rows = _norm(rows)
    idx = parse_row_span(span, len(rows))
    if len(idx) < 2:
        return rows
    merged_text = " ".join(rows[i][0].strip() for i in idx if rows[i][0].strip())
    speaker = next((rows[i][1] for i in idx if rows[i][1]), "")
    start = min(rows[i][2] for i in idx)
    end = max(rows[i][3] for i in idx)
    keep = [r for j, r in enumerate(rows) if j not in set(idx)]
    keep.insert(idx[0], [merged_text, speaker, start, end])
    keep.sort(key=lambda r: r[2])
    return keep


def split_row(rows, row_1based: int, first_n_words: int | None = None) -> list[list]:
    """Split one row into two at a word boundary, prorating the time span by word
    count. Default splits at the midpoint. A single-word row is returned unchanged."""
    rows = _norm(rows)
    i = int(row_1based) - 1
    if not (0 <= i < len(rows)):
        return rows
    text, speaker, start, end = rows[i]
    words = text.split()
    if len(words) < 2:
        return rows
    k = len(words) // 2 if first_n_words is None else int(first_n_words)
    k = max(1, min(k, len(words) - 1))
    mid = round(start + (end - start) * (k / len(words)), 3)
    first = [" ".join(words[:k]), speaker, start, mid]
    second = [" ".join(words[k:]), speaker, mid, end]
    return rows[:i] + [first, second] + rows[i + 1:]
