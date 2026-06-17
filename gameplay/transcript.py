"""The transcript data model and the adapter to the caption tuples.

This is the editable gate: the rows the GUI shows ARE the
`(text, start, end, speaker)` list `modules.karaoke_captions.build_ass` consumes,
so a corrected transcript flows straight into captions with no separate export.

Kept free of the heavy WhisperX/torch import so it (and its tests) run with no GPU.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# 4+ of the same character in a row — the signature of a Whisper repetition
# collapse ("Naaaaaa…"). Real words almost never have a run this long.
_CHAR_RUN = re.compile(r"(.)\1{3,}", re.UNICODE)


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: str | None = None


@dataclass
class Transcript:
    """Word-level transcript with optional speaker labels.

    `single_speaker` is True when diarization was skipped or found <=1 voice; in
    that case every word's speaker is None and captions use the style default
    colour (matching the lore look). `diarized` records whether diarization
    actually RAN (so the editor can tell "no token" apart from "ran but collapsed
    to one dominant speaker")."""
    words: list[Word] = field(default_factory=list)
    single_speaker: bool = False
    diarized: bool = False

    # ---- speakers ----

    @property
    def speakers(self) -> list[str]:
        """Distinct speaker labels in order of first appearance (excludes None)."""
        seen: list[str] = []
        for w in self.words:
            if w.speaker and w.speaker not in seen:
                seen.append(w.speaker)
        return seen

    def rename_speaker(self, old: str, new: str) -> None:
        new = (new or "").strip() or None
        for w in self.words:
            if w.speaker == old:
                w.speaker = new

    # ---- caption adapter ----

    def to_tuples(self) -> list[tuple]:
        """Emit the list `build_ass` consumes. 4-tuples when any speaker label is
        present (drives per-speaker colour); plain 3-tuples in the single-speaker
        case (so the renderer uses its default fill — identical to the lore path)."""
        any_speaker = any(w.speaker for w in self.words)
        if any_speaker and not self.single_speaker:
            return [(w.text, w.start, w.end, w.speaker) for w in self.words]
        return [(w.text, w.start, w.end) for w in self.words]

    # ---- editable grid (Gradio Dataframe) ----

    HEADERS = ["text", "speaker", "start", "end"]

    def to_rows(self) -> list[list]:
        return [[w.text, (w.speaker or ""), round(w.start, 2), round(w.end, 2)]
                for w in self.words]

    @staticmethod
    def _normalise_grid(rows) -> list[list]:
        """Coerce whatever the Gradio Dataframe hands us into a plain list-of-lists.

        `type="array"` yields list-of-lists, but be robust to a pandas DataFrame
        (older/other Gradio paths), a dict payload ({"data": [...]}), or a numpy
        array — and NEVER raise (a truthiness check on a DataFrame would). This is
        the seam where edited grid values become the build's transcript, so it must
        not silently drop edits."""
        if rows is None:
            return []
        # dict payload, e.g. {"headers": [...], "data": [[...], ...]}
        if isinstance(rows, dict):
            rows = rows.get("data", [])
        # pandas DataFrame / numpy array -> list of lists (avoid ambiguous truthiness)
        if hasattr(rows, "values") and hasattr(rows, "columns"):   # DataFrame
            rows = rows.values.tolist()
        elif hasattr(rows, "tolist") and not isinstance(rows, list):  # ndarray
            rows = rows.tolist()
        try:
            return [list(r) for r in rows]
        except TypeError:
            return []

    @classmethod
    def from_rows(cls, rows, single_speaker: bool = False) -> "Transcript":
        """Rebuild from edited grid rows. Tolerant of blank/half-filled trailing
        rows the editor may add, and of out-of-order/garbled timing."""
        words: list[Word] = []
        for row in cls._normalise_grid(rows):
            row = list(row) + ["", "", None, None]
            text = str(row[0] or "").strip()
            if not text:
                continue
            speaker = str(row[1] or "").strip() or None
            try:
                start = float(row[2])
                end = float(row[3])
            except (TypeError, ValueError):
                continue
            if end < start:
                start, end = end, start
            words.append(Word(text, start, end, speaker))
        words.sort(key=lambda w: w.start)
        has_speaker = any(w.speaker for w in words)
        return cls(words=words, single_speaker=single_speaker or not has_speaker)

    # ---- persistence (resumable cache) ----

    def to_dict(self) -> dict:
        return {"single_speaker": self.single_speaker, "diarized": self.diarized,
                "words": [vars(w) for w in self.words]}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        words = [Word(w.get("text", ""), float(w["start"]), float(w["end"]),
                      w.get("speaker")) for w in d.get("words", [])]
        return cls(words=words, single_speaker=bool(d.get("single_speaker")),
                   diarized=bool(d.get("diarized")))

    @classmethod
    def load(cls, path: str | Path) -> "Transcript":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def clamp_long_words(words: list[Word], max_dur: float) -> list[Word]:
    """Clamp the displayed END of any word longer than `max_dur` to start+max_dur.

    Noisy gameplay audio makes Whisper hallucinate a single token spanning tens of
    seconds (e.g. "Naaaa…" 3.1s→31.08s, burned as a screen-wide "AAAA…" wall that
    also overlaps every later word). Clamping here — before the editable grid — keeps
    the word (the user can fix/delete it) but stops the wall. `max_dur <= 0` disables."""
    if max_dur and max_dur > 0:
        for w in words:
            if w.end - w.start > max_dur:
                w.end = round(w.start + max_dur, 3)
    return words


def sanitize_runaway_tokens(words: list[Word], max_chars: int) -> list[Word]:
    """Repair or drop repetition-collapse tokens BEFORE the editable grid.

    A noisy/music chunk can collapse into a single "word" of hundreds of repeated
    characters ("Naaaaaa…"). `clamp_long_words` fixes such a token's DURATION but
    leaves its TEXT — so a 300-char wall still reaches the grid/captions. This is
    the text-side guard:

      1. collapse any run of 4+ identical chars to one ("Naaaaaa" -> "Na"), which
         repairs the common shout-collapse into a plausible, editable word;
      2. if the repaired token is STILL longer than `max_chars`, it is genuine
         garbage (not a real word) and the word is dropped entirely.

    Well-formed words pass through unchanged. `max_chars <= 0` disables the drop
    step (runs are still collapsed). Returns a NEW list (input left intact)."""
    out: list[Word] = []
    for w in words:
        repaired = _CHAR_RUN.sub(lambda m: m.group(1), w.text)
        if max_chars and max_chars > 0 and len(repaired) > max_chars:
            continue  # still absurd after repair -> drop (can't reach captions)
        out.append(Word(repaired, w.start, w.end, w.speaker) if repaired != w.text else w)
    return out


def from_whisperx(result: dict, max_word_s: float | None = None,
                  diarized: bool = False,
                  max_word_chars: int = 0) -> Transcript:
    """Adapt a WhisperX aligned (and optionally diarized) result into a Transcript.

    WhisperX yields result["segments"][i]["words"] with keys word/start/end and,
    after `assign_word_speakers`, a "speaker" key. Some aligned words (e.g. pure
    punctuation) can lack timestamps — those inherit the neighbouring time so no
    caption is dropped."""
    words: list[Word] = []
    last_end = 0.0
    for seg in result.get("segments", []):
        seg_speaker = seg.get("speaker")
        for w in seg.get("words", []):
            text = str(w.get("word", "")).strip()
            if not text:
                continue
            start = w.get("start")
            end = w.get("end")
            if start is None:
                start = last_end
            if end is None:
                end = start
            speaker = w.get("speaker", seg_speaker)
            words.append(Word(text, float(start), float(end), speaker))
            last_end = float(end)
    # text-side guard first: repair/drop repetition-collapse tokens ("Naaaaaa…"),
    # then clamp any remaining over-long DURATIONS — both BEFORE the editable grid.
    if max_word_chars:
        words = sanitize_runaway_tokens(words, max_word_chars)
    if max_word_s is not None:
        clamp_long_words(words, max_word_s)
    distinct = {w.speaker for w in words if w.speaker}
    single = len(distinct) <= 1
    if single:                       # normalise to the no-colour default
        for w in words:
            w.speaker = None
    return Transcript(words=words, single_speaker=single, diarized=diarized)


def speaker_color_map(rows_or_map) -> dict[str, tuple[int, int, int]]:
    """Normalise a {speaker: (r,g,b)} mapping (tuples or [r,g,b] lists) for
    CaptionStyle.speaker_colors. Skips entries with no colour."""
    out: dict[str, tuple[int, int, int]] = {}
    items = rows_or_map.items() if isinstance(rows_or_map, dict) else rows_or_map
    for name, color in items:
        if not name or color is None:
            continue
        r, g, b = (int(c) for c in color)
        out[str(name)] = (r, g, b)
    return out
