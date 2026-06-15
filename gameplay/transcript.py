"""The transcript data model and the adapter to the caption tuples.

This is the editable gate: the rows the GUI shows ARE the
`(text, start, end, speaker)` list `modules.karaoke_captions.build_ass` consumes,
so a corrected transcript flows straight into captions with no separate export.

Kept free of the heavy WhisperX/torch import so it (and its tests) run with no GPU.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


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
    colour (matching the lore look)."""
    words: list[Word] = field(default_factory=list)
    single_speaker: bool = False

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

    @classmethod
    def from_rows(cls, rows, single_speaker: bool = False) -> "Transcript":
        """Rebuild from edited grid rows. Tolerant of blank/half-filled trailing
        rows the editor may add, and of out-of-order/garbled timing."""
        words: list[Word] = []
        for row in rows or []:
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
        return {"single_speaker": self.single_speaker,
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
        return cls(words=words, single_speaker=bool(d.get("single_speaker")))

    @classmethod
    def load(cls, path: str | Path) -> "Transcript":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def from_whisperx(result: dict) -> Transcript:
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
    distinct = {w.speaker for w in words if w.speaker}
    single = len(distinct) <= 1
    if single:                       # normalise to the no-colour default
        for w in words:
            w.speaker = None
    return Transcript(words=words, single_speaker=single)


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
