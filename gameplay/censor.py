"""Profanity censor — one word-list, shared by the audio bleep and the caption mask.

A censored word is the same `(text, start, end)` span the captions use (the WhisperX
word alignment), so the audio censor and the caption mask hit exactly the same moment.
Pure logic here — matching, span merging, mask text, and ffmpeg audio-filter strings —
with no ffmpeg/Gradio import, so it unit-tests without a GPU. Deterministic given the
word-list + transcript.

Matching is WHOLE-WORD and case-insensitive (never substring): the bare token, stripped
of surrounding punctuation, must equal a word-list entry and not be in the allow-list —
so "Shaco", "assassin", "Cassiopeia" are never censored.
"""
from __future__ import annotations

from gameplay import config as gconf

# Punctuation stripped from a token's ends before the whole-word compare.
_STRIP = "\"'.,!?;:()[]{}…*-—–_/\\"


def _norm(token: str) -> str:
    return str(token or "").strip().strip(_STRIP).lower()


def _wordset(values) -> set[str]:
    return {_norm(v) for v in (values or []) if _norm(v)}


def is_censored(text: str, wordlist=None, allowlist=None) -> bool:
    """True if ANY whole-word token in `text` is a censored word (and not allow-listed).
    Case-insensitive, punctuation-tolerant, never a substring match."""
    wl = _wordset(gconf.CENSOR_WORDLIST if wordlist is None else wordlist)
    al = _wordset(gconf.CENSOR_ALLOWLIST if allowlist is None else allowlist)
    for tok in str(text or "").split():
        t = _norm(tok)
        if t and t not in al and t in wl:
            return True
    return False


def mask_token(token: str, style: str | None = None) -> str:
    """Mask one token: "stars" -> first letter + asterisks ("fuck"->"f***"); "block"
    -> "[bleep]". Leading/trailing punctuation is preserved around the masked core."""
    style = style or gconf.CENSOR_CAPTION_STYLE
    lead = len(token) - len(token.lstrip(_STRIP))
    trail = len(token) - len(token.rstrip(_STRIP))
    pre = token[:lead]
    post = token[len(token) - trail:] if trail else ""
    core = token[lead:len(token) - trail] if trail else token[lead:]
    if not core:
        return token
    if style == "block":
        masked = "[bleep]"
    else:
        masked = core[0] + "*" * (len(core) - 1)
    return pre + masked + post


def mask_text(text: str, style: str | None = None,
              wordlist=None, allowlist=None) -> str:
    """Mask every censored whole-word token in `text`, leaving the rest intact."""
    wl = _wordset(gconf.CENSOR_WORDLIST if wordlist is None else wordlist)
    al = _wordset(gconf.CENSOR_ALLOWLIST if allowlist is None else allowlist)
    out = []
    for tok in str(text or "").split():
        t = _norm(tok)
        out.append(mask_token(tok, style) if (t and t not in al and t in wl) else tok)
    return " ".join(out)


def merge_spans(spans, pad: float = 0.0, dur: float | None = None
                ) -> list[tuple[float, float]]:
    """Pad, clamp to [0, dur], sort, and merge overlapping/adjacent spans into one."""
    norm = []
    for s, e in spans or []:
        a, b = float(s) - pad, float(e) + pad
        if dur is not None:
            b = min(b, float(dur))
        a = max(0.0, a)
        if b > a:
            norm.append((a, b))
    norm.sort()
    merged: list[tuple[float, float]] = []
    for a, b in norm:
        if merged and a <= merged[-1][1]:          # overlap/adjacent after padding
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _enable_expr(spans) -> str:
    """ffmpeg timeline-`enable` expression: truthy inside any span."""
    return "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in spans) or "0"


def audio_graph(spans, dur: float, *, mode: str | None = None,
                hz: int | None = None, bleep_gain: float | None = None,
                duck_gain: float | None = None,
                src: str = "[0:a]", out: str = "[a]") -> str | None:
    """An ffmpeg filter_complex audio sub-graph (ending in `out`) that censors `src`
    over `spans`. Returns None when there's nothing to do (no spans). Modes:
      - mute: silence the spans
      - duck: drop the spans' volume to CENSOR_DUCK_GAIN
      - bleep: silence the spans and overlay a 1 kHz tone there (the default)
    No extra `-i` inputs needed — the tone is a `sine` source filter."""
    spans = list(spans or [])
    if not spans:
        return None
    mode = (mode or gconf.CENSOR_AUDIO_MODE).lower()
    expr = _enable_expr(spans)
    if mode == "mute":
        return f"{src}volume=0:enable='{expr}'{out}"
    if mode == "duck":
        g = gconf.CENSOR_DUCK_GAIN if duck_gain is None else duck_gain
        return f"{src}volume={g}:enable='{expr}'{out}"
    # bleep (default): mute voice inside spans, add a gated tone there, mix.
    hz = gconf.CENSOR_BLEEP_HZ if hz is None else hz
    gain = gconf.CENSOR_BLEEP_GAIN if bleep_gain is None else bleep_gain
    return (
        f"sine=frequency={hz}:sample_rate=48000:duration={float(dur):.3f},"
        f"volume={gain}[__tone];"
        f"[__tone]volume=0:enable='not({expr})'[__toneg];"
        f"{src}volume=0:enable='{expr}'[__voiceg];"
        f"[__voiceg][__toneg]amix=inputs=2:duration=first:normalize=0{out}"
    )
