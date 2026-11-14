"""Caption customisation + the player-captions on/off switch in the gameplay engine
(gameplay/manual.py + modules/karaoke_captions.py). No GPU; the run_manual tests
record ffmpeg invocations instead of encoding."""
import shutil
from pathlib import Path

import pytest

from gameplay import config as gconf
from gameplay import manual as manual_mod
from gameplay import overlay as overlay_mod
from gameplay import reframe as reframe_mod
from gameplay.manual import ManualOptions, caption_style, run_manual, write_captions
from gameplay.state import GameplayClip
from gameplay.transcript import Transcript, Word
from modules.karaoke_captions import CaptionStyle, build_ass
from orchestrator.errors import FriendlyError

T = Transcript([Word("no way", 0.0, 0.6, "Chan"), Word("what", 0.7, 1.2, "Jett")])


def _dialogue(ass_text):
    return [ln for ln in ass_text.splitlines() if ln.startswith("Dialogue:")]


# ---- defaults are the original look ---------------------------------------

def test_default_options_reproduce_the_original_style():
    st = caption_style(ManualOptions())
    assert st.fontsize == gconf.CAPTION_FONTSIZE
    assert st.outline == CaptionStyle().outline
    assert st.uppercase is True and st.animate is True
    assert st.fill == CaptionStyle().fill                # no single-colour override
    assert ManualOptions().captions_enabled is True


def test_default_preset_matches_the_defaults():
    p = gconf.CAPTION_PRESETS[gconf.CAPTION_DEFAULT_PRESET]
    o = ManualOptions()
    assert (p["font"], p["size"], p["outline"], p["uppercase"], p["animate"],
            p["mode"], p["by_player"]) == (o.caption_font, o.caption_size,
                                           o.caption_outline, o.caption_uppercase,
                                           o.caption_animate, o.caption_mode,
                                           o.caption_by_player)


def test_every_preset_is_complete():
    keys = {"font", "size", "outline", "uppercase", "animate", "mode", "by_player",
            "color"}
    for name, p in gconf.CAPTION_PRESETS.items():
        assert set(p) == keys, name
        assert p["font"] in gconf.CAPTION_FONTS and p["mode"] in ("phrase", "word")


# ---- look options reach the renderer --------------------------------------

def test_size_outline_case_reach_the_style():
    st = caption_style(ManualOptions(caption_size=96, caption_outline=3.5,
                                     caption_uppercase=False))
    assert (st.fontsize, st.outline, st.uppercase) == (96, 3.5, False)


def test_animate_off_emits_static_cues(tmp_path):
    on = write_captions(T, ManualOptions(), tmp_path / "on.ass").read_text("utf-8")
    off = write_captions(T, ManualOptions(caption_animate=False),
                         tmp_path / "off.ass").read_text("utf-8")
    assert "\\t(" in on and "\\fscx70" in on             # pop-in transform
    assert "\\t(" not in off and "\\fscx" not in off     # static text
    assert len(_dialogue(on)) == len(_dialogue(off))     # same cues


def test_animate_flag_defaults_on_for_the_lore_renderer():
    ass = build_ass([("hi", 0.0, 0.5)], CaptionStyle())
    assert "\\t(" in ass                                 # lore output unchanged


def test_uppercase_off_keeps_case(tmp_path):
    ass = write_captions(T, ManualOptions(caption_uppercase=False),
                         tmp_path / "c.ass").read_text("utf-8")
    assert "no way" in ass and "NO WAY" not in ass


def test_single_colour_drops_player_colours(tmp_path):
    by_player = write_captions(T, ManualOptions(speaker_colors={"Chan": (255, 0, 0)}),
                               tmp_path / "a.ass").read_text("utf-8")
    single = write_captions(T, ManualOptions(speaker_colors={"Chan": (255, 0, 0)},
                                             caption_by_player=False,
                                             caption_color="#00FF00"),
                            tmp_path / "b.ass").read_text("utf-8")
    assert "\\c&H" in by_player                          # per-player colour tags
    assert "\\c&H" not in single                         # one colour for all lines
    # the style fill carries the chosen colour (ASS is &HBBGGRR -> green = 00FF00)
    assert caption_style(ManualOptions(caption_by_player=False,
                                       caption_color="#00FF00")).fill == (0, 255, 0)


def test_single_colour_keeps_the_hook_narrator_colour(tmp_path):
    ass = write_captions(T, ManualOptions(caption_by_player=False),
                         tmp_path / "h.ass", hook=("story time", 1.0, 0.0)) \
        .read_text("utf-8")
    assert "\\c&H" in ass                                # NARRATOR still coloured


# ---- line wrapping follows font + size + case -------------------------------

def _lines(opts, text):
    """The rendered lines of one cue (after case + wrapping) under these options."""
    from modules.karaoke_captions import wrap_lines
    return wrap_lines(caption_style(opts), text)


def _look(name, **over):
    p = gconf.CAPTION_PRESETS[name]
    kw = dict(caption_font=p["font"], caption_size=p["size"],
              caption_outline=p["outline"], caption_uppercase=p["uppercase"],
              caption_animate=p["animate"], caption_by_player=p["by_player"])
    kw.update(over)
    return ManualOptions(**kw)


def test_default_wrapping_is_unchanged():
    # the default look (Anton at the default size) still breaks lines at 12 chars
    st = caption_style(ManualOptions())
    assert st.max_line_chars == gconf.CAPTION_MAX_LINE_CHARS
    assert _lines(ManualOptions(), "no way he just did that") == \
        ["NO WAY HE", "JUST DID", "THAT"]
    assert manual_mod.line_chars("Anton", gconf.CAPTION_FONTSIZE, outline=7.0) == \
        gconf.CAPTION_MAX_LINE_CHARS                      # the estimate fallback too


def test_common_words_are_never_split_when_they_fit():
    # Regression (review): Arial wrapped by a fixed per-character width that ignored
    # case, so the Clean look burned 'unbelievabl' / 'e' and Arial at 150 split
    # 'BASICALL' / 'Y' though both fit the frame easily.
    assert _lines(_look("Clean"), "unbelievable teammates") == ["unbelievable",
                                                                "teammates"]
    assert _lines(_look("Clean", caption_size=150), "basically seriously") == \
        ["basically", "seriously"]
    arial = ManualOptions(caption_font="Arial", caption_size=150)
    assert _lines(arial, "basically seriously") == ["BASICALLY", "SERIOUSLY"]


def test_a_word_is_split_only_when_it_really_cannot_fit():
    big = ManualOptions(caption_font="Arial", caption_size=220)
    st = caption_style(big)
    lines = _lines(big, "unbelievable")
    assert len(lines) > 1 and "".join(lines) == "UNBELIEVABLE"   # too wide: split...
    assert all(st.text_px(ln) <= st.max_line_px for ln in lines)  # ...into lines that fit


def test_estimate_fallback_without_the_font_file(monkeypatch):
    # no font file to measure -> a case-aware per-character estimate
    monkeypatch.setattr(manual_mod, "_measure_font", lambda font: None)
    st = caption_style(ManualOptions(caption_font="Arial", caption_size=150))
    assert st.text_px is None
    assert st.max_line_chars == manual_mod.line_chars("Arial", 150, outline=7.0)
    assert manual_mod.line_chars("Arial", 150, uppercase=False) > \
        manual_mod.line_chars("Arial", 150, uppercase=True)       # lowercase is narrower
    assert manual_mod.line_chars("Arial", 220) < manual_mod.line_chars("Arial", 112)
    assert manual_mod.line_chars("Arial", 220) >= 4


def test_a_cue_too_tall_for_its_height_is_split_in_time():
    # Regression (review): a big font placed low wrapped a phrase into more lines than
    # fit below its centre, running the last line off the bottom of the frame.
    opts = ManualOptions(caption_font="Arial", caption_size=220, caption_pos_y_frac=0.9)
    st = caption_style(opts)
    assert manual_mod.max_caption_lines(st) == 1
    cues = manual_mod.fit_cues_to_height([("played wanted killed", 1.0, 2.0, "Chan")], st)
    assert [c[0] for c in cues] == ["played", "wanted", "killed"]
    assert cues[0][1] == 1.0 and cues[-1][2] == 2.0                   # same span...
    assert all(a[2] == b[1] for a, b in zip(cues, cues[1:]))         # ...back to back
    assert all(c[3] == "Chan" for c in cues)                         # speaker kept


def test_a_cue_that_fits_is_untouched():
    st = caption_style(ManualOptions())
    cue = ("no way he just did that", 0.0, 1.0, "Chan")
    assert manual_mod.fit_cues_to_height([cue], st) == [cue]


def test_a_single_word_too_tall_for_its_height_is_split_in_time():
    # Regression (review): height fitting only regrouped whole words, so one long word
    # hard-split into 2 lines at a big size placed low still ran off the bottom.
    st = caption_style(ManualOptions(caption_font="Arial", caption_size=220,
                                     caption_pos_y_frac=0.9))
    cues = manual_mod.fit_cues_to_height([("unbelievable", 1.0, 2.0, "Chan")], st)
    assert len(cues) > 1 and "".join(c[0] for c in cues) == "UNBELIEVABLE"
    assert all(len(manual_mod.wrap_lines(st, c[0])) == 1 for c in cues)
    assert cues[0][1] == 1.0 and cues[-1][2] == 2.0


def test_characters_the_font_lacks_are_budgeted_wide():
    # Regression (review): emoji / CJK aren't in Anton or Arial; Pillow measured them as
    # the narrow .notdef box while libass draws them with a much wider fallback font,
    # so a row of emoji was predicted to fit and burned off both frame edges.
    assert manual_mod._has_glyph("Anton", "A") and manual_mod._has_glyph("Anton", "é")
    assert not manual_mod._has_glyph("Anton", "\U0001F480")      # skull emoji
    assert not manual_mod._has_glyph("Arial", "今")           # CJK
    st = caption_style(ManualOptions())
    skulls = "\U0001F480" * 8
    lines = manual_mod.wrap_lines(st, skulls)
    assert len(lines) > 1 and all(st.text_px(ln) <= st.max_line_px for ln in lines)
    assert manual_mod.wrap_lines(st, "no way \U0001F480\U0001F480") == \
        ["NO WAY \U0001F480\U0001F480"]                           # short ones still fit


def _burn_and_bbox(tmp_path, words, opts):
    """Burn `words` onto a grey 1080x1920 frame and return the (x0, y0, x1, y1) box of
    every pixel the caption changed (fill, outline and shadow), taken over every cue
    both at its pop-in peak (the text at its widest) and at rest."""
    import re
    import subprocess

    import numpy as np
    from PIL import Image
    base = tmp_path / "grey.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    f"color=0x808080:s={gconf.WIDTH}x{gconf.HEIGHT}:d=4:r=50",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(base)],
                   check=True, capture_output=True)
    ass = write_captions(Transcript(words), opts, tmp_path / "c.ass")
    starts = [int(h) * 3600 + int(m) * 60 + float(s) for h, m, s in
              re.findall(r"Dialogue: \d+,(\d+):(\d+):([\d.]+),", ass.read_text("utf-8"))]
    assert starts, "no caption cues written"
    out = manual_mod.burn_captions(base, ass, tmp_path / "o.mp4", final=False)
    box = None
    for i, at in enumerate(t + d for t in starts for d in (0.12, 0.3)):
        png = tmp_path / f"f{i}.png"
        subprocess.run(["ffmpeg", "-y", "-ss", f"{at:.3f}", "-i", str(out),
                        "-frames:v", "1", str(png)], check=True, capture_output=True)
        lit = np.abs(np.asarray(Image.open(png).convert("L"), dtype=int) - 128) > 40
        ys, xs = np.where(lit)
        if len(xs):
            b = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]),
                                         max(box[2], b[2]), max(box[3], b[3]))
    assert box, "no caption pixels found"
    return box


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_big_arial_caption_stays_inside_the_frame(tmp_path):
    # Regression (review): a big Arial size burned lines wider than the 1080px frame,
    # clipping letters off both edges. At the pop-in's widest moment the burned
    # caption must not touch either side of the frame.
    x0, _y0, x1, _y1 = _burn_and_bbox(
        tmp_path, [Word("unbelievable", 0.2, 1.4, "Chan")],
        ManualOptions(caption_font="Arial", caption_size=220))
    assert x0 > 0 and x1 < gconf.WIDTH - 1, f"caption spans x={x0}..{x1} (clipped)"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_big_caption_placed_low_stays_above_the_bottom_edge(tmp_path):
    words = [Word("played", 0.2, 0.6, "Chan"), Word("wanted", 0.6, 1.0, "Chan"),
             Word("killed", 1.0, 1.4, "Chan")]
    opts = ManualOptions(caption_font="Arial", caption_size=220, caption_pos_y_frac=0.9)
    _x0, y0, _x1, y1 = _burn_and_bbox(tmp_path, words, opts)
    assert y1 < gconf.HEIGHT - 1, f"caption reaches y={y1} (cut off at the bottom)"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_one_long_word_placed_low_stays_above_the_bottom_edge(tmp_path):
    # a single word too wide for one line at the biggest size, lowest position — incl.
    # a lowercase descender (the 'g'), which used to be cut for the whole cue
    words = [Word("unbelievable", 0.2, 1.2, "Chan"), Word("outplaying", 1.2, 2.2, "Chan")]
    opts = ManualOptions(caption_font="Arial", caption_size=220, caption_pos_y_frac=0.9,
                         caption_uppercase=False)
    _x0, _y0, _x1, y1 = _burn_and_bbox(tmp_path, words, opts)
    assert y1 < gconf.HEIGHT - 1, f"caption reaches y={y1} (cut off at the bottom)"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_a_row_of_emoji_stays_inside_the_frame(tmp_path):
    words = [Word("gg " + "\U0001F480" * 8, 0.2, 1.4, "Chan")]
    x0, _y0, x1, _y1 = _burn_and_bbox(tmp_path, words, ManualOptions())
    assert x0 > 0 and x1 < gconf.WIDTH - 1, f"caption spans x={x0}..{x1} (clipped)"


# ---- player captions off ---------------------------------------------------

def test_captions_off_writes_no_player_cues(tmp_path):
    ass = write_captions(T, ManualOptions(captions_enabled=False),
                         tmp_path / "c.ass").read_text("utf-8")
    assert _dialogue(ass) == []


def test_captions_off_still_captions_the_hook(tmp_path):
    ass = write_captions(T, ManualOptions(captions_enabled=False),
                         tmp_path / "c.ass", hook=("the time I", 1.0, 0.0)) \
        .read_text("utf-8")
    assert _dialogue(ass) and all("NO WAY" not in d for d in _dialogue(ass))


def test_burn_without_ass_uses_a_null_video_chain(tmp_path, monkeypatch):
    cmds = []
    monkeypatch.setattr(manual_mod, "_run", lambda cmd, cwd=None: cmds.append((cmd, cwd)))
    monkeypatch.setattr(manual_mod, "_has_audio", lambda p: False)
    manual_mod.burn_captions(tmp_path / "v.mp4", None, tmp_path / "o.mp4")
    cmd, cwd = cmds[0]
    vf = cmd[cmd.index("-vf") + 1]
    assert vf == "null" and "ass=" not in " ".join(cmd) and cwd is None


def _patch_runners(monkeypatch, tmp_path, cmds):
    monkeypatch.setattr(gconf, "GAMEPLAY_DIR", tmp_path)
    monkeypatch.setattr(manual_mod, "ensure_ffmpeg", lambda: None)
    for mod in (reframe_mod, manual_mod, overlay_mod):
        monkeypatch.setattr(mod, "_run", lambda cmd, cwd=None: cmds.append(cmd))
        monkeypatch.setattr(mod, "_has_audio", lambda p: False)


def _clip():
    clip = GameplayClip("capopts")
    (clip.dir / "source.mp4").write_bytes(b"")
    return clip


def test_run_manual_captions_off_needs_no_transcript(tmp_path, monkeypatch):
    cmds = []
    _patch_runners(monkeypatch, tmp_path, cmds)
    clip = _clip()
    events = list(run_manual(clip, Transcript([]),
                             ManualOptions(captions_enabled=False), force=True))
    assert events[-1].get("done") is True
    assert not any("ass=" in str(a) for c in cmds for a in c)   # no caption track
    assert any("captions off" in e["msg"] for e in events)
    assert not clip.ass_path.exists()


def test_run_manual_captions_on_still_requires_a_transcript(tmp_path, monkeypatch):
    _patch_runners(monkeypatch, tmp_path, [])
    with pytest.raises(FriendlyError, match="turn player captions off"):
        list(run_manual(_clip(), Transcript([]), ManualOptions(), force=True))


def test_run_manual_captions_on_burns_the_track(tmp_path, monkeypatch):
    cmds = []
    _patch_runners(monkeypatch, tmp_path, cmds)
    list(run_manual(_clip(), T, ManualOptions(), force=True))
    assert any("ass=" in str(a) for c in cmds for a in c)


def test_preview_refuses_when_captions_off(tmp_path, monkeypatch):
    monkeypatch.setattr(gconf, "GAMEPLAY_DIR", tmp_path)
    monkeypatch.setattr(manual_mod, "ensure_ffmpeg", lambda: None)
    with pytest.raises(FriendlyError, match="captions are off"):
        manual_mod.preview_captions(_clip(), T, ManualOptions(captions_enabled=False))
