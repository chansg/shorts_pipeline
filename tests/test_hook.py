"""Narrated hook — pure helpers, TTS caching (mocked ElevenLabs), caption colour,
and the composed audio graph. No GPU/network; ffmpeg only in test_hook_render.py."""
from pathlib import Path

from gameplay import hook
from gameplay import config as gconf


# ---- caption tuples (even distribution) ------------------------------------

def test_hook_caption_tuples_even_and_narrator():
    t = hook.hook_caption_tuples("one two three four", dur=4.0, lead=0.0)
    assert [x[0] for x in t] == ["one", "two", "three", "four"]
    assert all(x[3] == "NARRATOR" for x in t)
    assert t[0][1] == 0.0 and abs(t[-1][2] - 4.0) < 1e-6
    assert abs((t[1][1] - t[0][1]) - 1.0) < 1e-6        # even 1s steps


def test_hook_caption_tuples_lead_and_empty():
    assert hook.hook_caption_tuples("", 3.0) == []
    assert hook.hook_caption_tuples("hi", 0) == []
    assert hook.hook_caption_tuples("a b", 2.0, lead=0.5)[0][1] == 0.5


# ---- duck/mix audio graph (string) -----------------------------------------

def test_duck_mix_graph_contents():
    g = hook.duck_mix_graph("[0:a]", 1.2, duck=0.25, release=0.3)
    assert "amix=inputs=2" in g and "volume='if(lt(t," in g
    assert "[1:a]" in g and g.endswith("[a]")
    assert "0.25" in g                                  # duck level in the expr


# ---- TTS cache (mock the ElevenLabs client) --------------------------------

def test_synthesize_hook_caches_by_text_and_voice(tmp_path, monkeypatch):
    calls = []

    def fake_synth(text, out, voice_id=None):
        calls.append((text, voice_id))
        Path(out).write_bytes(b"RIFFfake")
        return Path(out)

    monkeypatch.setattr("modules.tts.synthesize", fake_synth)
    monkeypatch.setattr("modules.assemble._probe_duration", lambda p: 1.5)

    p1, d1 = hook.synthesize_hook("hello world", "V1", tmp_path)
    p2, d2 = hook.synthesize_hook("hello world", "V1", tmp_path)   # cached
    assert p1 == p2 and d1 == d2 == 1.5
    assert len(calls) == 1                              # only ONE bill for the repeat

    p3, _ = hook.synthesize_hook("hello world", "V2", tmp_path)    # different voice
    assert p3 != p1 and len(calls) == 2
    assert calls[1] == ("hello world", "V2")


def test_synthesize_hook_defaults_voice_to_config(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr("modules.tts.synthesize",
                        lambda t, o, voice_id=None: (seen.update(v=voice_id),
                                                     Path(o).write_bytes(b"x"))[1])
    monkeypatch.setattr("modules.assemble._probe_duration", lambda p: 1.0)
    hook.synthesize_hook("hi", None, tmp_path)
    assert seen["v"] == gconf.HOOK_VOICE


# ---- caption: NARRATOR colour during the hook, normal captions after -------

def test_write_captions_prepends_narrator_coloured_hook():
    import tempfile
    from gameplay.manual import ManualOptions, write_captions
    from gameplay.transcript import Transcript, Word
    from modules.karaoke_captions import _ass_color
    with tempfile.TemporaryDirectory() as d:
        t = Transcript([Word("gameplay", 5.0, 5.4, "S0")])
        ass = write_captions(t, ManualOptions(), Path(d) / "c.ass",
                             hook=("hello there", 2.0, 0.0))
        text = ass.read_text(encoding="utf-8")
    assert _ass_color(*gconf.NARRATOR_CAPTION_COLOR) in text   # reserved hook colour
    assert "HELLO" in text.upper() and "GAMEPLAY" in text.upper()
    # hook line is at the top of [Events] (t=0), the transcript word later (t=5)
    assert text.index("HELLO") < text.index("GAMEPLAY")


def test_write_captions_without_hook_is_unchanged():
    import tempfile
    from gameplay.manual import ManualOptions, write_captions
    from gameplay.transcript import Transcript, Word
    with tempfile.TemporaryDirectory() as d:
        t = Transcript([Word("gg", 1.0, 1.4, "S0")])
        a = write_captions(t, ManualOptions(), Path(d) / "a.ass").read_text("utf-8")
        b = write_captions(t, ManualOptions(), Path(d) / "b.ass",
                           hook=None).read_text("utf-8")
    assert a == b and "NARRATOR" not in a   # no hook -> today's caption exactly


# ---- burn copies audio when there's no graph (disabled hook == today) ------

def test_burn_captions_copies_audio_without_graph(tmp_path, monkeypatch):
    from gameplay import manual
    cmds = []
    monkeypatch.setattr(manual, "_run", lambda cmd, cwd=None: cmds.append(cmd))
    monkeypatch.setattr(manual, "_has_audio", lambda v: True)
    ass = tmp_path / "c.ass"; ass.write_text("[Events]\n")
    manual.burn_captions(tmp_path / "v.mp4", ass, tmp_path / "o.mp4")
    assert "-c:a" in cmds[0] and "copy" in cmds[0]
    assert "-filter_complex" not in cmds[0]


# ---- censor + hook compose into ONE final encode (no extra pass) -----------

def test_censor_and_hook_compose_in_one_graph(tmp_path, monkeypatch):
    from gameplay import manual, config as g
    from gameplay.state import GameplayClip
    from gameplay.transcript import Transcript, Word
    monkeypatch.setattr(g, "GAMEPLAY_DIR", tmp_path)
    monkeypatch.setattr(manual, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(manual.reframe_mod, "reframe",
                        lambda *a, **k: None)            # skip the real reframe encode
    monkeypatch.setattr(manual, "_probe_duration", lambda p: 5.0)
    monkeypatch.setattr(manual, "_has_audio", lambda v: True)
    monkeypatch.setattr(manual.hook_mod, "synthesize_hook",
                        lambda text, voice, d: (tmp_path / "_hook.wav", 1.5))
    cmds = []
    monkeypatch.setattr(manual, "_run", lambda cmd, cwd=None: cmds.append(cmd))

    clip = GameplayClip("c")
    (clip.dir / "source.mp4").write_bytes(b"")
    t = Transcript([Word("oh", 0.4, 0.8), Word("shit", 1.0, 1.4, censor=True)])
    opts = manual.ManualOptions(censor="both", censor_audio_mode="bleep",
                                hook_enabled=True, hook_text="watch this")
    list(manual.run_manual(clip, t, opts, force=True))

    encodes = [c for c in cmds if "libx264" in c]
    assert len(encodes) == 1                             # ONE final encode, no extra pass
    fc = encodes[0][encodes[0].index("-filter_complex") + 1]
    assert "amix=inputs=2" in fc                          # hook duck/mix
    assert "sine=frequency=" in fc or "volume=0:enable=" in fc   # censor (bleep)
    assert "-i" in encodes[0]                             # the narration input was added
