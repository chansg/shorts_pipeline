"""Music-montage builder: pure timecode/duration/filter-graph logic, the build_montage
orchestration (music offset wiring + validation, mocked ffmpeg), the GUI handler
(threaded streaming, mocked), and a real ffmpeg end-to-end (3 clips + music -> 9:16)."""
import shutil
import subprocess
from pathlib import Path

import pytest

from orchestrator.errors import FriendlyError
from gameplay import config as gconf
from gameplay import montage as m
from gameplay import montage_gui as mgui

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


# ---- PURE: timecode --------------------------------------------------------

def test_parse_timecode():
    assert m.parse_timecode("0:45") == 45
    assert m.parse_timecode("1:30") == 90
    assert m.parse_timecode("1:02:03") == 3723
    assert m.parse_timecode("90") == 90
    assert m.parse_timecode("") == 0.0
    with pytest.raises(FriendlyError):
        m.parse_timecode("abc")


# ---- PURE: durations / offsets ---------------------------------------------

def test_unquote_strips_copy_as_path_quotes():
    assert m._unquote('"C:\\Users\\me\\Royalty [NCS Release].mp3"') == "C:\\Users\\me\\Royalty [NCS Release].mp3"
    assert m._unquote("  'song.mp3'  ") == "song.mp3"
    assert m._unquote("plain.mp3") == "plain.mp3"
    assert m._unquote("") == ""


def test_build_montage_accepts_quoted_paths(tmp_path, monkeypatch):
    # regression: Windows 'Copy as path' wraps in quotes; both clips and music must work
    clip = tmp_path / "Video Project 157.mp4"
    clip.write_bytes(b"x")
    music = tmp_path / "Royalty [NCS Release].mp3"
    music.write_bytes(b"x")
    _patch_ffmpeg(monkeypatch)
    out = m.build_montage([f'"{clip}"'], f'"{music}"', "0:30", out_path=tmp_path / "o.mp4")
    assert out.exists()


def test_montage_duration_and_offsets():
    assert m.montage_duration([10, 10, 10], 0.4) == pytest.approx(29.2)
    assert m.montage_duration([10], 0.4) == 10
    assert m.xfade_offsets([10, 8, 6], 0.4) == [pytest.approx(9.6), pytest.approx(17.2)]
    assert m.xfade_offsets([10], 0.4) == []


# ---- PURE: filter graph ----------------------------------------------------

def _graph(durations, **over):
    kw = dict(transition="fade", tdur=0.4, game_gain=0.2, denoise=12, music_fadein=0.5,
              music_fadeout=1.5, music_fadeout_start=10.0, music_gain=1.0, fade_ends=True,
              fade_dur=0.5, montage_dur=sum(durations) - (len(durations) - 1) * 0.4)
    kw.update(over)
    return m.build_filtergraph(durations, **kw)


def test_filtergraph_three_clips_has_all_stages():
    g = _graph([10, 8, 6])
    assert g.count("xfade=transition=fade") == 2          # two joins for three clips
    assert "offset=9.6000" in g and "offset=17.2000" in g  # accumulated offsets
    assert g.count("acrossfade=d=0.4000") == 2            # audio bed crossfades
    assert "afftdn=nr=12" in g and "volume=0.2" in g      # denoise + duck the game bed
    assert "afade=t=in:st=0:d=0.500" in g and "afade=t=out:st=10.000" in g  # music fades
    assert "amix=inputs=2:duration=longest:normalize=0" in g               # music dominant
    assert g.endswith("[a]") and "[v]" in g


def test_filtergraph_single_clip_no_transition():
    g = _graph([12], montage_dur=12)
    assert "xfade" not in g and "acrossfade" not in g
    assert "volume=0.2" in g and "amix=inputs=2" in g and "[v]" in g


def test_filtergraph_denoise_off_drops_afftdn():
    g = _graph([10, 10], denoise=0)
    assert "afftdn" not in g and "volume=0.2" in g


# ---- build_montage: validation + music-offset wiring (mocked ffmpeg) -------

def _patch_ffmpeg(monkeypatch, music_dur=300.0, clip_dur=10.0):
    import modules.assemble as A
    cmds = []
    monkeypatch.setattr(A, "_run", lambda cmd, cwd=None: cmds.append(cmd) or "")
    monkeypatch.setattr(A, "_probe_duration",
                        lambda p: music_dur if str(p).endswith((".mp3", ".wav")) else clip_dur)
    monkeypatch.setattr(A, "_has_audio", lambda p: True)
    monkeypatch.setattr(m.reframe_mod, "reframe",
                        lambda src, out, **k: (Path(out).write_bytes(b"v"), Path(out))[1])
    monkeypatch.setattr(m.overlay_mod, "composite",
                        lambda b, ov, out, **k: (Path(out).write_bytes(b"o"), Path(out))[1])
    return cmds


def test_build_montage_seeks_into_music(tmp_path, monkeypatch):
    clips = [tmp_path / f"c{i}.mp4" for i in range(2)]
    for c in clips:
        c.write_bytes(b"x")
    music = tmp_path / "song.mp3"
    music.write_bytes(b"x")
    cmds = _patch_ffmpeg(monkeypatch)
    out = m.build_montage([str(c) for c in clips], str(music), "0:45",
                          out_path=tmp_path / "out.mp4")
    assert out == tmp_path / "out.mp4"
    body = next(c for c in cmds if "-filter_complex" in c)
    mi = body.index(str(music))
    # the music input is seeked: ... -ss 45.000 -i <music>
    assert body[mi - 1] == "-i" and body[mi - 2] == "45.000" and body[mi - 3] == "-ss"


def test_build_montage_rejects_offset_past_track(tmp_path, monkeypatch):
    c = tmp_path / "c.mp4"
    c.write_bytes(b"x")
    music = tmp_path / "song.mp3"
    music.write_bytes(b"x")
    _patch_ffmpeg(monkeypatch, music_dur=60.0)
    with pytest.raises(FriendlyError, match="at/after the track length"):
        m.build_montage([str(c)], str(music), "1:30", out_path=tmp_path / "o.mp4")


def test_build_montage_requires_clip_and_music(tmp_path, monkeypatch):
    _patch_ffmpeg(monkeypatch)
    with pytest.raises(FriendlyError, match="at least one"):
        m.build_montage([], "x.mp3", "0", out_path=tmp_path / "o.mp4")
    c = tmp_path / "c.mp4"
    c.write_bytes(b"x")
    with pytest.raises(FriendlyError, match="music file"):
        m.build_montage([str(c)], "", "0", out_path=tmp_path / "o.mp4")


# ---- GUI handler -----------------------------------------------------------

def _interactive(u):
    return u.get("interactive") if isinstance(u, dict) else getattr(u, "interactive", None)


def test_gui_clip_list_helpers():
    assert mgui._clip_lines("a.mp4\n  b.mp4 \n\n\"c.mp4\"") == ["a.mp4", "b.mp4", "c.mp4"]
    assert mgui._append_clips(["x.mp4", "a.mp4"], "a.mp4") == "a.mp4\nx.mp4"   # dedup, order


def test_gui_handler_streams_and_toggles_button(tmp_path, monkeypatch):
    def fake_build(clips, music, start, *, out_path=None, progress=None):
        progress("Reframing clip 1/2 to 9:16")
        progress("Applying the GamerChans overlay")
        Path(out_path).write_bytes(b"out")
        return out_path

    monkeypatch.setattr(mgui.montage_mod, "build_montage", fake_build)
    frames = list(mgui._do_montage("a.mp4\nb.mp4", "song.mp3", "0:30", str(tmp_path)))
    assert _interactive(frames[0][2]) is False           # Run disabled while building
    assert _interactive(frames[-1][2]) is True           # re-enabled on completion
    assert "Reframing clip 1/2" in frames[-1][0]
    assert str(frames[-1][1]).endswith(".mp4")           # result video path


def test_gui_handler_prompts_without_inputs():
    frames = list(mgui._do_montage("", "song.mp3", "0", ""))
    assert len(frames) == 1 and _interactive(frames[0][2]) is True
    assert "at least one gameplay clip" in frames[0][0]
    frames2 = list(mgui._do_montage("a.mp4", "", "0", ""))
    assert "music file" in frames2[0][0]


def test_gui_handler_survives_build_error(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise FriendlyError("Music start 90.0s is at/after the track length (60.0s)")

    monkeypatch.setattr(mgui.montage_mod, "build_montage", boom)
    frames = list(mgui._do_montage("a.mp4", "song.mp3", "1:30", str(tmp_path)))
    assert "ERROR: FriendlyError" in frames[-1][0]
    assert _interactive(frames[-1][2]) is True


def test_servable_preview_copies_external_file(tmp_path, monkeypatch):
    # a file outside CWD/temp must be copied into temp so gr.Video can serve it
    import tempfile
    fake_temp = tmp_path / "tmp"
    fake_temp.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_temp))
    external = tmp_path / "AshenChan" / "montage.mp4"
    external.parent.mkdir()
    external.write_bytes(b"video")
    served = mgui._servable_preview(str(external))
    assert Path(served).parent == fake_temp and Path(served).read_bytes() == b"video"
    # a file already under temp is served in place (no needless copy)
    inside = fake_temp / "already.mp4"
    inside.write_bytes(b"x")
    assert mgui._servable_preview(str(inside)) == str(inside)
    assert mgui._servable_preview(None) is None


def test_gui_open_folder(tmp_path, monkeypatch):
    opened = {}
    monkeypatch.setattr(mgui.os, "startfile", lambda p: opened.setdefault("p", p),
                        raising=False)
    msg = mgui._open_folder(str(tmp_path / "montage"))
    assert "Opened" in msg and (tmp_path / "montage").exists()


# ---- real ffmpeg end-to-end ------------------------------------------------

def _clip(path, color, dur=2):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=640x360:r=30:d={dur}",
         "-f", "lavfi", "-i", f"sine=frequency=300:duration={dur}",
         "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)], check=True, capture_output=True)
    return path


def _probe(path, entries, stream=None):
    cmd = ["ffprobe", "-v", "error"]
    if stream:
        cmd += ["-select_streams", stream]
    cmd += ["-show_entries", entries, "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    return subprocess.run(cmd, capture_output=True, text=True).stdout.split()


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_end_to_end_montage_9x16_with_audio(tmp_path):
    clips = [_clip(tmp_path / f"c{i}.mp4", c) for i, c in enumerate(["red", "green", "blue"])]
    music = tmp_path / "song.wav"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
                    str(music)], check=True, capture_output=True)
    out = m.build_montage([str(c) for c in clips], str(music), "0:02",
                          out_path=tmp_path / "montage.mp4", progress=lambda s: None)
    assert out.exists()
    assert _probe(out, "stream=width,height", "v:0") == [str(gconf.WIDTH), str(gconf.HEIGHT)]
    assert _probe(out, "stream=index", "a:0")           # has an audio stream
    # duration ~ 3*2 - 2*0.4 = 5.2s (tolerate encoder rounding)
    dur = float(_probe(out, "format=duration")[0])
    assert 4.5 < dur < 6.0
