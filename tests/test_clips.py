"""Full-auto clips orchestration — ranking, HUD-boost effect, manifest round-trip,
friendly empty, and a real 9:16 export probe (shared reframe/encode)."""
import json
import subprocess

import pytest

from fullauto import clips
from fullauto import hud as hud_mod
from fullauto.reaction import ReactionWindow
from gameplay.state import AutoSession


def _win(start, end, score):
    return ReactionWindow(start, end, score, [round((start + end) / 2, 2)])


# ---- ranking (no ffmpeg: HUD disabled short-circuits before any frame read) -

def test_rank_orders_by_score_desc_and_assigns_ranks():
    wins = [_win(0, 18, 0.4), _win(40, 58, 0.9), _win(80, 98, 0.6)]
    ranked = clips.rank_windows("x.mp4", wins, hud_enabled=False)
    assert [c.rank for c in ranked] == [1, 2, 3]
    assert [c.audio_score for c in ranked] == [0.9, 0.6, 0.4]
    assert all(c.hud_boost == 0.0 and c.score == c.audio_score for c in ranked)


def test_hud_boost_lifts_a_candidate_above_a_louder_one(monkeypatch):
    # a quieter reaction WITH a pentakill should outrank a louder one with no HUD
    def fake_scan(video, start, end, **kw):
        return [hud_mod.HudEvent("pentakill")] if start == 40 else []

    monkeypatch.setattr(hud_mod, "scan_window", fake_scan)
    wins = [_win(0, 18, 0.8), _win(40, 58, 0.5)]      # louder=0.8 vs quieter=0.5+penta
    ranked = clips.rank_windows("x.mp4", wins, hud_enabled=True)
    top = ranked[0]
    assert top.start == 40 and top.hud_events == ["pentakill"]
    assert top.score == pytest.approx(0.5 * (1 + 1.0))   # 1.0 > 0.8
    assert top.score > ranked[1].score


def test_hud_failure_never_blocks_ranking():
    # The real scan_window is fail-safe: with no OCR backend in the test env it returns
    # [] rather than raising, so ranking still yields the audio-only candidate.
    wins = [_win(0, 18, 0.7)]
    ranked = clips.rank_windows("x.mp4", wins, hud_enabled=True)
    assert len(ranked) == 1 and ranked[0].score == 0.7 and ranked[0].hud_events == []


# ---- manifest round-trip ----------------------------------------------------

def test_manifest_has_the_fields_the_gui_needs_and_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(clips.gconf, "GAMEPLAY_DIR", tmp_path)
    session = AutoSession("vod1")
    c = clips.HighlightClip(12.0, 30.0, 0.82, 0.4, 1.148, ["doublekill"], [20.0],
                            rank=1, clip_name="vod1_12s",
                            source_path="s.mp4", clip_path="r.mp4", preview_path="p.jpg")
    clips.write_manifest(session, [c])

    data = json.loads(session.candidates_path.read_text(encoding="utf-8"))
    for key in ("rank", "score", "start", "end", "duration", "audio_score",
                "hud_boost", "hud_events", "clip_name", "clip_path", "preview_path",
                "why"):
        assert key in data[0], key
    assert data[0]["why"].startswith("reaction 0.82")
    back = clips.load_manifest(session)
    assert back[0].clip_name == "vod1_12s" and back[0].hud_events == ["doublekill"]
    assert back[0].score == 1.148


# ---- friendly empty (no crash) ----------------------------------------------

def test_no_windows_returns_empty_with_message(tmp_path, monkeypatch):
    monkeypatch.setattr(clips.gconf, "GAMEPLAY_DIR", tmp_path)
    monkeypatch.setattr(clips, "detect_windows", lambda *a, **k: ([], 0.0))
    msgs = []
    video = tmp_path / "vid.mp4"
    video.write_bytes(b"x")                              # exists; detect is stubbed
    out, session = clips.run_highlight_detection(video, progress=msgs.append)
    assert out == []
    assert any("REACTION_THRESHOLD" in m for m in msgs)  # tells user how to fix
    assert session.candidates_path.read_text() == "[]"


# ---- real export: a generous raw 9:16 clip via the shared reframe/encode -----

def _probe_wh(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True).stdout.strip()
    w, h = out.split(",")[:2]
    return int(w), int(h)


def test_export_clip_produces_a_916_clip(tmp_path, monkeypatch):
    monkeypatch.setattr(clips.gconf, "GAMEPLAY_DIR", tmp_path)
    src = tmp_path / "long.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=30:duration=6",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-shortest",
         "-pix_fmt", "yuv420p", str(src)], check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    session = AutoSession("long")
    c = clips.HighlightClip(1.0, 5.0, 0.7)
    c.rank = 1
    clips.export_clip(src, session, c, 0)
    assert c.clip_name and c.clip_path
    w, h = _probe_wh(c.clip_path)
    assert (w, h) == (clips.gconf.WIDTH, clips.gconf.HEIGHT) == (1080, 1920)
