"""Full-auto candidate-export stage: pure scoring/selection/matching/manifest (no
ffmpeg) + the ffmpeg-gated acceptance checks — both audio tracks preserved on export,
voice-track extraction, and a full run completing on a synthetic source."""
import json
import shutil
import subprocess
import wave

import numpy as np
import pytest

from gameplay import config as gconf
from fullauto import candidates as cand
from fullauto.candidates import Candidate, OcrEvent

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


# ---- PURE: fuzzy keyword matching ------------------------------------------

def test_match_keyword_exact_and_fuzzy():
    assert cand.match_keyword("PENTAKILL", ["Pentakill"])[0] == "Pentakill"
    assert cand.match_keyword("you got a triple kill!", ["Triple Kill"])[0] == "Triple Kill"
    # Tesseract misread: 'Pemtaklll' should still fuzzy-match Pentakill
    m = cand.match_keyword("Pemtaklll", ["Pentakill", "Double Kill"], 0.7)
    assert m and m[0] == "Pentakill"


def test_match_keyword_rejects_below_threshold():
    assert cand.match_keyword("loading screen", ["Pentakill", "Ace"], 0.8) is None
    assert cand.match_keyword("", ["Ace"]) is None


def test_match_keyword_exact_substring_beats_fuzzy_neighbour():
    # 'triple kill' present verbatim must not be mis-snapped to the similar 'Double Kill'
    assert cand.match_keyword("TRIPLE KILL", ["Double Kill", "Triple Kill"])[0] == "Triple Kill"


# ---- PURE: window framing --------------------------------------------------

def test_frame_window_targets_60_90_and_clamps():
    s, e = cand.frame_window(600.0, 1200.0, 60, 90)
    assert 60 <= (e - s) <= 90 and 0 <= s < e <= 1200
    # peak near the start clamps to 0 without going negative
    s0, e0 = cand.frame_window(2.0, 1200.0, 60, 90)
    assert s0 == 0.0 and 60 <= (e0 - s0) <= 90
    # peak near the end clamps to the duration
    s1, e1 = cand.frame_window(1199.0, 1200.0, 60, 90)
    assert e1 == 1200.0 and 60 <= (e1 - s1) <= 90


def test_frame_window_short_source_is_whole_clip():
    assert cand.frame_window(20.0, 40.0, 60, 90) == (0.0, 40.0)


# ---- PURE: interest + selection --------------------------------------------

def _grid(n, dt=0.5):
    return (np.arange(n) + 0.5) * dt


def test_build_interest_bumps_ocr_bins():
    times = _grid(10)
    v = np.zeros(10)
    interest = cand.build_interest(times, v, [OcrEvent(2.4, "ACE", "Ace")],
                                   weight_voice=1.0, weight_ocr=1.5)
    hot = int(np.argmax(interest))
    assert abs(times[hot] - 2.4) < 0.5 and interest[hot] == pytest.approx(1.5)


def test_select_top_n_nonoverlap_min_gap_and_category():
    # two voice peaks far apart + one OCR event near the first -> first is `play`
    n = 1200                                  # 600s at 0.5s bins
    times = _grid(n)
    v = np.zeros(n)
    v[200] = 1.0                              # voice peak at 100s
    v[1000] = 0.9                             # voice peak at 500s
    ocr = [OcrEvent(float(times[200]), "PENTAKILL", "Pentakill")]
    interest = cand.build_interest(times, v, ocr)
    cands = cand.select_candidates(times, interest, v, ocr, duration=600.0,
                                   n=5, min_gap=120, min_s=60, max_s=90)
    assert len(cands) == 2
    assert cands[0].category == "play" and cands[0].ocr_events           # OCR-driven first
    assert cands[1].category == "banter"
    # non-overlapping + spaced
    assert cands[0].end <= cands[1].start or cands[1].end <= cands[0].start
    assert abs(cands[0].peak - cands[1].peak) >= 120
    # play peak snapped to the banner; voice score recorded
    assert abs(cands[0].peak - 100.0) < 1.0
    assert cands[0].voice_energy_score > 0


def test_select_respects_min_gap_collapsing_adjacent_peaks():
    n = 400
    times = _grid(n)
    v = np.zeros(n)
    v[100] = 1.0          # 50s
    v[110] = 0.95         # 55s — within MIN_GAP of the first, must be dropped
    interest = cand.build_interest(times, v, [])
    cands = cand.select_candidates(times, interest, v, [], duration=200.0,
                                   n=5, min_gap=120, min_s=60, max_s=90)
    assert len(cands) == 1


def test_select_floors_weak_peaks():
    times = _grid(50)
    v = np.full(50, 0.05)                     # all below the reaction threshold
    interest = cand.build_interest(times, v, [])
    assert cand.select_candidates(times, interest, v, [], duration=25.0,
                                  floor=0.35) == []


# ---- PURE: filename + manifest + encoder args ------------------------------

def test_candidate_filename():
    c = Candidate(1, "play", 0.9, 41.0, 116.0, 78.4, 0.6)
    assert cand.candidate_filename(c) == "clip_01_play_1m18s.mp4"


def test_manifest_dict_schema():
    c = Candidate(1, "play", 0.91, 41.0, 116.0, 78.4, 0.62,
                  ocr_events=[OcrEvent(79.1, "PENTAKILL", "Pentakill")],
                  output="clip_01_play_1m18s.mp4", why="Pentakill banner + reaction")
    d = cand.manifest_dict("2026-06-24 18-09-40.mp4", 83.3, 2, [c], note="x")
    assert d["source"] == "2026-06-24 18-09-40.mp4"
    assert d["audio_tracks"] == 2 and d["note"] == "x"
    cd = d["candidates"][0]
    assert set(cd) >= {"rank", "category", "score", "start_s", "end_s", "duration_s",
                       "peak_s", "voice_energy_score", "ocr_events", "output", "why"}
    assert cd["ocr_events"] == [{"t_s": 79.1, "text": "PENTAKILL"}]


def test_video_args_encoder_quality():
    assert cand._video_args("h264_nvenc", 20) == [
        "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "20", "-pix_fmt", "yuv420p"]
    x264 = cand._video_args("libx264", 18)
    assert "-crf" in x264 and "18" in x264 and "libx264" in x264


# ---- collect_inputs --------------------------------------------------------

def test_collect_inputs_dir_list_and_dedup(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "b.mkv").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    from_dir = cand.collect_inputs(tmp_path)
    assert [p.name for p in from_dir] == ["a.mp4", "b.mkv"]        # ext-filtered, sorted
    mixed = cand.collect_inputs([tmp_path, tmp_path / "a.mp4"])
    assert [p.name for p in mixed] == ["a.mp4", "b.mkv"]           # de-duplicated


# ---- ocr_scan: per-frame fail-safe + unavailable ---------------------------

def test_ocr_scan_skips_bad_frame_keeps_rest():
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    calls = {"n": 0}

    def flaky(crop):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("corrupt frame")
        return "PENTAKILL"

    evs = cand.ocr_scan("x.mp4", 4.0, frames=[(1.0, arr), (2.0, arr), (3.0, arr)],
                        recognizer=flaky, crop=None)
    assert [(e.text, e.t) for e in evs] == [("PENTAKILL", 1.0), ("PENTAKILL", 3.0)]


def test_ocr_scan_audio_only_when_tesseract_missing(monkeypatch):
    monkeypatch.setattr(cand.hud_mod, "ocr_available", lambda: False)
    log = []
    assert cand.ocr_scan("x.mp4", 10.0, progress=log.append) == []
    assert any("audio-only" in m.lower() for m in log)


# ---- ffmpeg: synthetic sources ---------------------------------------------

def _dominant_hz(wav_path):
    with wave.open(str(wav_path), "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(float)
    spec = np.abs(np.fft.rfft(data * np.hanning(len(data))))
    return float(np.fft.rfftfreq(len(data), 1 / sr)[int(np.argmax(spec))])


def _two_track(path, dur=8):
    # v + a:0 (300Hz steady "mix") + a:1 (500Hz "voice", quiet then loud burst at t=dur/2)
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=15:duration={dur}",
         "-f", "lavfi", "-i", f"sine=frequency=300:duration={dur}",
         "-f", "lavfi", "-i", f"sine=frequency=500:duration={dur}",
         "-filter_complex", f"[2:a]volume='if(gt(t,{dur/2}),1.0,0.04)':eval=frame[voice]",
         "-map", "0:v", "-map", "1:a", "-map", "[voice]",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         str(path)], check=True, capture_output=True)
    return path


def _one_track(path, dur=8):
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=15:duration={dur}",
         "-f", "lavfi", "-i", f"sine=frequency=400:duration={dur}",
         "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         str(path)], check=True, capture_output=True)
    return path


def _n_audio(path):
    # the acceptance check: how many audio tracks survived the export. (The task's
    # `stream=index -> 0,1` assumes audio-first ordering; with video at index 0 the
    # absolute audio indices are 1,2 — what matters is that BOTH tracks are present.)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return len([x for x in out.split() if x.strip()])


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_export_preserves_both_audio_tracks(tmp_path):
    # HARD acceptance: an exported candidate must keep a:0 AND a:1.
    src = _two_track(tmp_path / "src.mp4")
    c = Candidate(1, "banter", 0.5, 1.0, 5.0, 2.0, 0.5)
    out = cand.export_candidate(src, c, tmp_path / "o.mp4", tracks=2,
                                encoder="libx264", quality=30)
    assert _n_audio(out) == 2


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_export_single_track_source_keeps_one(tmp_path):
    src = _one_track(tmp_path / "src.mp4")
    c = Candidate(1, "banter", 0.5, 1.0, 5.0, 2.0, 0.5)
    out = cand.export_candidate(src, c, tmp_path / "o.mp4", tracks=1,
                                encoder="libx264", quality=30)
    assert _n_audio(out) == 1


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_extract_voice_wav_picks_track_2(tmp_path):
    src = _two_track(tmp_path / "src.mp4")
    wav, tracks = cand.extract_voice_wav(src, tmp_path / "v.wav")
    assert tracks == 2 and wav.exists()
    assert abs(_dominant_hz(wav) - 500) < 60          # a:1 voice (500Hz), not a:0 mix (300)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_run_source_end_to_end(tmp_path, monkeypatch):
    # Full run completes on a real 2-track source: writes candidates.json + a clip that
    # keeps BOTH audio tracks. (OCR runs over blank frames -> no banners -> banter.)
    monkeypatch.setattr(gconf, "OCR_SAMPLE_FPS", 1.0)
    src = _two_track(tmp_path / "2026-06-24 18-09-40.mp4")
    out_dir, manifest = cand.run_source(src, tmp_path / "cand", progress=lambda m: None)
    assert (out_dir / "candidates.json").exists()
    assert manifest["audio_tracks"] == 2
    data = json.loads((out_dir / "candidates.json").read_text("utf-8"))
    assert data["source"] == "2026-06-24 18-09-40.mp4"
    assert len(data["candidates"]) >= 1
    c0 = data["candidates"][0]
    assert c0["category"] == "banter"                 # no OCR banner in the testsrc
    clip = out_dir / c0["output"]
    assert clip.exists() and _n_audio(clip) == 2   # both tracks in the export
