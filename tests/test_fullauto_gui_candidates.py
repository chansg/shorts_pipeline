"""GUI front-end for the candidate-export batch: input resolution, the per-source
summary table, the threaded streaming handler (live progress + Run-button disable/
re-enable + one-bad-source continues), and the open-folder action. No Gradio server /
ffmpeg — run_batch is mocked; the handler is driven as a plain generator."""
from pathlib import Path

from fullauto import gui as fa_gui
from fullauto import candidates as cand_mod


def _interactive(update):
    """Extract the `interactive` flag from a gr.update() result (dict or object)."""
    if isinstance(update, dict):
        return update.get("interactive")
    return getattr(update, "interactive", None)


# ---- input resolution ------------------------------------------------------

def test_resolve_inputs_combines_files_and_folder(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "b.mkv").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    picked = str(tmp_path / "a.mp4")
    out = fa_gui._resolve_cand_inputs([picked], str(tmp_path))
    # folder contributes its videos, picked file de-dups, .txt filtered out
    assert sorted(p.name for p in out) == ["a.mp4", "b.mkv"]


def test_resolve_inputs_empty_is_empty():
    assert fa_gui._resolve_cand_inputs(None, "") == []
    assert fa_gui._resolve_cand_inputs([], "   \n  ") == []


# ---- per-source summary table ----------------------------------------------

def test_summary_rows_from_manifests(tmp_path):
    results = [
        (tmp_path / "s1", {"source": "s1.mp4", "candidates": [
            {"rank": 1, "category": "play", "peak_s": 78.4, "score": 0.91,
             "why": "Pentakill banner + reaction"},
            {"rank": 2, "category": "banter", "peak_s": 12.0, "score": 0.4,
             "why": "sustained squad reaction"}]}),
        (tmp_path / "s2", {"source": "s2.mp4", "candidates": [],
                           "note": "No qualifying highlight regions."}),
    ]
    rows = fa_gui._cand_summary_rows(results)
    assert rows[0][0] == "s1.mp4 (2)" and rows[0][2] == "play" and "1m18s" in rows[0][3]
    assert rows[1][0] == "" and rows[1][1] == "2"            # 2nd candidate, blank source
    assert rows[2][0] == "s2.mp4 (0)" and "No qualifying" in rows[2][4]


# ---- threaded streaming handler --------------------------------------------

def test_handler_streams_progress_and_toggles_button(tmp_path, monkeypatch):
    (tmp_path / "a.mp4").write_bytes(b"")

    def fake_run_batch(inputs, out_root, *, progress=None):
        progress("=== a.mp4 ===")
        progress("  extracting voice track")
        progress("  exporting clip 1/1")
        return [(Path(out_root) / "a", {"source": "a.mp4", "candidates": [
            {"rank": 1, "category": "banter", "peak_s": 5.0, "score": 0.5, "why": "loud"}]})]

    monkeypatch.setattr(cand_mod, "run_batch", fake_run_batch)
    gen = fa_gui._do_candidate_export([str(tmp_path / "a.mp4")], "", str(tmp_path / "out"))
    frames = list(gen)

    # first frame disables Run, last frame re-enables it
    assert _interactive(frames[0][2]) is False
    assert _interactive(frames[-1][2]) is True
    # the streamed stage lines reached the log live
    final_log = frames[-1][0]
    assert "extracting voice track" in final_log and "exporting clip 1/1" in final_log
    assert "Done -" in final_log
    # final frame carries the summary rows
    assert frames[-1][1][0][2] == "banter"


def test_handler_no_inputs_prompts_and_keeps_button(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(cand_mod, "run_batch",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    frames = list(fa_gui._do_candidate_export(None, "", ""))
    assert called["n"] == 0                                  # never started the batch
    assert len(frames) == 1 and _interactive(frames[0][2]) is True
    assert "Pick one or more" in frames[0][0]


def test_handler_survives_run_batch_failure(tmp_path, monkeypatch):
    (tmp_path / "a.mp4").write_bytes(b"")

    def boom(*a, **k):
        raise RuntimeError("disk gone")

    monkeypatch.setattr(cand_mod, "run_batch", boom)
    frames = list(fa_gui._do_candidate_export([str(tmp_path / "a.mp4")], "", str(tmp_path)))
    # never hangs; logs the failure; Run button re-enabled
    assert "FAILED: RuntimeError" in frames[-1][0]
    assert _interactive(frames[-1][2]) is True


# ---- open output folder ----------------------------------------------------

def test_open_folder_invokes_opener(tmp_path, monkeypatch):
    opened = {}
    monkeypatch.setattr(fa_gui.os, "startfile", lambda p: opened.setdefault("p", p),
                        raising=False)
    msg = fa_gui._open_cand_folder(str(tmp_path / "cand"))
    assert "Opened" in msg and (tmp_path / "cand").exists()   # created if missing
    assert opened["p"].endswith("cand")
