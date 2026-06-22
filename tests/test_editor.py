"""Fast transcript editor — the Python render/parse/colour transforms + round-trip and
caption-equivalence. The JS controller is browser-verified separately; here we lock the
data contract: rows -> render_editor -> (commit) -> parse_bridge -> rows is lossless, and
the editor feeds the SAME caption output as the rows did before."""
import html
import json
import re

from gameplay import editor as ed
from gameplay.transcript import Transcript


def _embedded_rows(html_str):
    """Pull the data-rows JSON back out of the rendered editor HTML (what the JS reads)."""
    m = re.search(r'data-rows="([^"]*)"', html_str)
    return json.loads(html.unescape(m.group(1)))


def test_render_embeds_rows_and_speaker_colours():
    rows = [["hey", "SPEAKER_00", 0.0, 0.4, False],
            ["nice", "SPEAKER_01", 0.4, 0.9, True]]
    out = ed.render_editor(rows, [["SPEAKER_00", "#ff0000"]])
    assert 'id="tx-root"' in out
    assert _embedded_rows(out) == ed.normalize_rows(rows)     # rows survive the embed
    colors = json.loads(html.unescape(re.search(r'data-speakers="([^"]*)"', out).group(1)))
    assert colors["SPEAKER_00"] == "#ff0000"                 # explicit hex wins
    assert colors["SPEAKER_01"].startswith("#")              # palette colour assigned


def test_render_empty_is_placeholder():
    assert "populate the fast editor" in ed.render_editor([])


def test_parse_bridge_roundtrips_and_is_tolerant():
    rows = [["go", "S0", 1.0, 1.4, False], ["clutch", "S1", 1.5, 2.0, True]]
    payload = json.dumps(rows)
    assert ed.parse_bridge(payload) == ed.normalize_rows(rows)
    assert ed.parse_bridge("[]") == []                       # delete-all -> valid empty
    assert ed.parse_bridge("not json") is None               # junk -> keep prior
    assert ed.parse_bridge("") is None
    assert ed.parse_bridge('{"x":1}') is None                # not a list -> None


def test_render_parse_roundtrip_is_lossless():
    rows = [["a word", "Chan", 0.1, 0.5, False],
            ["another", "Sam", 0.5, 1.2, True]]
    embedded = _embedded_rows(ed.render_editor(rows))         # what the JS receives
    committed = json.dumps(embedded)                          # what the JS commits back
    assert ed.parse_bridge(committed) == ed.normalize_rows(rows)


def test_speaker_colours_palette_by_order():
    rows = [["x", "A", 0, 0.1, False], ["y", "B", 0.1, 0.2, False],
            ["z", "A", 0.2, 0.3, False]]
    colors = ed.speaker_colors(rows)
    assert set(colors) == {"A", "B"} and colors["A"] != colors["B"]


def test_editor_feeds_same_caption_output_as_rows():
    # equivalence: rows that came from the old grid produce identical caption tuples
    # whether read directly or round-tripped through the editor's embed/commit/parse.
    rows = [["nice", "SPEAKER_00", 0.0, 0.4, False],
            ["shot", "SPEAKER_01", 0.4, 0.9, False]]
    before = Transcript.from_rows(rows).to_tuples()
    after = Transcript.from_rows(
        ed.parse_bridge(json.dumps(_embedded_rows(ed.render_editor(rows))))).to_tuples()
    assert before == after


def test_js_and_python_agree_on_element_ids():
    # the render HTML, the controller JS, and the commit reader must reference the same
    # element ids — guard against drift that would silently break commits.
    assert ed.ROOT_ELEM_ID == "tx-root" and ed.COMMIT_ELEM_ID == "tx-commit"
    assert 'id="tx-root"' in ed.render_editor([["a", "S0", 0, 0.1, False]])
    assert "tx-commit" in ed.SETUP_JS                 # commit() clicks the hidden button
    assert "tx-root" in ed.SETUP_JS
    assert "tx-root" in ed.READ_ROWS_JS and "__rows" in ed.READ_ROWS_JS


def test_html_escaping_safe_for_quotes_and_angle_brackets():
    # a transcript word with quotes/brackets must not break the data attribute
    rows = [['he said "go" <now>', "S0", 0.0, 0.4, False]]
    out = ed.render_editor(rows)
    assert _embedded_rows(out)[0][0] == 'he said "go" <now>'  # survives intact
