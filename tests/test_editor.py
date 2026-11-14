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
    pairs = json.loads(html.unescape(re.search(r'data-speakers="([^"]*)"', out).group(1)))
    colors = dict(pairs)
    assert colors["SPEAKER_00"] == "#ff0000"                 # explicit hex wins
    assert colors["SPEAKER_01"].startswith("#")              # palette colour assigned


def test_speakers_are_embedded_as_an_ordered_list():
    # an ordered [[name, colour], ...] list, not an object: JS would put integer-like
    # keys ('2', '420') first and shift the speaker buttons / Alt+N mapping
    out = ed.render_editor([["hi", "SPEAKER_00", 0, 0.4, False]],
                           [["SPEAKER_00", "#ff0000"], ["2", "#00ff00"]])
    pairs = json.loads(html.unescape(re.search(r'data-speakers="([^"]*)"', out).group(1)))
    assert [p[0] for p in pairs] == ["SPEAKER_00", "2"]


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


def test_speaker_colours_grid_order_stable():
    # the colour grid defines the order (and the Alt+N / button mapping); speakers used
    # in the rows but not in the grid are appended after.
    spk = [["SPEAKER_00", "#ff0000"], ["SPEAKER_01", ""], ["SPEAKER_02", ""]]
    rows = [["hi", "SPEAKER_02", 0, 0.1, False], ["yo", "Ghost", 0.1, 0.2, False]]
    colors = ed.speaker_colors(rows, spk)
    assert list(colors)[:3] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]  # grid order
    assert colors["SPEAKER_00"] == "#ff0000"                 # explicit hex kept
    assert colors["SPEAKER_01"].startswith("#")              # palette filled
    assert list(colors)[-1] == "Ghost"                       # extra speaker appended


def test_render_has_clickable_speaker_buttons():
    spk = [["SPEAKER_00", "#ff0000"], ["SPEAKER_01", "#00ff00"]]
    out = ed.render_editor([["hi", "SPEAKER_00", 0, 0.4, False]], spk)
    assert out.count('class="txe-spk"') == 2                 # one button per speaker
    assert 'data-idx="0"' in out and 'data-idx="1"' in out
    assert "--c:#ff0000" in out                              # button carries the colour
    # the controller JS wires the buttons -> setSpeaker
    assert "wireSpeakerButtons" in ed.SETUP_JS and "txe-spk" in ed.SETUP_JS


def test_editor_shows_per_row_start_time():
    # each row renders a read-only start-time label (M:SS.d) so you can scrub the video
    # to it; the timestamps live in the row data (start=r[2], end=r[3]).
    assert ".tx-time" in ed._STYLE                            # the column is styled
    assert "tx-time" in ed.SETUP_JS and "fmtT(" in ed.SETUP_JS  # rendered + formatted
    assert "function fmtT" in ed.SETUP_JS                     # M:SS.d formatter
    # the row still carries start/end in the embedded data (contract unchanged)
    rows = [["hey", "SPEAKER_00", 3.2, 3.8, False]]
    assert _embedded_rows(ed.render_editor(rows))[0][2] == 3.2


def test_editor_self_heals_after_in_place_html_patch():
    # Regression: Gradio 6 patches gr.HTML in place — it keeps #tx-root but strips the
    # injected row list. A colour-only re-render leaves data-rows identical, so the
    # controller must ALSO re-render when the list is missing or data-speakers changed
    # (otherwise recolouring a player blanked the editor).
    js = ed.SETUP_JS
    assert "function restyle" in js
    assert "!root.querySelector('.tx-list')" in js
    assert "root.dataset.speakers !== root.__seenSpk" in js
    assert "attributeFilter:['data-rows','data-speakers','data-rev']" in js


def test_authoritative_loads_carry_a_revision():
    # Regression (review): Revert re-sent rows identical to the last render, so the
    # editor never rebuilt and the build kept the discarded edits. Loads now carry a
    # data-rev the controller always rebuilds on; ordinary re-renders carry none.
    rows = [["hi", "S0", 0.0, 0.4, False]]
    assert 'data-rev="123"' in ed.render_editor(rows, rev=123)
    assert "data-rev" not in ed.render_editor(rows)
    js = ed.SETUP_JS
    assert "rev && rev !== root.__rev" in js


def test_player_rename_relabels_the_editor_in_place():
    # Regression (review): a rename used to re-render the editor (stealing focus from
    # a row just clicked into) and a Build clicked right after could read the old
    # labels. The cards now call this client-side relabel on blur instead.
    js = ed.SETUP_JS
    assert "window.__txeRename=function(oldN,newN)" in js
    # relabels rows/chips/spans/buttons without rebuilding inputs
    rename_body = js.split("window.__txeRename=function(oldN,newN)", 1)[1] \
        .split("new MutationObserver", 1)[0]
    assert "innerHTML" not in rename_body and "render(root)" not in rename_body


def test_render_keeps_the_targeted_row_highlighted():
    assert "if(root.__active!=null && root.__active<root.__rows.length) " \
           "setActive(root,root.__active);" in ed.SETUP_JS


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
