"""Behavioural tests for the transcript editor's browser controller (SETUP_JS), run in
Node against a tiny fake DOM. They cover what string checks can't: speaker-button
order with integer-like names, names that clash with Object.prototype, and carrying
live edits across a remount. Skipped when Node isn't installed."""
import html as html_mod
import json
import re
import shutil
import subprocess

import pytest

from gameplay import editor as ed

pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node not on PATH")

# A minimal DOM: just the selectors and properties the controller uses. The fake
# MutationObserver hands the controller's scan() to the test as `rescan()`.
FAKE_DOM = r"""
class El {
  constructor(tag){ this.tagName=String(tag).toUpperCase(); this.children=[]; this.parent=null;
    this.dataset={}; this.className=''; this.id=''; this._text=''; this.value='';
    this.style={ setProperty(k,v){ this[k]=v; } }; }
  get classList(){ const el=this; return {
    toggle(c,on){ const s=new Set(el.className.split(/\s+/).filter(Boolean));
      if(on===undefined) on=!s.has(c); if(on) s.add(c); else s.delete(c);
      el.className=[...s].join(' '); },
    contains(c){ return el.className.split(/\s+/).includes(c); } }; }
  appendChild(c){ if(c.parent) c.parent.removeChild(c); c.parent=this; this.children.push(c); return c; }
  removeChild(c){ this.children=this.children.filter(x=>x!==c); c.parent=null; return c; }
  append(...cs){ cs.forEach(c=>this.appendChild(c)); }
  set innerHTML(v){ this.children.forEach(c=>c.parent=null); this.children=[]; }
  set textContent(v){ this._text=String(v); this.children=[]; }
  get textContent(){ return this._text+this.children.map(c=>c.textContent).join(''); }
  get lastChild(){ return this.children[this.children.length-1]||null; }
  get isConnected(){ let p=this; while(p.parent) p=p.parent; return p===document.body; }
  addEventListener(){} focus(){} setSelectionRange(){}
  closest(sel){ let p=this; while(p){ if(matches(p,sel)) return p; p=p.parent; } return null; }
  querySelectorAll(sel){ const out=[];
    const walk=e=>e.children.forEach(c=>{ if(matches(c,sel)) out.push(c); walk(c); });
    walk(this); return out; }
  querySelector(sel){ return this.querySelectorAll(sel)[0]||null; }
}
function matches(el,sel){
  let m;
  if((m=/^#([\w-]+)$/.exec(sel))) return el.id===m[1];
  if((m=/^\.([\w-]+)$/.exec(sel))) return el.classList.contains(m[1]);
  if((m=/^\.([\w-]+)\[data-i="(\d+)"\]$/.exec(sel)))
    return el.classList.contains(m[1]) && String(el.dataset.i)===m[2];
  return false;
}
globalThis.document={ body:new El('body'),
  querySelectorAll(s){ return this.body.querySelectorAll(s); },
  querySelector(s){ return this.body.querySelector(s); },
  getElementById(id){ return this.body.querySelector('#'+id); },
  createElement(t){ return new El(t); } };
globalThis.window=globalThis;
globalThis.MutationObserver=class{ constructor(cb){ globalThis.rescan=cb; } observe(){} };
globalThis.setInterval=()=>0;

// mount an editor as render_editor lays it out: wrap > [speaker buttons..., #tx-root]
function mount(attrs){
  const wrap=new El('div'); wrap.className='txe-wrap';
  const n=JSON.parse(attrs.speakers).length;
  for(let i=0;i<n;i++){
    const b=new El('button'); b.className='txe-spk'; b.dataset.idx=String(i);
    const label=new El('#text'); label.textContent=(i+1)+' · (server label)';
    b.append(new El('b'), label); wrap.appendChild(b);
  }
  const root=new El('div'); root.id='tx-root'; Object.assign(root.dataset, attrs);
  wrap.appendChild(root); document.body.appendChild(wrap);
  return {wrap, root};
}
function labels(wrap){ return wrap.querySelectorAll('.txe-spk').map(b=>b.lastChild.textContent); }
function press(wrap, root, row, idx){            // target a row, click a speaker button
  root.__active=row; root.__sel.clear();
  wrap.querySelectorAll('.txe-spk')[idx].onclick();
  return root.__rows[row][1];
}
function report(o){ console.log(JSON.stringify(o)); }
"""


def _attrs(rows, players, rev=None):
    """The data-* attributes render_editor puts on #tx-root (exactly what the JS reads)."""
    out = ed.render_editor(rows, players, rev=rev)
    root = re.search(r'<div id="tx-root"([^>]*)>', out).group(1)
    return {k: html_mod.unescape(v) for k, v in re.findall(r'data-(\w+)="([^"]*)"', root)}


def _run(body: str, **data):
    """Install the controller, then run `body` (which mounts editors and calls
    rescan()); returns the JSON the body report()s last."""
    script = (FAKE_DOM
              + "".join(f"const {k}={json.dumps(v)};\n" for k, v in data.items())
              + f"(eval({json.dumps(ed.SETUP_JS)}))();\n" + body)
    res = subprocess.run(["node", "-e", script], capture_output=True, text=True,
                         encoding="utf-8", timeout=60)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout.strip().splitlines()[-1])


ROWS = [["hi", "SPEAKER_00", 0.0, 0.4, False], ["yo", "2", 0.4, 0.8, False],
        ["gg", "SPEAKER_02", 0.8, 1.2, False]]
PLAYERS = [["SPEAKER_00", "#ff0000"], ["2", "#00ff00"], ["SPEAKER_02", "#0000ff"]]


def test_integer_like_player_names_keep_the_button_order():
    # Regression (review): the speaker order came from Object.keys, which puts '2'
    # first — so button '1 · SPEAKER_00' (and Alt+1) assigned the player named '2'.
    out = _run("""
      const {wrap, root}=mount(A); rescan();
      const first=press(wrap, root, 2, 0), second=press(wrap, root, 0, 1);
      report({spk: root.__spk, first, second, labels: labels(wrap)});
    """, A=_attrs(ROWS, PLAYERS))
    assert out["spk"] == ["SPEAKER_00", "2", "SPEAKER_02"]
    assert out["first"] == "SPEAKER_00" and out["second"] == "2"
    assert out["labels"] == ["1 · SPEAKER_00", "2 · 2", "3 · SPEAKER_02"]


def test_renaming_to_an_integer_like_name_keeps_every_button_in_place():
    out = _run("""
      const {wrap, root}=mount(A); rescan();
      window.__txeRename('SPEAKER_02', '1');
      const got=press(wrap, root, 0, 2);
      report({spk: root.__spk, got, labels: labels(wrap), row2: root.__rows[2][1]});
    """, A=_attrs(ROWS, PLAYERS))
    assert out["spk"] == ["SPEAKER_00", "2", "1"]
    assert out["labels"] == ["1 · SPEAKER_00", "2 · 2", "3 · 1"]
    assert out["got"] == "1" and out["row2"] == "1"


def test_prototype_names_are_ordinary_player_names():
    # Regression (review): `name in colours` on a plain object made 'constructor' or
    # 'toString' look like an existing player, so a rename took the merge path and
    # shifted every speaker button by one.
    rows = [["a", "A", 0, 0.1, False], ["b", "B", 0.1, 0.2, False],
            ["c", "C", 0.2, 0.3, False]]
    players = [["A", "#ff0000"], ["B", "#00ff00"], ["C", "#0000ff"]]
    out = _run("""
      const {wrap, root}=mount(A); rescan();
      window.__txeRename('A', 'constructor');
      report({spk: root.__spk, labels: labels(wrap), b: press(wrap, root, 0, 1),
              colour: root.__colors.get('constructor'), row0: root.__rows[0][1]});
    """, A=_attrs(rows, players))
    assert out["spk"] == ["constructor", "B", "C"]
    assert out["labels"][0] == "1 · constructor"
    assert out["b"] == "B" and out["colour"] == "#ff0000"


def test_merge_keeps_buttons_aligned_until_the_server_refresh():
    out = _run("""
      const {wrap, root}=mount(A); rescan();
      window.__txeRename('SPEAKER_02', 'SPEAKER_00');          // merge
      report({spk: root.__spk, third: press(wrap, root, 1, 2), second: press(wrap, root, 0, 1),
              rows: root.__rows.map(r=>r[1])});
    """, A=_attrs(ROWS, PLAYERS))
    assert out["spk"] == ["SPEAKER_00", "2", "SPEAKER_00"]
    assert out["third"] == "SPEAKER_00" and out["second"] == "2"


def test_a_remounted_editor_keeps_the_live_edits():
    # Regression (review): hiding the editor's container unmounted it; showing it again
    # remounted it from the last SERVER value (no commit re-renders the editor), so the
    # stale rows came back and the next commit wrote them over the user's fixes.
    out = _run("""
      const a=mount(A); rescan();
      a.root.__rows[0][0]='fixed word';
      window.__txeRename('SPEAKER_00', 'Chan');
      document.body.removeChild(a.wrap);                        // unmount
      const b=mount(A); rescan();                               // remount, stale value
      report({text: b.root.__rows[0][0], spk: b.root.__spk, labels: labels(b.wrap),
              same: a.root.__rows===b.root.__rows});
    """, A=_attrs(ROWS, PLAYERS))
    assert out["text"] == "fixed word"
    assert out["spk"][0] == "Chan" and out["labels"][0] == "1 · Chan"


def test_a_new_load_is_never_mistaken_for_a_remount():
    # different rows, or the same rows with a new revision (Revert), rebuild from data
    out = _run("""
      const a=mount(A); rescan();
      a.root.__rows[0][0]='local edit';
      document.body.removeChild(a.wrap);
      const b=mount(B); rescan();                               // a new transcript
      document.body.removeChild(b.wrap);
      b.root.__rows[0][0]='local edit';
      const c=mount(C); rescan();                               // Revert: same rows, new rev
      report({b: b.root.__rows[0][0], c: c.root.__rows[0][0]});
    """, A=_attrs(ROWS, PLAYERS, rev=1),
        B=_attrs([["new", "SPEAKER_00", 0, 0.4, False]], PLAYERS, rev=2),
        C=_attrs([["new", "SPEAKER_00", 0, 0.4, False]], PLAYERS, rev=3))
    assert out["b"] == "local edit"        # b itself was edited after its build...
    assert out["c"] == "new"               # ...but a new rev rebuilds from the data


def test_same_root_patched_in_place_still_rebuilds_on_new_rows():
    out = _run("""
      const a=mount(A); rescan();
      a.root.dataset.rows=B.rows; a.root.dataset.speakers=B.speakers; rescan();
      report({text: a.root.__rows[0][0]});
    """, A=_attrs(ROWS, PLAYERS), B=_attrs([["bulk", "SPEAKER_00", 0, 0.4, False]],
                                          PLAYERS))
    assert out["text"] == "bulk"
