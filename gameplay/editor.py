"""Fast, keyboard-driven transcript editor (custom HTML/JS) for the gameplay gate.

WHY: the click-to-edit Gradio Dataframe is mouse-bound — double-click each cell to fix a
word, re-assign speakers row by row — which caps how many clips ship. This replaces it
with a row-list editor built for CORRECTING ASR fast:

  - whole-row text field; Enter commits + advances, Up/Down walk rows (no per-cell click)
  - Alt+1..N set the speaker of the active row OR a shift-selected range in ONE action
  - Alt+B set the speaker for all rows below (diariser often flips mid-clip)
  - Alt+D delete row; inline speaker colour chip; click 🔇 to censor

The DATA CONTRACT IS UNCHANGED — rows are `[text, speaker, start, end, censor]` — so
gameplay.editing's helpers, Transcript.from_rows, the caption preview, and the build all
keep working untouched. Client state commits to a hidden bridge textbox ONLY on commit
(blur / Enter / a speaker or row op), never per keystroke, so there is no change-event
echo loop. The controller is installed once via demo.load(js=SETUP_JS); a MutationObserver
rebuilds the editor whenever the server re-renders its data (load / bulk-op / reload).
"""
from __future__ import annotations

import html
import json

from modules.karaoke_captions import DEFAULT_SPEAKER_PALETTE

COMMIT_ELEM_ID = "tx-commit"          # hidden button the editor clicks to commit
ROOT_ELEM_ID = "tx-root"

# JS that reads the live editor rows straight from the DOM — used as the `js=` of the
# hidden commit button, so the committed rows reach Python via the button event's return
# (Gradio 6 does NOT react to a programmatic textbox value-set, but a real button click +
# its js reader is reliable). Maps the prior rows_state input -> [rows_json].
READ_ROWS_JS = ("(prior) => { const r = document.querySelector('#tx-root'); "
                "return [ (r && r.__rows) ? JSON.stringify(r.__rows) : '' ]; }")


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def speaker_colors(rows, spk_rows=None) -> dict:
    """speaker -> hex colour: an explicit hex from the colour grid wins, otherwise a
    palette colour assigned in order of first appearance (matches the caption renderer)."""
    explicit: dict[str, str] = {}
    for r in (spk_rows or []):
        r = list(r) + ["", ""]
        name = str(r[0] or "").strip()
        hexv = str(r[1] or "").strip()
        if name and hexv:
            explicit[name] = hexv if hexv.startswith("#") else "#" + hexv
    order: list[str] = []
    for row in (rows or []):
        s = str((list(row) + ["", ""])[1] or "").strip()
        if s and s not in order:
            order.append(s)
    pal = list(DEFAULT_SPEAKER_PALETTE)
    out: dict[str, str] = {}
    for i, name in enumerate(order):
        out[name] = explicit.get(name, _hex(pal[i % len(pal)]))
    for name, hx in explicit.items():           # explicit speakers not yet seen in text
        out.setdefault(name, hx)
    return out


def normalize_rows(rows) -> list[list]:
    """Coerce to [text, speaker, start, end, censor] — the editor/caption row shape."""
    out = []
    for r in (rows or []):
        r = list(r) + ["", "", 0.0, 0.0, False]
        out.append([str(r[0] or ""), str(r[1] or "").strip(), r[2], r[3], bool(r[4])])
    return out


def render_editor(rows, spk_rows=None) -> str:
    """Build the editor HTML for a gr.HTML value. Carries the rows + speaker colours as
    escaped data-attributes; the SETUP_JS controller turns it into the interactive list."""
    rows = normalize_rows(rows)
    colors = speaker_colors(rows, spk_rows)
    data = html.escape(json.dumps(rows), quote=True)
    spk = html.escape(json.dumps(colors), quote=True)
    legend = "".join(
        f'<span class="txe-leg"><b style="background:{html.escape(c)}"></b>'
        f'Alt+{i + 1} {html.escape(n)}</span>'
        for i, (n, c) in enumerate(colors.items()))
    if not rows:
        return ('<div class="txe-wrap"><div class="txe-help">Transcribe a clip to '
                'populate the fast editor.</div></div>')
    return (
        '<div class="txe-wrap">'
        + _STYLE
        + '<div class="txe-help">'
        '<b>↑/↓</b> or <b>Enter</b> move · type to fix the word · '
        '<b>Alt+1…N</b> set speaker (active row, or a shift-selected range) · '
        '<b>Alt+B</b> speaker to all below · <b>Alt+D</b> delete row · '
        'click a colour chip to select · click 🔇 to censor. '
        '<span id="txe-selinfo"></span></div>'
        f'<div class="txe-legend">{legend}</div>'
        f'<div id="{ROOT_ELEM_ID}" data-rows="{data}" data-speakers="{spk}"></div>'
        '</div>'
    )


def parse_bridge(payload):
    """Parse the committed rows JSON from the editor bridge into the row shape. Returns
    None on junk/empty payload (the caller then keeps its prior state) and a (possibly
    empty) row list for valid JSON — so deleting every row is distinguishable from junk."""
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except (TypeError, ValueError):
        return None
    return normalize_rows(data) if isinstance(data, list) else None


_STYLE = """<style>
.txe-wrap{font-family:ui-monospace,Consolas,monospace;}
.txe-help{font-size:12px;opacity:.8;margin:2px 0 6px;}
.txe-legend{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:6px;font-size:12px;}
.txe-leg b{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:4px;vertical-align:middle;}
#tx-root .tx-row{display:flex;align-items:center;gap:6px;padding:1px 2px;border-radius:4px;}
#tx-root .tx-row.active{background:rgba(120,120,255,.18);}
#tx-root .tx-row.sel{background:rgba(120,120,255,.30);}
#tx-root .tx-chip{flex:0 0 auto;width:34px;text-align:center;border-radius:4px;color:#000;
  font-weight:700;font-size:11px;cursor:pointer;user-select:none;padding:2px 0;}
#tx-root .tx-text{flex:1 1 auto;background:#0d1117;color:#e6edf3;border:1px solid #30363d;
  border-radius:4px;padding:3px 6px;font:inherit;}
#tx-root .tx-row.active .tx-text{border-color:#7d88ff;}
#tx-root .tx-spk{flex:0 0 auto;width:120px;font-size:11px;opacity:.7;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap;}
#tx-root .tx-cz{flex:0 0 auto;width:20px;text-align:center;cursor:pointer;opacity:.45;}
#tx-root .tx-cz.on{opacity:1;}
</style>"""


# Installed once via demo.load(js=SETUP_JS). Vanilla JS, no CDN. Idempotent.
SETUP_JS = r"""
() => {
  if (window.__txe_installed) return;
  window.__txe_installed = true;

  function commit(root){
    // click the hidden commit button; its js reader pulls root.__rows from the DOM and
    // returns it to Python. (Setting a Gradio textbox value programmatically does NOT
    // trigger an update in Gradio 6, but a real button click does.)
    const b = document.querySelector('#tx-commit button') || document.querySelector('button#tx-commit');
    if (b) b.click();
  }
  function rowsEls(root){ return root.querySelectorAll('.tx-row'); }
  function inputAt(root,i){ return root.querySelector('.tx-text[data-i="'+i+'"]'); }
  function setActive(root,i){ root.__active=i; rowsEls(root).forEach(el=>el.classList.toggle('active', +el.dataset.i===i)); }
  function focusRow(root,i){ const el=inputAt(root,i); if(el){ el.focus(); try{el.setSelectionRange(el.value.length,el.value.length);}catch(e){} setActive(root,i);} }
  function updateSel(root){
    rowsEls(root).forEach(el=>el.classList.toggle('sel', root.__sel.has(+el.dataset.i)));
    const info=document.getElementById('txe-selinfo');
    if(info) info.textContent = root.__sel.size ? (root.__sel.size+' row(s) selected') : '';
  }
  function selectRow(root,i,extend){
    if(extend && root.__active!=null){ root.__sel=new Set(); const a=Math.min(root.__active,i),b=Math.max(root.__active,i); for(let k=a;k<=b;k++) root.__sel.add(k); }
    else { root.__sel.has(i)?root.__sel.delete(i):root.__sel.add(i); }
    setActive(root,i); updateSel(root);
  }
  function setSpeaker(root,targets,idx){
    const name=root.__spk[idx]; if(name==null) return;
    targets.forEach(i=>{ if(root.__rows[i]) root.__rows[i][1]=name; });
    render(root); commit(root);
  }
  function onKey(root,e,i){
    if(e.key==='Enter'){ e.preventDefault(); root.__rows[i][0]=e.target.value; commit(root); if(i+1<root.__rows.length) focusRow(root,i+1); }
    else if(e.key==='ArrowDown' && !e.shiftKey){ e.preventDefault(); if(i+1<root.__rows.length) focusRow(root,i+1); }
    else if(e.key==='ArrowUp' && !e.shiftKey){ e.preventDefault(); if(i>0) focusRow(root,i-1); }
    else if(e.altKey && e.key>='1' && e.key<='9'){ e.preventDefault(); const t=root.__sel.size?[...root.__sel]:[i]; setSpeaker(root,t,(+e.key)-1); root.__sel.clear(); }
    else if(e.altKey && (e.key==='b'||e.key==='B')){ e.preventDefault(); const s=root.__rows[i][1]; for(let k=i;k<root.__rows.length;k++) root.__rows[k][1]=s; render(root); commit(root); focusRow(root,i); }
    else if(e.altKey && (e.key==='d'||e.key==='D')){ e.preventDefault(); root.__rows.splice(i,1); render(root); commit(root); focusRow(root,Math.min(i,root.__rows.length-1)); }
  }
  function render(root){
    let list=root.querySelector('.tx-list');
    if(!list){ list=document.createElement('div'); list.className='tx-list'; root.appendChild(list); }
    list.innerHTML='';
    root.__rows.forEach((r,i)=>{
      const spk=r[1]||''; const color=(root.__colors[spk])||'#7d8590';
      const row=document.createElement('div'); row.className='tx-row'; row.dataset.i=i;
      const chip=document.createElement('span'); chip.className='tx-chip'; chip.style.background=color;
      chip.textContent=(spk||'·').replace(/^SPEAKER_?/i,'S')||'·'; chip.title=spk||'(no speaker) — click to select';
      chip.onmousedown=(e)=>{ e.preventDefault(); selectRow(root,i,e.shiftKey); };
      const inp=document.createElement('input'); inp.className='tx-text'; inp.type='text'; inp.value=r[0]; inp.dataset.i=i;
      inp.addEventListener('keydown',(e)=>onKey(root,e,i));
      inp.addEventListener('input',()=>{ root.__rows[i][0]=inp.value; });
      inp.addEventListener('blur',()=>{ root.__rows[i][0]=inp.value; commit(root); });
      inp.addEventListener('focus',()=>setActive(root,i));
      const sp=document.createElement('span'); sp.className='tx-spk'; sp.textContent=spk;
      const cz=document.createElement('span'); cz.className='tx-cz'+(r[4]?' on':''); cz.textContent=r[4]?'🔇':'·'; cz.title='censor (click)';
      cz.onclick=()=>{ root.__rows[i][4]=!root.__rows[i][4]; cz.className='tx-cz'+(root.__rows[i][4]?' on':''); cz.textContent=root.__rows[i][4]?'🔇':'·'; commit(root); };
      row.append(chip,inp,sp,cz); list.appendChild(row);
    });
    updateSel(root);
  }
  function build(root){
    try{ root.__rows=JSON.parse(root.dataset.rows||'[]'); }catch(e){ root.__rows=[]; }
    try{ root.__colors=JSON.parse(root.dataset.speakers||'{}'); }catch(e){ root.__colors={}; }
    root.__spk=Object.keys(root.__colors); root.__sel=new Set(); root.__active=null;
    render(root);
  }
  function scan(){
    document.querySelectorAll('#tx-root').forEach(root=>{
      if(root.dataset.rows !== root.__seen){ root.__seen=root.dataset.rows; build(root); }
    });
  }
  new MutationObserver(scan).observe(document.body,{subtree:true,childList:true,attributes:true,attributeFilter:['data-rows']});
  scan();
  setInterval(scan, 1000);
}
"""
