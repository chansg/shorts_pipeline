"""Unified GUI: Dark Souls lore Short, script -> publish-ready 1080x1920 video.

    python app.py

One wizard, eight gated stages:
  1 Script        pick/edit scripts/<name>.txt (one sentence = one scene)
  2 References    REQUIRED style lock + optional character refs (gates stage 3)
  3 Prompts       reference-aware auto-draft -> editable aria-i2v manifest
  4 Stills        Nano Banana 2, gallery, per-scene regenerate
  5 Animate       choose the still/clip hybrid, Veo with cost guard rails
  6 Handoff       auto-place NN.png / NN.mp4 into assets/images (no renaming)
  7 Voice & Build ElevenLabs TTS -> Whisper captions -> ffmpeg assemble
  8 Review        spec check, contact sheet, approve -> ready to publish

Long stages stream progress; nothing blocks the UI. Every stage is idempotent:
existing outputs are skipped, so re-runs only fill gaps (and Veo/ElevenLabs are
never billed twice for the same work).
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

import config
import prompt_gen
from modules import script as script_mod
from orchestrator import manifest_gen, qc, stages
from orchestrator import state as st
from orchestrator.errors import FriendlyError

MAX_SCENES = 12
ROOT = Path(__file__).resolve().parent

STAGE_LABELS = {
    "script": "1 Script", "references": "2 References", "prompts": "3 Prompts",
    "stills": "4 Stills", "animate": "5 Animate", "handoff": "6 Handoff",
    "assemble": "7 Voice & Build", "qc": "8 Review",
}


# ----------------------------------------------------------------- helpers --

def _ep(name: str) -> st.Episode:
    if not name:
        raise gr.Error("Pick or create a script first (stage 1).")
    return st.Episode(name)


def _err(exc: Exception) -> gr.Error:
    from orchestrator.errors import friendly
    return gr.Error(str(friendly(exc)), duration=None)


def status_banner(name: str) -> str:
    if not name:
        return "**No episode selected** — pick a script in stage 1."
    ep = st.Episode(name)
    s = ep.status()
    chain = " → ".join(
        f"{'✅' if s[k] else '⬜'} {STAGE_LABELS[k]}" for k in st.STAGES)
    nxt = ep.first_incomplete()
    tail = ("🎉 **Episode approved and publish-ready.**" if nxt == "done"
            else f"**Next:** {STAGE_LABELS[nxt]}")
    return f"### `{name}`\n{chain}\n\n{tail}"


def script_stats(text: str) -> str:
    sents = prompt_gen.split_sentences(text or "")
    words = len((text or "").split())
    dur = words / 2.8
    note = "" if 5 <= len(sents) <= 10 else "  ⚠ episodes are usually 7–8 scenes"
    return (f"**{len(sents)} sentences → {len(sents)} scenes**{note} · "
            f"{words} words · ≈{dur:.0f}s narrated")


def _require(name: str, stage: str, what: str) -> st.Episode:
    """Enforce the dependency chain at action time."""
    ep = _ep(name)
    if not ep.status()[stage]:
        raise gr.Error(f"Blocked: {what} ({STAGE_LABELS[stage]} is not complete).",
                       duration=None)
    return ep


def _set_env_key(key: str, value: str) -> None:
    """Write KEY=value into the root .env (create/replace), set os.environ.
    The value is never logged or returned."""
    value = (value or "").strip()
    if not value:
        return
    env_path = ROOT / ".env"
    lines = (env_path.read_text(encoding="utf-8").splitlines()
             if env_path.exists() else [])
    out, replaced = [], False
    for ln in lines:
        if ln.split("=", 1)[0].strip() == key:
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(ln)
    if not replaced:
        out.append(f"{key}={value}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    os.environ[key] = value


def _key_status() -> str:
    rows = []
    for key, hint in (("GEMINI_API_KEY", "images + Veo + prompt drafting"),
                      ("ELEVENLABS_API_KEY", "voiceover")):
        val = os.environ.get(key, "")
        rows.append(f"- **{key}** ({hint}): "
                    + (f"✅ set (…{val[-4:]})" if val else "❌ missing"))
    import shutil as _sh
    rows.append("- **ffmpeg**: " + ("✅ on PATH" if _sh.which("ffmpeg")
                else "❌ not found — `winget install Gyan.FFmpeg`"))
    return "\n".join(rows)


# ------------------------------------------------------------ tab loaders --

def load_script_tab(name: str):
    if not name:
        return "", "Pick a script.", status_banner(name)
    ep = st.Episode(name)
    text = ep.script_text()
    return text, script_stats(text), status_banner(name)


def load_refs_tab(name: str):
    files = [str(p.relative_to(ROOT)) for p in st.list_ref_images()]
    if not name:
        return (gr.update(choices=files, value=[]), "_no episode_",
                gr.update(choices=[]))
    ep = st.Episode(name)
    chars = ep.characters()
    char_md = ("\n".join(f"- **{k}**: {', '.join(Path(p).name for p in v)}"
               for k, v in chars.items()) or "_No character refs yet (optional)._")
    return (gr.update(choices=files, value=ep.style_refs()),
            char_md, gr.update(choices=list(chars)))


def _rows_from_manifest(ep: st.Episode):
    """Flat component updates for the MAX_SCENES editor rows."""
    clips = ep.clips()
    chars = list(ep.characters())
    out = []
    for i in range(MAX_SCENES):
        if i < len(clips):
            c = clips[i]
            narr = c.get("narrates") or ""
            out += [
                gr.update(visible=True,
                          label=f"Scene {i+1:02d} — {c['name']}"
                                + ("  🎬 animate" if c.get("animate") else "")),
                gr.update(value=f"*narrates:* {narr}" if narr else ""),
                gr.update(value=c.get("prompt", "")),
                gr.update(value=c.get("motion_prompt", "")),
                gr.update(choices=chars,
                          value=[r for r in c.get("refs", []) if r in chars]),
                gr.update(value=bool(c.get("animate"))),
            ]
        else:
            out += [gr.update(visible=False), gr.update(value=""),
                    gr.update(value=""), gr.update(value=""),
                    gr.update(choices=chars, value=[]), gr.update(value=False)]
    return out


def load_prompts_tab(name: str):
    if not name:
        return [gr.update()] * (MAX_SCENES * 6) + [""]
    ep = st.Episode(name)
    note = ("" if ep.load_manifest()
            else "No prompts yet — set the style lock (stage 2), then generate.")
    return _rows_from_manifest(ep) + [note]


def _stills_gallery(ep: st.Episode):
    items = []
    for i, c in enumerate(ep.clips(), 1):
        p = ep.still_path(c)
        if p.exists():
            items.append((str(p), f"{i:02d} · {c['name']}"))
    return items


def load_stills_tab(name: str):
    if not name:
        return [], gr.update(choices=[]), ""
    ep = st.Episode(name)
    names = [c["name"] for c in ep.clips()]
    done = sum(1 for c in ep.clips() if ep.still_path(c).exists())
    msg = (f"{done}/{len(names)} stills on disk."
           if names else "Generate prompts first (stage 3).")
    return _stills_gallery(ep), gr.update(choices=names), msg


def load_animate_tab(name: str):
    if not name:
        return (gr.update(choices=[], value=[]), gr.update(choices=[]),
                None, "")
    ep = st.Episode(name)
    clips = ep.clips()
    names = [c["name"] for c in clips]
    marked = [c["name"] for c in clips if c.get("animate")]
    rendered = [c["name"] for c in clips if ep.clip_path(c).exists()]
    dur = (ep.load_manifest() or {}).get("defaults", {}).get("duration_seconds", 8)
    msg = (f"{len(marked)} scene(s) marked to animate · {len(rendered)} clip(s) "
           f"already rendered (will be skipped, not re-billed). "
           f"Batch ≈ {len(marked)}×{dur}s of Veo output.")
    return (gr.update(choices=names, value=marked),
            gr.update(choices=names, value=(marked[0] if marked else None)),
            None, msg)


def load_handoff_tab(name: str):
    if not name:
        return "_no episode_"
    ep = st.Episode(name)
    pairs = ep.expected_assets()
    if not pairs:
        return "Generate prompts first (stage 3)."
    lines = ["| # | source | → assets/images/ |", "|---|--------|-----------------|"]
    for src, dest in pairs:
        ok = "" if src.exists() else "  ⚠ missing"
        lines.append(f"| {dest.stem} | `{src.relative_to(ROOT)}`{ok} | `{dest.name}` |")
    done = "✅ Handoff already done for this episode." if ep.handoff_done() else ""
    return "\n".join(lines) + ("\n\n" + done if done else "")


def load_assemble_tab(name: str):
    music = stages.list_music()
    default_music = music[0] if music else None
    if name:
        saved = st.Episode(name).load_state().get("voice_settings") or {}
    else:
        saved = {}
    return (gr.update(value=saved.get("voice_id", config.ELEVENLABS_VOICE_ID)),
            saved.get("stability", config.ELEVENLABS_STABILITY),
            saved.get("similarity", config.ELEVENLABS_SIMILARITY),
            saved.get("style", config.ELEVENLABS_STYLE),
            gr.update(choices=["(no music)"] + music,
                      value=(default_music or "(no music)")),
            "")


def load_qc_tab(name: str):
    if not name:
        return None, "", [], "", "", ""
    ep = st.Episode(name)
    video = str(ep.output_path) if ep.output_path.exists() else None
    state = ep.load_state()
    return (video, "", [], state.get("title", ""), state.get("description", ""),
            "✅ Already approved." if state.get("approved") else "")


# ------------------------------------------------------------- handlers ----

def create_script(new_name: str):
    name = st.slugify(new_name or "", "")
    if not name:
        raise gr.Error("Give the new script a name.")
    path = config.SCRIPTS_DIR / f"{name}.txt"
    if not path.exists():
        path.write_text("", encoding="utf-8")
    return gr.update(choices=st.list_scripts(), value=name), name


def save_script(name: str, text: str):
    ep = _ep(name)
    ep.script_path.write_text((text or "").strip() + "\n", encoding="utf-8")
    return script_stats(text), status_banner(name)


def rewrite_script(name: str, text: str):
    _ep(name)
    if not (text or "").strip():
        raise gr.Error("Nothing to rewrite — the script box is empty.")
    try:
        out = script_mod.rewrite(text.strip())
    except Exception as exc:
        raise _err(exc)
    return out, script_stats(out)


def upload_refs(files: list[str] | None):
    saved = []
    for f in files or []:
        dest = st.REFS_DIR / Path(f).name
        if Path(f).resolve() != dest.resolve():
            import shutil
            shutil.copy2(f, dest)
        saved.append(dest.name)
    files_now = [str(p.relative_to(ROOT)) for p in st.list_ref_images()]
    msg = f"Saved to refs/: {', '.join(saved)}" if saved else ""
    return gr.update(choices=files_now), msg


def save_style_lock(name: str, style_files: list[str]):
    ep = _ep(name)
    valid = [f for f in (style_files or []) if (ROOT / f).exists()]
    ep.save_state(style_refs=valid)
    if not valid:
        return ("⚠ Style lock CLEARED — prompt generation is blocked until at "
                "least one style reference is set."), status_banner(name)
    return (f"✅ Style lock set ({len(valid)} image(s)) — applied to every "
            "scene. Prompt generation is unlocked."), status_banner(name)


def add_character(name: str, char_name: str, files: list[str] | None):
    ep = _ep(name)
    cname = st.slugify(char_name or "", "")
    if not cname:
        raise gr.Error("Give the character a short name (e.g. nito).")
    if not files:
        raise gr.Error("Attach at least one reference image for the character.")
    import shutil
    paths = []
    for f in files:
        dest = st.REFS_DIR / Path(f).name
        if Path(f).resolve() != dest.resolve():
            shutil.copy2(f, dest)
        paths.append(str(dest.relative_to(ROOT)))
    state = ep.load_state()
    chars = state.get("characters", {})
    chars[cname] = sorted(set(chars.get(cname, []) + paths))
    ep.save_state(characters=chars)
    return load_refs_tab(name) + (f"Added character '{cname}'.",)


def remove_character(name: str, cname: str):
    ep = _ep(name)
    state = ep.load_state()
    chars = state.get("characters", {})
    chars.pop(cname, None)
    ep.save_state(characters=chars)
    return load_refs_tab(name) + (f"Removed '{cname}'." if cname else "",)


def generate_prompts(name: str, use_llm: bool, duration: int):
    ep = _require(name, "references",
                  "set the style lock before generating prompts")
    try:
        manifest, notes = manifest_gen.generate_manifest(
            ep, use_llm=use_llm, duration_seconds=int(duration))
    except FriendlyError as fe:
        raise gr.Error(str(fe), duration=None)
    except Exception as exc:
        raise _err(exc)
    ep.save_manifest(manifest)
    n = len(manifest["clips"])
    note = f"Generated {n} scene prompt(s).\n" + "\n".join(notes)
    return _rows_from_manifest(ep) + [note, status_banner(name)]


def save_prompts(name: str, *row_values):
    ep = _ep(name)
    manifest = ep.load_manifest()
    if not manifest:
        raise gr.Error("Generate prompts first.")
    clips = manifest["clips"]
    warnings = []
    for i, c in enumerate(clips):
        prompt_v, motion_v, refs_v, anim_v = row_values[i * 4:i * 4 + 4]
        c["prompt"] = (prompt_v or "").strip()
        c["motion_prompt"] = (motion_v or "").strip() or prompt_gen.DEFAULT_MOTION
        c["refs"] = list(refs_v or [])
        c["animate"] = bool(anim_v)
        motion_to_lint = (c["motion_prompt"]
                          if c["motion_prompt"] != prompt_gen.DEFAULT_MOTION else "")
        hits = prompt_gen.lint(c["prompt"] + " " + motion_to_lint)
        if hits:
            warnings.append(f"Scene {i+1:02d}: filter-risk words -> {', '.join(hits)}")
        if not c["prompt"]:
            warnings.append(f"Scene {i+1:02d}: prompt is EMPTY.")
    ep.save_manifest(manifest)
    msg = f"Saved {len(clips)} scene(s) to {ep.manifest_path.name}."
    if warnings:
        msg += "\n" + "\n".join(warnings)
    return msg, status_banner(name)


def run_stills(name: str, force: bool):
    ep = _require(name, "prompts", "generate + save prompts first")
    clips = ep.clips()
    done = 0
    for i, c in enumerate(clips, 1):
        label = f"[{i}/{len(clips)}] {c['name']}"
        if ep.still_path(c).exists() and not force:
            yield f"{label}: exists — skipped.", _stills_gallery(ep), gr.update()
            continue
        yield f"{label}: generating (Nano Banana 2)...", _stills_gallery(ep), gr.update()
        try:
            res = stages.generate_stills(ep, only=[c["name"]], force=force)
        except FriendlyError as fe:
            raise gr.Error(str(fe), duration=None)
        status = res[0]["status"] if res else "failed"
        if status == "failed":
            err = res[0].get("error", "unknown error")
            yield (f"{label}: FAILED — {err}\n(You can rephrase the prompt in "
                   "stage 3 and re-run; finished stills are kept.)",
                   _stills_gallery(ep), gr.update())
        else:
            done += 1
            yield f"{label}: {status}.", _stills_gallery(ep), gr.update()
    missing = [c["name"] for c in clips if not ep.still_path(c).exists()]
    summary = (f"Done — {done} generated, all {len(clips)} stills present."
               if not missing else
               f"Done — {done} generated, still missing: {', '.join(missing)}.")
    yield summary, _stills_gallery(ep), status_banner(name)


def regen_still(name: str, scene: str):
    ep = _require(name, "prompts", "generate prompts first")
    if not scene:
        raise gr.Error("Pick the scene to regenerate.")
    yield f"Regenerating {scene} (force)...", gr.update()
    try:
        res = stages.generate_stills(ep, only=[scene], force=True)
    except FriendlyError as fe:
        raise gr.Error(str(fe), duration=None)
    status = res[0]["status"] if res else "failed"
    msg = (f"{scene}: regenerated." if status == "generated"
           else f"{scene}: {status} — {res[0].get('error', '')}")
    yield msg, _stills_gallery(ep)


def approve_stills(name: str):
    ep = _require(name, "stills", "every scene needs a still before approval")
    ep.save_state(stills_approved=True)
    return "✅ Stills approved — Animate unlocked.", status_banner(name)


def save_animate_selection(name: str, selected: list[str]):
    ep = _ep(name)
    manifest = ep.load_manifest()
    if not manifest:
        raise gr.Error("Generate prompts first.")
    selected = set(selected or [])
    for c in manifest["clips"]:
        c["animate"] = c["name"] in selected
    ep.save_manifest(manifest)
    return load_animate_tab(name)[3], status_banner(name)


def _check_stills_approved(ep: st.Episode):
    if not ep.load_state().get("stills_approved"):
        raise gr.Error("Approve the stills (stage 4) before animating.",
                       duration=None)


def test_one_clip(name: str, scene: str, force: bool):
    ep = _require(name, "stills", "generate all stills first")
    _check_stills_approved(ep)
    if not scene:
        raise gr.Error("Pick a scene to test.")
    existing = ep.clips_dir / f"{scene}.mp4"
    if existing.exists() and not force:
        yield f"{scene}: already rendered — showing it (not re-billed).", str(existing)
        return
    dur = (ep.load_manifest() or {}).get("defaults", {}).get("duration_seconds", 8)
    yield (f"Submitting {scene} to Veo ({dur}s — this one clip is billed). "
           "Rendering typically takes 1–3 minutes...", None)
    try:
        res = stages.animate(ep, only=[scene], force=force)
    except FriendlyError as fe:
        raise gr.Error(str(fe), duration=None)
    r = res[0]
    if r.status == "failed":
        raise gr.Error(f"{scene} failed: {r.error}", duration=None)
    yield f"{scene}: {r.status} ({r.seconds or '?'}s render).", str(ep.clips_dir / f"{scene}.mp4")


def run_animate_batch(name: str, selected: list[str], confirmed: bool, force: bool):
    ep = _require(name, "stills", "generate all stills first")
    _check_stills_approved(ep)
    selected = list(selected or [])
    save_animate_selection(name, selected)
    to_bill = [s for s in selected
               if force or not (ep.clips_dir / f"{s}.mp4").exists()]
    if to_bill and not confirmed:
        raise gr.Error(
            f"This batch would bill {len(to_bill)} Veo clip(s): "
            f"{', '.join(to_bill)}. Tick the confirmation box to proceed.",
            duration=None)
    done = 0
    for i, scene in enumerate(selected, 1):
        label = f"[{i}/{len(selected)}] {scene}"
        if (ep.clips_dir / f"{scene}.mp4").exists() and not force:
            yield f"{label}: already rendered — skipped (not re-billed).", gr.update()
            continue
        yield f"{label}: rendering with Veo (1–3 min)...", gr.update()
        try:
            res = stages.animate(ep, only=[scene], force=force)
        except FriendlyError as fe:
            raise gr.Error(str(fe), duration=None)
        r = res[0]
        if r.status == "failed":
            yield f"{label}: FAILED — {r.error} (continuing with the rest).", gr.update()
        else:
            done += 1
            yield f"{label}: rendered.", gr.update()
    ep.save_state(animate_confirmed=True)
    missing = [s for s in selected if not (ep.clips_dir / f"{s}.mp4").exists()]
    msg = (f"Batch done — {done} rendered. All selected scenes have clips; "
           "the rest stay Ken Burns stills. Handoff unlocked."
           if not missing else
           f"Batch done — {done} rendered, missing: {', '.join(missing)}. "
           "Re-run to fill gaps (existing clips are never re-billed).")
    yield msg, status_banner(name)


def confirm_no_animate(name: str):
    ep = _require(name, "stills", "generate all stills first")
    _check_stills_approved(ep)
    ep.save_state(animate_confirmed=True)
    return ("Confirmed — this episode ships as Ken Burns stills"
            " (plus any clips already rendered)."), status_banner(name)


def run_handoff(name: str):
    ep = _require(name, "animate",
                  "finish stage 5 first (confirm the animate choice)")
    try:
        placed, warnings = stages.handoff(ep)
    except FriendlyError as fe:
        raise gr.Error(str(fe), duration=None)
    msg = "✅ Handoff complete: " + ", ".join(placed)
    if warnings:
        msg += "\n" + "\n".join(f"⚠ {w}" for w in warnings)
    return msg, load_handoff_tab(name), status_banner(name)


def fetch_voices():
    try:
        voices = stages.list_voices()
    except FriendlyError as fe:
        raise gr.Error(str(fe), duration=None)
    return gr.update(choices=[v[0] for v in voices]), \
        f"{len(voices)} voices on the account. Picking one fills the ID box."


def voice_picked(label: str):
    # label format: "Name (voice_id)"
    if label and "(" in label:
        return label.rsplit("(", 1)[1].rstrip(")")
    return gr.update()


def run_assemble(name: str, voice_id: str, stability: float, similarity: float,
                 style: float, music_choice: str, force: bool):
    ep = _require(name, "handoff", "run the handoff first")
    stages.apply_voice_settings(voice_id, stability, similarity, style)
    music = None if music_choice in (None, "", "(no music)") else Path(music_choice)
    if ep.assemble_done() and not force:
        yield ("Final video already rendered for this script — tick "
               "'force re-render' to redo it.", str(ep.output_path),
               status_banner(name))
        return
    log: list[str] = []
    try:
        for event in stages.voice_assemble(ep, music, force=force):
            if event.get("done"):
                log.append(f"Done -> {event['output']}")
                yield "\n".join(log), str(event["output"]), status_banner(name)
            else:
                log.append(event["msg"])
                yield "\n".join(log), gr.update(), gr.update()
    except FriendlyError as fe:
        raise gr.Error(str(fe), duration=None)


def run_qc(name: str):
    ep = _require(name, "assemble", "render the video first")
    try:
        spec = qc.spec_check(ep.output_path)
        sheet, issues = qc.contact_sheet(ep)
    except FriendlyError as fe:
        raise gr.Error(str(fe), duration=None)
    marks = {True: "✅", False: "❌"}
    c = spec["checks"]
    md = (f"**Spec:** {marks[c['resolution']]} {spec['width']}×{spec['height']} · "
          f"{marks[c['fps']]} {spec['fps']} fps · "
          f"{marks[c['audio']]} audio · "
          f"{marks[c['duration']]} {spec['duration']}s\n\n")
    md += ("**All spec checks passed.**" if spec["ok"]
           else "**⚠ Spec problems — fix before publishing.**")
    if issues:
        md += "\n\n" + "\n".join(f"- ⚠ {i}" for i in issues)
    else:
        md += "\n\n- No near-black scenes detected."
    return md, sheet


def draft_meta(name: str):
    ep = _require(name, "script", "no script")
    title, desc = qc.draft_metadata(ep.script_text())
    return title, desc


def approve(name: str, title: str, desc: str):
    ep = _require(name, "assemble", "render the video first")
    ep.save_state(approved=True, title=title or "", description=desc or "")
    qc.open_output_folder(ep.output_path)
    return (f"🎉 **Approved — ready to publish.**\n\nFile: `{ep.output_path}`\n\n"
            "The output folder has been opened."), status_banner(name)


def save_keys(gemini: str, eleven: str):
    _set_env_key("GEMINI_API_KEY", gemini)
    _set_env_key("ELEVENLABS_API_KEY", eleven)
    return _key_status(), "", ""


# ------------------------------------------------------------------ UI -----

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Shorts Pipeline") as demo:
        gr.Markdown("# 🔥 Lore Shorts Studio\nScript → stills → motion → "
                    "voice → captions → **your approval** → publish-ready.")
        ep_state = gr.State("")
        banner = gr.Markdown(status_banner(""))

        with gr.Tabs():
            # ---------------------------------------------------- 1 Script
            with gr.Tab("1 · Script"):
                with gr.Row():
                    script_dd = gr.Dropdown(choices=st.list_scripts(), value=None,
                                            label="Episode script", scale=3)
                    new_name = gr.Textbox(label="…or new script name", scale=2)
                    new_btn = gr.Button("Create", scale=1)
                script_tb = gr.Textbox(label="Script (one sentence = one scene)",
                                       lines=10)
                stats_md = gr.Markdown("")
                with gr.Row():
                    save_script_btn = gr.Button("💾 Save script", variant="primary")
                    rewrite_btn = gr.Button(
                        f"♻ Rewrite seed → script ({config.REWRITE_BACKEND})")

            # ------------------------------------------------ 2 References
            with gr.Tab("2 · References"):
                gr.Markdown("**Style lock (required)** — applied to every scene. "
                            "Prompts can't be generated until this is set. Put "
                            "images in `refs/` or upload here.")
                ref_upload = gr.File(label="Upload reference image(s) → refs/",
                                     file_count="multiple",
                                     file_types=["image"])
                upload_msg = gr.Markdown("")
                style_dd = gr.Dropdown(choices=[], multiselect=True,
                                       label="Style lock image(s) (refs/style_*.png)")
                style_btn = gr.Button("🔒 Set style lock", variant="primary")
                refs_msg = gr.Markdown("")
                gr.Markdown("---\n**Character / subject refs (optional)** — e.g. "
                            "a boss design kept consistent across scenes.")
                with gr.Row():
                    char_name_tb = gr.Textbox(label="Character name (e.g. nito)")
                    char_files = gr.File(label="Reference image(s)",
                                         file_count="multiple",
                                         file_types=["image"])
                with gr.Row():
                    char_add_btn = gr.Button("Add character")
                    char_del_dd = gr.Dropdown(choices=[], label="Remove…", scale=2)
                    char_del_btn = gr.Button("Remove")
                chars_md = gr.Markdown("")

            # --------------------------------------------------- 3 Prompts
            with gr.Tab("3 · Prompts"):
                gr.Markdown("Auto-draft one still prompt per sentence (+ motion "
                            "prompt) into the i2v manifest — **reference-aware**: "
                            "character refs wire into their scenes, prompts stay "
                            "compositional because the style is locked by refs. "
                            "Then edit everything below.")
                with gr.Row():
                    use_llm_cb = gr.Checkbox(value=True,
                                             label="Draft with Gemini (cheap text "
                                                   "call; off = sentence verbatim)")
                    clip_dur = gr.Slider(4, 8, value=8, step=2,
                                         label="Veo clip duration (s)")
                    gen_prompts_btn = gr.Button("⚡ Generate scene prompts",
                                                variant="primary")
                prompts_note = gr.Markdown("")
                rows = []
                for i in range(MAX_SCENES):
                    with gr.Accordion(f"Scene {i+1:02d}", visible=False,
                                      open=(i < 8)) as acc:
                        sent_md = gr.Markdown("")
                        p_tb = gr.Textbox(label="Still prompt (composition/subject"
                                                " — style comes from the refs)",
                                          lines=2)
                        m_tb = gr.Textbox(label="Motion prompt (used only if "
                                                "animated)", lines=2)
                        with gr.Row():
                            r_cbg = gr.CheckboxGroup(choices=[],
                                                     label="Character refs in "
                                                           "this scene")
                            a_cb = gr.Checkbox(label="Animate with Veo (💰)")
                    rows.append((acc, sent_md, p_tb, m_tb, r_cbg, a_cb))
                row_comps = [c for r in rows for c in r]
                row_inputs = [c for r in rows for c in (r[2], r[3], r[4], r[5])]
                save_prompts_btn = gr.Button("💾 Save prompts", variant="primary")
                prompts_save_md = gr.Markdown("")

            # ---------------------------------------------------- 4 Stills
            with gr.Tab("4 · Stills"):
                gr.Markdown("Nano Banana 2 with the locked style + character "
                            "refs. Idempotent: existing stills are skipped.")
                with gr.Row():
                    stills_force_cb = gr.Checkbox(label="Force regenerate ALL")
                    stills_btn = gr.Button("🖼 Generate stills", variant="primary")
                stills_status = gr.Textbox(label="Progress", lines=4,
                                           interactive=False)
                stills_gallery = gr.Gallery(label="Stills (manifest order)",
                                            columns=4, height=420,
                                            object_fit="contain")
                with gr.Row():
                    regen_dd = gr.Dropdown(choices=[], label="Regenerate one scene")
                    regen_btn = gr.Button("♻ Regenerate selected")
                    approve_stills_btn = gr.Button("✅ Approve stills → Animate",
                                                   variant="primary")
                stills_msg = gr.Markdown("")

            # --------------------------------------------------- 5 Animate
            with gr.Tab("5 · Animate"):
                gr.Markdown("The hybrid: pick which scenes become Veo clips; the "
                            "rest stay Ken Burns stills (free). **Veo bills per "
                            "clip** — test one before the batch. Rendered clips "
                            "are never re-billed.")
                animate_cbg = gr.CheckboxGroup(choices=[],
                                               label="Scenes to animate")
                save_sel_btn = gr.Button("💾 Save selection")
                animate_info = gr.Markdown("")
                gr.Markdown("**Test one clip first:**")
                with gr.Row():
                    test_dd = gr.Dropdown(choices=[], label="Scene")
                    test_force_cb = gr.Checkbox(label="Force re-roll (💰 bills "
                                                      "again)")
                    test_btn = gr.Button("🎬 Render test clip (💰 1 clip)",
                                         variant="primary")
                clip_preview = gr.Video(label="Clip preview", height=420)
                gr.Markdown("**Then the batch:**")
                with gr.Row():
                    batch_confirm_cb = gr.Checkbox(
                        label="I confirm the batch spend (un-rendered clips only)")
                    batch_force_cb = gr.Checkbox(label="Force re-render all (💰💰)")
                    batch_btn = gr.Button("🎬 Run batch", variant="primary")
                    no_anim_btn = gr.Button("Skip animation → all stills")
                animate_status = gr.Textbox(label="Progress", lines=4,
                                            interactive=False)

            # --------------------------------------------------- 6 Handoff
            with gr.Tab("6 · Handoff"):
                gr.Markdown("Places this episode's media into `assets/images/` "
                            "as `NN.mp4` / `NN.png` in manifest order — the step "
                            "that used to be manual renaming. Anything else in "
                            "that folder is archived, not deleted.")
                handoff_md = gr.Markdown("")
                handoff_btn = gr.Button("📦 Run handoff", variant="primary")
                handoff_msg = gr.Markdown("")

            # ----------------------------------------------- 7 Voice/Build
            with gr.Tab("7 · Voice & Build"):
                gr.Markdown("ElevenLabs TTS → Whisper word timings → script-"
                            "aligned captions (3 words/line) → Ken Burns + clips "
                            "+ 0.4s crossfades + looped music → `output/<name>.mp4`.")
                with gr.Row():
                    voice_id_tb = gr.Textbox(label="ElevenLabs voice_id",
                                             value=config.ELEVENLABS_VOICE_ID)
                    voices_dd = gr.Dropdown(choices=[], label="…or pick from "
                                                              "account")
                    fetch_voices_btn = gr.Button("Fetch voices")
                with gr.Row():
                    stab_sl = gr.Slider(0, 1, value=config.ELEVENLABS_STABILITY,
                                        step=0.05, label="Stability")
                    sim_sl = gr.Slider(0, 1, value=config.ELEVENLABS_SIMILARITY,
                                       step=0.05, label="Similarity")
                    style_sl = gr.Slider(0, 1, value=config.ELEVENLABS_STYLE,
                                         step=0.05, label="Style")
                with gr.Row():
                    music_dd = gr.Dropdown(choices=["(no music)"],
                                           label="Music bed")
                    assemble_force_cb = gr.Checkbox(label="Force re-render "
                                                          "(re-bills TTS if voice "
                                                          "settings changed)")
                    assemble_btn = gr.Button("🎙 Build the video",
                                             variant="primary")
                assemble_status = gr.Textbox(label="Progress", lines=8,
                                             interactive=False)
                assemble_video = gr.Video(label="Result", height=480)

            # ---------------------------------------------------- 8 Review
            with gr.Tab("8 · Review & Approve"):
                gr.Markdown("**The human gate.** Watch it, scan the per-scene "
                            "contact sheet for morphs/wrong shots, then approve.")
                qc_video = gr.Video(label="Final video", height=480)
                qc_btn = gr.Button("🔍 Run QC checks", variant="primary")
                qc_md = gr.Markdown("")
                qc_gallery = gr.Gallery(label="Contact sheet — first frame of "
                                              "each scene", columns=4, height=340,
                                        object_fit="contain")
                gr.Markdown("---\n**Publish copy (optional):**")
                with gr.Row():
                    title_tb = gr.Textbox(label="Title")
                    draft_btn = gr.Button("✍ Draft with Gemini")
                desc_tb = gr.Textbox(label="Description", lines=3)
                approve_btn = gr.Button("✅ Approve → ready to publish",
                                        variant="primary")
                approve_md = gr.Markdown("")

            # --------------------------------------------------- Settings
            with gr.Tab("⚙ Settings"):
                keys_md = gr.Markdown(_key_status())
                with gr.Row():
                    gemini_tb = gr.Textbox(label="GEMINI_API_KEY",
                                           type="password")
                    eleven_tb = gr.Textbox(label="ELEVENLABS_API_KEY",
                                           type="password")
                save_keys_btn = gr.Button("Save keys → .env (gitignored)")
                gr.Markdown(
                    f"**Paths** — scripts: `scripts/` · style/character refs: "
                    f"`refs/` · manifests: `manifests/` · episode media: "
                    f"`episodes/<name>/` · pipeline input: `assets/images/` · "
                    f"music: `assets/music/` · final videos: `output/`\n\n"
                    f"Repo root: `{ROOT}`")

        # ------------------------------------------------------- wiring ----
        refs_outputs = [style_dd, chars_md, char_del_dd]

        def set_episode(name):
            return name or ""

        load_chain = script_dd.change(set_episode, script_dd, ep_state)
        load_chain = load_chain.then(load_script_tab, ep_state,
                                     [script_tb, stats_md, banner])
        load_chain = load_chain.then(load_refs_tab, ep_state, refs_outputs)
        load_chain = load_chain.then(load_prompts_tab, ep_state,
                                     row_comps + [prompts_note])
        load_chain = load_chain.then(load_stills_tab, ep_state,
                                     [stills_gallery, regen_dd, stills_msg])
        load_chain = load_chain.then(load_animate_tab, ep_state,
                                     [animate_cbg, test_dd, clip_preview,
                                      animate_info])
        load_chain = load_chain.then(load_handoff_tab, ep_state, handoff_md)
        load_chain = load_chain.then(load_assemble_tab, ep_state,
                                     [voice_id_tb, stab_sl, sim_sl, style_sl,
                                      music_dd, assemble_status])
        load_chain.then(load_qc_tab, ep_state,
                        [qc_video, qc_md, qc_gallery, title_tb, desc_tb,
                         approve_md])

        new_btn.click(create_script, new_name, [script_dd, ep_state])
        save_script_btn.click(save_script, [ep_state, script_tb],
                              [stats_md, banner])
        script_tb.change(lambda t: script_stats(t), script_tb, stats_md)
        rewrite_btn.click(rewrite_script, [ep_state, script_tb],
                          [script_tb, stats_md])

        ref_upload.upload(upload_refs, ref_upload, [style_dd, upload_msg])
        style_btn.click(save_style_lock, [ep_state, style_dd],
                        [refs_msg, banner])
        char_add_btn.click(add_character,
                           [ep_state, char_name_tb, char_files],
                           refs_outputs + [refs_msg])
        char_del_btn.click(remove_character, [ep_state, char_del_dd],
                           refs_outputs + [refs_msg])

        gen_prompts_btn.click(generate_prompts,
                              [ep_state, use_llm_cb, clip_dur],
                              row_comps + [prompts_note, banner])
        save_prompts_btn.click(save_prompts, [ep_state] + row_inputs,
                               [prompts_save_md, banner])

        stills_btn.click(run_stills, [ep_state, stills_force_cb],
                         [stills_status, stills_gallery, banner])
        regen_btn.click(regen_still, [ep_state, regen_dd],
                        [stills_status, stills_gallery])
        approve_stills_btn.click(approve_stills, ep_state,
                                 [stills_msg, banner])

        save_sel_btn.click(save_animate_selection, [ep_state, animate_cbg],
                           [animate_info, banner])
        test_btn.click(test_one_clip, [ep_state, test_dd, test_force_cb],
                       [animate_status, clip_preview])
        batch_btn.click(run_animate_batch,
                        [ep_state, animate_cbg, batch_confirm_cb,
                         batch_force_cb],
                        [animate_status, banner])
        no_anim_btn.click(confirm_no_animate, ep_state,
                          [animate_status, banner])

        handoff_btn.click(run_handoff, ep_state,
                          [handoff_msg, handoff_md, banner])

        fetch_voices_btn.click(fetch_voices, None, [voices_dd, assemble_status])
        voices_dd.change(voice_picked, voices_dd, voice_id_tb)
        assemble_btn.click(run_assemble,
                           [ep_state, voice_id_tb, stab_sl, sim_sl, style_sl,
                            music_dd, assemble_force_cb],
                           [assemble_status, assemble_video, banner])
        assemble_btn.click(lambda n: load_qc_tab(n)[0], ep_state, qc_video)

        qc_btn.click(run_qc, ep_state, [qc_md, qc_gallery])
        draft_btn.click(draft_meta, ep_state, [title_tb, desc_tb])
        approve_btn.click(approve, [ep_state, title_tb, desc_tb],
                          [approve_md, banner])

        save_keys_btn.click(save_keys, [gemini_tb, eleven_tb],
                            [keys_md, gemini_tb, eleven_tb])
    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue().launch(inbrowser=True, show_error=True, theme=gr.themes.Soft())
