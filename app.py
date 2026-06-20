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
                None, "", gr.update(choices=[]))
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
            None, msg, gr.update(choices=names))


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


def use_own_clip(name: str, scene: str, file: str | None):
    """Install a user-supplied .mp4 as the scene's clip. From here on the
    pipeline treats it exactly like a Veo render: the batch skips it (never
    billed) and the handoff places it as NN.mp4 instead of the still."""
    ep = _require(name, "prompts", "generate prompts first")
    if not scene:
        raise gr.Error("Pick the scene this clip replaces.")
    if not file:
        raise gr.Error("Choose an .mp4 file first.")
    src = Path(file)
    if src.suffix.lower() != ".mp4":
        raise gr.Error("Only .mp4 files — the build step keys on NN.mp4 "
                       "filenames in assets/images.")
    import shutil
    ep.clips_dir.mkdir(parents=True, exist_ok=True)
    dest = ep.clips_dir / f"{scene}.mp4"
    shutil.copy2(src, dest)
    manifest = ep.load_manifest()
    for c in manifest["clips"]:
        if c["name"] == scene:
            c["animate"] = True
    ep.save_manifest(manifest)
    cbg_u, _, _, info, own_u = load_animate_tab(name)
    return (f"{scene}: your clip is installed as {dest.name} — marked animate; "
            "Veo will skip it (rendered clips are never re-billed).",
            str(dest), cbg_u, info, own_u, status_banner(name))


def run_animate_batch(name: str, selected: list[str], confirmed: bool, force: bool):
    ep = _require(name, "stills", "generate all stills first")
    _check_stills_approved(ep)
    selected = list(selected or [])
    if not selected:
        raise gr.Error(
            "No scenes are selected to animate. Tick scenes above and re-run — "
            "or, if this episode should be all stills, use the "
            "'Skip animation → all stills' button instead.", duration=None)
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
                 style: float, music_choice: str, force: bool,
                 cap_style: str = "active_word", cap_font: str = "Anton"):
    ep = _require(name, "handoff", "run the handoff first")
    stages.apply_voice_settings(voice_id, stability, similarity, style)
    stages.apply_caption_settings(cap_style, cap_font)
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


# ----------------------------------------------------------------- SFX panel

def _sfx_sources() -> list[str]:
    """All referenceable cue tags: library + imported, from the tag map."""
    from orchestrator.audio_spec import load_sfx_map
    return sorted(load_sfx_map().keys())


def sfx_summary(name: str) -> str:
    """A human-readable list of the episode's current cues (manifest-backed)."""
    from orchestrator.audio_spec import parse_audio_spec, AudioSpecError
    if not name:
        return "_Pick an episode first._"
    ep = _ep(name)
    manifest = ep.load_manifest() or {}
    try:
        spec = parse_audio_spec(manifest)
    except AudioSpecError as exc:
        return f"⚠ **Spec has problems:**\n\n```\n{exc}\n```"
    if spec.is_empty():
        return ("_No SFX yet._ Upload an mp3 or pick a library tag, choose a "
                "layer, and **Add cue**. The VO never speaks a cue — cues live "
                "in the manifest's `audio`/`sfx` blocks, never in the script.")
    rows = ["| layer | source | placement | gain | extras |",
            "|---|---|---|---|---|"]

    def _anchor_str(c) -> str:
        a = c.anchor or {}
        if "word" in a:
            return f"word “{a['word']}” +{a.get('offset', 0)}s"
        if "scene" in a:
            return f"scene {a['scene']} +{a.get('offset', 0)}s"
        if "time" in a:
            return f"@{a['time']}s"
        return "whole video"

    for c in spec.all_cues():
        extras = []
        if c.pan is not None:
            extras.append(f"pan {c.pan:+.1f}")
        if c.fade_in:
            extras.append(f"fade-in {c.fade_in}s")
        if c.fade_out:
            extras.append(f"fade-out {c.fade_out}s")
        if c.loop:
            extras.append("loop")
        rows.append(f"| {c.layer} | `{c.source}` | {_anchor_str(c)} | "
                    f"{c.gain_db:+.0f} dB | {', '.join(extras) or '—'} |")
    duck = ("on" if spec.duck_enabled else "off")
    return "\n".join(rows) + f"\n\n**Ducking:** {duck} (SFX dip under the VO)."


def import_sfx(files) -> tuple:
    """Import + loudnorm uploaded audio; register and refresh the source list."""
    if not files:
        raise gr.Error("Choose one or more audio files to import.")
    paths = [f.name if hasattr(f, "name") else f for f in files]
    try:
        from orchestrator.audio_import import import_many
        tags = import_many(paths)
    except Exception as exc:
        raise gr.Error(f"Import failed: {exc}", duration=None)
    srcs = _sfx_sources()
    return (f"✅ Imported & normalized: {', '.join(tags)}. "
            f"They're now in the source list.",
            gr.update(choices=srcs, value=tags[0]))


def add_sfx_cue(name: str, source: str, layer: str, host_scene: float,
                anchor_type: str, word: str, time_s: float, offset: float,
                gain_db: float, pan: float, fade_in: float, fade_out: float,
                loop: bool, ducking: bool):
    """Append a cue to the manifest, validating before save (no silent fails)."""
    from orchestrator.audio_spec import parse_audio_spec, AudioSpecError
    if not name:
        raise gr.Error("Pick an episode first.")
    if not source:
        raise gr.Error("Pick a cue source (a library tag or an imported track).")
    ep = _ep(name)
    manifest = ep.load_manifest()
    if not manifest:
        raise gr.Error("Generate prompts first — SFX attach to the manifest.")

    cue: dict = {"source": source, "gain_db": round(float(gain_db), 1)}
    if abs(float(pan)) > 1e-6:
        cue["pan"] = round(float(pan), 2)
    if float(fade_in) > 0:
        cue["fade_in"] = round(float(fade_in), 2)
    if float(fade_out) > 0:
        cue["fade_out"] = round(float(fade_out), 2)

    host = int(host_scene or 1)
    if layer in ("ambient_bed", "music_bed"):
        cue["loop"] = bool(loop) if loop is not None else True
        manifest.setdefault("audio", {})[layer] = cue
    else:
        if bool(loop):
            cue["loop"] = True
        if anchor_type == "word":
            if not (word or "").strip():
                raise gr.Error("Enter the word to anchor the cue to.")
            cue["at"] = {"word": word.strip(), "offset": round(float(offset), 2)}
        elif anchor_type == "time":
            cue["at"] = {"time": round(float(time_s), 2)}
        else:  # scene
            cue["at"] = {"scene": host, "offset": round(float(offset), 2)}
        if layer == "motif":
            manifest.setdefault("audio", {}).setdefault("motifs", []).append(cue)
        else:  # oneshot → lives under its host clip
            clips = manifest.get("clips", [])
            if host < 1 or host > len(clips):
                raise gr.Error(f"Host scene {host} is out of range "
                               f"(1..{len(clips)}).")
            clips[host - 1].setdefault("sfx", []).append(cue)

    manifest.setdefault("audio", {}).setdefault("ducking", {})["enabled"] = \
        bool(ducking)

    # Validate the WHOLE spec before persisting — reject bad additions loudly.
    try:
        parse_audio_spec(manifest)
    except AudioSpecError as exc:
        raise gr.Error(f"Cue rejected:\n{exc}", duration=None)
    ep.save_manifest(manifest)
    return f"✅ Added {layer} cue `{source}`.", sfx_summary(name)


def clear_sfx(name: str):
    """Strip all SFX from the manifest (audio block + per-clip sfx)."""
    if not name:
        raise gr.Error("Pick an episode first.")
    ep = _ep(name)
    manifest = ep.load_manifest()
    if not manifest:
        return "_Nothing to clear._", sfx_summary(name)
    manifest.pop("audio", None)
    for c in manifest.get("clips", []):
        c.pop("sfx", None)
    ep.save_manifest(manifest)
    return "🧹 Cleared all SFX for this episode.", sfx_summary(name)


def load_sfx_tab(name: str):
    """Populate the source dropdown + cue summary on episode load."""
    srcs = _sfx_sources()
    return gr.update(choices=srcs, value=(srcs[0] if srcs else None)), \
        sfx_summary(name)


# ------------------------------------------------------------------ UI -----

# Landing shell: the app opens on a landing screen with two entry cards that route
# into one of the two pipelines (Shorts / Gaming). The wizard, gameplay tab, and
# Settings are wrapped — unchanged — in visibility-toggled containers so they all
# stay in one session and every existing handler keeps firing.
_MODES = ("landing", "shorts", "gaming", "fullauto", "settings")


def _route(target: str) -> tuple:
    """Visibility updates for [landing, shorts, gaming, fullauto, settings] — exactly
    one container visible for `target`."""
    return tuple(gr.update(visible=(m == target)) for m in _MODES)


_LANDING_CSS = """
.landing-wrap { max-width: 920px; margin: 0 auto; padding: 24px 8px; }
.landing-wrap .brand-title { text-align: center; }
.entry-card { border: 1px solid var(--border-color-primary);
  border-radius: 16px; padding: 20px 22px; min-height: 188px;
  background: var(--block-background-fill);
  transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease; }
.entry-card:hover { transform: translateY(-3px);
  box-shadow: 0 8px 24px rgba(0,0,0,.12); border-color: var(--color-accent); }
/* Experimental card: muted, dashed accent + badge so it doesn't read as a
   finished, equal-status pipeline next to the two production cards. */
.entry-card.experimental { border-style: dashed; opacity: .92;
  background: var(--block-background-fill);
  border-color: var(--border-color-accent, var(--border-color-primary)); }
.entry-card.experimental:hover { border-color: var(--color-accent); opacity: 1; }
.exp-badge { display: inline-block; font-size: .72em; font-weight: 700;
  letter-spacing: .06em; padding: 2px 8px; border-radius: 999px;
  background: var(--color-accent-soft, rgba(255,170,0,.18));
  border: 1px solid var(--color-accent, #d98a00); color: var(--body-text-color); }
.mode-header { align-items: center; gap: 8px; margin-bottom: 4px; }
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Lore Shorts Studio") as demo:
        ep_state = gr.State("")
        prev_mode = gr.State("landing")   # which mode Settings' "Back" returns to

        # ---- Landing (shown on startup) ----
        with gr.Column(visible=True, elem_classes="landing-wrap",
                       elem_id="mode-landing") as landing:
            gr.Markdown("# 🎬 Lore Shorts Studio", elem_classes="brand-title")
            gr.Markdown("### Choose a pipeline", elem_classes="brand-title")
            with gr.Row(equal_height=True):
                with gr.Group(elem_classes="entry-card"):
                    gr.Markdown("## 🎬 Shorts Pipeline\nCinematic lore Shorts: "
                                "script → stills → motion → voice → captions → "
                                "approve.")
                    shorts_card_btn = gr.Button("Open Shorts →", variant="primary")
                with gr.Group(elem_classes="entry-card"):
                    gr.Markdown("## 🎮 Gaming\nTurn a pre-trimmed gameplay clip into "
                                "a captioned 9:16 Short.")
                    gaming_card_btn = gr.Button("Open Gaming →", variant="primary")
                with gr.Group(elem_classes="entry-card experimental"):
                    gr.Markdown(
                        "## ⚗ Full-Auto Experiment <span class='exp-badge'>"
                        "EXPERIMENTAL</span>\nDrop in raw YouTube footage or long "
                        "gameplay; auto-find, categorise and cut highlights into a "
                        "**16:9 YouTube video**.")
                    fullauto_card_btn = gr.Button("Open Full-Auto →",
                                                  variant="secondary")
            with gr.Row():
                landing_settings_btn = gr.Button("⚙ Settings", scale=0)

        # ---- Shorts (the lore wizard) ----
        with gr.Column(visible=False, elem_id="mode-shorts") as shorts_view:
          with gr.Row(elem_classes="mode-header"):
              shorts_home_btn = gr.Button("← Home / Switch mode", scale=0)
              shorts_settings_btn = gr.Button("⚙ Settings", scale=0)
              gr.Markdown("### 🎬 Shorts Pipeline")
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
                with gr.Row():
                    save_sel_btn = gr.Button("💾 Save selection")
                    refresh_anim_btn = gr.Button("🔄 Refresh from disk")
                animate_info = gr.Markdown("")
                gr.Markdown("**Test one clip first:**")
                with gr.Row():
                    test_dd = gr.Dropdown(choices=[], label="Scene")
                    test_force_cb = gr.Checkbox(label="Force re-roll (💰 bills "
                                                      "again)")
                    test_btn = gr.Button("🎬 Render test clip (💰 1 clip)",
                                         variant="primary")
                clip_preview = gr.Video(label="Clip preview", height=420)
                gr.Markdown("**Or bring your own clip** — replaces the still "
                            "for that scene with your .mp4 (no Veo, never "
                            "billed). Use 9:16 to match the build.")
                with gr.Row():
                    own_scene_dd = gr.Dropdown(choices=[], label="Scene")
                    own_clip_file = gr.File(label="Your .mp4",
                                            file_types=[".mp4"])
                    own_clip_btn = gr.Button("📥 Use this clip for the scene")
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

                with gr.Accordion("🔊 Sound effects & custom audio (optional)",
                                  open=False):
                    gr.Markdown(
                        "Layer sound on top of the narration. Cues live in the "
                        "manifest's `audio`/`sfx` blocks — **never in the script**, "
                        "so the voice can't speak a cue tag. The Build button "
                        "above picks these up automatically.")
                    with gr.Row():
                        sfx_upload = gr.File(label="Import your own audio "
                                             "(mp3/wav…) — loudnorm'd on import",
                                             file_count="multiple",
                                             file_types=["audio"])
                        sfx_import_btn = gr.Button("⬇ Import & normalize")
                    with gr.Row():
                        sfx_source_dd = gr.Dropdown(choices=[], label="Cue source "
                                                    "(library tag or imported)")
                        sfx_layer_dd = gr.Dropdown(
                            choices=["ambient_bed", "music_bed", "motif",
                                     "oneshot"], value="oneshot", label="Layer")
                        sfx_scene_n = gr.Number(value=1, precision=0,
                                                label="Host scene # (one-shot / "
                                                      "scene anchor)")
                    with gr.Row():
                        sfx_anchor_rb = gr.Radio(
                            choices=["scene", "word", "time"], value="scene",
                            label="Anchor (ignored for beds)")
                        sfx_word_tb = gr.Textbox(label="…on word", value="")
                        sfx_time_n = gr.Number(value=0.0, label="…at time (s)")
                        sfx_offset_n = gr.Number(value=0.0,
                                                 label="offset (s, scene/word)")
                    with gr.Row():
                        sfx_gain_sl = gr.Slider(-40, 6, value=-12, step=1,
                                                label="Gain (dB)")
                        sfx_pan_sl = gr.Slider(-1, 1, value=0, step=0.1,
                                               label="Pan (L–R)")
                        sfx_fadein_n = gr.Number(value=0.0, label="Fade in (s)")
                        sfx_fadeout_n = gr.Number(value=0.0, label="Fade out (s)")
                    with gr.Row():
                        sfx_loop_cb = gr.Checkbox(label="Loop", value=False)
                        sfx_duck_cb = gr.Checkbox(
                            label="Duck SFX under VO (sidechain)", value=True)
                        sfx_add_btn = gr.Button("➕ Add cue", variant="primary")
                        sfx_clear_btn = gr.Button("🧹 Clear all SFX")
                    sfx_msg = gr.Markdown("")
                    sfx_summary_md = gr.Markdown("")

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

        # ---- Gaming (the gameplay pipeline) ----
        with gr.Column(visible=False, elem_id="mode-gaming") as gaming_view:
          with gr.Row(elem_classes="mode-header"):
              gaming_home_btn = gr.Button("← Home / Switch mode", scale=0)
              gaming_settings_btn = gr.Button("⚙ Settings", scale=0)
              gr.Markdown("### 🎮 Gaming")
          with gr.Tabs():
            # Second, parallel pipeline (gameplay clips). Self-contained in
            # gameplay/gui.py; the lore wizard is untouched.
            from gameplay.gui import build_gameplay_tab
            gaming_clip_video = build_gameplay_tab()

        # ---- Full-Auto Experiment (its own mode; 16:9 YouTube output) ----
        with gr.Column(visible=False, elem_id="mode-fullauto") as fullauto_view:
          with gr.Row(elem_classes="mode-header"):
              fullauto_home_btn = gr.Button("← Home / Switch mode", scale=0)
              fullauto_settings_btn = gr.Button("⚙ Settings", scale=0)
              gr.Markdown("### ⚗ Full-Auto Experiment")
          with gr.Tabs():
            # Experimental long-form processor — its own package; exports a 16:9
            # YouTube video and never touches the 9:16 Shorts backend.
            from fullauto.gui import build_fullauto_view
            fullauto_handles = build_fullauto_view(manual_clip_video=gaming_clip_video)

        # ---- Settings (shared — reachable from all modes) ----
        with gr.Column(visible=False, elem_id="mode-settings") as settings_view:
          with gr.Row(elem_classes="mode-header"):
              settings_back_btn = gr.Button("← Back", scale=0)
              gr.Markdown("### ⚙ Settings")
          with gr.Tabs():
            with gr.Tab("⚙ Settings"):
                keys_md = gr.Markdown(_key_status())
                with gr.Row():
                    gemini_tb = gr.Textbox(label="GEMINI_API_KEY",
                                           type="password")
                    eleven_tb = gr.Textbox(label="ELEVENLABS_API_KEY",
                                           type="password")
                save_keys_btn = gr.Button("Save keys → .env (gitignored)")

                gr.Markdown("### Captions")
                with gr.Row():
                    cap_style_dd = gr.Dropdown(
                        label="Caption style",
                        choices=[("Active word (big yellow, pops in)", "active_word"),
                                 ("Classic (3 words/line)", "classic")],
                        value=config.CAPTION_STYLE)
                    cap_font_dd = gr.Dropdown(
                        label="Active-word font",
                        choices=["Anton", "Arial"],
                        value=config.CAPTION_AW_FONT)
                gr.Markdown(
                    "Applies to the next build. Fine-tuning (size, colour, position, "
                    "words-per-cue) lives in `config.py` under the `CAPTION_AW_*` "
                    "settings. Anton is bundled in `fonts/` — no system install needed.")

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
                                      animate_info, own_scene_dd])
        load_chain = load_chain.then(load_handoff_tab, ep_state, handoff_md)
        load_chain = load_chain.then(load_assemble_tab, ep_state,
                                     [voice_id_tb, stab_sl, sim_sl, style_sl,
                                      music_dd, assemble_status])
        load_chain = load_chain.then(load_sfx_tab, ep_state,
                                     [sfx_source_dd, sfx_summary_md])
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

        # The Stills/Animate/Handoff tabs list scenes from the manifest, so any
        # event that (re)writes the manifest must refresh them — otherwise an
        # episode whose prompts were generated mid-session shows empty scene
        # pickers until the page is reloaded.
        def _chain_scene_refresh(evt):
            (evt.then(load_stills_tab, ep_state,
                      [stills_gallery, regen_dd, stills_msg])
                .then(load_animate_tab, ep_state,
                      [animate_cbg, test_dd, clip_preview, animate_info,
                       own_scene_dd])
                .then(load_handoff_tab, ep_state, handoff_md))

        _chain_scene_refresh(
            gen_prompts_btn.click(generate_prompts,
                                  [ep_state, use_llm_cb, clip_dur],
                                  row_comps + [prompts_note, banner]))
        _chain_scene_refresh(
            save_prompts_btn.click(save_prompts, [ep_state] + row_inputs,
                                   [prompts_save_md, banner]))

        stills_btn.click(run_stills, [ep_state, stills_force_cb],
                         [stills_status, stills_gallery, banner])
        regen_btn.click(regen_still, [ep_state, regen_dd],
                        [stills_status, stills_gallery])
        approve_stills_btn.click(approve_stills, ep_state,
                                 [stills_msg, banner]) \
            .then(load_animate_tab, ep_state,
                  [animate_cbg, test_dd, clip_preview, animate_info,
                   own_scene_dd])

        save_sel_btn.click(save_animate_selection, [ep_state, animate_cbg],
                           [animate_info, banner])
        # Re-read the manifest and on-disk stills/clips so files edited
        # outside the GUI show up without a page reload.
        refresh_anim_btn.click(load_animate_tab, ep_state,
                               [animate_cbg, test_dd, clip_preview,
                                animate_info, own_scene_dd]) \
            .then(load_stills_tab, ep_state,
                  [stills_gallery, regen_dd, stills_msg]) \
            .then(load_handoff_tab, ep_state, handoff_md)
        test_btn.click(test_one_clip, [ep_state, test_dd, test_force_cb],
                       [animate_status, clip_preview])
        own_clip_btn.click(use_own_clip,
                           [ep_state, own_scene_dd, own_clip_file],
                           [animate_status, clip_preview, animate_cbg,
                            animate_info, own_scene_dd, banner]) \
            .then(load_handoff_tab, ep_state, handoff_md)
        batch_btn.click(run_animate_batch,
                        [ep_state, animate_cbg, batch_confirm_cb,
                         batch_force_cb],
                        [animate_status, banner]) \
            .then(lambda n: load_animate_tab(n)[3], ep_state, animate_info) \
            .then(load_handoff_tab, ep_state, handoff_md)
        no_anim_btn.click(confirm_no_animate, ep_state,
                          [animate_status, banner])

        handoff_btn.click(run_handoff, ep_state,
                          [handoff_msg, handoff_md, banner])

        fetch_voices_btn.click(fetch_voices, None, [voices_dd, assemble_status])
        voices_dd.change(voice_picked, voices_dd, voice_id_tb)

        sfx_import_btn.click(import_sfx, sfx_upload, [sfx_msg, sfx_source_dd])
        sfx_add_btn.click(
            add_sfx_cue,
            [ep_state, sfx_source_dd, sfx_layer_dd, sfx_scene_n, sfx_anchor_rb,
             sfx_word_tb, sfx_time_n, sfx_offset_n, sfx_gain_sl, sfx_pan_sl,
             sfx_fadein_n, sfx_fadeout_n, sfx_loop_cb, sfx_duck_cb],
            [sfx_msg, sfx_summary_md])
        sfx_clear_btn.click(clear_sfx, ep_state, [sfx_msg, sfx_summary_md])
        assemble_btn.click(run_assemble,
                           [ep_state, voice_id_tb, stab_sl, sim_sl, style_sl,
                            music_dd, assemble_force_cb, cap_style_dd, cap_font_dd],
                           [assemble_status, assemble_video, banner])
        assemble_btn.click(lambda n: load_qc_tab(n)[0], ep_state, qc_video)

        qc_btn.click(run_qc, ep_state, [qc_md, qc_gallery])
        draft_btn.click(draft_meta, ep_state, [title_tb, desc_tb])
        approve_btn.click(approve, [ep_state, title_tb, desc_tb],
                          [approve_md, banner])

        save_keys_btn.click(save_keys, [gemini_tb, eleven_tb],
                            [keys_md, gemini_tb, eleven_tb])

        # ----------------------------------------------- landing / nav ----
        # Visibility-only routing between the landing and the three containers.
        # No pipeline state is touched, so switching modes preserves everything.
        nav = [landing, shorts_view, gaming_view, fullauto_view, settings_view]
        shorts_card_btn.click(lambda: _route("shorts"), None, nav)
        gaming_card_btn.click(lambda: _route("gaming"), None, nav)
        fullauto_card_btn.click(lambda: _route("fullauto"), None, nav)
        shorts_home_btn.click(lambda: _route("landing"), None, nav)
        gaming_home_btn.click(lambda: _route("landing"), None, nav)
        fullauto_home_btn.click(lambda: _route("landing"), None, nav)
        # Settings remembers where it was opened from, so "← Back" returns there.
        landing_settings_btn.click(lambda: "landing", None, prev_mode) \
            .then(lambda: _route("settings"), None, nav)
        shorts_settings_btn.click(lambda: "shorts", None, prev_mode) \
            .then(lambda: _route("settings"), None, nav)
        gaming_settings_btn.click(lambda: "gaming", None, prev_mode) \
            .then(lambda: _route("settings"), None, nav)
        fullauto_settings_btn.click(lambda: "fullauto", None, prev_mode) \
            .then(lambda: _route("settings"), None, nav)
        settings_back_btn.click(lambda p: _route(p), prev_mode, nav)

        # Full-auto "Refine in manual mode": load the chosen candidate's raw clip into
        # the Gameplay uploader, then route to the Gaming tab so it's one click.
        if fullauto_handles.get("refine_btn") is not None and gaming_clip_video is not None:
            from fullauto.gui import _refine_source_for
            fullauto_handles["refine_btn"].click(
                _refine_source_for,
                [fullauto_handles["video"], fullauto_handles["dd"]],
                gaming_clip_video) \
                .then(lambda: _route("gaming"), None, nav)
    return demo


if __name__ == "__main__":
    app = build_app()
    # ssr_mode=False: this is one large single-page Blocks (lore wizard + gaming +
    # full-auto + settings all mounted at once). With Gradio's default server-side
    # rendering, every interaction re-runs a full-tree render/hydration, which on a
    # session with loaded media (stills galleries, clip video, transcript grids)
    # blanks the page for several seconds when switching modes — the "black screen,
    # work missing" symptom. Client-side rendering toggles container visibility
    # without that stall. (Session state is unaffected — it lives server-side.)
    app.queue().launch(inbrowser=True, show_error=True, theme=gr.themes.Soft(),
                       css=_LANDING_CSS, ssr_mode=False)
