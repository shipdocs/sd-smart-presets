"""
Wizard 2 UI: "Leer een persoon, personage, voorwerp of stijl".

  1. Photos + name. What it is gets guessed from the photos.
  2. The plan: photos, steps, expected time, base model; consent for a real person.
  3. Progress (survives a page reload), then the result: test images per version,
     the best one is installed as a LoRA and added as a preset.
"""

from pathlib import Path

import gradio as gr

from . import training

STAGES = {
    "queued": "In de wachtrij",
    "start": "Starten",
    "prepare": "1/3 · Foto's voorbereiden",
    "train": "2/3 · Trainen",
    "evaluate": "3/3 · Testen welke versie het best is",
    "done": "✅ Klaar",
    "error": "⚠️ Mislukt",
    "cancelled": "Gestopt",
}


def _bar(fraction: float) -> str:
    pct = int(round(max(0.0, min(1.0, fraction or 0.0)) * 100))
    return (f"<div style='background:var(--block-background-fill);border:1px solid var(--border-color-primary);"
            f"border-radius:6px;height:14px;overflow:hidden'><div style='width:{pct}%;height:100%;"
            f"background:var(--color-accent)'></div></div><small>{pct}%</small>")


def _status_view(st: dict):
    stage = st.get("stage", "queued")
    lines = [f"### {STAGES.get(stage, stage)}", "" if stage == "done" else st.get("message", "")]
    ds = st.get("dataset")
    if ds:
        dropped = ", ".join(f"{v} {k}" for k, v in ds.get("dropped", {}).items()) or "geen"
        lines.append(f"**Foto's:** {ds['used']} gebruikt · afgevallen: {dropped} · trigger: `{ds['caption']}`")
        if ds.get("upscaled"):
            how = f"met {ds['upscaler']}" if ds.get("upscaler") else "gewoon (geen upscaler gevonden)"
            lines.append(f"ℹ️ {ds['upscaled']} foto's vergroot {how}; echte grotere foto's geven de scherpste ogen.")
    if stage == "done":
        res = st.get("result", {})
        ev = st.get("evaluation", {})
        lines.append(f"**Geïnstalleerd als LoRA:** `{res.get('lora_name')}` (gekozen: {ev.get('best_reason', '')})")
        lines.append(f"Gebruik in een prompt: `{res.get('trigger')}, <lora:{res.get('lora_name')}:0.8>`")
        if st.get("preset"):
            lines.append(f"Er staat een preset **{st['preset']}** klaar in 'Maak een afbeelding' → 'Of begin met een opgeslagen preset'.")
    html = _bar(st.get("progress", 0) if stage not in ("done",) else 1.0) if stage not in ("error", "cancelled") else ""
    log = "\n".join(st.get("log", [])[-12:])
    return "\n\n".join(x for x in lines if x), html, log


def refresh(last_key=""):
    """Called every few seconds. Gallery and version choice only change when the job's result changes,
    so they don't flicker or undo what you clicked."""
    job = training.latest_job()
    if job is None:
        return "", "", "", gr.update(), gr.update(), gr.update(visible=False), last_key
    st = training.read_status(job)
    text, bar, log = _status_view(st)
    ev = st.get("evaluation") or {}
    active = st.get("stage") not in ("done", "error", "cancelled")
    key = f"{job.name}:{st.get('stage')}:{len(ev.get('versions', []))}"
    if key == last_key:
        return text, bar, log, gr.update(), gr.update(), gr.update(visible=active), last_key
    versions = [(f"stap {v['step']}" + (f" · gelijkenis {v['score']:.2f}" if v.get("score") is not None else ""), v["file"])
                for v in ev.get("versions", [])]
    return (text, bar, log, gr.update(value=training.sample_images(st) or None),
            gr.update(choices=versions, value=ev.get("best"), visible=bool(versions)),
            gr.update(visible=active), key)


IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _paths(files, folder=None) -> list[str]:
    """Dropped photos plus the images of a chosen folder (other files in it are skipped)."""
    out = []
    for f in list(files or []) + list(folder or []):
        p = f if isinstance(f, str) else f.name
        if Path(p).suffix.lower() in IMAGE_EXT and p not in out:
            out.append(p)
    return out


def on_upload(files, folder):
    paths = _paths(files, folder)
    if not paths:
        return gr.update(), "", gr.update()
    kind, why = training.guess_kind(paths)
    return gr.update(value=kind), f"🔍 {len(paths)} foto's. {why}", gr.update(visible=(kind == "person"))


def on_plan(files, folder, name, kind, checkpoint):
    paths = _paths(files, folder)
    problems = []
    if len(paths) < 5:
        problems.append("Geef minstens 5 foto's (beter 15 tot 30, verschillende kanten, kleding en licht).")
    if not (name or "").strip():
        problems.append("Geef een naam.")
    from . import plan as plan_lib
    info = plan_lib.checkpoint_info(checkpoint)
    arch = plan_lib.detect_arch(info) if info else None
    if arch not in training.RESOLUTION:
        problems.append("Kies een SD 1.5- of SDXL-checkpoint.")
    if problems:
        return "### ⚠️ Nog niet compleet\n" + "\n".join(f"- {p}" for p in problems), gr.update(interactive=False)
    s = training.auto_settings(arch, kind, len(paths))
    minutes = training.estimate_minutes(arch, s["steps"])
    label = "SDXL" if arch == "xl" else "SD 1.5"
    warnings = []
    if training.is_distilled(checkpoint):
        normal = [c for c in training.training_checkpoints() if not training.is_distilled(c)]
        warnings.append("⚠️ **Dit is een snel (gedestilleerd) model** (Hyper/Lightning/Turbo/LCM). Daarop trainen "
                        "geeft vaak misvormde ogen en vlekken." + (f" Beter: `{normal[0]}`; de LoRA werkt daarna ook "
                        "op dit model." if normal else ""))
    sizes = training.photo_sizes(paths, arch)
    if kind in ("person", "character") and sizes["small"]:
        up = training.find_upscaler()
        how = (f"Ze worden automatisch scherp vergroot met **{Path(up).stem}**; alleen piepkleine vallen af. "
               if up else "Ze worden gewoon vergroot (wazig): zet **4x-UltraSharp** in `models/ESRGAN` voor scherpe "
                          "vergrotingen; sterk vergrote vallen af. ")
        warnings.append(f"ℹ️ **{sizes['small']} van de {sizes['total']} foto's zijn klein** (korte kant onder "
                        f"{int(s['resolution'] * 1.5)} px). " + how +
                        f"Echte foto's van {sizes['advised']} px of groter blijven het best.")
    text = "\n\n".join(warnings + [
        f"**Wat:** {training.KINDS[kind]} · **naam/trigger:** `{training.token_for(name)}`",
        f"**Basismodel:** {checkpoint} ({label}) · trainen op {s['resolution']}×{s['resolution']}",
        f"**Training:** {len(paths)} foto's → ± {s['steps']} stappen, rank {s['rank']} · **± {minutes} minuten**",
        "**Automatisch:** onscherpe, dubbele en (bij een persoon) foto's van iemand anders vallen af; "
        "uitsnijden rond het gezicht; 4 versies opslaan, met elk 4 testbeelden; de beste wordt geïnstalleerd "
        "en als preset klaargezet.",
        "⏸️ Tijdens de training kan Forge geen afbeeldingen maken (de GPU is nodig); opdrachten wachten tot hij klaar is.",
    ])
    return text, gr.update(interactive=True)


def on_start(files, folder, name, kind, checkpoint, consent):
    paths = _paths(files, folder)
    try:
        training.start(paths, name, kind, checkpoint, consent)
    except Exception as e:
        gr.Warning(str(e))
        return gr.update(), gr.update()
    return gr.update(visible=False), gr.update(visible=True)


def on_install(version):
    job = training.latest_job()
    if not version or job is None:
        return ""
    try:
        name = training.install_version(job, version)
    except Exception as e:
        return f"⚠️ {e}"
    return f"✅ Versie {Path(version).stem} staat nu in `{name}`"


def build():
    with gr.Column(visible=True) as setup:
        gr.Markdown("### 1 · Foto's")
        with gr.Row():
            with gr.Column(scale=3):
                files = gr.File(label="Sleep hier 10 tot 40 foto's", file_count="multiple", file_types=["image"],
                                type="filepath")
                folder = gr.File(label="… of kies een map (alleen de afbeeldingen erin worden gebruikt)",
                                 file_count="directory", type="filepath")
            with gr.Column(scale=2):
                name = gr.Textbox(label="Naam", placeholder="Bijv. Anna, of Oma's theepot")
                kind = gr.Radio(choices=[(v, k) for k, v in training.KINDS.items()], value="person", label="Wat is het?")
                kind_md = gr.Markdown("")
                consent = gr.Checkbox(label="Deze persoon heeft toestemming gegeven om een model van zijn of haar "
                                            "gezicht te maken", value=False, visible=True)
        checkpoint = gr.Dropdown(choices=[], label="Trainen op basismodel (SD 1.5 / SDXL)")
        gr.Markdown("### 2 · Het plan")
        plan_md = gr.Markdown("Voeg foto's toe en geef een naam.")
        start_btn = gr.Button("🎓 Start training", variant="primary", interactive=False)

    with gr.Column(visible=False) as progress_col:
        gr.Markdown("### 3 · Voortgang")
        status_md = gr.Markdown("")
        bar = gr.HTML("")
        with gr.Accordion("Logboek", open=False):
            log = gr.Textbox(lines=10, show_label=False, interactive=False)
        cancel_btn = gr.Button("⏹️ Stoppen", visible=True)
        samples = gr.Gallery(label="Testbeelden per versie (★ = geïnstalleerd)", columns=4, height=520)
        version = gr.Radio(label="Liever een andere versie?", choices=[], visible=False)
        install_btn = gr.Button("Deze versie installeren")
        install_md = gr.Markdown("")
        new_btn = gr.Button("Nieuwe training")

    timer = gr.Timer(3.0)
    last_key = gr.State("")

    plan_inputs = [files, folder, name, kind, checkpoint]
    for src in (files, folder):
        src.upload(on_upload, [files, folder], [kind, kind_md, consent]).then(on_plan, plan_inputs, [plan_md, start_btn])
        src.clear(on_plan, plan_inputs, [plan_md, start_btn])
    for c in (name, kind, checkpoint):
        c.change(on_plan, plan_inputs, [plan_md, start_btn], show_progress=False)
    kind.change(lambda k: gr.update(visible=(k == "person")), [kind], [consent], show_progress=False)
    start_btn.click(on_start, [files, folder, name, kind, checkpoint, consent], [setup, progress_col])

    view = [status_md, bar, log, samples, version, cancel_btn, last_key]
    timer.tick(refresh, [last_key], view, show_progress=False)
    cancel_btn.click(lambda: training.cancel(), None, None).then(refresh, [last_key], view)
    install_btn.click(on_install, [version], [install_md])
    new_btn.click(lambda: (gr.update(visible=True), gr.update(visible=False)), None, [setup, progress_col])

    def on_load():
        # show a running or finished job after a page reload; fill the checkpoint list
        cps = training.training_checkpoints()
        job = training.latest_job()
        show_progress = job is not None and training.read_status(job).get("stage") not in ("error", "cancelled")
        return (gr.update(choices=cps, value=cps[0] if cps else None),
                gr.update(visible=not show_progress), gr.update(visible=show_progress))

    return on_load, [checkpoint, setup, progress_col]
