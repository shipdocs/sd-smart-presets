"""
Smart Presets wizard: a "Smart Presets" tab that asks only what it can't work out itself.

  1. What do you want to make? (any language) + an optional example photo
  2. What do you want to take over from the photo? (pre-ticked from what's in the photo)
  3. The plan, in plain language: model, settings, ControlNet units, downloads, VRAM
  4. The result: again, open in txt2img, or save as a preset

The logic lives in lib_smart_presets/; this file is only the UI.
"""

import sys
from pathlib import Path

import gradio as gr

EXT_DIR = Path(__file__).resolve().parents[1]
if str(EXT_DIR) not in sys.path:
    sys.path.insert(0, str(EXT_DIR))

from lib_smart_presets import analyze, catalog, downloads, executor, prompting, store, ui_training  # noqa: E402
from lib_smart_presets import plan as plan_lib  # noqa: E402

from modules import infotext_utils, script_callbacks  # noqa: E402

ASPECT_CHOICES = [(f"{catalog.ASPECTS[k][0]} – {catalog.ASPECTS[k][1]}", k) for k in catalog.ASPECT_ORDER]


def _steps(n):
    """Visibility updates for the four step columns."""
    return [gr.update(visible=(i == n)) for i in range(1, 5)]


def _go_label(missing):
    if missing:
        return gr.update(value=f"⬇ {downloads.human(sum(f.size for f in missing))} ophalen en afbeelding maken", interactive=True)
    return gr.update(value="✨ Maak afbeelding", interactive=True)


def _show_plan(plan):
    text, missing = plan_lib.describe(plan)
    ok = not plan.get("problems")
    go = _go_label(missing) if ok else gr.update(value="Kan niet uitvoeren", interactive=False)
    return text, go


# ------------------------------------------------------------------ step handlers

def on_next_from_start(description, image):
    if not (description or "").strip() and image is None:
        gr.Warning("Beschrijf wat je wilt maken, of geef een voorbeeldfoto.")
        return [gr.update()] * 6
    if image is None:
        return [gr.update()] * 4 + ["", gr.update(value=[])]  # no photo: straight to the plan, see after_start
    result = analyze.analyze(image)
    return [*_steps(2), f"🔍 {result['summary']}", gr.update(value=result["aspects"])]


def prepare_plan(description, image, aspects, use_llm, prompt, negative, prompt_for, progress=gr.Progress()):
    """Builds the plan; writes a prompt first when the description changed."""
    info = plan_lib.current_checkpoint()
    arch = plan_lib.detect_arch(info)
    note = ""
    if description and (description != prompt_for or not prompt):
        if use_llm:
            progress(0.2, desc="Prompt schrijven met het lokale taalmodel ...")
            try:
                prompt, negative, note = prompting.make_prompt(description, catalog.ARCH_LABELS.get(arch, "SD"))
            except Exception as e:
                prompt, negative, note = description, prompting.DEFAULT_NEGATIVE, f"Taalmodel mislukt ({e}); je tekst is als prompt gebruikt."
        else:
            prompt, negative = description, negative or prompting.DEFAULT_NEGATIVE
        prompt_for = description
    size = image.size if image is not None else None
    chosen = (aspects or []) if image is not None else []
    plan = plan_lib.build(description, prompt, negative, chosen, size)
    if note:
        plan["notes"].insert(0, note)
    text, go = _show_plan(plan)
    return [plan, text, go, prompt, negative, prompt_for, *_steps(3)]


def after_start(description, image, aspects, use_llm, prompt, negative, prompt_for, progress=gr.Progress()):
    """Without a photo there is nothing to ask in step 2: build the plan right away."""
    if image is not None or not (description or "").strip():
        return [gr.update()] * 10
    return prepare_plan(description, image, aspects, use_llm, prompt, negative, prompt_for, progress)


def run_plan(plan, prompt, negative, image, description, aspects, progress=gr.Progress()):
    if not plan or ("settings" not in plan and not plan.get("problems")):
        # the plan lives on the server: after a Forge restart the page still shows it, the server doesn't
        chosen = (aspects or []) if image is not None else []
        plan = plan_lib.build(description, prompt, negative, chosen, image.size if image is not None else None)
    plan = dict(plan, prompt=prompt, negative_prompt=negative)
    try:
        images, infotext, seed = executor.generate(plan, image, progress)
    except Exception as e:
        import traceback
        traceback.print_exc()
        gr.Warning(str(e) or type(e).__name__)
        print(f"[Smart Presets] mislukt: {e!r}")
        text, go = _show_plan(plan)
        return [plan, gr.update(), "", f"⚠️ {e}", text, go, *_steps(3)]
    plan["last_seed"] = seed
    text, go = _show_plan(plan)
    return [plan, images, infotext, f"Seed {seed}", text, go, *_steps(4)]


def save_preset(name, plan, image):
    if not (name or "").strip():
        gr.Warning("Geef de preset een naam.")
        return gr.update(), ""
    slug = store.save(name.strip(), plan, image if plan.get("units") else None)
    return gr.update(choices=store.names(), value=slug), f"✅ Opgeslagen als **{slug}**"


def load_preset(slug):
    if not slug:
        return [gr.update()] * 13
    data, image = store.load(slug)
    plan = plan_lib.from_preset(data, image.size if image is not None else None)
    text, go = _show_plan(plan)
    desc = data.get("description", "")
    return [plan, desc, image, gr.update(value=plan.get("aspects", [])), plan.get("prompt", ""),
            plan.get("negative_prompt", ""), desc, text, go, *_steps(3)]


def import_preset(file):
    if file is None:
        return gr.update(), ""
    try:
        slug = store.import_file(file if isinstance(file, str) else file.name)
    except Exception as e:
        return gr.update(), f"⚠️ {e}"
    return gr.update(choices=store.names(), value=slug), f"✅ Geïmporteerd als **{slug}**"


def export_preset(slug):
    if not slug:
        return gr.update(value=None, visible=False)
    return gr.update(value=store.export_path(slug), visible=True)


def delete_preset(slug):
    if slug:
        store.delete(slug)
    return gr.update(choices=store.names(), value=None)


# ------------------------------------------------------------------ UI

def on_ui_tabs():
    llm = prompting.local_llm_available()

    with gr.Blocks(analytics_enabled=False) as tab:
        plan_state = gr.State({})
        prompt_for = gr.State("")

        gr.Markdown("## 🧙 Smart Presets\nZeg wat je wilt maken; de wizard regelt model-instellingen, ControlNet en downloads.")

        with gr.Tabs():
            with gr.Tab("🖼️ Maak een afbeelding"):
                # 1 ------------------------------------------------------------
                with gr.Column(visible=True) as step1:
                    gr.Markdown("### 1 · Wat wil je maken?")
                    with gr.Row():
                        with gr.Column(scale=3):
                            description = gr.Textbox(
                                label="Beschrijving (Nederlands mag)", lines=4,
                                placeholder="Bijv.: een vrouw die op een terras koffie drinkt, avondzon")
                            use_llm = gr.Checkbox(value=llm, visible=llm,
                                                  label="Prompt laten schrijven door het lokale taalmodel")
                            if not llm:
                                gr.Markdown("<small>Geen lokaal taalmodel gevonden (Prompt Refiner + Qwen): je tekst wordt "
                                            "letterlijk als prompt gebruikt, dus schrijf hem bij voorkeur in het Engels.</small>")
                        with gr.Column(scale=2):
                            reference = gr.Image(type="pil", label="Voorbeeldfoto (mag leeg)", sources=["upload", "clipboard"], height=260)
                    with gr.Accordion("Of begin met een opgeslagen preset", open=False):
                        with gr.Row():
                            preset_dd = gr.Dropdown(choices=store.names(), label="Preset", scale=3)
                            load_btn = gr.Button("Laden", scale=1)
                            export_btn = gr.Button("Exporteren", scale=1)
                            delete_btn = gr.Button("🗑️ Verwijderen", scale=1)
                        with gr.Row():
                            import_file = gr.File(label="Preset importeren (.json)", file_types=[".json"], type="filepath")
                            export_file = gr.File(label="Export", visible=False, interactive=False)
                        preset_status = gr.Markdown("")
                    next1 = gr.Button("Volgende →", variant="primary")

                # 2 ------------------------------------------------------------
                with gr.Column(visible=False) as step2:
                    gr.Markdown("### 2 · Wat wil je van de foto overnemen?")
                    analysis_md = gr.Markdown("")
                    aspects = gr.CheckboxGroup(choices=ASPECT_CHOICES, label="Overnemen", value=[])
                    with gr.Row():
                        back2 = gr.Button("← Terug")
                        next2 = gr.Button("Volgende →", variant="primary")

                # 3 ------------------------------------------------------------
                with gr.Column(visible=False) as step3:
                    gr.Markdown("### 3 · Het plan")
                    plan_md = gr.Markdown("")
                    with gr.Accordion("Prompt bekijken of aanpassen", open=False):
                        prompt_tb = gr.Textbox(label="Prompt", lines=3)
                        negative_tb = gr.Textbox(label="Negatieve prompt", lines=2)
                    with gr.Row():
                        back3 = gr.Button("← Terug")
                        go_btn = gr.Button("✨ Maak afbeelding", variant="primary")

                # 4 ------------------------------------------------------------
                with gr.Column(visible=False) as step4:
                    gr.Markdown("### 4 · Resultaat")
                    gallery = gr.Gallery(label="Resultaat", columns=3, height=520, object_fit="contain")
                    result_md = gr.Markdown("")
                    infotext = gr.Textbox(visible=False)
                    with gr.Row():
                        again_btn = gr.Button("🎲 Nog een (andere seed)", variant="primary")
                        send_btn = gr.Button("Openen in txt2img", elem_id="smart_presets_to_txt2img")
                        edit_btn = gr.Button("← Plan aanpassen")
                        new_btn = gr.Button("Nieuwe opdracht")
                    with gr.Row():
                        preset_name = gr.Textbox(label="Bevalt het? Sla op als preset", placeholder="Naam, bijv. 'Anna op terras'", scale=3)
                        save_btn = gr.Button("👍 Opslaan", scale=1)
                    save_status = gr.Markdown("")

            with gr.Tab("🎓 Leer iets nieuws (LoRA-training)"):
                training_load, training_load_outputs = ui_training.build()

        steps = [step1, step2, step3, step4]

        # -- wiring
        prep_inputs = [description, reference, aspects, use_llm, prompt_tb, negative_tb, prompt_for]
        prep_outputs = [plan_state, plan_md, go_btn, prompt_tb, negative_tb, prompt_for, *steps]

        next1.click(on_next_from_start, [description, reference], [*steps, analysis_md, aspects]).then(
            after_start, prep_inputs, prep_outputs)
        next2.click(prepare_plan, prep_inputs, prep_outputs)
        back2.click(lambda: _steps(1), None, steps)
        back3.click(lambda img: _steps(2 if img is not None else 1), [reference], steps)

        run_outputs = [plan_state, gallery, infotext, result_md, plan_md, go_btn, *steps]
        run_inputs = [plan_state, prompt_tb, negative_tb, reference, description, aspects]
        go_btn.click(run_plan, run_inputs, run_outputs)
        again_btn.click(lambda p: dict(p or {}, seed=-1), [plan_state], [plan_state]).then(run_plan, run_inputs, run_outputs)
        edit_btn.click(lambda: _steps(3), None, steps)
        new_btn.click(lambda: [*_steps(1), "", None, [], "", "", "", {}],
                      None, [*steps, description, reference, aspects, prompt_tb, negative_tb, prompt_for, plan_state])

        save_btn.click(save_preset, [preset_name, plan_state, reference], [preset_dd, save_status])
        load_btn.click(load_preset, [preset_dd],
                       [plan_state, description, reference, aspects, prompt_tb, negative_tb, prompt_for, plan_md, go_btn, *steps])
        export_btn.click(export_preset, [preset_dd], [export_file])
        delete_btn.click(delete_preset, [preset_dd], [preset_dd])
        import_file.upload(import_preset, [import_file], [preset_dd, preset_status])

        # "Openen in txt2img": Forge's own paste mechanism (prompt, settings and ControlNet unit
        # settings from the infotext); the reference image itself has to be dropped in again there
        infotext_utils.register_paste_params_button(infotext_utils.ParamBinding(
            paste_button=send_btn, tabname="txt2img", source_text_component=infotext))
        send_btn.click(fn=None, js="switch_to_txt2img")

        def on_load():
            return [gr.update(choices=store.names()), *training_load()]

        tab.load(on_load, None, [preset_dd, *training_load_outputs])

    return [(tab, "Smart Presets", "smart_presets")]


script_callbacks.on_ui_tabs(on_ui_tabs)


# ------------------------------------------------------------------ API (only with --api)
#
#   POST /smart-presets/plan      {"description", "prompt"?, "negative_prompt"?, "aspects"?, "image"?: base64, "use_llm"?}
#   POST /smart-presets/generate  {"plan": {...}, "image"?: base64}  -> {"images": [base64], "info", "seed"}
#   POST /smart-presets/analyze   {"image": base64}
#   POST /smart-presets/train     {"images": [base64], "name", "kind"?, "checkpoint"?, "consent"}
#   GET  /smart-presets/train/status

def on_app_started(_demo, app):
    from modules import shared
    if not getattr(shared.cmd_opts, "api", False):
        return
    from fastapi import Body, HTTPException
    from modules.api.api import decode_base64_to_image, encode_pil_to_base64

    def img(b64):
        return decode_base64_to_image(b64).convert("RGB") if b64 else None

    @app.post("/smart-presets/analyze")
    def api_analyze(req: dict = Body(...)):
        return analyze.analyze(img(req.get("image")))

    @app.post("/smart-presets/plan")
    def api_plan(req: dict = Body(...)):
        image = img(req.get("image"))
        description = req.get("description", "")
        prompt, negative = req.get("prompt"), req.get("negative_prompt")
        if not prompt:
            if req.get("use_llm", True) and prompting.local_llm_available():
                arch = plan_lib.detect_arch(plan_lib.current_checkpoint())
                prompt, negative, _ = prompting.make_prompt(description, catalog.ARCH_LABELS.get(arch, "SD"))
            else:
                prompt = description
        aspects = req.get("aspects")
        if aspects is None:
            aspects = analyze.analyze(image)["aspects"] if image is not None else []
        p = plan_lib.build(description, prompt, negative or prompting.DEFAULT_NEGATIVE, aspects,
                           image.size if image is not None else None)
        text, missing = plan_lib.describe(p)
        return {"plan": p, "summary": text, "missing": [f.name for f in missing]}

    @app.post("/smart-presets/generate")
    def api_generate(req: dict = Body(...)):
        try:
            images, info, seed = executor.generate(req["plan"], img(req.get("image")))
        except Exception as e:
            raise HTTPException(400, str(e))
        return {"images": [encode_pil_to_base64(i) for i in images], "info": info, "seed": seed}


    @app.post("/smart-presets/train")
    def api_train(req: dict = Body(...)):
        """{"images": [base64, ...], "name", "kind"?, "checkpoint"?, "consent": bool}"""
        import tempfile
        from lib_smart_presets import training
        tmp = Path(tempfile.mkdtemp(prefix="smart_presets_"))
        paths = []
        for i, b64 in enumerate(req.get("images", [])):
            p = tmp / f"{i:03d}.png"
            img(b64).save(p)
            paths.append(str(p))
        kind = req.get("kind") or training.guess_kind(paths)[0]
        checkpoint = req.get("checkpoint") or (training.training_checkpoints() or [None])[0]
        try:
            job = training.start(paths, req.get("name", "lora"), kind, checkpoint, bool(req.get("consent")))
        except Exception as e:
            raise HTTPException(400, str(e))
        return {"job": job.name, "kind": kind, "checkpoint": checkpoint}

    @app.get("/smart-presets/train/status")
    def api_train_status():
        from lib_smart_presets import training
        job = training.latest_job()
        return training.read_status(job) if job else {}


script_callbacks.on_app_started(on_app_started)
