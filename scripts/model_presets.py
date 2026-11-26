"""
Model Presets - per-checkpoint generation settings for Forge Neo.

When you switch checkpoints:
  1. Forge Neo's "UI Preset" is switched to the model's architecture (sd / xl / klein /
     flux3 / ...), detected from the file's tensor names. Neo's preset then selects the
     matching VAE / text encoder and its default sampler, steps and CFG.
  2. The steps / sampler / scheduler / CFG / size for that model are applied on top.

Where the per-model settings come from (later wins, per field):
  1. CivitAI: the settings the model creator used for their showcase images
     (looked up once by the file's SHA256 and cached). Only used for SD 1.5 / SDXL:
     for Flux-style models Neo's preset values are the reliable ones.
  2. Your own: saved with the "Opslaan voor dit model" button.

Everything is stored in model_presets.json in this extension's folder.
"""

import json
import os
import threading
import time
from collections import Counter
from pathlib import Path

import gradio as gr
import requests

from modules import hashes, script_callbacks, scripts, sd_models, sd_samplers, shared

EXT_DIR = Path(__file__).resolve().parents[1]
PRESETS_FILE = EXT_DIR / "model_presets.json"

FIELDS = ["steps", "sampler", "scheduler", "cfg_scale", "width", "height"]
FIELD_LABELS = {
    "steps": "Steps",
    "sampler": "Sampler",
    "scheduler": "Scheduler",
    "cfg_scale": "CFG",
    "width": "Breedte",
    "height": "Hoogte",
}
LABEL_TO_FIELD = {v: k for k, v in FIELD_LABELS.items()}

_lock = threading.RLock()


# ---------------------------------------------------------------- storage

def load_presets() -> dict:
    with _lock:
        if not PRESETS_FILE.exists():
            return {}
        try:
            return json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[Model Presets] kan {PRESETS_FILE} niet lezen: {e}")
            return {}


def save_presets(data: dict):
    with _lock:
        tmp = PRESETS_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, PRESETS_FILE)


# ---------------------------------------------------------------- checkpoints

def find_checkpoint(name: str):
    """Exact lookup only: the JS side may briefly send partial text while the user types in the dropdown."""
    if not name:
        return None
    return sd_models.checkpoint_aliases.get(name)


# Neo UI presets whose sampling recipe comes from CivitAI showcase images; for the others
# (distilled Flux / Klein / Flux 3 ...) creators' sample settings are often wrong for Neo.
CIVITAI_PRESETS = {"sd", "xl"}
_arch_cache = {}


def detect_preset(info) -> str | None:
    """Neo UI preset for a checkpoint, from its safetensors tensor names (header only, no weights)."""
    path = info.filename
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return None
    if key in _arch_cache:
        return _arch_cache[key]

    name = os.path.basename(path).lower()
    keys = []
    if path.endswith(".safetensors"):
        try:
            import struct
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                keys = [k for k in json.loads(f.read(n)) if k != "__metadata__"]
        except Exception as e:
            print(f"[Model Presets] kan header van {name} niet lezen: {e}")

    def has(fragment):
        return any(fragment in k for k in keys)

    preset = None
    if has("conditioner.embedders.1"):
        preset = "xl"
    elif has("model.diffusion_model.input_blocks"):
        preset = "sd"
    elif has("content_mode_blocks") or "flux3" in name or "flux-3" in name:
        preset = "flux3"
    elif has("double_blocks"):
        preset = "klein" if ("klein" in name or "flux2" in name or "flux-2" in name) else "flux"

    available = available_presets()
    if preset not in available:
        preset = None  # e.g. flux3 without the FLUX 3 extension
    if available:
        # before Neo's UI exists the preset list is empty: don't remember "unknown" for good
        _arch_cache[key] = preset
    return preset


def available_presets() -> set:
    from modules_forge import main_entry
    dropdown = getattr(main_entry, "ui_forge_preset", None)
    choices = getattr(dropdown, "choices", None) or []
    return {c[0] if isinstance(c, (list, tuple)) else c for c in choices}


def current_preset() -> str | None:
    return getattr(shared.opts, "forge_preset", None)


def current_checkpoint(trigger_value: str):
    name = (trigger_value or "").split("|")[0]
    return find_checkpoint(name) or find_checkpoint(shared.opts.sd_model_checkpoint)


def model_sha256(info) -> str | None:
    # CivitAI Browser+ writes the hash next to the model; that saves hashing several GB
    sidecar = os.path.splitext(info.filename)[0] + ".json"
    if os.path.exists(sidecar):
        try:
            sha = json.loads(Path(sidecar).read_text(encoding="utf-8")).get("sha256")
            if sha:
                return sha.upper()
        except Exception:
            pass
    sha = hashes.sha256(info.filename, f"checkpoint/{info.name}")
    return sha.upper() if sha else None


# ---------------------------------------------------------------- civitai

def _most_common(values):
    values = [v for v in values if v not in (None, "")]
    if not values:
        return None, 0
    value, count = Counter(values).most_common(1)[0]
    return value, len(values)


def fetch_civitai(info) -> dict:
    """Return {"settings": {...}, "info": {...}} based on the creator's showcase images."""
    sha = model_sha256(info)
    if not sha:
        return {"settings": {}, "info": {"status": "no_hash"}}

    headers = {}
    api_key = getattr(shared.opts, "custom_api_key", "")  # CivitAI Browser+ setting, optional
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    r = requests.get(f"https://civitai.com/api/v1/model-versions/by-hash/{sha}", headers=headers, timeout=20)
    if r.status_code == 404:
        return {"settings": {}, "info": {"status": "not_found", "fetched": time.strftime("%Y-%m-%d")}}
    r.raise_for_status()
    data = r.json()

    metas = [img["meta"] for img in data.get("images", []) if img.get("meta")]

    settings = {}
    steps, n = _most_common([m.get("steps") for m in metas])
    if steps:
        settings["steps"] = int(steps)
    cfg, _ = _most_common([m.get("cfgScale") for m in metas])
    if cfg:
        settings["cfg_scale"] = float(cfg)

    combos = [(m.get("sampler"), m.get("Schedule type") or "") for m in metas if m.get("sampler")]
    combo, _ = _most_common(combos)
    if combo:
        sampler, scheduler = sd_samplers.get_sampler_and_scheduler(combo[0], combo[1])
        settings["sampler"] = sampler
        settings["scheduler"] = scheduler

    size, _ = _most_common([m.get("Size") for m in metas])
    if size and "x" in str(size):
        try:
            w, h = (int(x) for x in str(size).lower().split("x")[:2])
            settings["width"], settings["height"] = w, h
        except ValueError:
            pass

    return {
        "settings": settings,
        "info": {
            "status": "ok" if settings else "no_settings",
            "model": (data.get("model") or {}).get("name"),
            "version": data.get("name"),
            "base_model": data.get("baseModel"),
            "url": f"https://civitai.com/models/{data.get('modelId')}?modelVersionId={data.get('id')}",
            "samples": len(metas),
            "fetched": time.strftime("%Y-%m-%d"),
        },
    }


def get_entry(info, force_refresh=False) -> dict:
    """Preset entry for a checkpoint; fetches CivitAI data once (or again when forced)."""
    with _lock:
        presets = load_presets()
        entry = presets.get(info.name, {})
        need_fetch = force_refresh or ("civitai_info" not in entry)
        if need_fetch and (force_refresh or shared.opts.data.get("model_presets_fetch_civitai", True)):
            try:
                print(f"[Model Presets] instellingen ophalen van CivitAI voor {info.name} ...")
                result = fetch_civitai(info)
                entry["civitai"] = result["settings"]
                entry["civitai_info"] = result["info"]
                presets[info.name] = entry
                save_presets(presets)
            except Exception as e:
                # don't cache network errors: try again next time
                print(f"[Model Presets] CivitAI ophalen mislukt voor {info.name}: {e}")
                entry = dict(entry, _error=str(e))
        return entry


def effective_settings(entry: dict, preset: str | None = None) -> dict:
    civitai = entry.get("civitai", {}) if (preset is None or preset in CIVITAI_PRESETS) else {}
    return {**civitai, **entry.get("user", {})}


# ---------------------------------------------------------------- status text

def status_html(info, entry, applied: dict | None, note: str = "") -> str:
    if info is None:
        return "<p>Geen model geselecteerd.</p>"
    parts = [f"<b>{info.name_for_extra}</b>"]
    user = entry.get("user", {})
    civ = entry.get("civitai", {})
    ci = entry.get("civitai_info", {})

    sources = []
    if user:
        sources.append("eigen instellingen")
    if civ:
        link = f'<a href="{ci.get("url")}" target="_blank">CivitAI</a>' if ci.get("url") else "CivitAI"
        sources.append(f"{link} ({ci.get('samples', 0)} voorbeelden van de maker)")
    if not sources:
        if entry.get("_error"):
            sources.append(f"CivitAI niet bereikbaar: {entry['_error']}")
        elif ci.get("status") == "not_found":
            sources.append("niet gevonden op CivitAI; sla je eigen instellingen op")
        elif ci.get("status") == "no_settings":
            sources.append("CivitAI heeft geen instellingen bij de voorbeelden")
        else:
            sources.append("nog geen preset")
    parts.append("Bron: " + " + ".join(sources))

    if applied:
        parts.append("Toegepast: " + ", ".join(f"{FIELD_LABELS[k]} {v}" for k, v in applied.items()))
    if note:
        parts.append(note)
    return "<p style='margin:0'>" + "<br>".join(parts) + "</p>"


# ---------------------------------------------------------------- settings

def on_ui_settings():
    section = ("model_presets", "Model Presets")
    shared.opts.add_option("model_presets_enabled", shared.OptionInfo(
        True, "Instellingen automatisch toepassen bij wisselen van model", section=section))
    shared.opts.add_option("model_presets_fetch_civitai", shared.OptionInfo(
        True, "Aanbevolen instellingen automatisch ophalen van CivitAI (eenmalig per model)", section=section))
    shared.opts.add_option("model_presets_fields", shared.OptionInfo(
        list(FIELD_LABELS.values()), "Welke instellingen toepassen", gr.CheckboxGroup,
        {"choices": list(FIELD_LABELS.values())}, section=section))


script_callbacks.on_ui_settings(on_ui_settings)


# ---------------------------------------------------------------- UI script

class ModelPresetsScript(scripts.Script):
    sorting_priority = 100

    def __init__(self):
        self.targets = {}
        self.ui_parts = None
        self.wired = False

    def title(self):
        return "Model Presets"

    def show(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"
        ids = {
            f"{tab}_steps": "steps",
            f"{tab}_sampling": "sampler",
            f"{tab}_scheduler": "scheduler",
            f"{tab}_cfg_scale": "cfg_scale",
            f"{tab}_width": "width",
            f"{tab}_height": "height",
        }
        for elem_id, field in ids.items():
            self.on_after_component(lambda c, f=field: self._found(f, c.component), elem_id=elem_id)
        return scripts.AlwaysVisible

    def _found(self, field, component):
        self.targets[field] = component
        self._wire()

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"
        trigger = gr.Textbox(elem_id=f"model_presets_trigger_{tab}", elem_classes=["hidden"], show_label=False)
        with gr.Accordion("Model Presets", open=False, elem_id=f"model_presets_{tab}"):
            status = gr.HTML("<p>Kies een model om zijn instellingen toe te passen.</p>")
            with gr.Row():
                apply_btn = gr.Button("Nu toepassen", size="sm")
                save_btn = gr.Button("Opslaan voor dit model", size="sm", variant="primary")
                clear_btn = gr.Button("Eigen instellingen wissen", size="sm")
                refresh_btn = gr.Button("Opnieuw van CivitAI", size="sm")
        self.ui_parts = dict(trigger=trigger, status=status, apply=apply_btn, save=save_btn, clear=clear_btn, refresh=refresh_btn)
        if not is_img2img:
            _preset_switch["trigger"] = trigger  # wired to Neo's UI Preset dropdown later, see below
        self._wire()
        return []

    # -- wiring

    def _scheduler_component(self):
        if "scheduler" in self.targets:
            return self.targets["scheduler"]  # visible dropdown
        # scheduler dropdown hidden in settings: the Sampler script keeps it in a gr.State
        runner = scripts.scripts_img2img if self.is_img2img else scripts.scripts_txt2img
        for s in getattr(runner, "alwayson_scripts", []):
            if type(s).__name__ == "ScriptSampler" and getattr(s, "scheduler", None) is not None:
                return s.scheduler
        return None

    def _wire(self):
        if self.wired or self.ui_parts is None:
            return
        needed = ["steps", "sampler", "cfg_scale", "width", "height"]
        if any(f not in self.targets for f in needed):
            return
        scheduler = self._scheduler_component()
        if scheduler is None:
            return
        self.wired = True

        t = self.targets
        p = self.ui_parts
        scheduler_is_state = isinstance(scheduler, gr.State)
        values_in = [t["steps"], t["sampler"], scheduler, t["cfg_scale"], t["width"], t["height"]]
        outputs = [t["steps"], t["sampler"], scheduler, t["cfg_scale"], t["width"], t["height"], p["status"]]

        def build(info, entry, current, note=""):
            current = dict(zip(FIELDS, current))
            preset = detect_preset(info) if info else None
            settings = effective_settings(entry, preset) if info else {}
            if info and preset and preset not in CIVITAI_PRESETS and entry.get("civitai") and not entry.get("user"):
                note = (note + " " if note else "") + f"CivitAI-waarden niet gebruikt: de {preset}-preset van Neo bepaalt de instellingen."
            enabled = {LABEL_TO_FIELD[x] for x in shared.opts.data.get("model_presets_fields", list(FIELD_LABELS.values())) if x in LABEL_TO_FIELD}
            applied = {}
            result = []
            for field in FIELDS:
                value = settings.get(field)
                if value is not None and field in enabled:
                    applied[field] = value
                    result.append(value if (field == "scheduler" and scheduler_is_state) else gr.update(value=value))
                else:
                    # a gr.State can't take gr.update(); give back its current value
                    result.append(current[field] if (field == "scheduler" and scheduler_is_state) else gr.update())
            return result + [status_html(info, entry, applied, note)]

        def unchanged(current, status=None):
            result = build(None, {}, current)
            result[-1] = gr.update() if status is None else status
            return result

        def on_model_change(trigger_value, *current):
            info = find_checkpoint((trigger_value or "").split("|")[0])
            if info is None:
                return unchanged(current)
            if not shared.opts.data.get("model_presets_enabled", True):
                return unchanged(current, status_html(info, load_presets().get(info.name, {}), None, "Automatisch toepassen staat uit (Settings → Model Presets)."))
            target = detect_preset(info)
            if target and target != current_preset():
                # the UI preset is switched first (see _wire_preset_switch); its defaults would
                # overwrite ours, so the per-model settings follow once the preset has changed
                return unchanged(current, status_html(info, {}, None, f"UI Preset → {target} ..."))
            if target:
                remember_checkpoint(target, info.name)
            return build(info, get_entry(info), current)

        def on_apply(trigger_value, *current):
            info = current_checkpoint(trigger_value)
            entry = get_entry(info) if info else {}
            return build(info, entry, current)

        def on_refresh(trigger_value, *current):
            info = current_checkpoint(trigger_value)
            entry = get_entry(info, force_refresh=True) if info else {}
            return build(info, entry, current, "CivitAI-gegevens vernieuwd.")

        def on_save(trigger_value, *current):
            info = current_checkpoint(trigger_value)
            if info is None:
                return status_html(None, {}, None)
            values = dict(zip(FIELDS, current))
            values["steps"] = int(values["steps"])
            values["width"], values["height"] = int(values["width"]), int(values["height"])
            values["cfg_scale"] = float(values["cfg_scale"])
            with _lock:
                presets = load_presets()
                entry = presets.setdefault(info.name, {})
                entry["user"] = values
                save_presets(presets)
            return status_html(info, entry, None, "Opgeslagen: " + ", ".join(f"{FIELD_LABELS[k]} {v}" for k, v in values.items()))

        def on_clear(trigger_value, *current):
            info = current_checkpoint(trigger_value)
            if info is None:
                return build(None, {}, current)
            with _lock:
                presets = load_presets()
                entry = presets.get(info.name, {})
                entry.pop("user", None)
                presets[info.name] = entry
                save_presets(presets)
            return build(info, get_entry(info), current, "Eigen instellingen gewist.")

        inputs = [p["trigger"]] + values_in
        p["trigger"].change(fn=on_model_change, inputs=inputs, outputs=outputs, show_progress=False)
        p["apply"].click(fn=on_apply, inputs=inputs, outputs=outputs)
        p["refresh"].click(fn=on_refresh, inputs=inputs, outputs=outputs)
        p["clear"].click(fn=on_clear, inputs=inputs, outputs=outputs)
        p["save"].click(fn=on_save, inputs=inputs, outputs=[p["status"]])


# ---------------------------------------------------------------- UI preset switching
#
# Neo's "UI Preset" dropdown is created after the txt2img tab, so it can't be wired in
# Script.ui(). forge_main_entry() runs inside the main Blocks once every component exists;
# we wrap it to add one event: when the checkpoint changes to another architecture, select
# the matching preset. Neo's own preset handler then loads that preset's VAE / text encoder
# and defaults, and model_presets.js re-sends the checkpoint so our per-model settings follow.

_preset_switch = {"trigger": None}
LAST_KEY = "__last_checkpoint_per_preset__"


def remember_checkpoint(preset: str, name: str):
    with _lock:
        presets = load_presets()
        last = presets.setdefault(LAST_KEY, {})
        if last.get(preset) != name:
            last[preset] = name
            save_presets(presets)


def _checkpoint_for_preset(preset: str):
    """Last checkpoint used with this preset, else the first one of that architecture."""
    last = load_presets().get(LAST_KEY, {}).get(preset)
    if last and find_checkpoint(last) and detect_preset(find_checkpoint(last)) == preset:
        return last
    for info in sd_models.checkpoints_list.values():
        if detect_preset(info) == preset:
            return info.name
    return None


def on_switch_preset(trigger_value, preset_value):
    name = (trigger_value or "").split("|")[0]
    info = find_checkpoint(name)
    if info is None or not shared.opts.data.get("model_presets_enabled", True):
        return gr.update()
    target = detect_preset(info)
    if not target or target == preset_value:
        return gr.update()

    # Neo already stored the new checkpoint under the *old* preset; give that preset back
    # a checkpoint of its own kind, so switching back to it later loads the right model.
    if preset_value and getattr(shared.opts, f"forge_checkpoint_{preset_value}", None) == name:
        shared.opts.set(f"forge_checkpoint_{preset_value}", _checkpoint_for_preset(preset_value))
    # make Neo's preset handler select the checkpoint the user just picked
    shared.opts.set(f"forge_checkpoint_{target}", name)
    shared.opts.save(shared.config_filename)
    print(f"[Model Presets] {info.name}: UI Preset {preset_value} -> {target}")
    return gr.update(value=target)


def _wire_preset_switch():
    from modules_forge import main_entry

    trigger = _preset_switch["trigger"]
    dropdown = getattr(main_entry, "ui_forge_preset", None)
    if trigger is None or dropdown is None:
        print("[Model Presets] UI Preset-koppeling niet gelukt (component niet gevonden)")
        return
    trigger.change(fn=on_switch_preset, inputs=[trigger, dropdown], outputs=[dropdown], show_progress=False)


def _patch_main_entry():
    from modules_forge import main_entry

    # on "Reload UI" this module is executed again: wrap the original, not our old wrapper
    original = getattr(main_entry.forge_main_entry, "_original", main_entry.forge_main_entry)

    def forge_main_entry(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            _wire_preset_switch()
        except Exception as e:
            print(f"[Model Presets] UI Preset-koppeling mislukt: {e}")
        return result

    forge_main_entry._original = original
    main_entry.forge_main_entry = forge_main_entry


_patch_main_entry()
