"""
A plan is everything needed to make one image, as plain JSON. The wizard builds it,
the executor runs it, and a saved preset is simply a plan with a name.
"""

import math
import os
from pathlib import Path

from . import catalog, downloads

PLAN_VERSION = 1


def _model_presets_module():
    """The Model Presets script Forge already loaded (detects architecture, knows CivitAI / own settings)."""
    from modules import script_loading
    for path, module in script_loading.loaded_scripts.items():
        if Path(path).name == "model_presets.py" and hasattr(module, "detect_preset"):
            return module
    return None


def current_checkpoint():
    from modules import sd_models, shared
    return sd_models.checkpoint_aliases.get(shared.opts.sd_model_checkpoint)


def checkpoint_info(name: str):
    from modules import sd_models
    return sd_models.checkpoint_aliases.get(name) if name else None


def model_settings(info, arch: str) -> tuple[dict, str]:
    """Sampler / steps / CFG / size for this checkpoint: own saved > CivitAI > defaults."""
    settings = dict(catalog.DEFAULT_SETTINGS.get(arch, {}))
    source = "standaardwaarden"
    mp = _model_presets_module()
    if mp and info:
        try:
            entry = mp.get_entry(info)
            found = mp.effective_settings(entry, arch)
            if found:
                settings.update(found)
                source = "je eigen instellingen" if entry.get("user") else "CivitAI (instellingen van de maker)"
        except Exception as e:
            print(f"[Smart Presets] model-instellingen niet gelezen: {e}")
    return settings, source


def detect_arch(info) -> str | None:
    mp = _model_presets_module()
    if mp and info:
        return mp.detect_preset(info)
    return None


def fit_aspect(width: int, height: int, ref_w: int, ref_h: int) -> tuple[int, int]:
    """Same number of pixels as the model's preferred size, aspect ratio of the reference, multiples of 64."""
    area = width * height
    ratio = ref_w / ref_h
    w = math.sqrt(area * ratio)
    h = w / ratio
    return max(256, int(round(w / 64)) * 64), max(256, int(round(h / 64)) * 64)


def vram_total_gb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except Exception:
        pass
    return None


def build(description: str, prompt: str, negative: str, aspects: list[str], reference_size=None,
          checkpoint: str | None = None) -> dict:
    info = checkpoint_info(checkpoint) if checkpoint else current_checkpoint()
    arch = detect_arch(info)
    plan = {
        "version": PLAN_VERSION,
        "description": description,
        "checkpoint": info.name if info else None,
        "arch": arch,
        "prompt": prompt,
        "negative_prompt": negative,
        "aspects": [a for a in catalog.ASPECT_ORDER if a in aspects],
        "units": [],
        "loras": [],
        "files": [],
        "seed": -1,
        "notes": [],
        "problems": [],
    }
    if info is None:
        plan["problems"].append("Kies eerst bovenaan een checkpoint.")
        return plan
    if arch not in catalog.RECIPES:
        plan["problems"].append(
            f"Deze versie van de wizard werkt met SD 1.5 en SDXL; dit model is {catalog.ARCH_LABELS.get(arch, arch or 'onbekend')}.")
        return plan

    settings, source = model_settings(info, arch)
    plan["settings_source"] = source
    if reference_size and any(a in ("pose", "composition") for a in plan["aspects"]):
        settings["width"], settings["height"] = fit_aspect(settings["width"], settings["height"], *reference_size)
        plan["notes"].append("Formaat aangepast aan de verhouding van je voorbeeldfoto.")
    plan["settings"] = settings

    from modules import shared
    max_units = int(shared.opts.data.get("control_net_unit_count", 3))
    chosen = plan["aspects"]
    if len(chosen) > max_units:
        plan["notes"].append(f"ControlNet heeft {max_units} units (Settings → ControlNet); '{catalog.ASPECTS[chosen[-1]][0]}' valt af.")
        chosen = chosen[:max_units]
        plan["aspects"] = chosen

    files = []
    for aspect in chosen:
        r = catalog.RECIPES[arch][aspect]
        plan["units"].append({
            "aspect": aspect, "module": r.module, "model": r.model.stem, "weight": r.weight,
            "guidance_end": r.guidance_end, "pixel_perfect": r.pixel_perfect,
        })
        files.append(r.model)
        if r.lora:
            plan["loras"].append({"name": r.lora.stem, "weight": r.lora_weight})
            files.append(r.lora)
    plan["files"] = [f.__dict__ for f in files]

    # VRAM: Forge moves parts to system RAM when it doesn't fit, which works but is slower
    total = vram_total_gb()
    need = catalog.VRAM_BASE[arch] + sum(catalog.VRAM_PER_UNIT[arch][a] for a in chosen)
    plan["vram"] = {"need_gb": round(need, 1), "total_gb": round(total, 1) if total else None}
    return plan


def from_preset(data: dict, reference_size=None) -> dict:
    """Rebuild a plan from a saved or imported preset for the *current* model.

    Files and download URLs always come from catalog.py, never from the preset file, so an
    imported preset can't make Forge download something else. Only values are taken over.
    """
    aspects = [a for a in data.get("aspects", []) if a in catalog.ASPECTS]
    new = build(data.get("description", ""), str(data.get("prompt", "")), str(data.get("negative_prompt", "")),
                aspects, reference_size)
    if new.get("problems"):
        return new
    saved_arch = data.get("arch")
    if saved_arch == new["arch"]:
        # same kind of model: keep the tuned sampler settings and unit weights
        s = data.get("settings") or {}
        for key, cast in (("steps", int), ("cfg_scale", float), ("width", int), ("height", int),
                          ("sampler", str), ("scheduler", str)):
            if key in s:
                try:
                    new["settings"][key] = cast(s[key])
                except (TypeError, ValueError):
                    pass
        weights = {u.get("aspect"): u for u in data.get("units", []) if isinstance(u, dict)}
        for u in new["units"]:
            old = weights.get(u["aspect"], {})
            for key in ("weight", "guidance_end"):
                try:
                    u[key] = min(2.0, max(0.0, float(old[key])))
                except (KeyError, TypeError, ValueError):
                    pass
        new["settings_source"] = "de preset"
    elif saved_arch:
        new["notes"].append(f"Preset is gemaakt voor {catalog.ARCH_LABELS.get(saved_arch, saved_arch)}; "
                            "instellingen van dit model gebruikt.")
    new["name"] = data.get("name")
    return new


def model_files(plan: dict) -> list[catalog.ModelFile]:
    # resolve by name against the catalog: the plan may come from an edited or imported file
    known = {f.name: f for f in catalog.F.values()}
    return [known[f["name"]] for f in plan.get("files", []) if isinstance(f, dict) and f.get("name") in known]


def describe(plan: dict) -> tuple[str, list]:
    """Markdown summary of the plan, plus the files that still have to be downloaded."""
    if plan.get("problems"):
        return "### ⚠️ Dit kan nog niet\n" + "\n".join(f"- {p}" for p in plan["problems"]), []

    s = plan["settings"]
    arch = catalog.ARCH_LABELS.get(plan["arch"], plan["arch"])
    lines = [
        f"**Model:** {plan['checkpoint']} ({arch})",
        f"**Instellingen** (uit {plan.get('settings_source', '?')}): {s['sampler']} · {s['scheduler']} · "
        f"{s['steps']} steps · CFG {s['cfg_scale']} · {s['width']}×{s['height']}",
    ]
    if plan["units"]:
        parts = [f"{catalog.ASPECTS[u['aspect']][0].lower()} via `{u['model']}`" for u in plan["units"]]
        lines.append("**Van je voorbeeldfoto:** " + ", ".join(parts))
    else:
        lines.append("**Van je voorbeeldfoto:** niets (alleen de prompt)")

    missing = downloads.missing(model_files(plan))
    if missing:
        total = sum(f.size for f in missing)
        lines.append(f"**⬇ Nog op te halen:** {len(missing)} bestand(en), {downloads.human(total)}")
        for f in missing:
            lines.append(f"  - `{f.name}` ({downloads.human(f.size)}, licentie: {f.license})")
        try:
            free = downloads.free_space(missing[0].folder)
            if free < total * 1.1:
                lines.append(f"  - ⚠️ Niet genoeg schijfruimte: {downloads.human(free)} vrij")
        except Exception:
            pass
    else:
        lines.append("**Bestanden:** alles is al aanwezig")

    v = plan.get("vram") or {}
    if v.get("total_gb"):
        if v["need_gb"] <= v["total_gb"] * 0.9:
            lines.append(f"**VRAM:** ± {v['need_gb']} GB nodig, past in je {v['total_gb']} GB")
        else:
            lines.append(f"**VRAM:** ± {v['need_gb']} GB nodig, je hebt {v['total_gb']} GB: Forge zet delen tijdelijk "
                         "in het werkgeheugen, dat werkt maar is trager")
    for n in plan.get("notes", []):
        lines.append(f"ℹ️ {n}")
    return "\n\n".join(lines), missing
