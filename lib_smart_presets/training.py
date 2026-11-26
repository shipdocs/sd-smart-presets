"""
Wizard 2, the Forge side: start a LoRA training job, keep the GPU free for it, follow
its progress and, when it's done, make the new LoRA usable right away (LoRA list + preset).

The training itself runs in trainer/train_lora.py as a separate process. While it runs,
this module holds Forge's queue lock, so image generation waits instead of running out
of VRAM; it starts again by itself when the training ends.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import analyze, downloads, plan as plan_lib, store

EXT_DIR = Path(__file__).resolve().parents[1]
JOBS = EXT_DIR / "training_jobs"
TRAINER = EXT_DIR / "trainer" / "train_lora.py"

KINDS = {
    "person": "Persoon",
    "character": "Personage (getekend, anime, game)",
    "object": "Voorwerp of product",
    "style": "Stijl",
}
RESOLUTION = {"sd": 512, "xl": 1024}
SECONDS_PER_STEP = {"sd": 0.3, "xl": 1.9}        # RTX 3060 12 GB: sd measured, xl estimated
SECONDS_PER_SAMPLE = {"sd": 3.0, "xl": 18.0}

_current = {"job": None, "proc": None, "thread": None}
_lock = threading.Lock()


# ------------------------------------------------------------------ helpers

def token_for(name: str) -> str:
    t = re.sub(r"[^a-z0-9]", "", name.lower())
    return t or "sks"


def guess_kind(image_paths: list[str]) -> tuple[str, str]:
    """Faces in most photos -> a person. Otherwise we can't tell object from style: ask."""
    from PIL import Image
    sample = image_paths[:6]
    faces = 0
    for p in sample:
        try:
            if analyze.analyze(Image.open(p))["faces"] > 0:
                faces += 1
        except Exception:
            pass
    if sample and faces / len(sample) >= 0.6:
        return "person", f"In {faces} van {len(sample)} bekeken foto's staat een gezicht: waarschijnlijk een persoon."
    return "object", "Geen (duidelijke) gezichten: kies zelf of het een voorwerp, personage of stijl is."


def auto_settings(arch: str, kind: str, n_images: int) -> dict:
    per = 100 if arch == "sd" else 80
    lo, hi = (800, 1600) if arch == "sd" else (600, 1200)
    steps = int(min(max(n_images * per, lo), hi))
    if kind == "style":
        steps = int(steps * 1.2)
    return {"steps": steps, "rank": 16, "lr": 1e-4, "resolution": RESOLUTION[arch]}


def estimate_minutes(arch: str, steps: int) -> int:
    samples = 4 * 4  # 4 versions x 4 test images
    return max(1, round((steps * SECONDS_PER_STEP[arch] + samples * SECONDS_PER_SAMPLE[arch] + 90) / 60))


DISTILLED = re.compile(r"hyper|lightning|turbo|lcm|dmd|flash|schnell", re.I)


def is_distilled(checkpoint: str) -> bool:
    """Few-step models (Hyper, Lightning, Turbo, LCM, DMD): fine for making images, poor to train on."""
    return bool(DISTILLED.search(checkpoint or ""))


def training_checkpoints() -> list[str]:
    """Checkpoints a LoRA can be trained on (SD 1.5 / SDXL): normal models first, the current one
    first among those; few-step (distilled) models last."""
    from modules import sd_models
    names = []
    for info in sd_models.checkpoints_list.values():
        if plan_lib.detect_arch(info) in RESOLUTION:
            names.append(info.name)
    cur = plan_lib.current_checkpoint()
    if cur and cur.name in names:
        names.remove(cur.name)
        names.insert(0, cur.name)
    return sorted(names, key=is_distilled)  # stable: keeps the order above within each group


def photo_sizes(image_paths: list[str], arch: str) -> dict:
    """How many photos are too small to give a sharp face at the training resolution.
    Only reads the image headers."""
    from PIL import Image
    res = RESOLUTION[arch]
    small = 0
    for p in image_paths:
        try:
            with Image.open(p) as im:
                if min(im.size) < res * 1.5:
                    small += 1
        except Exception:
            pass
    return {"small": small, "total": len(image_paths), "advised": int(res * 2)}


# ------------------------------------------------------------------ jobs

def read_status(job_dir: Path) -> dict:
    try:
        st = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
    except Exception:
        st = {"stage": "queued", "progress": 0.0, "message": "Wachten tot Forge klaar is met de huidige opdracht ...", "log": []}
    st["job"] = job_dir.name
    if st.get("stage") not in ("done", "error", "cancelled", "queued") and not running(job_dir):
        st["stage"] = "error"
        st["message"] = st.get("error") or "De training is onderbroken (Forge of de computer is herstart)."
    return st


def running(job_dir: Path | None = None) -> bool:
    proc = _current["proc"]
    t = _current["thread"]
    busy = (proc is not None and proc.poll() is None) or (t is not None and t.is_alive())
    if job_dir is None:
        return busy
    return busy and _current["job"] == job_dir


def latest_job() -> Path | None:
    if _current["job"] is not None:
        return _current["job"]
    if not JOBS.exists():
        return None
    jobs = sorted((p for p in JOBS.iterdir() if (p / "config.json").exists()), key=lambda p: p.name)
    return jobs[-1] if jobs else None


def start(image_paths: list[str], name: str, kind: str, checkpoint: str, consent: bool) -> Path:
    from modules import paths, shared

    with _lock:
        if running():
            raise RuntimeError("Er loopt al een training.")
        if kind == "person" and not consent:
            raise RuntimeError("Bevestig eerst dat deze persoon toestemming heeft gegeven.")
        info = plan_lib.checkpoint_info(checkpoint)
        arch = plan_lib.detect_arch(info) if info else None
        if arch not in RESOLUTION:
            raise RuntimeError("Kies een SD 1.5- of SDXL-checkpoint om op te trainen.")
        if len(image_paths) < 5:
            raise RuntimeError("Geef minstens 5 foto's (beter 15 tot 30).")

        slug = re.sub(r"[^\w\-]", "_", name.strip())[:40] or "lora"
        job = JOBS / f"{time.strftime('%Y%m%d_%H%M%S')}_{slug}"
        (job / "input").mkdir(parents=True)
        images = []
        for i, p in enumerate(image_paths):
            dst = job / "input" / f"{i:03d}{Path(p).suffix.lower() or '.png'}"
            shutil.copy2(p, dst)  # gradio's upload files are temporary
            images.append(str(dst))

        settings = auto_settings(arch, kind, len(images))
        model_settings, _ = plan_lib.model_settings(info, arch)
        config = {
            "name": slug,
            "display_name": name.strip(),
            "token": token_for(name),
            "kind": kind,
            "arch": arch,
            "checkpoint": info.name,
            "checkpoint_path": info.filename,
            "images": images,
            "lora_dir": downloads.folder_paths("Lora")[0],
            "insightface_root": os.path.join(paths.models_path, "insightface"),
            # test images like the model is normally used, but never fewer than 12 steps
            "eval_steps": max(12, min(30, int(model_settings.get("steps", 25)))),
            "eval_cfg": float(model_settings.get("cfg_scale", 6.0)),
            "consent": bool(consent),
            "created": time.strftime("%Y-%m-%d %H:%M"),
            **settings,
        }
        (job / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

        t = threading.Thread(target=_run, args=(job,), daemon=True, name="smart-presets-training")
        _current.update(job=job, proc=None, thread=t)
        t.start()
        print(f"[Smart Presets] training gestart: {job.name} ({arch}, {settings['steps']} stappen)")
        return job


def _free_vram():
    try:
        from backend import memory_management
        memory_management.unload_all_models()
        memory_management.soft_empty_cache(force=True)
    except Exception as e:
        print(f"[Smart Presets] VRAM vrijmaken mislukt: {e}")


def _run(job: Path):
    from modules.call_queue import queue_lock

    with queue_lock:  # no image generation while training: both don't fit in the GPU
        if (job / "cancelled").exists():
            return
        _free_vram()
        log = open(job / "train.log", "w", encoding="utf-8")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen([sys.executable, str(TRAINER), str(job)], stdout=log, stderr=subprocess.STDOUT,
                                cwd=str(job), env=env)
        _current["proc"] = proc
        proc.wait()
        log.close()

    st = read_status(job)
    if st.get("stage") == "done" and st.get("result"):
        try:
            _after_success(job, st)
        except Exception as e:
            print(f"[Smart Presets] na de training: {e}")


def _after_success(job: Path, st: dict):
    """Make the LoRA visible in Forge and add a ready-to-use preset."""
    downloads.refresh_lists()
    cfg = json.loads((job / "config.json").read_text(encoding="utf-8"))
    res = st["result"]
    prompt = f"{res['trigger']}, <lora:{res['lora_name']}:0.8>"
    if cfg["kind"] == "person":
        prompt = f"{res['trigger']}, portrait, natural light, RAW photo, sharp focus, <lora:{res['lora_name']}:0.8>"
    p = plan_lib.build(f"LoRA {cfg['display_name']}", prompt, "blurry, lowres, deformed, bad anatomy, watermark, text",
                       [], checkpoint=cfg["checkpoint"])
    p["lora_trained"] = {"name": res["lora_name"], "trigger": res["trigger"], "job": job.name}
    slug = store.save(f"LoRA {cfg['display_name']}", p)
    st_path = job / "status.json"
    data = json.loads(st_path.read_text(encoding="utf-8"))
    data["preset"] = slug
    st_path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[Smart Presets] LoRA {res['lora_name']} klaar, preset '{slug}' aangemaakt")


def cancel():
    job = _current["job"]
    if job is None:
        return
    (job / "cancelled").touch()
    proc = _current["proc"]
    if proc is not None and proc.poll() is None:
        proc.terminate()
    st_path = job / "status.json"
    try:
        data = json.loads(st_path.read_text(encoding="utf-8")) if st_path.exists() else {"log": []}
    except Exception:
        data = {"log": []}
    data.update(stage="cancelled", message="Gestopt.")
    st_path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def install_version(job_dir: Path, version_file: str) -> str:
    """Install another version than the automatic pick (replaces the installed file)."""
    st = read_status(job_dir)
    res = st.get("result") or {}
    target = res.get("lora_path")
    if not target:
        raise RuntimeError("Deze training heeft nog geen geïnstalleerde LoRA.")
    if Path(version_file).parent != job_dir:
        raise RuntimeError("Onbekende versie")
    shutil.copy2(version_file, target)
    downloads.refresh_lists()
    return Path(target).stem


def sample_images(st: dict) -> list[tuple[str, str]]:
    ev = st.get("evaluation") or {}
    out = []
    for v in ev.get("versions", []):
        label = f"stap {v['step']}" + (f" · gelijkenis {v['score']:.2f}" if v.get("score") is not None else "")
        if v["file"] == ev.get("best"):
            label = "★ " + label
        out += [(img, label) for img in v["images"]]
    return out
