"""
Saved presets: one JSON file per preset in user_presets/, with the reference image next
to it (it stays on your PC; export leaves it out unless you ask for it).
"""

import json
import os
import re
import time
from pathlib import Path

EXT_DIR = Path(__file__).resolve().parents[1]
STORE = EXT_DIR / "user_presets"


def _slug(name: str) -> str:
    return re.sub(r"[^\w\- ]", "", name).strip().replace(" ", "_")[:60] or time.strftime("preset_%Y%m%d_%H%M%S")


def names() -> list[str]:
    if not STORE.exists():
        return []
    return sorted(p.stem for p in STORE.glob("*.json"))


def save(name: str, plan: dict, image=None) -> str:
    STORE.mkdir(exist_ok=True)
    slug = _slug(name)
    data = dict(plan, name=name, saved=time.strftime("%Y-%m-%d %H:%M"))
    data.pop("problems", None)
    if image is not None:
        image.convert("RGB").save(STORE / f"{slug}.png")
        data["reference_image"] = f"{slug}.png"
    else:
        data.pop("reference_image", None)
    tmp = STORE / f"{slug}.json.tmp"
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STORE / f"{slug}.json")
    return slug


def load(slug: str) -> tuple[dict, object]:
    from PIL import Image
    data = json.loads((STORE / f"{slug}.json").read_text(encoding="utf-8"))
    image = None
    ref = data.get("reference_image")
    if ref and (STORE / ref).exists():
        image = Image.open(STORE / ref).convert("RGB")
    return data, image


def delete(slug: str):
    for p in (STORE / f"{slug}.json", STORE / f"{slug}.png"):
        if p.exists():
            p.unlink()


def export_path(slug: str) -> str:
    """A shareable copy without the reference image and machine-specific fields."""
    data, _ = load(slug)
    for key in ("reference_image", "vram", "seed"):
        data.pop(key, None)
    out = STORE / "export"
    out.mkdir(exist_ok=True)
    path = out / f"{slug}.smartpreset.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def import_file(path: str) -> str:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "units" not in data or "prompt" not in data:
        raise ValueError("Dit is geen Smart Preset-bestand")
    data.pop("reference_image", None)
    name = str(data.get("name") or Path(path).stem.replace(".smartpreset", ""))
    # never overwrite a preset you already have (and its photo)
    base, n = name, 2
    while _slug(name) in names():
        name, n = f"{base} ({n})", n + 1
    return save(name, data)
