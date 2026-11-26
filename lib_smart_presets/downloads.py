"""
Finds model files on disk and downloads the missing ones (resumable, to a .part file first).
"""

import os
import shutil
import sys

import requests

from .catalog import ModelFile

CHUNK = 1 << 20


def folder_paths(folder: str) -> list[str]:
    from modules import shared
    if folder == "ControlNet":
        from modules_forge.shared import controlnet_dir
        extra = [getattr(shared.opts, "control_net_models_path", None), *getattr(shared.cmd_opts, "controlnet_dirs", [])]
        return [p for p in [controlnet_dir, *extra] if p and os.path.isdir(p)]
    if folder == "Lora":
        return [p for p in [shared.cmd_opts.lora_dir, *getattr(shared.cmd_opts, "lora_dirs", [])] if p and os.path.isdir(p)]
    raise ValueError(folder)


def find_local(f: ModelFile) -> str | None:
    """Path of the file if present, matched by name without extension (case-insensitive)."""
    wanted = f.stem.lower()
    for root in folder_paths(f.folder):
        for dirpath, _dirs, files in os.walk(root, followlinks=True):
            for name in files:
                if name.endswith((".part", ".aria2")):
                    continue
                if os.path.splitext(name)[0].lower() == wanted:
                    return os.path.join(dirpath, name)
    return None


def missing(files) -> list[ModelFile]:
    seen, result = set(), []
    for f in files:
        if f.name not in seen and find_local(f) is None:
            seen.add(f.name)
            result.append(f)
    return result


def free_space(folder: str) -> int:
    return shutil.disk_usage(folder_paths(folder)[0]).free


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit == "GB" else f"{n:.0f} {unit}"
        n /= 1024


def download(f: ModelFile, progress=None) -> str:
    """Download one file into the first folder of its kind. progress(fraction, text) is optional."""
    target_dir = folder_paths(f.folder)[0]
    target = os.path.join(target_dir, f.name)
    part = target + ".part"
    done = os.path.getsize(part) if os.path.exists(part) else 0

    headers = {"User-Agent": "sd-smart-presets"}
    if done:
        headers["Range"] = f"bytes={done}-"
    with requests.get(f.url, headers=headers, stream=True, timeout=30, allow_redirects=True) as r:
        if r.status_code == 416:  # already complete
            pass
        else:
            r.raise_for_status()
            if done and r.status_code != 206:
                done = 0  # server ignored the range: start over
            total = done + int(r.headers.get("content-length", 0)) or f.size
            with open(part, "ab" if done else "wb") as out:
                for chunk in r.iter_content(CHUNK):
                    out.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(min(done / total, 1.0), f"{f.name}: {human(done)} / {human(total)}")

    if f.size and os.path.getsize(part) != f.size:
        raise RuntimeError(f"{f.name}: download onvolledig ({human(os.path.getsize(part))} van {human(f.size)}); probeer opnieuw, hij gaat verder waar hij was")
    os.replace(part, target)
    print(f"[Smart Presets] gedownload: {target}")
    return target


def refresh_lists():
    """Let ControlNet and the LoRA list see newly downloaded files without a restart."""
    try:
        from lib_controlnet import global_state
        global_state.update_controlnet_filenames()
    except Exception as e:
        print(f"[Smart Presets] ControlNet-lijst verversen mislukt: {e}")
    networks = sys.modules.get("networks")
    if networks is not None and hasattr(networks, "list_available_networks"):
        try:
            networks.list_available_networks()
        except Exception as e:
            print(f"[Smart Presets] LoRA-lijst verversen mislukt: {e}")
