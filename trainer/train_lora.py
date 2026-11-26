"""
Smart Presets LoRA trainer: photos in, a tested LoRA out. Runs as its own process
(started by lib_smart_presets/training.py) so its VRAM is completely freed afterwards.

    python train_lora.py <job_dir>

Reads <job_dir>/config.json, writes progress to <job_dir>/status.json.

Stages
  1. prepare   filter blurry / duplicate / other-person photos, crop around the face, square
  2. train     LoRA on the UNet attention layers (diffusers + peft); text embeddings and
               latents are computed once, so the text encoders and VAE leave the GPU
  3. evaluate  test images for every saved version; for a person the version whose faces
               look most like the photos wins (InsightFace similarity)
  4. install   best version -> Forge's LoRA folder, with preview and metadata
"""

import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

JOB = Path(sys.argv[1]).resolve()
CFG = json.loads((JOB / "config.json").read_text(encoding="utf-8"))
STATUS = JOB / "status.json"
_status = {"stage": "start", "progress": 0.0, "message": "", "log": [], "started": time.time(), "pid": os.getpid()}


def status(stage=None, progress=None, message=None, log=None, **extra):
    if stage is not None:
        _status["stage"] = stage
    if progress is not None:
        _status["progress"] = round(float(progress), 4)
    if message is not None:
        _status["message"] = message
    if log:
        _status["log"].append(f"{time.strftime('%H:%M:%S')} {log}")
        print(f"[Smart Presets trainer] {log}", flush=True)
    _status.update(extra)
    _status["updated"] = time.time()
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(_status, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STATUS)


# ====================================================================== 1. prepare

def _dhash(img: Image.Image) -> int:
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _sharpness(img: Image.Image) -> float:
    import cv2
    small = img.convert("L")
    small.thumbnail((512, 512))
    return float(cv2.Laplacian(np.asarray(small), cv2.CV_64F).var())


def _square_crop(img: Image.Image, cx: float, cy: float, side: float) -> Image.Image:
    w, h = img.size
    side = min(side, w, h)
    x0 = min(max(cx - side / 2, 0), w - side)
    y0 = min(max(cy - side / 2, 0), h - side)
    return img.crop((int(x0), int(y0), int(x0 + side), int(y0 + side)))


def prepare() -> list[Path]:
    res = CFG["resolution"]
    out_dir = JOB / "dataset"
    out_dir.mkdir(exist_ok=True)
    kind = CFG["kind"]
    dropped = {"onscherp": 0, "dubbel": 0, "geen gezicht": 0, "ander gezicht": 0, "onleesbaar": 0}

    items = []
    seen = []
    for i, path in enumerate(CFG["images"]):
        status("prepare", i / max(1, len(CFG["images"])) * 0.5, f"Foto's controleren ({i + 1}/{len(CFG['images'])})")
        try:
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        except Exception:
            dropped["onleesbaar"] += 1
            continue
        if _sharpness(img) < CFG.get("min_sharpness", 40):
            dropped["onscherp"] += 1
            continue
        h = _dhash(img)
        if any(bin(h ^ s).count("1") <= 4 for s in seen):
            dropped["dubbel"] += 1
            continue
        seen.append(h)
        items.append({"img": img, "path": path})

    class_word = CFG.get("class_word") or {"person": "person", "character": "character", "object": "object", "style": ""}[kind]

    if kind == "person":
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", root=CFG["insightface_root"], providers=["CPUExecutionProvider"],
                           allowed_modules=["detection", "recognition", "genderage"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        with_face = []
        for i, it in enumerate(items):
            status("prepare", 0.5 + i / max(1, len(items)) * 0.4, f"Gezichten zoeken ({i + 1}/{len(items)})")
            faces = [f for f in app.get(np.asarray(it["img"])[:, :, ::-1]) if float(f.det_score) >= 0.5]
            if not faces:
                dropped["geen gezicht"] += 1
                continue
            f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            it["face"] = f
            it["emb"] = f.normed_embedding
            with_face.append(it)
        items = with_face
        # who is this about: the face that most photos share
        for _ in range(2):
            if len(items) < 2:
                break
            ref = np.mean([it["emb"] for it in items], axis=0)
            ref /= np.linalg.norm(ref)
            keep = [it for it in items if float(np.dot(it["emb"], ref)) >= 0.35]
            dropped["ander gezicht"] += len(items) - len(keep)
            items = keep
        if items:
            ref = np.mean([it["emb"] for it in items], axis=0)
            np.save(JOB / "reference_embedding.npy", ref / np.linalg.norm(ref))
            if not CFG.get("class_word"):
                genders = [int(getattr(it["face"], "gender", -1)) for it in items]
                male = sum(1 for g in genders if g == 1)
                female = sum(1 for g in genders if g == 0)
                class_word = "man" if male > female * 2 else "woman" if female > male * 2 else "person"

    paths = []
    for n, it in enumerate(items):
        img = it["img"]
        if "face" in it:
            x1, y1, x2, y2 = it["face"].bbox
            side = max(x2 - x1, y2 - y1) * 2.8  # head and shoulders
            crop = _square_crop(img, (x1 + x2) / 2, (y1 + y2) / 2 + (y2 - y1) * 0.35, side)
        else:
            s = min(img.size)
            crop = _square_crop(img, img.size[0] / 2, img.size[1] / 2, s)
        crop = crop.resize((res, res), Image.LANCZOS)
        p = out_dir / f"{n:03d}.png"
        crop.save(p)
        paths.append(p)

    caption = f"{CFG['token']} style" if kind == "style" else f"photo of {CFG['token']} {class_word}".strip()
    summary = {"used": len(paths), "dropped": {k: v for k, v in dropped.items() if v}, "class_word": class_word,
               "caption": caption}
    status("prepare", 1.0, f"{len(paths)} foto's bruikbaar", log=f"dataset: {summary}", dataset=summary)
    if len(paths) < CFG.get("min_images", 5):
        raise RuntimeError(f"Te weinig bruikbare foto's ({len(paths)}); er zijn er minstens {CFG.get('min_images', 5)} nodig. "
                           f"Afgevallen: {summary['dropped']}")
    return paths


# ====================================================================== 2. train

def load_pipeline(dtype):
    import torch
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
    cls = StableDiffusionXLPipeline if CFG["arch"] == "xl" else StableDiffusionPipeline
    pipe = cls.from_single_file(CFG["checkpoint_path"], torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def train(paths: list[Path]) -> list[Path]:
    import torch
    import torch.nn.functional as F
    from diffusers import DDPMScheduler
    from diffusers.utils.state_dict_utils import (convert_all_state_dict_to_peft, convert_state_dict_to_diffusers,
                                                  convert_state_dict_to_kohya)
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file

    dev = "cuda"
    xl = CFG["arch"] == "xl"
    res = CFG["resolution"]
    caption = _status["dataset"]["caption"]
    torch.manual_seed(CFG.get("seed", 1234))
    random.seed(CFG.get("seed", 1234))

    status("train", 0.0, "Basismodel laden ...", log=f"basismodel {Path(CFG['checkpoint_path']).name}")
    pipe = load_pipeline(torch.float16)

    # --- text embeddings: one caption for all images, computed once
    with torch.no_grad():
        if xl:
            pipe.text_encoder.to(dev)
            pipe.text_encoder_2.to(dev)
            embeds, _, pooled, _ = pipe.encode_prompt(caption, device=dev, num_images_per_prompt=1,
                                                      do_classifier_free_guidance=False)
            time_ids = torch.tensor([[res, res, 0, 0, res, res]], device=dev, dtype=torch.float16)
        else:
            pipe.text_encoder.to(dev)
            embeds, _ = pipe.encode_prompt(caption, dev, 1, False)
            pooled = time_ids = None
    pipe.text_encoder.to("cpu")
    if xl:
        pipe.text_encoder_2.to("cpu")

    # --- latents (with mirrored copies), VAE in fp32: the SDXL VAE overflows in fp16
    status("train", 0.0, "Foto's voorbereiden voor training ...")
    vae = pipe.vae.to(dev, dtype=torch.float32)
    latents = []
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB")
            variants = [img] if CFG["kind"] == "person" and CFG.get("no_flip") else [img, ImageOps.mirror(img)]
            for v in variants:
                x = torch.from_numpy(np.asarray(v, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None].to(dev)
                lat = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
                latents.append(lat.to(torch.float16).cpu())
    vae.to("cpu")
    torch.cuda.empty_cache()

    # --- UNet + LoRA
    unet = pipe.unet.to(dev)
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    rank = CFG["rank"]
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    params = []
    for prm in unet.parameters():
        if prm.requires_grad:
            prm.data = prm.data.float()
            params.append(prm)
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=CFG["lr"], weight_decay=1e-2)
    except Exception:
        opt = torch.optim.AdamW(params, lr=CFG["lr"], weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda")
    sched = DDPMScheduler.from_config(pipe.scheduler.config)
    sched_alphas = sched.alphas_cumprod.to(dev)

    steps = CFG["steps"]
    save_at = sorted(set([round(steps * f) for f in CFG.get("save_fractions", [0.4, 0.6, 0.8, 1.0])]))
    versions = []
    unet.train()
    t0 = time.time()
    order = []
    for step in range(1, steps + 1):
        if not order:
            order = list(range(len(latents)))
            random.shuffle(order)
        lat = latents[order.pop()].to(dev)
        noise = torch.randn_like(lat)
        noise += 0.03 * torch.randn((1, lat.shape[1], 1, 1), device=dev, dtype=lat.dtype)  # small noise offset
        t = torch.randint(0, sched.config.num_train_timesteps, (1,), device=dev).long()
        noisy = sched.add_noise(lat, noise, t)
        with torch.autocast("cuda", dtype=torch.float16):
            kwargs = {}
            if xl:
                kwargs["added_cond_kwargs"] = {"text_embeds": pooled, "time_ids": time_ids}
            pred = unet(noisy, t, encoder_hidden_states=embeds, **kwargs).sample
        target = noise if sched.config.prediction_type == "epsilon" else sched.get_velocity(lat, noise, t)
        loss = F.mse_loss(pred.float(), target.float())
        # min-SNR weighting (gamma 5): steadier training, less overcooking
        snr = sched_alphas[t] / (1 - sched_alphas[t])
        loss = loss * torch.clamp(snr, max=5.0).mean() / snr.mean() if sched.config.prediction_type == "epsilon" else loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()

        if step % 10 == 0 or step == steps:
            per_step = (time.time() - t0) / step
            left = per_step * (steps - step)
            status("train", step / steps, f"Trainen: stap {step}/{steps} · nog ± {math.ceil(left / 60)} min",
                   loss=round(float(loss.detach()), 4))
        if step in save_at:
            sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
            sd = convert_state_dict_to_kohya(convert_all_state_dict_to_peft({f"unet.{k}": v for k, v in sd.items()}))
            sd = {k: v.detach().to("cpu", torch.float16).contiguous() for k, v in sd.items()}
            path = JOB / f"version_{step:05d}.safetensors"
            save_file(sd, str(path), metadata=_metadata(step))
            versions.append(path)
            status(log=f"versie opgeslagen na {step} stappen")

    del unet, opt, pipe, latents
    torch.cuda.empty_cache()
    return versions


def _metadata(step):
    return {
        "ss_network_module": "networks.lora",
        "ss_network_dim": str(CFG["rank"]),
        "ss_network_alpha": str(CFG["rank"]),
        "ss_base_model_version": "sdxl_base_v1-0" if CFG["arch"] == "xl" else "sd_v1",
        "ss_output_name": CFG["name"],
        "ss_steps": str(step),
        "ss_tag_frequency": json.dumps({"smart_presets": {CFG["token"]: 1}}),
        "smart_presets_trigger": _status["dataset"]["caption"],
        "smart_presets_kind": CFG["kind"],
    }


# ====================================================================== 3. evaluate

EVAL_PROMPTS = {
    "person": ["{c}, portrait, looking at the camera, natural light",
               "{c}, smiling, outdoors in a park, upper body",
               "{c}, sitting in a cafe, side view",
               "{c}, wearing a suit, studio photo"],
    "character": ["{c}, full body", "{c}, portrait", "{c}, walking in a city", "{c}, sitting on a chair"],
    "object": ["{c} on a wooden table", "{c} in a garden", "{c}, product photo, white background", "{c} in a living room"],
    "style": ["a lighthouse at the sea, {c}", "portrait of an old woman, {c}", "a city street at night, {c}", "a bowl of fruit, {c}"],
}


def evaluate(versions: list[Path]) -> dict:
    import torch
    from diffusers import DPMSolverMultistepScheduler

    caption = _status["dataset"]["caption"]
    prompts = [p.format(c=caption) for p in EVAL_PROMPTS[CFG["kind"]]]
    negative = "blurry, lowres, deformed, bad anatomy, watermark, text"
    res = CFG["resolution"]

    status("evaluate", 0.0, "Testbeelden maken ...")
    pipe = load_pipeline(torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe.to("cuda")

    ref = None
    app = None
    if CFG["kind"] == "person" and (JOB / "reference_embedding.npy").exists():
        from insightface.app import FaceAnalysis
        ref = np.load(JOB / "reference_embedding.npy")
        app = FaceAnalysis(name="buffalo_l", root=CFG["insightface_root"], providers=["CPUExecutionProvider"],
                           allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=-1, det_size=(640, 640))

    results = []
    total = len(versions) * len(prompts)
    done = 0
    for v in versions:
        pipe.load_lora_weights(str(v), adapter_name="test")
        pipe.set_adapters(["test"], adapter_weights=[CFG.get("eval_weight", 0.9)])
        images, scores = [], []
        for i, prompt in enumerate(prompts):
            g = torch.Generator("cuda").manual_seed(1000 + i)
            img = pipe(prompt, negative_prompt=negative, num_inference_steps=CFG.get("eval_steps", 25),
                       guidance_scale=CFG.get("eval_cfg", 6.0), width=res, height=res, generator=g).images[0]
            out = JOB / "samples" / v.stem
            out.mkdir(parents=True, exist_ok=True)
            img_path = out / f"{i}.png"
            img.save(img_path)
            images.append(str(img_path))
            if app is not None:
                faces = app.get(np.asarray(img)[:, :, ::-1])
                scores.append(max((float(np.dot(f.normed_embedding, ref)) for f in faces), default=0.0))
            done += 1
            status("evaluate", done / total, f"Testbeelden ({done}/{total})")
        pipe.delete_adapters("test")
        step = int(v.stem.split("_")[1])
        score = round(float(np.mean(scores)), 3) if scores else None
        results.append({"file": str(v), "step": step, "images": images, "score": score})
        status(log=f"versie {step}: " + (f"gelijkenis {score}" if score is not None else "testbeelden klaar"))

    if any(r["score"] is not None for r in results):
        best = max(results, key=lambda r: (r["score"] or 0))
        how = "meeste gelijkenis met je foto's"
    else:
        # no objective measure for styles / objects: the version at ~80% is usually the sweet spot
        best = min(results, key=lambda r: abs(r["step"] - CFG["steps"] * 0.8))
        how = "versie op ± 80% van de training"
    del pipe
    torch.cuda.empty_cache()
    return {"versions": results, "best": best["file"], "best_reason": how}


# ====================================================================== 4. install

def install(version_file: str) -> dict:
    lora_dir = Path(CFG["lora_dir"])
    lora_dir.mkdir(parents=True, exist_ok=True)
    name = CFG["name"]
    target = lora_dir / f"{name}.safetensors"
    n = 2
    while target.exists():
        target = lora_dir / f"{name}_{n}.safetensors"
        n += 1
    shutil.copy2(version_file, target)
    stem = target.with_suffix("")
    version = next((v for v in _status.get("evaluation", {}).get("versions", []) if v["file"] == version_file), None)
    if version and version["images"]:
        shutil.copy2(version["images"][0], f"{stem}.preview.png")
    caption = _status["dataset"]["caption"]
    Path(f"{stem}.json").write_text(json.dumps({
        "description": f"Getraind met Smart Presets op {len(list((JOB / 'dataset').glob('*.png')))} foto's.",
        "activation text": caption,
        "preferred weight": 0.8,
        "sd version": "SDXL" if CFG["arch"] == "xl" else "SD1",
        "notes": f"Basismodel: {Path(CFG['checkpoint_path']).name}",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"lora_name": target.stem, "lora_path": str(target), "trigger": caption}


def main():
    status("prepare", 0.0, "Starten ...", log=f"job {JOB.name}")
    try:
        paths = prepare()
        versions = train(paths)
        ev = evaluate(versions)
        _status["evaluation"] = ev
        installed = install(ev["best"])
        status("done", 1.0, "Klaar", log=f"geïnstalleerd als {installed['lora_name']} ({ev['best_reason']})",
               result=installed)
    except Exception as e:
        traceback.print_exc()
        status("error", None, str(e), log=f"fout: {e}", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
