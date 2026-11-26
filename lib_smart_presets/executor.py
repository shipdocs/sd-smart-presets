"""
Runs a plan: downloads what's missing, then generates with Forge's own txt2img pipeline.

ControlNet units are passed the same way the /sdapi/v1/txt2img API does it (as
always-on script arguments), which is far more robust than clicking through the UI.
"""

from contextlib import closing

import numpy as np

from . import downloads, plan as plan_lib


def _default_script_args(runner) -> list:
    """Default values of every script's UI components, like the API's init_default_script_args."""
    last = max([1] + [s.args_to for s in runner.scripts])
    args = [None] * last
    args[0] = 0  # no selectable script
    for s in runner.scripts:
        if s.api_info is not None:
            args[s.args_from:s.args_to] = [a.value for a in s.api_info.args]
    return args


def _controlnet_script(runner):
    for s in runner.alwayson_scripts:
        if s.title() == "ControlNet":
            return s
    return None


def _units(plan: dict, image) -> list:
    from lib_controlnet.external_code import ControlNetUnit

    arr = np.array(image.convert("RGB")).astype("uint8") if image is not None else None
    units = []
    for u in plan["units"]:
        units.append(ControlNetUnit(
            enabled=True, module=u["module"], model=u["model"], weight=float(u["weight"]),
            image=arr, guidance_start=0.0, guidance_end=float(u["guidance_end"]),
            pixel_perfect=bool(u["pixel_perfect"]), resize_mode="Crop and Resize",
        ))
    return units


def ensure_files(plan: dict, progress=None):
    missing = downloads.missing(plan_lib.model_files(plan))
    for i, f in enumerate(missing):
        def cb(frac, text, i=i):
            if progress:
                progress((i + frac) / len(missing), desc=f"Downloaden ({i + 1}/{len(missing)}) {text}")
        downloads.download(f, cb)
    if missing:
        downloads.refresh_lists()


def generate(plan: dict, image, progress=None):
    """Returns (images, infotext, seed)."""
    from modules import scripts, shared
    from modules.call_queue import queue_lock
    from modules.processing import StableDiffusionProcessingTxt2Img, process_images

    if plan.get("problems"):
        raise RuntimeError(" ".join(plan["problems"]))
    if plan["units"] and image is None:
        raise RuntimeError("Dit plan gebruikt een voorbeeldfoto, maar er is er geen.")
    current = plan_lib.current_checkpoint()
    if current is None or current.name != plan["checkpoint"]:
        raise RuntimeError(f"Kies bovenaan eerst het model {plan['checkpoint']} (nu: {current.name if current else 'geen'}).")

    ensure_files(plan, progress)

    runner = scripts.scripts_txt2img
    args = _default_script_args(runner)
    units = _units(plan, image)
    if units:
        cn = _controlnet_script(runner)
        if cn is None:
            raise RuntimeError("ControlNet-extensie niet gevonden")
        slots = cn.args_to - cn.args_from
        if len(units) > slots:
            raise RuntimeError(f"ControlNet heeft maar {slots} units")
        args[cn.args_from:cn.args_from + len(units)] = units

    # Neo's built-in "Sampler" and "Seed" scripts set p.steps / p.sampler_name / p.scheduler /
    # p.seed from their own arguments in setup(), overriding the processing object
    s = plan["settings"]
    for script in runner.alwayson_scripts:
        name = type(script).__name__
        if name == "ScriptSampler":
            args[script.args_from:script.args_from + 3] = [int(s["steps"]), s["sampler"], s.get("scheduler") or "Automatic"]
        elif name == "ScriptSeed":
            args[script.args_from] = int(plan.get("seed", -1))

    prompt = plan["prompt"]
    for lora in plan.get("loras", []):
        prompt += f" <lora:{lora['name']}:{lora['weight']}>"

    s = plan["settings"]
    if progress:
        progress(0, desc="Afbeelding maken ...")
    with queue_lock:
        with closing(StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            prompt=prompt,
            negative_prompt=plan.get("negative_prompt", ""),
            steps=int(s["steps"]),
            sampler_name=s["sampler"],
            scheduler=s.get("scheduler"),
            cfg_scale=float(s["cfg_scale"]),
            width=int(s["width"]),
            height=int(s["height"]),
            seed=int(plan.get("seed", -1)),
            batch_size=1,
            n_iter=1,
            outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples,
            outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids,
        )) as p:
            p.scripts = runner
            p.script_args = tuple(args)
            try:
                shared.state.begin(job="smart_presets")
                processed = process_images(p)
            finally:
                shared.state.end()
                shared.total_tqdm.clear()

    infotext = processed.infotexts[0] if processed.infotexts else processed.info
    # ControlNet adds its detected maps (pose skeleton, depth) after the real images
    return processed.images, infotext, processed.seed
