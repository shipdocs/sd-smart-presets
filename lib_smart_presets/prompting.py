"""
Turns "what do you want to make" (any language) into a prompt that fits the model.

Uses a local LLM when one is available (the Prompt Refiner extension with a Qwen text
encoder, the same backend Prompt Brain uses). Nothing is sent to the internet: without
a local model your text is used as the prompt as-is.
"""

import re
from pathlib import Path

SYSTEM_TAGS = """You write prompts for Stable Diffusion ({arch}) checkpoints.
The user's idea may be in Dutch or another language: translate it to English first and keep its main subject exactly.

Positive prompt rules:
- Short comma-separated tags, not sentences. At most 40 words in total (it must fit in 75 CLIP tokens).
- Order: main subject, subject details (appearance, clothing, pose, expression), setting, lighting, camera, quality.
- Include every object, person and detail the user mentions; drop nothing.
- Give the main subject weight 1.2 like (cat:1.2). Use no other weights.
- Default style is a real photograph: use terms like RAW photo, DSLR, 50mm or 85mm, natural lighting, film grain, sharp focus, detailed skin when there is a person.
- Only if the user asks for another style (anime, painting, 3d, illustration) use that style instead of photo terms.
- End with 2-3 quality tags such as masterpiece, best quality, highly detailed.

Negative prompt rules:
- 12-20 short comma-separated tags against what would ruin THIS image.
- For a photo: cartoon, anime, painting, 3d render, plus defects (blurry, lowres, jpeg artifacts, watermark, text).
- Add bad anatomy, deformed hands, extra fingers, mutated face only when there is a person or animal.
- Never put photo, realistic or photorealistic in the negative of a photo.
- Only when the user asked for a non-photo style: replace cartoon, anime with photo, realistic.

Answer in exactly this format and nothing else:
PROMPT: <positive prompt>
NEGATIVE: <negative prompt>"""

DEFAULT_NEGATIVE = "blurry, lowres, jpeg artifacts, watermark, text, bad anatomy, deformed hands, extra fingers"

SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, rep_penalty=1.0, presence_penalty=0.0)
PREFERRED = ["qwen_3_4b", "qwen3_4b", "qwen3-4b", "qwen"]


def _refiner():
    from modules import script_loading
    for path, module in script_loading.loaded_scripts.items():
        if Path(path).name == "prompt_refiner.py" and hasattr(module, "run_refine"):
            return module
    return None


def _encoder(refiner):
    names = refiner.list_encoders()
    for wanted in PREFERRED:
        for name in names:
            if wanted in name.lower():
                return name
    return names[0] if names else None


def local_llm_available() -> bool:
    try:
        refiner = _refiner()
        return bool(refiner and _encoder(refiner))
    except Exception:
        return False


def _clean(s: str) -> str:
    s = (s or "").strip().strip("\"'`")
    s = re.sub(r",?\s*--[a-z]+(\s+[\w:.]+)?", "", s, flags=re.I)
    return s.rstrip(", ")


def _fix_weights(s: str) -> str:
    # small local models sometimes write "old fisherman:1.2" without the brackets
    return re.sub(r"(^|,)(\s*)([^,()]+?):(\d(?:\.\d+)?)(?=\s*(?:,|$))", r"\1\2(\3:\4)", s)


def parse(out: str) -> tuple[str, str]:
    pos = re.search(r"PROMPT:\s*([\s\S]*?)(?=\n\s*NEGATIVE:|$)", out, re.I)
    neg = re.search(r"NEGATIVE:\s*([\s\S]*)$", out, re.I)
    return _fix_weights(_clean(pos.group(1) if pos else out)), _clean(neg.group(1) if neg else "")


def make_prompt(description: str, arch_label: str) -> tuple[str, str, str]:
    """Returns (prompt, negative, note)."""
    description = (description or "").strip()
    if not description:
        return "", DEFAULT_NEGATIVE, ""
    refiner = _refiner() if local_llm_available() else None
    if refiner is None:
        return description, DEFAULT_NEGATIVE, "Geen lokaal taalmodel: je tekst is als prompt gebruikt (schrijf hem bij voorkeur in het Engels)."

    from modules.call_queue import queue_lock
    encoder = _encoder(refiner)
    result, message = None, ""
    with queue_lock:  # don't unload the image model in the middle of a generation
        for out, msg, _bar in refiner.run_refine(
            encoder, description, SYSTEM_TAGS.format(arch=arch_label),
            SAMPLING["temperature"], SAMPLING["top_p"], SAMPLING["top_k"], SAMPLING["min_p"],
            SAMPLING["rep_penalty"], 400, -1,
            "auto (GPU if available)", "auto", "Auto (checkpoint configuration / model profile)",
            False, True, True,  # keep_loaded, free_vram, disable thinking
            presence_penalty=SAMPLING["presence_penalty"],
        ):
            if isinstance(out, str):
                result = out
            message = msg
    if not result:
        return description, DEFAULT_NEGATIVE, f"Taalmodel gaf geen antwoord ({message}); je tekst is als prompt gebruikt."
    prompt, negative = parse(result)
    return prompt or description, negative or DEFAULT_NEGATIVE, f"Prompt geschreven door {encoder} (lokaal)."
