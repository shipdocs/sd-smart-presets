"""
Looks at a reference image and guesses what you probably want to take over from it,
so the wizard can tick those boxes for you. Runs on the CPU (no VRAM needed).
"""

import os
import threading

import numpy as np

_detector = None
_lock = threading.Lock()


def _insightface_root() -> str:
    from modules import paths
    return os.path.join(paths.models_path, "insightface")


def _get_detector():
    global _detector
    with _lock:
        if _detector is None:
            from insightface.app import FaceAnalysis
            det = FaceAnalysis(name="buffalo_l", root=_insightface_root(),
                               providers=["CPUExecutionProvider"], allowed_modules=["detection"])
            det.prepare(ctx_id=-1, det_size=(640, 640))
            _detector = det
        return _detector


def analyze(image) -> dict:
    """image: PIL image. Returns {"aspects": [...], "summary": str, "faces": int}."""
    if image is None:
        return {"aspects": [], "summary": "", "faces": 0}

    rgb = np.array(image.convert("RGB"))
    faces = []
    error = None
    try:
        # insightface expects BGR
        faces = [f for f in _get_detector().get(rgb[:, :, ::-1]) if float(f.det_score) >= 0.5]
    except Exception as e:  # missing buffalo_l model, broken onnxruntime, ...
        error = str(e)

    h, w = rgb.shape[:2]
    aspects = []
    if faces:
        biggest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = biggest.bbox
        ratio = max(0.0, (x2 - x1) * (y2 - y1)) / float(w * h)
        aspects.append("face")
        if ratio < 0.08:
            # small face: the body is in the picture, so the pose is probably part of the idea
            aspects.append("pose")
            summary = "Persoon gevonden, met lichaam in beeld."
        else:
            summary = "Portret gevonden (gezicht groot in beeld)."
        if len(faces) > 1:
            summary += f" Let op: {len(faces)} gezichten, het grootste wordt gebruikt."
    else:
        aspects.append("composition")
        summary = "Geen gezicht gevonden: ik neem de opbouw van het beeld over."
        if error:
            summary += f" (Gezichtsherkenning werkt niet: {error})"

    return {"aspects": aspects, "summary": summary, "faces": len(faces)}
