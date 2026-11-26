"""
What the wizard knows: for every "aspect" you can take over from a reference image
(face, pose, composition, style) and every model architecture, which ControlNet /
IP-Adapter unit to use and which files that needs.

Everything the wizard does automatically comes from these tables, so adding support
for a new model type or a better model is a matter of editing this file.
"""

from dataclasses import dataclass, field

HF = "https://huggingface.co"


@dataclass(frozen=True)
class ModelFile:
    name: str          # file name on disk (also the ControlNet model name without extension)
    folder: str        # "ControlNet" or "Lora", resolved against Forge's own paths
    url: str
    size: int          # bytes, shown before downloading
    license: str

    @property
    def stem(self) -> str:
        return self.name.rsplit(".", 1)[0]


@dataclass(frozen=True)
class UnitRecipe:
    module: str                    # preprocessor name in Forge Neo
    model: ModelFile
    weight: float
    guidance_end: float = 1.0
    pixel_perfect: bool = True
    lora: ModelFile | None = None  # IP-Adapter FaceID needs its LoRA in the prompt
    lora_weight: float = 0.6
    extra_files: tuple = field(default_factory=tuple)


# ------------------------------------------------------------------ files

F = {
    # SD 1.5
    "openpose_sd15": ModelFile(
        "control_v11p_sd15_openpose_fp16.safetensors", "ControlNet",
        f"{HF}/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_openpose_fp16.safetensors",
        722_601_100, "OpenRAIL"),
    "depth_sd15": ModelFile(
        "control_v11f1p_sd15_depth_fp16.safetensors", "ControlNet",
        f"{HF}/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors",
        722_601_100, "OpenRAIL"),
    "faceid_sd15": ModelFile(
        "ip-adapter-faceid-plusv2_sd15.bin", "ControlNet",
        f"{HF}/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin",
        156_558_509, "non-commercial (InsightFace)"),
    "faceid_lora_sd15": ModelFile(
        "ip-adapter-faceid-plusv2_sd15_lora.safetensors", "Lora",
        f"{HF}/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15_lora.safetensors",
        51_059_544, "non-commercial (InsightFace)"),
    "style_sd15": ModelFile(
        "ip-adapter-plus_sd15.safetensors", "ControlNet",
        f"{HF}/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors",
        98_183_288, "Apache-2.0"),
    # SDXL
    "openpose_xl": ModelFile(
        "xinsir-controlnet-openpose-sdxl-1.0.safetensors", "ControlNet",
        f"{HF}/xinsir/controlnet-openpose-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors",
        2_502_139_104, "Apache-2.0"),
    "depth_xl": ModelFile(
        "xinsir-controlnet-depth-sdxl-1.0.safetensors", "ControlNet",
        f"{HF}/xinsir/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors",
        2_502_139_104, "Apache-2.0"),
    "faceid_xl": ModelFile(
        "ip-adapter-faceid-plusv2_sdxl.bin", "ControlNet",
        f"{HF}/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl.bin",
        1_487_555_181, "non-commercial (InsightFace)"),
    "faceid_lora_xl": ModelFile(
        "ip-adapter-faceid-plusv2_sdxl_lora.safetensors", "Lora",
        f"{HF}/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl_lora.safetensors",
        371_842_896, "non-commercial (InsightFace)"),
    "style_xl": ModelFile(
        "ip-adapter-plus_sdxl_vit-h.safetensors", "ControlNet",
        f"{HF}/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
        847_517_512, "Apache-2.0"),
}

# ------------------------------------------------------------------ aspects

ASPECTS = {
    # key: (label in the wizard, short explanation)
    "face": ("Gezicht", "dezelfde persoon (IP-Adapter FaceID)"),
    "pose": ("Houding", "dezelfde lichaamshouding (OpenPose)"),
    "composition": ("Compositie", "dezelfde opbouw en diepte (Depth)"),
    "style": ("Stijl", "dezelfde kleuren en sfeer (IP-Adapter)"),
}
ASPECT_ORDER = ["face", "pose", "composition", "style"]

RECIPES = {
    "sd": {
        "face": UnitRecipe("InsightFace+CLIP-H (IPAdapter)", F["faceid_sd15"], 0.85, pixel_perfect=False,
                           lora=F["faceid_lora_sd15"], lora_weight=0.6),
        "pose": UnitRecipe("dw_openpose_full", F["openpose_sd15"], 1.0),
        "composition": UnitRecipe("depth_anything_v2", F["depth_sd15"], 0.6, guidance_end=0.8),
        "style": UnitRecipe("CLIP-ViT-H (IPAdapter)", F["style_sd15"], 0.6, pixel_perfect=False),
    },
    "xl": {
        "face": UnitRecipe("InsightFace+CLIP-H (IPAdapter)", F["faceid_xl"], 0.85, pixel_perfect=False,
                           lora=F["faceid_lora_xl"], lora_weight=0.6),
        "pose": UnitRecipe("dw_openpose_full", F["openpose_xl"], 0.9),
        "composition": UnitRecipe("depth_anything_v2", F["depth_xl"], 0.6, guidance_end=0.8),
        "style": UnitRecipe("CLIP-ViT-H (IPAdapter)", F["style_xl"], 0.6, pixel_perfect=False),
    },
}

ARCH_LABELS = {"sd": "SD 1.5", "xl": "SDXL", "flux": "Flux", "klein": "Flux.2 Klein", "flux3": "FLUX 3"}

# Used when neither CivitAI nor your own saved values know the model
DEFAULT_SETTINGS = {
    "sd": dict(steps=28, sampler="DPM++ 2M", scheduler="Karras", cfg_scale=7.0, width=512, height=768),
    "xl": dict(steps=30, sampler="DPM++ 2M", scheduler="Karras", cfg_scale=6.0, width=832, height=1216),
}

# Rough VRAM need in GB while sampling (fp16), for the "fits in your GPU" estimate
VRAM_BASE = {"sd": 2.5, "xl": 7.0}
VRAM_PER_UNIT = {
    "sd": {"face": 1.2, "pose": 0.8, "composition": 0.8, "style": 1.0},
    "xl": {"face": 2.0, "pose": 2.6, "composition": 2.6, "style": 1.6},
}
