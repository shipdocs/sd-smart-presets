# Smart Presets for Forge Neo

Tell it what you want to make; it works out the rest. A wizard tab plus automatic
per-model settings for [Stable Diffusion WebUI Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo).

> UI is in Dutch for now.

## Wizard (tab "Smart Presets")

1. **What do you want to make?** Write it in any language. With the
   [Prompt Refiner](https://github.com/AMGarkin/sd-webui-prompt-refiner) extension and a Qwen
   text encoder installed, a local LLM turns it into a prompt for your model. Nothing leaves your PC.
2. **Example photo (optional): what should be taken over?** Face, pose, composition and/or style.
   The photo is checked first (InsightFace, on the CPU) and the likely choices are pre-ticked.
3. **The plan**, in plain language: model, sampler settings, ControlNet / IP-Adapter units,
   which files still have to be downloaded (size and licence) and whether it fits in your VRAM.
   Nothing is downloaded or generated before you click.
4. **Result**: make another one, open it in txt2img to fine-tune, or save it as a preset.

What is decided automatically:

| | SD 1.5 | SDXL |
|---|---|---|
| Face | IP-Adapter FaceID Plus v2 + its LoRA | same, SDXL version |
| Pose | DW OpenPose + ControlNet v1.1 openpose | DW OpenPose + xinsir openpose |
| Composition | Depth Anything v2 + ControlNet v1.1 depth | Depth Anything v2 + xinsir depth |
| Style | IP-Adapter Plus (ViT-H) | IP-Adapter Plus SDXL (ViT-H) |

All of this lives in `lib_smart_presets/catalog.py`.
Missing files come from Hugging Face; downloads resume after an interruption.

The image is generated through Forge's own txt2img pipeline. ControlNet units are passed as
script arguments, the same way the `/sdapi/v1/txt2img` API does it, so no UI fields are clicked behind your back.

### Presets

A preset is a saved plan (JSON) in `user_presets/`, with the example photo next to it.
**Export** writes a shareable copy *without* the photo. When you load or import a preset, the files to download
always come from the built-in catalog, never from the preset file itself.

## Wizard 2: train a LoRA (tab "Leer iets nieuws")

Drop 10 to 40 photos and give it a name. Everything else is automatic:

1. **Guess what it is.** If most photos have a face, it's a person; otherwise you choose
   between object, character and style. For a real person you must confirm they gave consent.
2. **Check the photos.** Blurry photos and duplicates are dropped. For a person, photos of
   someone else are dropped too: the face most photos share wins (InsightFace embeddings).
   Gender is detected to pick the class word (`man` / `woman`). Then each photo is cropped
   to head and shoulders and made square.
3. **Train** a LoRA on the UNet attention layers (diffusers + peft, rank 16, 8-bit AdamW, min-SNR).
   The number of steps depends on the number of photos. Captions and latents are computed once,
   so only the UNet stays on the GPU: SD 1.5 fits easily in 12 GB, SDXL fits with gradient checkpointing.
4. **Pick the best version.** Four versions are saved and each makes four test images.
   For a person, the version whose faces look most like your photos wins.
   For other kinds it's the version at about 80% of the training. You can install another version with one click.
5. **Install it.** The LoRA goes into `models/Lora`, with a preview, its trigger words and the
   base model in the `.json`. A preset is also added to Wizard 1.

The training runs as a separate process. While it runs, Forge holds its generation queue
(image requests wait), and afterwards the VRAM is completely free again. Progress survives a page reload.

## Per-model settings (accordion "Model Presets" in txt2img/img2img)

When you switch checkpoints:

1. Forge Neo's **UI Preset** is set to the model's architecture (sd / xl / klein / flux3 / flux),
   detected from the tensor names in the file.
2. Steps, sampler, scheduler, CFG and size are applied:
   - from your own saved values (**Opslaan voor dit model**),
   - otherwise from the settings the creator used for their CivitAI showcase images (SD 1.5 / SDXL only,
     looked up once by SHA256).

Settings → Model Presets turns this on or off.

## Limitations

- The wizard supports SD 1.5 and SDXL. Flux-type models only get the per-model settings for now.
- It uses the checkpoint selected at the top of the page.
- IP-Adapter FaceID and InsightFace models are for non-commercial use only.

## Roadmap

- Pick the checkpoint automatically from what you describe.
- Training: start it at night (agent scheduler), or caption with WD14 tags for styles.

## License

AGPL-3.0, the same licence as Forge Neo.
