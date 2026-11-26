# Run by Forge when the extension is installed or Forge starts.
# peft is needed for LoRA training (Wizard 2); everything else ships with Forge Neo.
import launch

if not launch.is_installed("peft"):
    launch.run_pip("install peft", "peft for Smart Presets LoRA training")
