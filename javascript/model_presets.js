// Model Presets: tell the Python side when the selected checkpoint or the UI Preset changes.
// A checkpoint of another architecture first switches Neo's UI Preset (server side); when the
// preset has changed the checkpoint is sent again, so the per-model settings are applied after
// the preset's own defaults instead of being overwritten by them.
(function () {
    let lastCkpt = null;
    let lastPreset = null;
    let counter = 0;

    function value(selector) {
        const input = gradioApp().querySelector(selector);
        // while a dropdown has focus its text is the search filter, not a real value
        if (!input || document.activeElement === input) return null;
        return input.value || null;
    }

    function push(name) {
        counter++;
        for (const tab of ["txt2img", "img2img"]) {
            const ta = gradioApp().querySelector(`#model_presets_trigger_${tab} textarea`);
            if (ta) {
                // counter makes it a "change" even when the same model is chosen again
                ta.value = `${name}|${counter}`;
                updateInput(ta);
            }
        }
    }

    onUiLoaded(function () {
        setInterval(function () {
            const ckpt = value("#setting_sd_model_checkpoint input");
            const preset = value("#forge_ui_preset input");
            if (!ckpt) return;
            const presetChanged = preset !== null && lastPreset !== null && preset !== lastPreset;
            if (ckpt !== lastCkpt || presetChanged) {
                push(ckpt);
            }
            lastCkpt = ckpt;
            if (preset !== null) lastPreset = preset;
        }, 1000);
    });
})();
