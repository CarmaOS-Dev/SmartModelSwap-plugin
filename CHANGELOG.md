# Changelog

## 0.4.0

**Automatic routing**
- **Bundled skill `smart-model-swap:smart-model-routing`.** Decision rules for big vs small (GPU jobs, deep work, quick or voice chat), plus `templates/gpu_job.sh`, which frees VRAM, runs a job and always restores the primary.
- **Idle restore.** A watcher brings the primary back after `idle_restore_min` minutes without engine activity, once its VRAM is free.
- **Hints** (`pre_llm_call`). One line is added to the turn when a GPU job is requested on the primary, deep work on the small model, or a voice conversation is detected (`pre_transcription`).
- **Guard** (`pre_tool_call`). Vetoes commands that stop, kill or launch `llama-server` behind SmartModelProvider, ComfyUI launches while the primary holds the GPU, and `gpu_job.sh` run in the foreground.
- **Unloaded on purpose.** While a GPU job or a model load is in flight, a new turn ends at once with an explanation instead of failing with `HTTP 503 Loading model`.

**Swaps**
- **Progress-aware drain.** In-flight replies may finish while they keep generating (`drain_max_s`); a request is cut only when it stalls (`drain_timeout_s`).
- **Release gate.** Before loading a model, every GPU it uses must have room for it. The per-GPU need is measured on each load and stored as a median per context size. If the VRAM is not freed, the previous model is reloaded instead.
- **`sms_plan`** works per GPU from the measured usage, assumes single-GPU tasks such as ComfyUI by default, and returns an `action` (`stay` / `swap` / `unload` / `impossible`).
- **Blocking CLI.** `hermes sms swap|restore --wait`, `hermes sms wait`, `hermes sms check`, `hermes sms stop --graceful`.
- A swap or restore to the model that is already loaded is a no-op; `--force` reloads it.

**Models and UI**
- **Shared context.** Saving a model's `-c` applies it to every model of the engine, because Hermes caches one context length per model id.
- **Model names** accept upper and lower case and the basic signs `. _ - + ( ) [ ] ,`. Lookups are case-insensitive.
- **English and Spanish.** Both UIs follow the Hermes language picker. Backend messages follow the UI language, or `display.language` elsewhere.
- **GPU job status.** The UIs show "LLM unloaded for a GPU job" instead of "server down" while a `gpu_job.sh` runs.
- `models.example.json` seeds the registry on a fresh install; the live `models.json` is not tracked.

**Fixes**
- Plugin settings are written through the public `hermes_cli.config.save_config` (catalog rule 9).
- A zombie idle watcher no longer prevents a new one from starting.

## 0.3.0
- **SmartModelProvider** launches and stops the inference engine itself; the plugin no longer depends on a systemd unit.
- The primary model autostarts with the gateway.
- **Launch-parameter editor** with a `--help`-driven catalog, command-line import, preview and validation.
- Ollama, vLLM and LM Studio engines (detection only).
