# Changelog

## Unreleased

- **Fast reply after a GPU job.** `gpu_job.sh` no longer reloads the primary when the job ends: the small model (already loaded, or the smallest that fits after `none`) delivers the result. The primary returns on demand (`sms_restore`) or through the idle restore. New setting `after_gpu_job` (`small` default, `primary` = old behaviour) and CLI `hermes sms after-job`.
- **`gpu_job.sh auto`.** Machine-aware placement: with 2+ GPUs the small model is loaded whole on one GPU (`hermes sms swap <model> --gpu N`, `sms_swap(gpu=N)`) and the job gets another GPU (`$SMS_TASK_GPU`), so no reload is needed when it ends; with one GPU the LLM is unloaded during the job. New CLI `hermes sms job-layout`; `sms_plan` returns it as `gpu_job_auto`. Pinned layouts get their own VRAM profile.
- **GPU job mode selector** in the dashboard and Desktop (setting `gpu_job_mode`): **Auto** (best for the machine), **None** (unload the LLM), **Nano** (keep the small model), **Multi** (small model on one GPU, another GPU for the job) and **Keep** (leave the loaded model alone). `gpu_job.sh auto` applies the selected mode; a mode that is not possible degrades to None and says why. API: `GET /state` → `gpu_job_modes`, `POST /gpu_job/mode`; CLI: `hermes sms job-layout --mode`.
- **Auto-recover** (setting `auto_recover`, default on). A model that fails to load, or whose server dies while serving, is replaced by another model that fits (the primary first, then from small to big) instead of leaving the agent without an LLM. The watcher now runs for every model, the primary included. A model that just failed is not retried automatically for 30 minutes.
- **Fix: crash on a single-GPU launch.** `-sm tensor|row` is dropped when the model can only use one GPU (pin, `--device X`, single `CUDA_VISIBLE_DEVICES`): llama-server loaded and then aborted on the first decode (`GGML_BACKEND_SPLIT_AXIS_UNKNOWN`).
- **Fix: release gate inflated the need** of a model whose context was reduced (stale auto-raised `min_free_vram_gb`), blocking the primary although it fitted.
- **Fix: idle restore never fired** while a resident process (e.g. an embeddings daemon) held any VRAM.
- **Fix: drain cut live generations** after `drain_timeout_s`: llama.cpp's token totals only move when a request ends; progress now also follows `n_decode_total`.
- No grace countdown when nothing is serving.

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
