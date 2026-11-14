# Changelog

## 0.5.1

**Incident of 2026-11-13: the primary did not come back**
- **Fix: a killed command took the model load with it.** `hermes sms restore --wait` run in a foreground terminal hit the tool's 180 s timeout, and Hermes killed its whole process tree. That tree included the runner and the `llama-server` it was loading: `systemd-run --scope` keeps the process a child of whoever spawned it.
  - The runner and the watcher are now transient user **services** (`sms-runner-*.service`, parent: systemd), or double-forked without a user systemd. They are never descendants of the gateway, the CLI or `gpu_job.sh`.
  - Models keep their own `sms-llm-*` scope.
  - The runner registers its own PID and process start time.
  - A freshly spawned runner counts as busy for 30 s until it does.
- **Interrupted swaps resume.** A swap/restore/start whose runner died mid-flight is relaunched once by the watcher, without counting it as a model failure.
  - Before, a SIGKILLed load was logged as "died on its own", the primary was excluded for 30 min, and the small model was loaded instead.
  - A watcher that finds a failed swap with nothing serving loads a model again instead of exiting with "stopped on purpose".
- **Fix: an idle ComfyUI blocked the idle restore forever.** Only work in ComfyUI's queue (`/queue`) now pauses the idle clock.
  - When the primary does not fit because an idle ComfyUI caches models, ComfyUI is asked to unload them (`POST /free`; new setting `comfy_free_on_idle`, default on, in both UIs). It is never killed.
  - Start-up uses the same rule.
  - The watcher logs when ComfyUI pauses it; before, it did so silently.

**Guard**
- **Blocks blocking `hermes sms` commands in the foreground:** `swap|restore --wait`, `wait`, `after-job`, `stop --graceful`. They are allowed with `background=true` and `notify=true`. The message points to `sms_restore` as the last tool.
- **No more blanket exemptions.** Mentioning `hermes sms` or `gpu_job.sh` used to exempt a whole command: `echo hermes sms; pkill llama-server` passed. Every segment is checked now, and only an executed `gpu_job.sh` lets ComfyUI through.
- **ComfyUI launched by hand is blocked while any model is loaded** or a swap runs, not only the primary.
- `guard_tools`, `block_when_unloaded` and `comfy_free_on_idle` are parsed as booleans (`"false"` was truthy).

**Waits and GPU jobs**
- **Swap budget.** Every `--wait`/`--timeout` defaults to the worst case of that swap, computed from the settings (`sms_status` → `swap_budget_s`; ≈ 22 min for a 420 s load). Before, the fixed 900 s could give up on a swap that was still draining a long reply.
- **`--wait` waits for a running swap** and asks again, instead of failing at once. `wait` ignores a result older than its launch, so a killed runner is no longer reported with the previous swap's outcome.
- **`after-job` no longer measures VRAM mid-swap.** It waits for a running swap first, and treats "another swap holds the lock" as wait-and-retry; `hermes sms check` does the same. This fixes a job whose FREE step timed out, then failed to bring any model back while the swap finished behind it.
- **`gpu_job.sh`:**
  - one job at a time (`flock`; a second one exits 3);
  - ComfyUI URL and `nvidia-smi` come from the settings (`job-layout --plain` gained fields);
  - optional `JOB_TIMEOUT`;
  - the marker records the applied mode (correct veto message);
  - job logs are kept for 14 days.

**State, agent guidance, cleanup**
- **`state.json` and `vram_profile.json` are updated under a file lock.** Concurrent writers (gateway, CLI, runner, watcher) used to drop each other's fields. A state file that does not parse is kept as `state.json.corrupt`.
- **One watcher at a time,** enforced by a lock held for its whole life instead of a PID file.
- **Agent guidance:**
  - the skill (1.1.0), the deep-work hint and the tool schemas no longer recommend a foreground `restore --wait`; `sms_restore` as the last tool is the way;
  - "never start ComfyUI outside `gpu_job.sh`";
  - what to do after a timeout or `free=FAILED`;
  - `force` added to `sms_swap`/`sms_restore`;
  - the long small-model hint is shown once per cooldown;
  - the unloaded veto also covers an interrupted swap and no longer depends on `inject_status`.
- **Runner and watcher environment.** They get an allow-listed environment, without provider API keys.
- **UIs.** Both show the watcher, other GPU users and recent failures. `sms_status` returns `watcher`, `failures` and `swap_budget_s`.
- **Removed:** a duplicate `_pid_alive` (the one in use treated an unreadable process as alive), `stop_llm`, `vram_available_for_swap`. The `/proc` scan for the engine reads `models.json` once instead of once per process. `/swap` usage messages are localized.
- **Docs:** README sections Timeouts, Troubleshooting and Development; INSTALL uninstall order; dashboard manifest version.

**Earlier unreleased changes, shipped in 0.5.1**

- **Fix: a GPU job result that was never delivered.** The guard now also refuses `gpu_job.sh` in the background without `notify=true`. With a list of patterns (`notify=["^SAVED="]`) Hermes sends no notification when the job ends, so the agent stayed silent until asked. The guard message, the skill and the template name the real parameter (`notify=true`, not `notify_on_complete`), and the hint for the small model says to send the file of the `SAVED=` line.
- **Fix: the guard blocked commands that only read `gpu_job.sh`** (`head`, `cat`). It now acts when the script is executed.
- **Fix: a swap cut a request that had just started.** The drain waits until the engine has been idle for `drain_quiet_s` (new setting, default 8 s) instead of trusting one idle read, and looks once more right before the stop.
- **Fix: the watcher quit when the engine was slow to answer.** A server whose process is alive but does not answer `/health` (a host under memory pressure) is still watched; before, the watcher logged "stopped on purpose" and exited.
- **Auto-recover reloads a model that was stopped from outside.** A model whose log ends with a clean shutdown (a signal that did not come from SmartModelProvider) is loaded again once, instead of being replaced by a smaller one for 30 minutes.
- A value-less switch repeated in the engine's `common_args` and in the model's args (`--metrics`) is passed once.
- `runner.log` and the engine logs keep one previous generation after 5 MB; expired failure records are dropped.
- Manifest: removed the `entry` field, which Hermes does not know and reported on every plugin load.
- Tests (`tests/`, standard library `unittest`).

## 0.5.0

**Catalog review**
- **Minimal environment for the engine and `nvidia-smi`.** They no longer inherit the Hermes process environment (provider API keys included): only `PATH`, `HOME`, `LANG`/`LC_*`, `LD_LIBRARY_PATH`, `TMPDIR`, `USER`/`LOGNAME`, `CUDA_*`, `GGML_*`, the user-bus variables `systemd-run --user --scope` needs (`XDG_RUNTIME_DIR`, `DBUS_SESSION_BUS_ADDRESS`), plus the `env` overrides of the engine and the model in `models.json`. A model that needs another variable (e.g. `HF_TOKEN`) declares it there.
- **Active profile.** State, logs and the provider install resolve the home through `hermes_constants.get_hermes_home()`; the detached runner and `gpu_job.sh` follow the same home (`HERMES_HOME`).
- Gateway detection uses `gateway.status.looks_like_gateway_command_line`; the dashboard no longer injects its stylesheet (the host loads the manifest css); settings are written with `ctx.set_config` when the plugin context exists in the process.

**Features and fixes**
- **Start-up always loads the primary.** When the gateway starts and a non-primary model was left serving (e.g. across a gateway restart), it is swapped for the primary through a normal restore (grace, drain, release gate; falls back to what was serving). Before, autostart only acted when nothing held the port. A running `gpu_job.sh` or ComfyUI is left alone (the idle restore brings the primary back). `hermes sms start` is unchanged: it only loads when nothing is serving.
- **Fast reply after a GPU job.** `gpu_job.sh` no longer reloads the primary when the job ends: the small model (already loaded, or the smallest that fits after `none`) delivers the result. The primary returns on demand (`sms_restore`) or through the idle restore. New setting `after_gpu_job` (`small` default, `primary` = old behaviour) and CLI `hermes sms after-job`.
- **`gpu_job.sh auto`.** Machine-aware placement: with 2+ GPUs the small model is loaded whole on one GPU (`hermes sms swap <model> --gpu N`, `sms_swap(gpu=N)`) and the job gets another GPU (`$SMS_TASK_GPU`), so no reload is needed when it ends; with one GPU the LLM is unloaded during the job. New CLI `hermes sms job-layout`; `sms_plan` returns it as `gpu_job_auto`. Pinned layouts get their own VRAM profile.
- **GPU job mode selector** in the dashboard and Desktop (setting `gpu_job_mode`): **Auto** (best for the machine), **None** (unload the LLM), **Nano** (keep the small model), **Multi** (small model on one GPU, another GPU for the job) and **Keep** (leave the loaded model alone). `gpu_job.sh auto` applies the selected mode; a mode that is not possible degrades to None and says why. API: `GET /state` → `gpu_job_modes`, `POST /gpu_job/mode`; CLI: `hermes sms job-layout --mode`.
- **Automation settings in both UIs** (dashboard card and Desktop section, `POST /settings`): which model answers after a GPU job, idle minutes before the primary returns, auto-recover and primary autostart. The Desktop pane also gains per-GPU VRAM bars and a clearer GPU-mode picker.
- **Fix (Desktop): styles that never applied.** The Desktop app ships precompiled CSS, so Tailwind classes it does not use itself do nothing in a plugin; the warning colour, error borders and the switch geometry now use inline styles.
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
