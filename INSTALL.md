# Smart-Model-Swap: installation guide (v0.5.0)

## Requirements
- Hermes Agent **0.23.0** or newer.
- Linux or WSL2 with an NVIDIA GPU and `nvidia-smi`. On WSL2 the plugin also looks for `/usr/lib/wsl/lib/nvidia-smi`.
- A built `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp) and at least one GGUF model. Two are needed to swap: a big **primary** model and a small one.
- Optional: a systemd user session (`systemd-run --user`). With it, the engine runs in its own transient scope and survives gateway restarts; without it, the engine runs in its own process session.
- No Python dependencies: the plugin uses the standard library only.

## 1. Install
From the plugin catalog:

```bash
hermes plugins install smart-model-swap
hermes plugins enable smart-model-swap
```

Or from source:

```bash
git clone https://github.com/CarmaOS-Dev/SmartModelSwap-plugin ~/.hermes/plugins/smart-model-swap
hermes plugins enable smart-model-swap
```

## 2. Configure the engine and your models
On first load the plugin copies `models.example.json` to `models.json`. That file is the live registry, and git does not track it. Edit it from the dashboard tab (**Engines · SmartModelProvider → configure**, **Models → edit**), from the Desktop pane, or by hand.

**`engines.llamacpp`**
- `bin`: path to `llama-server` (`~` is expanded).
- `host` / `port`: where the engine listens, `127.0.0.1:8080` by default.
- `common_args`: flags shared by every model.
- `env`: extra environment, e.g. `CUDA_VISIBLE_DEVICES=0,1`.

**`models[]`**
- `name`: letters, digits and `. _ - + ( ) [ ] ,`.
- `gguf`, `mmproj` and `args` (the model's own flags).
- `min_free_vram_gb` and `load_timeout_s`.

**Rules**
- Keep the **same `-c` on every model**: Hermes caches one context length per model id. The editor enforces it.
- Set the plugin setting `primary_model` to the big model (or pick it in the **primary** slot of the UI).

**Plugin settings** live under **Settings → Plugins → smart-model-swap** in Hermes Desktop, or under `plugins.entries.smart-model-swap.settings` in `config.yaml`. The full list is in the README. The ones you are most likely to change are also in the plugin's own UIs (dashboard card and Desktop section):

| Setting | Default | What it does |
|---|---|---|
| `autostart_primary` | on | The gateway always starts on the primary model (see step 3). |
| `idle_restore_min` | 15 | Minutes of inactivity on a smaller model before the primary comes back. 0 disables. |
| `auto_recover` | on | If a model fails to load or its server dies, load another one that fits. |
| `gpu_job_mode` | `auto` | What `gpu_job.sh auto` does with the LLM during a GPU job (step 4). |
| `after_gpu_job` | `small` | Which model answers when a GPU job ends: `small`, or `primary` to reload the big one at once. |

**Coming from a `llama-server` systemd unit?** Disable it so it does not race SMP for the port at boot: `systemctl --user disable --now llama-server`. SMP never reads or starts it.

## 3. Start and check
Restart the gateway so the plugin registers its tools and hooks and loads the primary model.

**Start-up rule.** With `autostart_primary` on (the default), the gateway always ends its start-up on the primary:
- nothing is serving: the primary is loaded in the background, after waiting up to `gpu_wait_s` (180 s) for the NVIDIA driver and the model file;
- a smaller model was left serving (for example across a gateway restart): it is swapped for the primary through a normal restore (grace countdown, drain, release gate; if the primary does not fit, the model that was serving stays);
- a `gpu_job.sh` or ComfyUI is running: nothing is touched, and the primary comes back through the idle restore.

`hermes sms start` does only the first case: it never replaces a model that is serving.

Then check:

```bash
hermes sms wait          # block until the start-up load finishes
hermes sms status        # engine, current model, per-GPU VRAM, swap state
hermes sms plan --needed-free-gb 8
hermes sms swap <small-model> --wait
hermes sms check         # may the primary load now? lists who holds the GPU
hermes sms restore --wait
```

Each load records the VRAM the model takes per GPU (`~/.hermes/cache/smart-model-swap/vram_profile.json`). `sms_plan` and the release gate use these numbers after the first load of each model.

## 4. GPU jobs (ComfyUI, training, …)
Run GPU-heavy work through the bundled script, in the background. It frees VRAM, runs the job and always leaves a model serving:

```bash
bash ~/.hermes/plugins/smart-model-swap/skills/smart-model-routing/templates/gpu_job.sh auto -- bash /path/to/my_job.sh
```

- `auto` applies the mode chosen in the UI (`gpu_job_mode`): **Auto** (best for the machine: small model on one GPU and the job on another with 2+ GPUs, no LLM with one GPU), **None** (unload the LLM), **Nano** (keep the small model), **Multi** (small model on one GPU, another GPU for the job) or **Keep** (leave the loaded model alone). A mode that is not possible degrades to None.
- The job reads its GPU from `$SMS_TASK_GPU` (ComfyUI: `--cuda-device "${SMS_TASK_GPU:-0}"`).
- `hermes sms job-layout` prints what `auto` will do on this machine.
- When the job ends, the small model answers (`after_gpu_job`); the primary returns on demand or when idle.
- Each run writes `gpu_job_<date>.log` in the cache folder; read its `RESULT` lines.

## 5. Optional
- **SmartModelProvider as a Hermes model provider:** use **Install** in the dashboard tab or Desktop pane, or `hermes sms provider install`. Then pick it with `/model`.
- **Bundled agent skill:** `skill_view("smart-model-swap:smart-model-routing")`. It holds the routing rules and `skills/smart-model-routing/templates/gpu_job.sh`, which runs GPU jobs with the LLM swapped out and always leaves a model serving (the small one by default).

## State and logs
Everything lives in `~/.hermes/cache/smart-model-swap/`:
- `runner.log`: swaps, restores, start-up, idle watcher and auto-recover;
- `llama-<model>.log`: engine output;
- `gpu_job_<date>.log`: one per `gpu_job.sh` run; `gpu_job.active` marks a job in progress;
- `state.json`: swap state machine; `idle_watch.json`: the running watcher;
- `vram_profile.json`: measured VRAM;
- `guard.log`: blocked commands;
- `ui.json`: UI slots.

With a non-default Hermes profile (`HERMES_HOME`), the folder is under that home.

## Uninstall
```bash
hermes sms stop
hermes plugins disable smart-model-swap
hermes sms provider remove   # only if you installed the provider
```
Then delete `~/.hermes/plugins/smart-model-swap` and, optionally, `~/.hermes/cache/smart-model-swap`.

## Troubleshooting
| Symptom | What to check |
|---|---|
| `nvidia-smi: No such file` on WSL | Set the `nvidia_smi` setting to `/usr/lib/wsl/lib/nvidia-smi`, or put `nvidia-smi` on `PATH`. |
| Model never comes up | Read `runner.log` and `llama-<model>.log`. Run `hermes sms cmd <model>` to see the exact launch line. |
| `RESOURCES NOT RELEASED` in `runner.log` | Another process holds the VRAM. `hermes sms check` lists who. |
| The small model was loaded and the big one came back after a gateway restart | Expected: start-up always loads the primary. Turn `autostart_primary` off to keep whatever is serving. |
| The gateway started but a smaller model is still serving | A GPU job or ComfyUI was running, or the primary did not fit: see `smart-model-swap autostart` in the gateway log and `runner.log`. `hermes sms check` lists who holds the VRAM; then `hermes sms restore`. |
| A different model than the one requested is serving | Auto-recover replaced a model that failed. `hermes sms status` → `last_result` and `runner.log` (`auto-recover`) say why. |
| `server.mode: external` in `sms_status` | Something started `llama-server` outside SMP. Stop it and run `hermes sms start`. |
| Slow loads on WSL2 | GGUFs on a Windows drive (`/mnt/<drive>`, 9p) read at ~150 MB/s. Copies on the Linux filesystem load about 10× faster. |
