# Smart-Model-Swap: installation guide (v0.4.0)

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
- Set the plugin setting `primary_model` to the big model; it autostarts with the gateway when `autostart_primary` is on.

Plugin settings live under **Settings → Plugins → smart-model-swap** in Hermes Desktop, or under `plugins.entries.smart-model-swap.settings` in `config.yaml`. The full list is in the README.

## 3. Start and check
Restart the gateway so the plugin registers its tools and hooks and autostarts the primary model, then:

```bash
hermes sms status        # engine, current model, per-GPU VRAM, swap state
hermes sms plan --needed-free-gb 8
hermes sms swap <small-model> --wait
hermes sms restore --wait
```

Each load records the VRAM the model takes per GPU (`~/.hermes/cache/smart-model-swap/vram_profile.json`). `sms_plan` and the release gate use these numbers after the first load of each model.

## 4. Optional
- **SmartModelProvider as a Hermes model provider:** use **Install** in the dashboard tab or Desktop pane, or `hermes sms provider install`. Then pick it with `/model`.
- **Bundled agent skill:** `skill_view("smart-model-swap:smart-model-routing")`. It holds the routing rules and `skills/smart-model-routing/templates/gpu_job.sh`, which runs GPU jobs with the LLM swapped out and always leaves a model serving (the small one by default).

## State and logs
Everything lives in `~/.hermes/cache/smart-model-swap/`:
- `runner.log`: swaps, restores, idle watcher;
- `llama-<model>.log`: engine output;
- `state.json`: swap state machine;
- `vram_profile.json`: measured VRAM;
- `guard.log`: blocked commands;
- `ui.json`: UI slots.

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
| `server.mode: external` in `sms_status` | Something started `llama-server` outside SMP. Stop it and run `hermes sms start`. |
| Slow loads on WSL2 | GGUFs on a Windows drive (`/mnt/<drive>`, 9p) read at ~150 MB/s. Copies on the Linux filesystem load about 10× faster. |
