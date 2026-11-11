# Smart-Model-Swap — Installation Guide (v0.3.0)

This is a Hermes Agent plugin for hot-swapping local LLM models by task, with VRAM auto-detection and CoMAs image-generation cohabitation. No external Python packages required beyond the standard library.

## Quick checklist

- [ ] You have Hermes Agent v2.x (`hermes --version`) running as your gateway process.
- [ ] NVIDIA + `nvidia-smi` installed on your Linux host (WSL mounts `/mnt/`).
- [ ] A llama.cpp build ready and configured in `~/.hermes/plugins/smart-model-swap/engine.llamacpp`.
- [ ] At least one GGUF model loaded into the model registry (`models.json`).
- [ ] The desktop pane installed (or you're running headless via `hermes sms`).

## Option A: Install from the catalog (if listed)

```bash
# After an official release, this should work automatically or with:
hermes plugins install smart-model-swap --force
hermes plugins enable smart-model-swap
```

(If it's not yet in the official catalog, proceed to **Option B** below.)

## Option B: Manual installation (current, works today)

1. **Copy the plugin directory**:

   ```bash
   cp -r /path/to/smart-model-swap ~/.hermes/plugins/
   hermes plugins list  # should show smart-model-swap v0.3.0
   ```

2. **Enable it (if not already enabled)**:

   ```bash
   hermes plugins enable smart-model-swap
   ```

   This registers the 4 tools (`sms_status`, `sms_plan`, `sms_swap`, `sms_restore`), the slash commands (`/swap` …), and CLI surface (`hermes sms`). If you've already enabled it, just verify:

   ```bash
   hermes sms status   # should print JSON state
   ```

3. **Configure the engine and your models**:

   On first load the plugin copies `models.example.json` to `models.json`, the live registry, which is not tracked by git. Edit it in the dashboard tab (Engines · SmartModelProvider → *configure*, Models → *edit*), in the Desktop pane, or by hand:

   - `engines.llamacpp`:
     - `bin`: path to `llama-server` (`~` is expanded);
     - `host` / `port`: the port must not clash with Ollama, vLLM or LM Studio;
     - `common_args`: flags shared by every model;
     - `env`: extra environment variables, e.g. `CUDA_VISIBLE_DEVICES=0,1`.
   - `models[]`, one entry per model:
     - `name`;
     - `gguf` and `mmproj`;
     - `args`, the model's own flags;
     - `min_free_vram_gb` and `load_timeout_s`.
   - Keep the **same `-c` on every model**: Hermes caches one context length per model id. The editor enforces it.
   - `primary_model` (a plugin setting) names the big model that autostarts with the gateway.

4. **Start the gateway**:

   The SmartModelProvider daemon will only start when the Hermes gateway launches and nothing holds the port (`hermes sms wait`). If you're already running `hermes` with an active session, just run:

   ```bash
   hermes sms status
   ```

   You should see: `current_model = signal`, `llm_up = true`, plus per-GPU VRAM and ComfyUI state.

5. **First smoke test** (5 min):

   - Check current model:

     ```bash
     hermes sms status       # JSON output
     ```

   - Plan a swap to a smaller model for 12 GiB of VRAM:

     ```bash
     hermes sms plan 12      # should recommend qwen or similar
     ```

   - Dry run (no actual load):

     ```bash
     hermes sms swap qwen --dry-run
     hermes sms restore --dry-run
     ```

   - Real, short test:

     ```bash
     hermes sms swap qwen --wait      # blocks ~45 s until the model is serving
     hermes sms status                # should now show current_model = qwen
     hermes sms restore --wait        # restores signal in ~210 s
     hermes sms status                # back to primary=signal, llm_up=true
     ```

   If all 4 commands exit with code 0 and `status` ends with `"ok": true`, you're ready.

## Option C: From source (for contributors / debugging)

If you cloned the plugin's repo or edited files in-place:

```bash
cd ~/.hermes/plugins/smart-model-swap
# No build step — it is stdlib-only.
hermes plugins enable smart-model-swap   # re-enables from the latest changes
hermes sms status                       # verify registration
```

## Desktop pane & SmartModelProvider (optional but recommended)

- **Install**: `hermes sms provider install` creates the FastAPI dashboard backend and updates the desktop SDK manifest.
- **Desktop UI**: After login in the gateway, open the Model Swap pane with ⌘K → Reload Desktop Plugins or via the top-right status-bar chip. It shows: current model (dot), per-GPU VRAM free/total, ComfyUI state, a swap/restore area, secondary selector from `models.json`, and an inline launch-parameter editor synced to your local llama.cpp `--help`.
- **SmartModelProvider**: The provider half manages engine lifecycle (`llama-server` start/stop), port detection, model-provider registration with Hermes. It's separate but recommended: `hermes sms provider remove` uninstalls it; re-install if the daemon goes stale (the gateway detects and relaunches it).

## Migration notes (from v0.1 → v0.2 → v0.3)

- **Removed systemd unit (`llama-server.service`)**. Now managed by SmartModelProvider inside the gateway process with `isolate_scope=true`.
- **Registry format**: `models.json` now uses engine + model registry semantics. The bundled defaults match Oscar's CoMAS stack (signal/qwen). Old GGUF paths auto-migrate to the new schema on first load.
- **Start mode**: replaced by an explicit daemon (`llama-server`) with `--host`, `--port`, and environment variables set from `engine.llamacpp`. If your unit file is still enabled, disable it:

  ```bash
  sudo systemctl disable --now llama-server.service   # if you have one
  ```

## State & logs (all in ~/.hermes/cache/smart-model-swap/)

| File | Purpose |
|---|---|
| `state.json` | current model, engine status, last swap result, ComfyUI state, primary/secondary roles |
| `models.json` | bundled defaults; your local registry overrides it on load |
| `runner.log`, `model_<name>.log` | per-model lifecycle logs: graceful stop → VRAM settle → health-check → start → serving → error tails (if any) |
| `vram_profile.json` | measured per-GPU used-after-loads; gates each new model to avoid false negatives |
| `ui.json` | desktop pane state (slots, active server selection) — independent of core state |

## Uninstall / rollback

```bash
# Disable tools & slash:
hermes plugins disable smart-model-swap

# Remove registry entries:
# edit ~/.hermes/plugins/smart-model-swap/models.json (or use the UI) and delete your model(s)

# Delete files (or move aside):
rm -r ~/.hermes/plugins/smart-model-swap ~/.hermes/desktop-plugins/smart-model-swap

# Reboot the gateway to clear cached state; SmartModelProvider will not auto-start unless re-enabled.
```

## Troubleshooting quick table

| Symptom | Fix |
|---|---|
| "ok: false" + `primary_restored: true` after swap/restore | VRAM release gate tripped → add 30 s to `release_wait_s`; or check for ComfyUI lingering (kill it if idle). |
| `nvidia-smi: No such file` on WSL | Prefix all calls with `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH; /usr/lib/wsl/lib/nvidia-smi` or use `/usr/bin/nvidia-smi`. |
| Load times out (`load_timeout_s`) | Increase to 300 in `models.json`; WSL mounts for `/mnt/` paths can be slow. |
| Port conflict (8080 already in use) | Change the port in `engine.llamacpp` or run another Ollama/vLLM on a different port and let SmartModelProvider auto-detect it (`/detect`). |
| Runner log stuck at "waiting for health" > 3 min | The daemon is dead → restart it via desktop pane button or `hermes sms provider install`; check the runner's stdout/stderr in `model_<name>.log`. |

## Where to go from here

- **Catalog PR**: when you're ready, submit a merge request against the official Hermes plugins repo with this README + install guide.
- **Model-provider installation**: run `hermes sms provider install` for the dashboard backend + desktop UI.
- **Next upgrade**: v0.4 will add multi-model concurrency (up to N parallel engines) and auto-detection of Ollama/vLLM/LM Studio via their health endpoints.
