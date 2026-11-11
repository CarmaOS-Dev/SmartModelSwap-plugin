---
name: smart-model-routing
description: "Pick the big or small local model and free VRAM."
version: 1.0.0
author: Oscar Carreto (DraksoN)
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [SmartModelSwap, SmartModelProvider, VRAM, routing, voice, ComfyUI]
---

# Smart Model Routing: big or small local model?

The local LLM is served by **SmartModelProvider (SMP)**, part of the `smart-model-swap` plugin. You usually run two profiles:

- **Big (primary):** reasons better and runs slower. It is the `primary_model` setting, and `sms_restore` brings it back.
- **Small:** fast and light, and leaves VRAM free for other GPU work. It is the registry model with the lowest `min_free_vram_gb`.

This skill decides which one to run and how to switch **without breaking the session**. Check `sms_status` for the real names, context size and measured VRAM.

## When to use it
- Before any GPU task: ComfyUI (image, video, 3D), voice design, training, or any other model that needs VRAM.
- When a casual or voice chat starts that wants quick replies.
- When you are on the small model and heavy work arrives: deep analysis, code, research, planning, careful writing.
- When a `[smart-model-swap]` line tells you the primary is unloaded.
- When a swap fails or the LLM stops answering.

## What a switch costs
- A swap leaves **every session without an LLM** while it runs (messaging, cron jobs, subagents). How long depends on model size and disk speed: typically 1–4 min.
- **Never switch for a single message.** Switch only if the new mode will last: 3+ exchanges, 10+ minutes of voice, or a GPU job.
- **Hysteresis:** after a swap, do not switch back within ~10 min, unless the user asks, a GPU job starts, or the loaded model clearly cannot do the task.
- **When in doubt, stay** on the model that is already loaded.
- **The user's explicit request always wins** ("use the small one", "load the big one", `/swap`).

## Decision matrix
| Situation | Model | How |
|---|---|---|
| GPU job (ComfyUI, video, 3D, voice design…) | small, or `none` if it does not fit | **Procedure A** |
| Deep analysis, code, debugging, research, planning, reviews, writing for others | big | **Procedure B** |
| Casual or voice chat that needs quick replies (greetings, small talk, quick facts, reminders) | small | **Procedure C** |
| GPU job or voice session over, or heavy work arrives on the small model | big | **Procedure D** |
| Cron / scheduled task | big, unless the cron *is* the GPU job | a cron **never** leaves the small model loaded |
| A swap is in progress (`sms_status` → `swap.state` ≠ `idle`) | none | wait (`hermes sms wait`); never start another |

## Tools
- `sms_status`: current model, swap state, per-GPU VRAM, ComfyUI, the running `gpu_job`. It is the source of truth.
- `sms_plan(needed_free_gb, task, single_gpu=true)` uses what each model **really takes per GPU** (measured on every load). By default it assumes the task runs on **one GPU** (ComfyUI `--cuda-device N`). It returns:
  - `action`: `stay`, `swap` (to `target`), `unload` (use `gpu_job.sh none`) or `impossible`;
  - `task_gpu`: which GPU to run the task on;
  - `free_after_per_gpu_gib`: the room each option leaves.
- `sms_swap(model)` and `sms_restore()` launch the switch in the background and **return at once**. The model running you is replaced, so call them as the **LAST tool of the turn**, then reply in one short sentence. The next turn runs on the new model.
- Blocking CLI, through the terminal tool, to continue **in the same turn** or inside scripts:
  - `hermes sms swap <model> --wait`
  - `hermes sms restore --wait`
  - `hermes sms wait [--target <model>]`

  They print JSON with `ok`, `current` and `waited_s`. The exit code is 0 only if the requested model is serving. In the foreground, use `timeout` ≥ 420 s (max 600).
- `hermes sms check [model] [--wait N]`: the release gate. It reports whether other heavy GPU processes have freed the VRAM the model needs on every GPU.
- `hermes sms stop [--graceful]` / `hermes sms start`: unload the LLM, or load the primary when nothing is serving.
- A swap or restore to the model that is **already loaded** is a free no-op (`"skipped"`). Use `--force` only to reload after editing its flags.

## Release check (always, before loading the big model)
The big model usually fills most of the VRAM of every GPU it uses. It loads only after other heavy processes (ComfyUI, other models, training) have freed that VRAM.
1. **Check first:** run `hermes sms check`, which lists who holds the GPU. Do not call `sms_restore` while it fails.
2. **If it fails:**
   - if the holder is something **you** started and it has finished (typically ComfyUI after a generation), stop it by exact PID and run `hermes sms check --wait 60`;
   - if it belongs to the **user**, or is still working, leave it alone: stay on the small model and tell the user what holds the GPU.
3. **The plugin enforces it as well:**
   - the preflight of `sms_restore` / `sms_swap` fails without cutting the current model;
   - after stopping the current model, the runner waits up to `release_wait_s` for the VRAM;
   - if it is not freed, the runner reloads the previous (or smallest fitting) model instead, so you are never left without a brain. `runner.log` shows `RESOURCES NOT RELEASED`.

## Procedure A: GPU-heavy job (free → work → back to big)
1. **Measure.** Call `sms_plan(needed_free_gb=<the job's VRAM>, task=…)`.
2. **Choose by `action`:**
   - `stay`: change nothing and run the task on `task_gpu`;
   - `swap`: switch to `target`, usually the small model;
   - `unload`: use `none`, which means no LLM during the job;
   - `impossible`: tell the user it does not fit.
3. **Run everything as one background job** with the bundled template:
   ```bash
   bash ~/.hermes/plugins/smart-model-swap/skills/smart-model-routing/templates/gpu_job.sh <small-model|none> -- bash /path/to/my_job.sh
   ```
   - Launch it with the terminal tool, `background=true`, `notify_on_complete=true`. The plugin blocks it in the foreground.
   - `my_job.sh` must start ComfyUI (on `task_gpu`), submit the workflow, fetch the result, **stop ComfyUI** by exact PID, and print `SAVED=<path>`.
   - The template frees VRAM through SMP, runs the job and **always** restores the big model before it exits.
4. **Reply in one sentence and END the turn**, for example "Generating it; freeing the GPU, back in a few minutes". After launching `gpu_job.sh`, make **no further tool or model calls**, not even to poll: the model running you is about to be unloaded. The swap waits 10 s and lets your reply finish first.
5. **When the completion notification arrives** (you are on the big model again), check the log for `RESULT job_rc=0`, `RESULT restore=ok` and `SAVED=`, then deliver the result. `RESULT restore=BLOCKED` means the VRAM was not freed: apply the release check, then `sms_restore`.

## Procedure B: deep work (load the big model)
1. If `sms_status` shows the big model, just work.
2. If the small one is loaded, run `hermes sms check`. If it passes:
   - say in one line "Loading the big model for this, back in a moment";
   - run `hermes sms restore --wait` in the foreground (`timeout=600`);
   - on `ok: true`, continue **in the same turn**.
3. If a GPU job still holds the VRAM, **do not** restore. Tell the user the deep work starts when the job ends, or do it on the small model with a caveat, and let them choose.

## Procedure C: casual / quick voice chat (switch to small)
1. Make sure the mode will last: voice mode or a call, "let's chat", or a run of short messages. A single greeting is not enough.
2. **Do not make the user wait.** Answer this message on the current model, then call `sms_swap(<small>)` as the **last tool**, adding "switching to quick mode, replies get faster in a moment".
3. While on the small model, keep replies short and conversational.
4. Exception: if TTS/ASR fails for lack of VRAM with the big model loaded, switching to small fixes it even for a short exchange.

## Procedure D: back to the big model
Restore when any of these holds:
- the GPU job is over (the `gpu_job.sh` template already does it);
- the voice or casual session ended;
- a task for the big model arrives (Procedure B);
- a `[smart-model-swap]` line reminds you the primary is unloaded and there is no reason left for the small one.

To restore: `hermes sms check --wait 120`, then `sms_restore()` as the last tool, or `hermes sms restore --wait` to continue in the same turn.

## What the plugin does on its own
- **Idle restore.** After any swap to a smaller model, a watcher brings the primary back once the engine has been idle for `idle_restore_min` minutes (default 15) and the release check passes.
- **Guard.** `terminal` / `execute_code` calls that stop, kill or launch `llama-server` by hand are blocked. So is a ComfyUI launch while the primary holds the GPU, unless it goes through `gpu_job.sh` or carries `SMS_ALLOW_GPU=1` (only after `sms_plan` said `stay`). The block message says what to use instead.
- **Hints.** You get one `[smart-model-swap]` line when:
  - an image or video is requested on the big model;
  - deep work is requested on the small model;
  - a voice conversation is detected on the big model.

  It is advice, not an order: apply the decision matrix.
- **Unloaded on purpose.** If a message arrives while a GPU job holds the GPU or a model is loading, the turn ends at once and the user is asked to resend after the notification.
- **Drain.** Before cutting a model, in-flight replies may finish while they keep generating (up to 10 min). Still, after `sms_swap` keep the reply short.

## Never (it breaks SMP)
- `systemctl … llama-server`, `pkill` / `kill` on `llama-server`, launching `llama-server` by hand, or running an old launcher script.
- Starting a swap while another is in flight, or killing the runner.
- SMP is the **only** owner of the engine port. Free VRAM with `sms_swap` / `hermes sms stop`; come back with `sms_restore` / `hermes sms start`.

## If something fails
1. Call `sms_status` and read `swap.state`, `last_result.error`, `llm.up` and `server.mode`. A `server.mode` of `external` means someone started the engine outside SMP.
2. Read the real logs, not the tool's own report: `~/.hermes/cache/smart-model-swap/runner.log` and `llama-<model>.log`.
3. If no model is serving and no swap is running: `hermes sms start`, then `hermes sms wait`.
4. If the restore fails on VRAM, run `hermes sms check` to see who holds it, then apply the release check.
5. After two failures in a row, stop and report `last_result.error` and the last lines of `runner.log` to the user. Do not improvise a manual start.
