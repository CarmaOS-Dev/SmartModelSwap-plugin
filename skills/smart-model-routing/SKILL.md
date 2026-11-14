---
name: smart-model-routing
description: "Pick the big or small local model and free VRAM."
version: 1.1.0
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
- A swap leaves **every session without an LLM** while it runs (messaging, cron jobs, subagents). How long depends on model size, disk speed and whether a reply is still being generated: typically 1–4 min, worst case `sms_status` → `swap_budget_s`.
- **Never switch for a single message.** Switch only if the new mode will last: 3+ exchanges, 10+ minutes of voice, or a GPU job.
- **Hysteresis:** after a swap, do not switch back within ~10 min, unless the user asks, a GPU job starts, or the loaded model clearly cannot do the task.
- **When in doubt, stay** on the model that is already loaded.
- **The user's explicit request always wins** ("use the small one", "load the big one", `/swap`).

## Decision matrix
| Situation | Model | How |
|---|---|---|
| GPU job (ComfyUI, video, 3D, voice design…) | `auto`: small on one GPU + job on another (2+ GPUs), no LLM on a single GPU | **Procedure A** |
| Deep analysis, code, debugging, research, planning, reviews, writing for others | big | **Procedure B** |
| Casual or voice chat that needs quick replies (greetings, small talk, quick facts, reminders) | small | **Procedure C** |
| GPU job over: deliver the result, comment on it, simple follow-ups | **small** (the one left loaded) | change nothing; the big one returns by itself when idle |
| Voice session over, or heavy work arrives on the small model | big | **Procedure D** |
| Cron / scheduled task | big, unless the cron *is* the GPU job | a cron **never** leaves the small model loaded |
| A swap is in progress (`sms_status` → `swap.state` ≠ `idle`) | none | end the turn; never start another and never re-run a swap that timed out |

## Tools
- `sms_status`: current model, swap state, per-GPU VRAM, ComfyUI, the running `gpu_job`. It is the source of truth.
- `sms_plan(needed_free_gb, task, single_gpu=true)` uses what each model **really takes per GPU** (measured on every load). By default it assumes the task runs on **one GPU** (ComfyUI `--cuda-device N`). It returns:
  - `action`: `stay`, `swap` (to `target`), `unload` (use `gpu_job.sh none`) or `impossible`;
  - `task_gpu`: which GPU to run the task on;
  - `free_after_per_gpu_gib`: the room each option leaves.
- `sms_swap(model)` and `sms_restore()` launch the switch in the background and **return at once**. The model running you is replaced, so call them as the **LAST tool of the turn**, then reply in one short sentence. The next turn runs on the new model. This is the only way to switch from a conversation.
- **Never run a blocking `hermes sms` command in a foreground terminal**: `swap … --wait`, `restore --wait`, `wait`, `after-job`, `stop --graceful`. A swap can outlast any terminal timeout (worst case: `sms_status` → `swap_budget_s`). The guard blocks them; they are for scripts such as `gpu_job.sh`. If you must follow a swap from the terminal, run it with `background=true` and `notify=true`.
- `hermes sms check [model] [--wait N]`: the release gate. It reports whether other heavy GPU processes have freed the VRAM the model needs on every GPU.
- `hermes sms stop [--graceful]` / `hermes sms start`: unload the LLM, or load the primary when nothing is serving (`start` never replaces a model that is serving; use `sms_restore` for that).
- A swap or restore to the model that is **already loaded** is a free no-op (`"skipped"`). Use `--force` only to reload after editing its flags.

## Release check (always, before loading the big model)
The big model usually fills most of the VRAM of every GPU it uses. It loads only after other heavy processes (ComfyUI, other models, training) have freed that VRAM.
1. **Check first:** run `hermes sms check`, which lists who holds the GPU. Do not call `sms_restore` while it fails.
2. **If it fails:**
   - an **idle** ComfyUI is not a problem: the idle restore asks it to release its VRAM (`/free`, setting `comfy_free_on_idle`) without stopping it;
   - if the holder is something **you** started and it has finished (another model, a training run), stop it by exact PID and run `hermes sms check --wait 60`;
   - if it belongs to the **user**, or is still working, leave it alone: stay on the small model and tell the user what holds the GPU.
3. **The plugin enforces it as well:**
   - the preflight of `sms_restore` / `sms_swap` fails without cutting the current model;
   - after stopping the current model, the runner waits up to `release_wait_s` for the VRAM;
   - if it is not freed, the runner reloads the previous (or smallest fitting) model instead, so you are never left without a brain. `runner.log` shows `RESOURCES NOT RELEASED`.

## Procedure A: GPU-heavy job (free → work → answer on the small model)
1. **Measure.** Call `sms_plan(needed_free_gb=<the job's VRAM>, task=…)`.
2. **Choose by `action`:**
   - `stay`: run it through `gpu_job.sh keep -- …` (the loaded model is left alone, the job gets `task_gpu`);
   - `swap`: switch to `target`, usually the small model;
   - `unload`: use `none`, which means no LLM during the job;
   - `impossible`: tell the user it does not fit.
3. **Run everything as one background job** with the bundled template, normally in `auto` mode:
   ```bash
   bash ~/.hermes/plugins/smart-model-swap/skills/smart-model-routing/templates/gpu_job.sh auto -- bash /path/to/my_job.sh
   ```
   - `auto` applies the mode **the user chose in the UI** (setting `gpu_job_mode`): `auto`, `none`, `nano` (small model kept in its normal layout), `multi` (small model on one GPU, another for the job) or `keep` (the loaded model is left alone). Do not override the user's choice with an explicit mode unless they ask. `sms_plan` → `gpu_job_auto` (or `hermes sms job-layout`) shows what will happen; the log line `MODE: asked=… applied=…` shows what did.
   - With the default mode (`auto`) the plugin places things for the machine:
     - **2+ GPUs:** the small model is loaded whole on one GPU and the job gets another GPU to itself. You keep a model during the job and answer as soon as it ends, with no reload.
     - **1 GPU** (or the small model does not fit on a single GPU): the LLM is unloaded during the job, like `none`.
   - The job must run on the GPU in `$SMS_TASK_GPU` (ComfyUI: `--cuda-device "${SMS_TASK_GPU:-0}"`). Prefix `TASK_GPU=<n>` to force it.
   - If the job needs more VRAM than one GPU has free (`gpu_job_auto.free_without_llm_gib`), it does not fit in `auto`: use `none` when the job can spread over all GPUs, otherwise tell the user.
   - A model name or `none` instead of `auto` still forces that mode.
   - Launch it with the terminal tool, `background=true` and `notify=true`. `notify` must be exactly `true`: a list of patterns (`notify=["^SAVED="]`) does **not** tell you when the job ends, and the user never gets the result. The plugin blocks the job in the foreground and without `notify=true`.
   - Run only the template: append nothing after the job command (no `; sms_restore`; `sms_restore` is a tool, not a shell command, and the template already leaves a model serving).
   - `my_job.sh` must start ComfyUI (on `task_gpu`), submit the workflow, fetch the result, **stop ComfyUI** by exact PID, and print `SAVED=<path>`.
   - **Never start ComfyUI (or any GPU job) by hand, outside `gpu_job.sh`.** The guard blocks it while a model is loaded. A ComfyUI left open is also seen by the plugin; while its queue works, the primary cannot come back.
   - The template frees VRAM through SMP, runs the job and **always** leaves a model serving before it exits: the **small** one (the one already loaded; with `none`, the smallest that fits). It does not reload the big model, so the result is delivered without a multi-minute wait (setting `after_gpu_job`; `primary` brings back the old behaviour).
4. **Reply in one sentence and END the turn**, for example "Generating it; freeing the GPU, back in a few minutes". After launching `gpu_job.sh`, make **no further tool or model calls**, not even to poll: the model running you is about to be unloaded. The swap waits 10 s and lets your reply finish first.
5. **When the completion notification arrives** (you are on the small model), check the log for `RESULT job_rc=0`, `RESULT model=<name> (restore=ok)` and `SAVED=`, then deliver the result **on that model, without switching**: send the user the file named in `SAVED=` (the file itself, not only a description of it). Stay on it for comments and simple requests; load the big model only when the user asks for deep work (Procedure B). `RESULT restore=FAILED` means no model is serving: see **If something fails**.

**Any other long job that ends** (a CPU render, ffmpeg, a download) follows the same delivery rule even though it does not use `gpu_job.sh`: launch it with `background=true` and `notify=true`, never with a list of patterns, and send the result when the notification arrives. Without `notify=true` the job ends silently and the user has to ask for it.

## Procedure B: deep work (load the big model)
1. If `sms_status` shows the big model, just work.
2. If the small one is loaded, run `hermes sms check`. If it passes:
   - tell the user in one line: "Loading the big model for this; send me the request again in a couple of minutes" (or "…I'll pick it up on your next message");
   - call `sms_restore()` as the **last tool** of the turn. The next turn runs on the big model.
3. If a GPU job still holds the VRAM, **do not** restore. Tell the user the deep work starts when the job ends, or do it on the small model with a caveat, and let them choose.

## Procedure C: casual / quick voice chat (switch to small)
1. Make sure the mode will last: voice mode or a call, "let's chat", or a run of short messages. A single greeting is not enough.
2. **Do not make the user wait.** Answer this message on the current model, then call `sms_swap(<small>)` as the **last tool**, adding "switching to quick mode, replies get faster in a moment".
3. While on the small model, keep replies short and conversational.
4. Exception: if TTS/ASR fails for lack of VRAM with the big model loaded, switching to small fixes it even for a short exchange.

## Procedure D: back to the big model
The end of a GPU job is **not** a reason to restore: the result is delivered on the small model. Restore when any of these holds:
- the voice or casual session ended;
- a task for the big model arrives (Procedure B);
- a `[smart-model-swap]` line reminds you the primary is unloaded and there is no reason left for the small one.

To restore: `hermes sms check`, then `sms_restore()` as the last tool of the turn.

## What the plugin does on its own
- **Start-up on the primary.** When the gateway starts, the primary is loaded. If a smaller model was left serving (e.g. across a gateway restart), it is swapped for the primary, unless a `gpu_job.sh` or ComfyUI is running. So after a gateway restart, expect the big model; check `sms_status` before assuming the small one is still loaded.
- **Idle restore.** After any swap to a smaller model, a watcher brings the primary back once the engine has been idle for `idle_restore_min` minutes (default 15) and the release check passes. A ComfyUI with work in its queue pauses the clock; an idle one is asked to release its VRAM first.
- **Interrupted swaps resume.** If a swap's runner dies mid-flight (killed together with whatever started it), the watcher relaunches the same swap once. It is not counted as a model failure.
- **Auto-recover.** If a model fails to load, or its server dies while serving, the plugin loads another model that fits (the primary first, then from small to big). `sms_status` → `last_result.detail.fallback` and `runner.log` (`auto-recover`) say which one. A model that just failed is not retried automatically for 30 min: do not force it back; report the error from `llama-<model>.log`.
- **Guard.** `terminal` / `execute_code` calls are blocked when they:
  - stop, kill or launch `llama-server` by hand;
  - launch ComfyUI while any model is loaded or a swap is running, unless through `gpu_job.sh` or with `SMS_ALLOW_GPU=1` (only after `sms_plan` said `stay`);
  - run a blocking `hermes sms` command in the foreground;
  - run `gpu_job.sh` in the foreground or without `notify=true`.

  The block message says what to use instead.
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
2. Read the real logs, not the tool's own report: `$HERMES_HOME/cache/smart-model-swap/runner.log` and `llama-<model>.log`.
3. A command that timed out did **not** cancel the swap: check `sms_status` and do not launch it again.
4. If no model is serving and no swap is running: call `sms_restore()` as the last tool (`hermes sms start` only works when the `autostart_primary` setting is on).
5. `RESULT free=FAILED — job NOT run` in a `gpu_job_*.log`: the swap before the job did not finish. Check `sms_status`, tell the user, and only relaunch the job once `swap.state` is `idle`.
6. If the restore fails on VRAM, run `hermes sms check` to see who holds it, then apply the release check.
7. After two failures in a row, stop and report `last_result.error` and the last lines of `runner.log` to the user. Do not improvise a manual start.
