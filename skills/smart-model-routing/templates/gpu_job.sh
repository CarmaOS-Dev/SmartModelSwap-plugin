#!/usr/bin/env bash
# gpu_job.sh — run a GPU-heavy job with the small model loaded (or none), then ALWAYS leave a model serving.
#
#   gpu_job.sh <auto|none|nano|multi|keep|model-name> -- <job command...>
#
#   auto    (recommended) the mode the user picked in the UI (setting gpu_job_mode). Its default,
#           also called "auto", chooses for THIS machine: multi with 2+ GPUs, otherwise none.
#   none    unload the LLM during the job: all the VRAM goes to the job.
#   nano    keep the small model loaded (its normal layout) so the agent has a brain meanwhile.
#   multi   small model whole on ONE GPU; another GPU is left to the job. One GPU only -> none.
#   keep    do not touch the loaded model (light jobs that fit beside it).
#   <name>  swap to that registry model for the job (legacy form).
#
#   The job reads the GPU it must use from $SMS_TASK_GPU (ComfyUI --cuda-device).
#   Force the job's GPU with TASK_GPU=<n> in the environment.
#
# Run it from the terminal tool with background=true and notify_on_complete=true. The completion
# notification wakes the agent on the model named in "RESULT model=" (the small one by default,
# setting after_gpu_job; the big one comes back on demand or when idle). Read the log/RESULT
# lines, not the exit code alone.
#
# Uses ONLY the SmartModelProvider CLI (`hermes sms ...`). Never `systemctl llama-server`,
# `pkill llama-server`, or a hand-launched llama-server: SMP owns the engine port.
set -uo pipefail

SMALL="${1:?usage: gpu_job.sh <auto|none|nano|multi|keep|model> -- <command...>}"; shift
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "usage: gpu_job.sh <auto|none|nano|multi|keep|model> -- <command...>" >&2; exit 2; }

HERMES="${HERMES_BIN:-hermes}"
LOG="${GPU_JOB_LOG:-$HOME/.hermes/cache/smart-model-swap/gpu_job_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Marker: while it exists with our PID, SMP's idle watcher will not restore the primary mid-job.
MARK="$HOME/.hermes/cache/smart-model-swap/gpu_job.active"
echo $$ > "$MARK"

restored=0
restore(){
  [ "$restored" = 1 ] && return
  restored=1
  if curl -sf --max-time 3 http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
    log "WARNING: ComfyUI (:8188) is still up and holds VRAM — the job must stop it"
  fi
  # Which model delivers the result is the plugin's call (setting after_gpu_job):
  #   small   (default) keep the small model that is loaded, or load the smallest that fits if
  #           the LLM was unloaded: the reply comes fast. The primary returns on demand
  #           (sms_restore) or by itself after idle_restore_min.
  #   primary reload the primary now (slower); falls back to the small one if its VRAM is held.
  log "AFTER-JOB: making sure a model is serving to deliver the result"
  local prefer=()
  [ "$SMALL" != none ] && [ "$SMALL" != keep ] && prefer=(--prefer "$SMALL")
  local out rc model
  out=$("$HERMES" sms after-job "${prefer[@]}" --timeout 900 2>&1); rc=$?
  echo "$out" >>"$LOG"
  model=$(echo "$out" | sed -n 's/^ *"current": *"\([^"]*\)".*/\1/p' | tail -1)
  if [ $rc -eq 0 ]; then log "RESULT model=${model:-?} (restore=ok)"; else
    log "RESULT restore=FAILED (rc=$rc) — no model serving; see ~/.hermes/cache/smart-model-swap/runner.log; retry: hermes sms start"
  fi
}
trap 'restore; rm -f "$MARK"' EXIT
trap 'log "INTERRUPTED"; exit 130' INT TERM

log "job: $*"
LLM_GPU=""
case "$SMALL" in
  auto|none|nano|multi|keep)
    # named mode: the plugin decides what it means on this machine (auto = the UI setting)
    asked="$SMALL"
    mode_arg=(); [ "$asked" != auto ] && mode_arg=(--mode "$asked")
    layout=$("$HERMES" sms job-layout --plain "${mode_arg[@]}" ${TASK_GPU:+--task-gpu "$TASK_GPU"} 2>>"$LOG" | tail -1)
    read -r l_do l_model l_llm l_task l_applied <<<"$layout"
    case "$l_do" in
      small) SMALL="$l_model"; [ "${l_llm:--}" != "-" ] && LLM_GPU="$l_llm" ;;
      keep)  SMALL=keep ;;
      none)  SMALL=none ;;
      *)     SMALL=none; l_applied="none (no layout from 'hermes sms job-layout': $layout)" ;;
    esac
    log "MODE: asked=$asked applied=${l_applied:-?} model=${l_model:--} llm_gpu=${l_llm:--} task_gpu=${l_task:--}"
    [ "${l_task:--}" != "-" ] && export SMS_TASK_GPU="$l_task"
    ;;
esac
[ -n "${TASK_GPU:-}" ] && export SMS_TASK_GPU="$TASK_GPU"
case "$SMALL" in
  none) log "FREE: unloading the LLM"
        # graceful: 10 s grace + drain, so the agent's last reply is not cut mid-stream
        "$HERMES" sms stop --graceful >>"$LOG" 2>&1; rc=$? ;;
  keep) log "FREE: nothing to do (the loaded model stays)"; rc=0 ;;
  *)    log "FREE: swapping to '$SMALL'${LLM_GPU:+ on GPU $LLM_GPU}"
        "$HERMES" sms swap "$SMALL" ${LLM_GPU:+--gpu "$LLM_GPU"} --wait --timeout 900 >>"$LOG" 2>&1; rc=$? ;;
esac
if [ $rc -ne 0 ]; then
  log "RESULT free=FAILED (rc=$rc) — job NOT run"
  exit 1
fi
log "FREE ok"
/usr/lib/wsl/lib/nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null | tee -a "$LOG"

log "JOB start"
"$@" 2>&1 | tee -a "$LOG"
job_rc=${PIPESTATUS[0]}
log "RESULT job_rc=$job_rc"
exit "$job_rc"   # the EXIT trap brings a model back before the notification fires
