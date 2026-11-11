#!/usr/bin/env bash
# gpu_job.sh — run a GPU-heavy job with the small model loaded, then ALWAYS restore the big one.
#
#   gpu_job.sh <small-model|none> -- <job command...>
#
#   small-model  registry name of the small model to load during the job. The agent keeps a brain.
#   none         unload the LLM completely (only if even the small model leaves too little VRAM).
#
# Run it from the terminal tool with background=true and notify_on_complete=true. The completion
# notification wakes the agent on the restored big model. Read the log/RESULT lines, not the
# exit code alone.
#
# Uses ONLY the SmartModelProvider CLI (`hermes sms ...`). Never `systemctl llama-server`,
# `pkill llama-server`, or a hand-launched llama-server: SMP owns the engine port.
set -uo pipefail

SMALL="${1:?usage: gpu_job.sh <small-model|none> -- <command...>}"; shift
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "usage: gpu_job.sh <small-model|none> -- <command...>" >&2; exit 2; }

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
  # Release gate: the big model loads only once the heavy processes freed the VRAM it needs
  # on EVERY GPU. Waits up to 120 s; if they don't, keep the current (small) model.
  log "CHECK: waiting for heavy GPU processes to release VRAM"
  if ! "$HERMES" sms check --wait "${RELEASE_WAIT_S:-120}" >>"$LOG" 2>&1; then
    log "RESULT restore=BLOCKED — VRAM still held by other processes (see the check JSON above); big model NOT loaded"
    return
  fi
  log "RESTORE: loading the primary model"
  if [ "$SMALL" = none ]; then
    "$HERMES" sms start >>"$LOG" 2>&1
    "$HERMES" sms wait --timeout 900 >>"$LOG" 2>&1
  else
    "$HERMES" sms restore --wait --timeout 900 >>"$LOG" 2>&1
  fi
  local rc=$?
  if [ $rc -eq 0 ]; then log "RESULT restore=ok"; else
    log "RESULT restore=FAILED (rc=$rc) — see ~/.hermes/cache/smart-model-swap/runner.log; retry: hermes sms start"
  fi
}
trap 'restore; rm -f "$MARK"' EXIT
trap 'log "INTERRUPTED"; exit 130' INT TERM

log "job: $*"
log "FREE: $( [ "$SMALL" = none ] && echo 'unloading the LLM' || echo "swapping to '$SMALL'")"
if [ "$SMALL" = none ]; then
  # graceful: 10 s grace + drain, so the agent's last reply is not cut mid-stream
  "$HERMES" sms stop --graceful >>"$LOG" 2>&1; rc=$?
else
  "$HERMES" sms swap "$SMALL" --wait --timeout 900 >>"$LOG" 2>&1; rc=$?
fi
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
exit "$job_rc"   # the EXIT trap restores the big model before the notification fires
