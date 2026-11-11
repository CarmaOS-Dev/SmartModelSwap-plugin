"""Per-turn context for the agent (pre_llm_call) + voice detection (pre_transcription).

Everything here is advisory: one short line injected into the turn so the
agent applies skill smart-model-routing at the right moment. Nothing swaps by
itself (except the idle watcher, which lives in the runner).

  * status   — a swap is in flight / a non-primary model is serving.
  * voice    — the user is talking by voice (several transcribed clips in a
               short window) while the primary is loaded: suggest the small
               model for a quick casual chat.
  * keywords — the message looks like a GPU job (image/video/ComfyUI …) while
               the primary is loaded, or like deep work while a small model is
               loaded.

Voice and keyword hints are throttled per session (one per kind every
HINT_COOLDOWN_S) and respect the swap hysteresis (no hint within
HYSTERESIS_S of the last swap/restore).
"""
from __future__ import annotations

import re
import threading
import time
from datetime import datetime

VOICE_TURN_S = 120          # a transcription this recent means "this turn came by voice"
VOICE_WINDOW_S = 15 * 60    # voice clips counted over this window
VOICE_MIN_CLIPS = 2         # …at least this many to call it a voice conversation
HINT_COOLDOWN_S = 20 * 60
HYSTERESIS_S = 10 * 60

_lock = threading.Lock()
_voice_ts: list[float] = []
_hinted: dict[tuple[str, str], float] = {}

GPU_RX = re.compile(
    r"\b(imagen(es)?|image(s)?|picture|foto(s)?|ilustraci[oó]n|dib[uú]ja|dibujo|render|retrato|portrait|"
    r"wallpaper|comfy\w*|video(s)?|v[ií]deo|animaci[oó]n|modelo 3d|3d model|lora|flux|sdxl|"
    r"genera(r|me)?\s+(una|un|la|el)\s+(imagen|foto|video|ilustraci[oó]n))\b", re.I)
DEEP_RX = re.compile(
    r"\b(anal[ií]za|an[aá]lisis|exhaustiv\w*|a fondo|en profundidad|profund\w*|investiga\w*|"
    r"revisa (todo|el|la|los|las)|code review|depura\w*|debug\w*|refactor\w*|planifica\w*|"
    r"plan detallado|estrategia|compara (a fondo|en detalle)|deep dive|thorough|in depth|analy[sz]e)\b", re.I)


def note_voice(**_kw):
    """pre_transcription hook: remember that a voice clip came in. Changes nothing."""
    now = time.time()
    with _lock:
        _voice_ts.append(now)
        del _voice_ts[:-50]
    return None


def _voice_state(now: float) -> tuple[bool, int]:
    with _lock:
        recent = [t for t in _voice_ts if now - t <= VOICE_WINDOW_S]
        turn = bool(_voice_ts) and now - _voice_ts[-1] <= VOICE_TURN_S
    return turn, len(recent)


def _cooldown_ok(session: str, kind: str, now: float) -> bool:
    with _lock:
        last = _hinted.get((session, kind), 0.0)
        if now - last < HINT_COOLDOWN_S:
            return False
        _hinted[(session, kind)] = now
        if len(_hinted) > 500:
            for k in sorted(_hinted, key=_hinted.get)[:250]:
                _hinted.pop(k, None)
    return True


def _since_last_swap(st: dict, now: float) -> float:
    at = (st.get("last_result") or {}).get("at")
    try:
        return now - datetime.strptime(at, "%Y-%m-%dT%H:%M:%S").timestamp()
    except (TypeError, ValueError):
        return 1e9


def _smallest(tools) -> str | None:
    models = tools.load_models().get("models", [])
    if not models:
        return None
    return min(models, key=lambda m: float(m.get("min_free_vram_gb") or 0)).get("name")


def unloaded_veto(tools, busy_state: dict | None) -> dict | None:
    """The LLM is not serving ON PURPOSE (a gpu_job.sh holds the GPU, or a swap /
    load is in flight): end the turn at once with an explanation instead of
    letting Hermes retry for minutes and fail. Hermes drops the vetoed message,
    so the user is asked to send it again."""
    job = tools.gpu_job_info()
    if job:
        mins = max(1, int(job.get("elapsed_s") or 0) // 60)
        what = job.get("job") or "GPU job"
        why = ("el modelo está descargado mientras corre un trabajo de GPU"
               if job.get("mode") == "none" else "hay un trabajo de GPU en curso y el modelo está cambiando")
        return {"action": "block", "message":
                f"⏳ Ahora mismo no puedo responder: {why} ({what}, lleva ~{mins} min). "
                "Al terminar, el modelo principal vuelve solo y recibirás el resultado. "
                "Vuelve a enviar tu mensaje después de ese aviso."}
    if busy_state:
        action = busy_state.get("action") or "swap"
        target = busy_state.get("target") or "?"
        return {"action": "block", "message":
                f"⏳ Ahora mismo no puedo responder: se está cargando el modelo '{target}' ({action}, "
                "2–4 min). Vuelve a enviar tu mensaje en unos minutos."}
    return None


def build_context(tools, **kwargs) -> dict | None:
    cfg = tools.get_cfg()
    if not cfg.get("inject_status", True):
        return None
    st = tools.read_state()
    busy = st.get("state") in tools.BUSY_STATES and tools.runner_alive(st)
    health = tools.llm_health(timeout=1.0)
    if not health.get("up") and cfg.get("block_when_unloaded", True):
        veto = unloaded_veto(tools, st if busy else None)
        if veto:
            return veto
    if busy:
        return {"context": f"[smart-model-swap] A model {st.get('action')} to '{st.get('target')}' "
                           "is in progress; the local LLM will be unavailable for a moment. "
                           "Keep this reply short."}
    now = time.time()
    current = tools.match_model(health)
    primary = cfg.get("primary_model")
    on_primary = bool(current) and tools.same_name(current, primary)
    session = str(kwargs.get("session_id") or "")
    msg = kwargs.get("user_message")
    msg = msg if isinstance(msg, str) else ""
    settled = _since_last_swap(st, now) >= HYSTERESIS_S
    lines: list[str] = []

    if current and not on_primary:
        entry = tools.find_model(current) or {}
        ctx_n = tools.model_ctx(entry)
        idle = float(cfg.get("idle_restore_min") or 0)
        lines.append(
            f"[smart-model-swap] Active local model: '{current}'"
            + (f" (context {ctx_n} tokens)" if ctx_n else "")
            + f". The primary '{primary}' is unloaded to free VRAM"
            + (f"; it comes back by itself after {idle:g} idle minutes" if idle > 0 else "")
            + ". Call sms_restore sooner if deep work is needed now (skill smart-model-routing).")
        if msg and DEEP_RX.search(msg) and _cooldown_ok(session, "deep", now):
            lines.append("[smart-model-swap] This looks like deep work and you are on the small model: "
                         "run `hermes sms check`, then `hermes sms restore --wait` (procedure B) "
                         "unless a GPU job still holds the VRAM.")
    elif on_primary:
        if msg and GPU_RX.search(msg) and _cooldown_ok(session, "gpu", now):
            lines.append("[smart-model-swap] If this needs ComfyUI/GPU work: the primary model holds "
                         "the GPU. Check `sms_plan(needed_free_gb=…, single_gpu=true)` and run the job "
                         "through gpu_job.sh (skill smart-model-routing, procedure A).")
        voice_turn, clips = _voice_state(now)
        small = _smallest(tools)
        if (voice_turn and clips >= VOICE_MIN_CLIPS and settled and small
                and not tools.same_name(small, primary) and _cooldown_ok(session, "voice", now)):
            lines.append(f"[smart-model-swap] Voice conversation detected ({clips} voice messages in "
                         f"{VOICE_WINDOW_S // 60} min). If it is casual and needs quick replies, answer this "
                         f"message and then call sms_swap('{small}') as the LAST tool (procedure C); stay "
                         "on the primary for deep work.")
    return {"context": "\n".join(lines)} if lines else None
