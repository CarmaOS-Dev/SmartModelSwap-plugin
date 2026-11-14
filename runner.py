"""Detached swap runner and watcher. Launched by tools.spawn_runner /
tools.spawn_idle_watch (a transient user service, or a double fork - never a
child of the caller) with:
  runner.py <state_file> <swap|restore|start|idle_watch> <target> <cfg_base64>

swap/restore: grace sleep -> drain in-flight requests -> stop the current
server (signal) -> start target -> verify -> update state. On failure another
model that fits is loaded (auto_recover) and the result stays ok=False.
start: autostart of the primary (gateway start-up, replaces the old systemd
unit): wait for the GPU driver and the model file (gpu_wait_s), then load.
idle_watch: restore the primary when idle, recover a dead server, resume a
swap whose runner was killed.
Writes progress to runner.log and state.json; never raises to the parent.
The state file argument is informational: the state lives under HERMES_HOME.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tools  # noqa: E402  (local plugin module)


def _watch(name: str, log) -> None:
    """Start the watcher for the model that ended up serving: idle restore for a
    non-primary model, crash recovery (auto_recover) for any model."""
    try:
        cfg = tools.get_cfg()
        r = tools.spawn_idle_watch(name)
        if r.get("pid") or r.get("unit"):
            what = []
            if (not tools.same_name(name, cfg.get("primary_model"))
                    and float(cfg.get("idle_restore_min") or 0) > 0):
                what.append(f"primary comes back after {cfg.get('idle_restore_min')} min idle")
            if cfg.get("auto_recover", True):
                what.append("swaps to another model if this one dies")
            log(f"watcher started ({r.get('unit') or 'pid %s' % r.get('pid')}): " + "; ".join(what))
    except Exception as e:  # noqa: BLE001
        log(f"watcher not started: {type(e).__name__}: {e}")


def _load_first(names: list, instead_of: str, log, label: str) -> str | None:
    """Load the first of `names` that fits right now (release gate on the VRAM
    free at this moment). A candidate that fails to load is noted and cleared.
    The one fallback loop for every path: failed load, release timeout,
    autostart. Returns the name now serving, or None."""
    for name in names:
        fb = tools.find_model(name)
        if not fb:
            continue
        rc = tools.release_check(fb)
        if not rc["ok"]:
            log(f"  '{name}' does not fit now: {rc['summary']}")
            continue
        log(f"{label}: loading '{name}' instead of '{instead_of}'")
        res = tools.start_model(fb)
        if res.get("ok"):
            log(f"{label} OK: '{name}' up in {res.get('load_s')}s")
            return name
        log(f"{label}: '{name}' FAILED too: {res.get('error')}")
        tools.note_failure(name, "load failed")
        tools.stop_server()
    return None


def load_instead(failed: dict, prefer: list, log) -> str | None:
    """`failed` could not be loaded: bring up another model that fits so the
    agent keeps a brain (setting auto_recover). Returns the name now serving."""
    tools.note_failure(failed["name"], "load failed")
    if not tools.get_cfg().get("auto_recover", True):
        return None
    name = _load_first(tools.recovery_candidates(failed["name"], prefer), failed["name"], log,
                       "auto-recover")
    if not name:
        log("auto-recover: no other model could be loaded; the LLM stays unloaded")
    return name


def release_or_fallback(entry: dict, prev: str | None, log) -> dict | None:
    """Gate before loading `entry`: other heavy GPU processes must have released
    their VRAM. On timeout the target is NOT loaded (and NOT noted as failed:
    the model is fine, the GPU is busy); instead the previous model, then the
    primary, then the smallest that fits is brought up so the agent keeps a
    brain. Returns None when the gate passed, else the failure detail."""
    cfg = tools.get_cfg()
    rel = tools.wait_release(entry, float(cfg.get("release_wait_s") or 0), log)
    if rel["ok"]:
        return None
    log(f"RESOURCES NOT RELEASED for '{entry['name']}': {rel['summary']}")
    fallback = _load_first(tools.recovery_candidates(entry["name"], [prev] if prev else []),
                           entry["name"], log, "fallback")
    if not fallback:
        log("no fallback model fits: the LLM stays unloaded")
    return {"error": f"other GPU processes still hold the VRAM '{entry['name']}' needs: "
                     f"{rel['summary']}", "release": rel, "fallback": fallback}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _resume_lost(lost: dict, primary: str, lost_since: float, say) -> bool:
    """A swap/restore/start whose runner died mid-flight (killed together with
    whatever started it). That is not a model failure: nothing is noted and the
    same request is relaunched once. Returns True when a new runner was
    launched (the watcher exits; that runner starts the next watcher)."""
    st = tools.read_state()
    action = lost.get("action") or "restore"
    want = str(lost.get("target") or "") if action == "swap" else primary
    current = tools.match_model(tools.llm_health(timeout=2.0))
    if current and tools.same_name(current, want):
        tools.write_state({"state": "idle", "current": current, "spawned_at": None, "last_result": {
            "action": action, "target": want, "ok": True, "current": current, "at": _now(),
            "detail": {"note": "the runner died, but the model came up"}}})
        say(f"the {action} runner died, but '{current}' came up: state fixed")
        return False
    if tools.server_unresponsive():
        entry = tools.find_model(want) or {}
        if time.time() - lost_since < float(entry.get("load_timeout_s") or 120) + 60:
            return False  # its engine is still loading on its own: give it a load's time
    prev = st.get("resumed") or {}
    if (tools.same_name(prev.get("target"), want)
            and time.time() - float(prev.get("at") or 0) < tools.FAILURE_WINDOW_S):
        say(f"the {action} to '{want}' was cut again (its runner died): not resuming it twice")
        tools.write_state({"state": "failed", "spawned_at": None, "last_result": {
            "action": action, "target": want, "ok": False, "at": _now(),
            "detail": {"error": "the runner died mid-flight twice"}}})
        return False
    say(f"the {action} to '{want}' was cut: its runner died mid-flight (killed together with "
        "the process that started it). Not a model failure: resuming it")
    tools.write_state({"state": "failed", "spawned_at": None,
                       "resumed": {"target": want, "at": time.time()}})
    pin = st.get("pin_gpu")
    if tools.same_name(want, primary):
        out = json.loads(tools.h_restore({}))
    else:
        out = json.loads(tools.h_swap({"model": want, **({"gpu": pin} if pin is not None else {})}))
    if out.get("ok") and not out.get("skipped"):
        say(f"resume: {action} to '{want}' launched")
        return True
    say(f"resume refused: {out.get('error') or out.get('skipped')}")
    return False


def _recover(dead: str | None, primary: str, say, note: bool = True) -> int | None:
    """Load another model after `dead` stopped serving. Returns the exit code
    when the watcher should exit, else None."""
    again = bool(dead) and tools.stopped_by_signal(dead) and dead not in tools.recent_failures()
    if dead:
        if again:
            say(f"'{dead}' was stopped by a signal from outside SMP (see llama-{dead}.log): loading it again")
        else:
            say(f"'{dead}' died on its own (see llama-{dead}.log): swapping to another model")
        if note:
            tools.note_failure(dead, "stopped by a signal" if again else "server died")
    tools.write_state({"server": None})
    for name in ([dead] if again else []) + tools.recovery_candidates(dead):
        if tools.same_name(name, primary):
            out = json.loads(tools.h_restore({}))
        else:
            out = json.loads(tools.h_swap({"model": name}))
        if out.get("ok"):
            say(f"auto-recover: loading '{name}' (its runner starts the next watcher)")
            return 0
        say(f"auto-recover: '{name}' refused: {out.get('error')}")
    say("auto-recover: no other model can be loaded; the LLM stays unloaded")
    return 1


def idle_watch(target: str) -> int:
    """Restore the primary after `idle_restore_min` minutes without work on a
    non-primary model; reload a model whose server died (auto_recover); resume
    a swap whose runner died mid-flight. One watcher at a time (a flock held
    for its whole life). Never holds the swap lock: it only calls h_restore /
    h_swap, which run the normal preflight + release gate."""
    cfg = tools.get_cfg()
    minutes = float(cfg.get("idle_restore_min") or 0)
    primary = str(cfg.get("primary_model") or "")
    recover = bool(cfg.get("auto_recover", True))
    free_comfy = bool(cfg.get("comfy_free_on_idle", True))
    poll = 5.0 if 0 < minutes < 2 else (15.0 if recover else 30.0)

    def say(msg: str) -> None:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] idle-watch: {msg}", flush=True)

    if minutes <= 0 and not recover:
        return 0
    # The previous watcher may still be exiting (its runner spawned us).
    lock_fd = tools.watch_lock(timeout_s=20.0)
    if lock_fd is None:
        return 0
    tools._atomic_json(tools.idle_watch_file(), {"pid": os.getpid(), "target": target,
                                                 "minutes": minutes, "started": _now()})
    if minutes > 0 and not tools.same_name(target, primary):
        say(f"restore '{primary}' after {minutes:g} min without activity on '{target}'")
    if recover:
        say(f"watching '{target}': if it dies, another model is loaded")
    try:
        last_counter = tools.llm_activity()
        last_turn = tools.last_turn_at()
        last_active = time.time()
        silent = 0
        lost_since = 0.0
        comfy_said = False
        while True:
            time.sleep(poll)
            now = time.time()
            if tools.busy_state():
                last_active = now
                lost_since = 0.0
                continue
            lost = tools.runner_died_midflight()
            if lost and recover:
                lost_since = lost_since or now
                if _resume_lost(lost, primary, lost_since, say):
                    return 0
                continue
            lost_since = 0.0
            health = tools.llm_health(timeout=2.0)
            current = tools.match_model(health)
            if current is None:
                if health.get("up"):
                    say("a model outside the registry is serving; exiting")
                    return 0
                dead = tools.crashed_model() if recover else None
                if not dead:
                    stuck = tools.server_unresponsive()
                    if stuck:
                        # The process is alive but did not answer (a loaded
                        # host, a long decode): that is not a stop. Keep watching.
                        silent += 1
                        if silent == 4:
                            say(f"'{stuck}' does not answer /health but its process is alive; still watching")
                        continue
                    if recover and tools.read_state().get("state") == "failed" and not tools.gpu_job_active():
                        # a swap/restore ended with nothing serving: bring a model back
                        say("the last swap/restore left no model serving")
                        return _recover(None, primary, say) or 0
                    say("no model serving (stopped on purpose); exiting")
                    return 0
                if tools.gpu_job_active():
                    continue  # gpu_job.sh brings a model back when the job ends
                time.sleep(3)  # a swap may have just started: look again
                if tools.busy_state() or tools.llm_health(timeout=2.0).get("up"):
                    continue
                return _recover(dead, primary, say) or 0
            if silent >= 4:
                say(f"'{current}' answers again")
            silent = 0
            if tools.same_name(current, primary) or minutes <= 0:
                if not recover:
                    say("primary is back; exiting")
                    return 0
                last_active = now
                continue  # nothing to restore: only watch that it stays alive
            counter = tools.llm_activity()
            working = (counter is not None and last_counter is not None and counter != last_counter)
            working = working or (tools.requests_processing() or 0) > 0
            last_counter = counter if counter is not None else last_counter
            turn = tools.last_turn_at()  # the agent called the LLM (any engine)
            working = working or (turn is not None and last_turn is not None and turn > last_turn)
            last_turn = turn if turn is not None else last_turn
            # Only a ComfyUI with work queued or running counts as GPU work. An
            # idle one (even with models cached in VRAM) does not stop the
            # clock: when the primary does not fit, it is asked to free them.
            comfy = tools.comfy_status(max_age=10.0)
            comfy_busy = bool(comfy.get("up") and comfy.get("busy"))
            if comfy_busy and not comfy_said:
                say("ComfyUI is working: the idle clock waits for its queue to empty")
            comfy_said = comfy_busy
            if working or tools.gpu_job_active() or comfy_busy:
                last_active = now
                continue
            if now - last_active < minutes * 60:
                continue
            rc = tools.check_release(primary)
            if not rc.get("ok") and comfy.get("up") and free_comfy:
                say(f"idle {minutes:g} min and ComfyUI is idle: asking it to release its VRAM (/free)")
                fr = tools.comfy_free()
                if fr.get("ok"):
                    time.sleep(5)
                    rc = tools.check_release(primary)
                else:
                    say(f"ComfyUI /free failed: {fr.get('error')}")
            if not rc.get("ok"):
                say("idle, but the primary does not fit yet: " + str(rc.get("summary")))
                last_active = now - minutes * 60 + 120  # look again in ~2 min
                continue
            say(f"{minutes:g} min idle on '{current}' -> restoring '{primary}'")
            out = json.loads(tools.h_restore({}))
            say("restore " + ("launched" if out.get("ok") else "refused: " + str(out.get("error"))))
            if out.get("ok"):
                return 0
            last_active = now - minutes * 60 + 120
    finally:
        try:
            info = json.loads(tools.idle_watch_file().read_text())
            if int(info.get("pid", 0)) == os.getpid():
                tools.idle_watch_file().unlink()
        except (OSError, ValueError):
            pass
        os.close(lock_fd)


def main() -> int:
    if len(sys.argv) < 5:
        print("usage: runner.py <state_file> <swap|restore|start|idle_watch> <target> <cfg_b64>")
        return 2
    action, target, cfg_b64 = sys.argv[2:5]
    cfg = json.loads(base64.b64decode(cfg_b64).decode())
    tools.apply_cfg(cfg)
    cfg = tools.get_cfg()
    if action == "idle_watch":
        return idle_watch(target)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)

        def add(st: dict) -> None:
            st["log_tail"] = ((st.get("log_tail") or []) + [line])[-20:]
        tools.update_state(add)

    def finish(ok: bool, detail: dict, current: str | None) -> int:
        tools.write_state({
            "state": "idle" if ok else "failed",
            "spawned_at": None,
            "current": current,
            "last_result": {
                "action": action, "target": target, "ok": ok, "current": current,
                "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "detail": detail,
            },
        })
        return 0 if ok else 1

    def drain() -> None:
        tools.drain_requests(log)

    def stop() -> dict:
        res = tools.stop_server()
        for act in res.get("actions", []):
            log("  " + act)
        if not res.get("ok"):
            log("  STOP FAILED: " + str(res.get("error")))
        return res

    try:
        # M4: hold the swap lock for our whole lifetime. The kernel drops it
        # if we die, so a crashed runner can never wedge the state machine.
        with tools.swap_lock(timeout_s=15.0) as got:
            if not got:
                log("another runner holds the swap lock - bailing out")
                tools.write_state({"last_result": {
                    "action": action, "target": target, "ok": False,
                    "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "detail": {"error": "swap lock held by another runner"}}})
                return 1
            # C3: self-assert our pid so busy_state() tracks the real runner
            # (the spawner may not know it: a service or a double fork).
            tools.write_state({"runner_pid": os.getpid(), "runner_start": tools.proc_start(os.getpid())})

            if action == "start":
                return autostart(log, finish)

            # 1) grace: let the agent finish its turn before the brain is cut
            wait = max(0.0, float(tools.read_state().get("grace_until") or 0) - time.time())
            if wait > 0:
                log(f"grace: sleeping {wait:.0f}s before starting {action}")
                time.sleep(wait)
            tools.write_state({"state": "swapping" if action == "swap" else "restoring",
                               "action": action, "target": target})

            # 2) drain, then stop the current model the way it is being run.
            # Looking up the server takes a moment: a request that starts in
            # that gap gets its own drain instead of being cut by the stop.
            for attempt in (1, 2, 3):
                drain()
                prev = tools.match_model(tools.llm_health())
                servers = tools.detect_servers()
                if attempt == 3 or (tools.requests_processing() or 0) <= 0:
                    break
                log("a request started right before the stop: draining again")
            log("stopping current model: "
                + ("; ".join(tools.describe_server(s) for s in servers) or "nothing running"))
            res = stop()
            if not res.get("ok"):
                current = tools.match_model(tools.llm_health())
                return finish(False, {"error": "could not stop the current server",
                                      "stop": res}, current)

            # 3) start target
            primary = tools.find_model(str(cfg.get("primary_model") or ""))
            if action == "swap":
                entry = tools.find_model(target)
                if not entry:
                    return finish(False, {"error": f"unknown model '{target}'"}, None)
                pin = tools.read_state().get("pin_gpu")
                if pin is not None:
                    entry = dict(entry, pin_gpu=int(pin))
                    log(f"'{target}' goes whole on GPU {pin} (the other GPUs stay free)")
            else:
                entry = primary
                if not entry:
                    return finish(False, {"error": f"unknown primary model '{target}'"}, None)

            blocked = release_or_fallback(entry, prev, log)
            if blocked:
                rc = finish(False, blocked, blocked.get("fallback"))
                if blocked.get("fallback"):
                    _watch(blocked["fallback"], log)
                return rc

            log(f"starting '{entry['name']}' on {entry.get('engine') or 'llamacpp'}")
            res = tools.start_model(entry)
            if res.get("ok"):
                log(f"OK: '{entry['name']}' up in {res.get('load_s')}s ({res.get('started_by')})")
                rc = finish(True, res, entry["name"])
                _watch(entry["name"], log)
                return rc
            log(f"START FAILED: {res.get('error')}")
            for ln in res.get("log_tail") or []:
                log("    | " + ln)

            # 4) failed: clear the broken target, then load another model so the
            # agent keeps a brain (the one that was serving first, then the primary,
            # then from small to big). The result stays ok=False: the model that
            # was asked for is NOT running.
            stop()  # do not leave a half-loaded server holding VRAM
            now_serving = load_instead(entry, [prev] if action != "swap" else [], log)
            rc = finish(False, {"error": f"{action} to '{entry['name']}' failed: {res.get('error')}",
                                "target_failed": res, "fallback": now_serving,
                                "primary_restored": bool(primary) and now_serving == primary.get("name")},
                        now_serving)
            if now_serving:
                _watch(now_serving, log)
            return rc
    except Exception as e:  # noqa: BLE001
        try:
            log(f"RUNNER EXCEPTION: {type(e).__name__}: {e}")
            tools.write_state({"state": "failed", "last_result": {
                "action": action, "target": target, "ok": False,
                "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "detail": {"error": f"{type(e).__name__}: {e}"}}})
        except Exception:  # noqa: BLE001
            pass
        return 1


def autostart(log, finish) -> int:
    """action=start: wait for GPU + model file, then load the primary."""
    cfg = tools.get_cfg()
    target = str(cfg.get("primary_model") or "")
    entry = tools.find_model(target)
    if not entry:
        return finish(False, {"error": f"unknown primary model '{target}'"}, None)
    limit = int(cfg.get("gpu_wait_s") or 0)
    deadline = time.time() + max(0, limit)
    gguf = str(entry.get("gguf") or "")
    file_based = tools.model_engine(entry)[0].file_based
    waited = False
    while True:
        gpu_ok = bool(tools.get_vram()) or not tools.nvidia_smi_path()
        file_ok = (bool(gguf) and os.path.exists(gguf)) if file_based else True
        if gpu_ok and file_ok:
            break
        if time.time() >= deadline:
            why = [w for w, ok in (("GPU driver not ready", gpu_ok),
                                   (f"model file missing: {gguf}", file_ok)) if not ok]
            log("autostart: gave up after %ds: %s" % (limit, "; ".join(why)))
            return finish(False, {"error": "; ".join(why)}, None)
        if not waited:
            log(f"autostart: waiting for {'GPU' if not gpu_ok else 'model file'} (up to {limit}s)")
            waited = True
        time.sleep(5)
    servers = tools.detect_servers()
    if servers:  # something came up while we waited: leave it alone
        log("autostart: server already running: " + "; ".join(tools.describe_server(s) for s in servers))
        current = tools.match_model(tools.llm_health())
        rc = finish(True, {"note": "already running"}, current)
        if current:
            _watch(current, log)
        return rc
    tools.write_state({"state": "starting", "action": "start", "target": target})
    blocked = release_or_fallback(entry, None, log)
    if blocked:
        return finish(False, blocked, blocked.get("fallback"))
    log(f"autostart: starting '{target}' on {entry.get('engine') or 'llamacpp'}")
    res = tools.start_model(entry)
    if res.get("ok"):
        log(f"OK: '{target}' up in {res.get('load_s')}s ({res.get('started_by')})")
        rc = finish(True, res, target)
        _watch(target, log)
        return rc
    log(f"START FAILED: {res.get('error')}")
    for ln in res.get("log_tail") or []:
        log("    | " + ln)
    tools.stop_server()  # do not leave a half-loaded server holding VRAM
    now_serving = load_instead(entry, [], log)
    rc = finish(False, {"error": f"start of '{target}' failed: {res.get('error')}",
                        "target_failed": res, "fallback": now_serving}, now_serving)
    if now_serving:
        _watch(now_serving, log)
    return rc


if __name__ == "__main__":
    sys.exit(main())
