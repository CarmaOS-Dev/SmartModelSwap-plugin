"""Detached swap runner. Launched by tools.spawn_runner with:
  runner.py <state_file> <swap|restore|start> <target> <cfg_base64>

swap/restore: grace sleep -> drain in-flight requests -> stop the current
server (signal) -> start target -> verify -> update state. On swap failure it
restores the primary (brain back > target) and still reports the swap as
failed.
start: autostart of the primary (gateway start-up, replaces the old systemd
unit): wait for the GPU driver and the model file (gpu_wait_s), then load.
Writes progress to runner.log and state.json; never raises to the parent.
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
    """Start the idle watcher when a non-primary model ended up serving."""
    try:
        if not tools.same_name(name, tools.get_cfg().get("primary_model")):
            r = tools.spawn_idle_watch(name)
            if r.get("pid"):
                log(f"idle watcher started (pid {r['pid']}): primary comes back after "
                    f"{tools.get_cfg().get('idle_restore_min')} min idle")
    except Exception as e:  # noqa: BLE001
        log(f"idle watcher not started: {type(e).__name__}: {e}")


def release_or_fallback(entry: dict, prev: str | None, log) -> dict | None:
    """Gate before loading `entry`: other heavy GPU processes must have released
    their VRAM. On timeout the target is NOT loaded; instead the previous model
    (or the smallest one that fits) is brought up so the agent keeps a brain.
    Returns None when the gate passed, else the failure detail."""
    cfg = tools.get_cfg()
    rel = tools.wait_release(entry, float(cfg.get("release_wait_s") or 0), log)
    if rel["ok"]:
        return None
    log(f"RESOURCES NOT RELEASED for '{entry['name']}': {rel['summary']}")
    cands = []
    if prev:
        cands.append(prev)
    for m in sorted(tools.load_models().get("models", []),
                    key=lambda x: float(x.get("min_free_vram_gb", 0))):
        cands.append(m["name"])
    fallback = None
    for name in dict.fromkeys(cands):
        if name == entry["name"]:
            continue
        fb = tools.find_model(name)
        if not fb or not tools.release_check(fb)["ok"]:
            continue
        log(f"loading fallback '{name}' instead (it fits in the VRAM that is free)")
        res = tools.start_model(fb)
        if res.get("ok"):
            log(f"fallback OK: '{name}' up in {res.get('load_s')}s")
            fallback = name
            break
        log(f"fallback '{name}' FAILED: {res.get('error')}")
        tools.stop_server()
    if not fallback:
        log("no fallback model fits: the LLM stays unloaded")
    return {"error": f"other GPU processes still hold the VRAM '{entry['name']}' needs: "
                     f"{rel['summary']}", "release": rel, "fallback": fallback}


def idle_watch(target: str) -> int:
    """Restore the primary after `idle_restore_min` minutes without work on a
    non-primary model. Never holds the swap lock and never writes the swap
    state: it only calls h_restore() (preflight + release gate) when idle."""
    cfg = tools.get_cfg()
    minutes = float(cfg.get("idle_restore_min") or 0)
    primary = str(cfg.get("primary_model") or "")
    poll = 30.0 if minutes >= 2 else 5.0

    def say(msg: str) -> None:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] idle-watch: {msg}", flush=True)

    if minutes <= 0:
        return 0
    if tools.idle_watch_status():
        return 0
    tools.idle_watch_file().write_text(json.dumps(
        {"pid": os.getpid(), "target": target, "minutes": minutes,
         "started": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    say(f"restore '{primary}' after {minutes:g} min without activity on '{target}'")
    try:
        last_counter = tools.llm_activity()
        last_active = time.time()
        while True:
            time.sleep(poll)
            now = time.time()
            if tools.busy_state():
                last_active = now
                continue
            current = tools.match_model(tools.llm_health(timeout=2.0))
            if current is None or tools.same_name(current, primary):
                say(f"{'primary is back' if current else 'no model serving'}; exiting")
                return 0
            counter = tools.llm_activity()
            working = (counter is not None and last_counter is not None and counter != last_counter)
            working = working or (tools.requests_processing() or 0) > 0
            last_counter = counter if counter is not None else last_counter
            if working or tools.gpu_job_active() or tools.gpu_holders():
                last_active = now
                continue
            if now - last_active < minutes * 60:
                continue
            rc = tools.check_release(primary)
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


def main() -> int:
    if len(sys.argv) < 5:
        print("usage: runner.py <state_file> <swap|restore> <target> <cfg_b64>")
        return 2
    _state_path, action, target, cfg_b64 = sys.argv[1:5]
    cfg = json.loads(base64.b64decode(cfg_b64).decode())
    tools.apply_cfg(cfg)
    cfg = tools.get_cfg()
    if action == "idle_watch":
        return idle_watch(target)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        tail = (tools.read_state().get("log_tail") or []) + [line]
        tools.write_state({"log_tail": tail[-20:]})

    def finish(ok: bool, detail: dict, current: str | None) -> int:
        tools.write_state({
            "state": "idle" if ok else "failed",
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
            # C3: self-assert our pid so busy_state() tracks the real runner.
            tools.write_state({"runner_pid": os.getpid()})

            if action == "start":
                return autostart(log, finish)

            # 1) grace: let the agent finish its turn before the brain is cut
            wait = max(0.0, float(tools.read_state().get("grace_until") or 0) - time.time())
            if wait > 0:
                log(f"grace: sleeping {wait:.0f}s before starting {action}")
                time.sleep(wait)
            tools.write_state({"state": "swapping" if action == "swap" else "restoring",
                               "action": action, "target": target})

            # 2) drain, then stop the current model the way it is being run
            drain()
            prev = tools.match_model(tools.llm_health())
            servers = tools.detect_servers()
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

            # 4) failed: clear the broken target, then bring the brain back.
            # The result stays ok=False: the model that was asked for is NOT running.
            if action == "swap" and primary and primary.get("name") != entry.get("name"):
                log("restoring primary after failed swap")
                stop()
                res2 = tools.start_model(primary)
                ok2 = bool(res2.get("ok"))
                log(f"primary restore: {'OK' if ok2 else 'FAILED'} {res2}")
                return finish(False, {"error": f"swap to '{target}' failed: {res.get('error')}",
                                      "target_failed": res, "primary_restored": ok2,
                                      "primary": res2},
                              primary["name"] if ok2 else None)
            stop()  # do not leave a half-loaded server holding VRAM
            return finish(False, res, None)
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
    waited = False
    while True:
        gpu_ok = bool(tools.get_vram()) or not tools.nvidia_smi_path()
        file_ok = bool(gguf) and os.path.exists(gguf)
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
        return finish(True, {"note": "already running"}, tools.match_model(tools.llm_health()))
    tools.write_state({"state": "starting", "action": "start", "target": target})
    blocked = release_or_fallback(entry, None, log)
    if blocked:
        return finish(False, blocked, blocked.get("fallback"))
    log(f"autostart: starting '{target}' on {entry.get('engine') or 'llamacpp'}")
    res = tools.start_model(entry)
    if res.get("ok"):
        log(f"OK: '{target}' up in {res.get('load_s')}s ({res.get('started_by')})")
        return finish(True, res, target)
    log(f"START FAILED: {res.get('error')}")
    for ln in res.get("log_tail") or []:
        log("    | " + ln)
    tools.stop_server()  # do not leave a half-loaded server holding VRAM
    return finish(False, res, None)


if __name__ == "__main__":
    sys.exit(main())
