"""pre_tool_call guard: keep the agent from managing the LLM behind SMP's back.

SmartModelProvider owns the engine port. The agent (and scripts it writes)
used to stop the brain with `systemctl --user stop llama-server` / `pkill`
and bring it back with the old unit — which leaves SMP with an "external"
server, races the runner and, on failure, no brain at all. This hook vetoes
those commands (the veto message becomes the tool result, so the agent learns
the right path), launches of ComfyUI while a model holds the GPU (unless they
go through gpu_job.sh or carry SMS_ALLOW_GPU=1), and blocking `hermes sms`
commands run in the foreground: a terminal timeout kills the command, and
they can outlive any foreground timeout (swap --wait, restore --wait, wait,
after-job, stop --graceful).

Read-only commands (status, journalctl, grep, ps, cat …) are never blocked.
Shell scripts run by the command are scanned too (comment lines ignored).
"""
from __future__ import annotations

import os
import re
import shlex
import time
from pathlib import Path

def skill_hint(tools) -> str:
    return ("Use sms_swap / sms_restore (as the LAST tool call of the turn), or for GPU work "
            f"`bash {tools.gpu_job_script()} auto -- <job>` in the background with notify=true "
            f"(frees VRAM through SMP and always leaves a model serving). Rules: {tools.SKILL_REF}.")


# `hermes sms` subcommands that block until a swap ends (minutes, up to the
# swap budget): never in a foreground terminal.
_SMS_BLOCKING = "hermes sms swap/restore --wait, wait, after-job, stop --graceful"

_SEG_SPLIT = re.compile(r"\|\||&&|[;|&\n]")
_WRAPPERS = {"sudo", "nohup", "exec", "setsid", "env", "time", "command", "stdbuf", "nice", "ionice",
             "if", "then", "elif", "else", "while", "until", "do", "!"}
_SYSTEMCTL_VERBS = {"start", "stop", "restart", "try-restart", "reload-or-restart", "kill",
                    "enable", "reenable", "unmask", "link"}
_SHELLS = {"bash", "sh", "zsh", "dash"}
_SCRIPT_MAX = 256 * 1024

# Loose patterns for code that is not shell (execute_code) and for kill pipelines.
_RX_SYSTEMCTL = re.compile(r"systemctl\b[^\n]*\b(?:start|stop|restart|kill|enable|reenable)\b[^\n]*llama-server")
_RX_PKILL = re.compile(r"\b(?:pkill|killall)\b[^\n]*llama")
_RX_KILL_PGREP = re.compile(r"(?:\bkill\b[^\n]*(?:\$\(|`)\s*pgrep[^\n]*llama)|(?:pgrep[^\n|]*llama[^\n]*\|\s*xargs\b[^\n]*\bkill\b)")
_RX_LAUNCH = re.compile(r"llama-server['\"]?\s*(?:,\s*['\"])?\s*(?:-m|--model|-hf|--hf-repo)\b")
_RX_COMFY = re.compile(r"main\.py\b[^\n]*(?:--port\s*['\"]?8188|--listen)|ComfyUI[^\n]*main\.py")
_RX_SMS_BLOCKING = re.compile(r"hermes\s+sms\s+(?:(?:swap|restore)\b[^\n]*--wait\b|wait\b|after-job\b"
                              r"|stop\b[^\n]*--graceful\b)")


def _words(segment: str) -> list[str]:
    try:
        return shlex.split(segment, comments=True)
    except ValueError:
        return segment.split()


def _program(words: list[str]) -> tuple[str, list[str]]:
    """Skip VAR=value assignments and wrappers (sudo, nohup, timeout 10 …)."""
    i = 0
    while i < len(words):
        w = words[i]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", w):
            i += 1
            continue
        base = os.path.basename(w)
        if base in _WRAPPERS:
            i += 1
            continue
        if base == "timeout":
            i += 1
            while i < len(words) and (words[i].startswith("-") or re.match(r"^\d+[smhd]?$", words[i])):
                i += 1
            continue
        return base, words[i + 1:]
    return "", []


def _sms_blocking(prog: str, args: list[str]) -> str | None:
    """`hermes sms <sub>` that blocks until a swap ends, else None."""
    if prog != "hermes" or len(args) < 2 or args[0] != "sms":
        return None
    sub, rest = args[1], args[2:]
    if sub in ("swap", "restore") and "--wait" in rest and "--dry-run" not in rest:
        return f"hermes sms {sub} --wait"
    if sub in ("wait", "after-job"):
        return f"hermes sms {sub}"
    if sub == "stop" and "--graceful" in rest:
        return "hermes sms stop --graceful"
    return None


def _lms_manual(prog: str, args: list[str]) -> str | None:
    """`lms load|unload` by hand (LM Studio's CLI); estimates are read-only."""
    if prog not in ("lms", "lms.exe") or not args or args[0] not in ("load", "unload"):
        return None
    if "--estimate-only" in args:
        return None
    return f"`lms {args[0]}` by hand: SmartModelProvider loads and unloads LM Studio models"


def _shell_findings(text: str, cwd: str | None = None) -> tuple[list[str], list[str], bool, list[str]]:
    """(llm_violations, scripts_to_scan, launches_comfy, blocking_sms) for one
    shell command text. Follows `cd` so relative script paths resolve (scripts
    come back absolute when possible). LM Studio hand loads come back in
    blocking_sms prefixed with "lms:" (they only matter when it is the active engine)."""
    bad: list[str] = []
    scripts: list[str] = []
    comfy = False
    blocking: list[str] = []

    def resolve(path: str) -> str:
        path = os.path.expanduser(path)
        return path if os.path.isabs(path) or not cwd else os.path.join(cwd, path)
    for line in text.splitlines():
        if line.strip().startswith("#"):
            continue
        for seg in _SEG_SPLIT.split(line):
            seg = seg.strip().lstrip("({").strip()
            if not seg:
                continue
            prog, args = _program(_words(seg))
            if not prog:
                continue
            joined = " ".join(args)
            blk = _sms_blocking(prog, args)
            if blk:
                blocking.append(blk)
            lms = _lms_manual(prog, args)
            if lms:
                blocking.append("lms:" + lms)
            if prog == "cd":
                cwd = resolve(args[0]) if args else os.path.expanduser("~")
                continue
            if prog == "systemctl":
                verbs = {a for a in args if not a.startswith("-")}
                if verbs & _SYSTEMCTL_VERBS and "llama-server" in joined:
                    bad.append(f"`systemctl {joined}`: the old llama-server unit is retired")
            elif prog in ("pkill", "killall") and "llama" in joined:
                bad.append(f"`{prog} {joined}`: killing the engine behind SMP")
            elif prog == "llama-server" or prog.endswith("/llama-server"):
                if not ({"--version", "--help", "-h", "--list-devices"} & set(args)):
                    bad.append("launching llama-server by hand: SMP owns the engine port")
            elif prog == "llama-server.sh" or (prog in _SHELLS and args and os.path.basename(args[0]) == "llama-server.sh"):
                bad.append("running ~/llama-server.sh: the old launcher is retired")
            if prog in _SHELLS and len(args) >= 2 and args[0] in ("-c", "-lc", "-lic", "-ic"):
                b2, s2, c2, k2 = _shell_findings(args[1], cwd)
                bad += b2
                scripts += s2
                comfy |= c2
                blocking += k2
            elif prog in _SHELLS and args and not args[0].startswith("-"):
                scripts.append(resolve(args[0]))
            elif prog.endswith(".sh"):
                first = next((w for w in _words(seg) if w.endswith(".sh")), prog)
                scripts.append(resolve(first))
            if re.search(r"(^|/)python[0-9.]*$", prog) and "main.py" in args:
                comfy = True
        if _RX_KILL_PGREP.search(line):
            bad.append("`kill` on pgrep'd llama-server PIDs: killing the engine behind SMP")
        if _RX_COMFY.search(line):
            comfy = True
    return bad, scripts, comfy, blocking


def _scan_script(path: str, cwd: str | None) -> str | None:
    p = Path(os.path.expanduser(path))
    if not p.is_absolute():
        if not cwd:
            return None
        p = Path(cwd) / p
    try:
        if p.is_file() and p.stat().st_size <= _SCRIPT_MAX:
            return p.read_text(errors="replace")
    except OSError:
        return None
    return None


def inspect(tool_name: str, args: dict) -> dict:
    """{'violations': [...], 'comfy': bool, 'gpu_job': bool, 'runs_gpu_job': bool,
    'allow_gpu': bool, 'sms_blocking': [...]}

    gpu_job / runs_gpu_job: the command EXECUTES gpu_job.sh (reading it with
    cat/head, or naming it in an echo, does not count).
    sms_blocking: `hermes sms` commands that block until a swap ends."""
    out = {"violations": [], "comfy": False, "gpu_job": False, "runs_gpu_job": False,
           "allow_gpu": False, "sms_blocking": []}
    if tool_name == "terminal":
        text = str(args.get("command") or "")
        cwd = args.get("workdir") or args.get("cwd")
        out["allow_gpu"] = bool(re.search(r"(^|[\s;&|(])SMS_ALLOW_GPU=1\b", text))
        bad, scripts, comfy, blocking = _shell_findings(text, cwd)
        out["runs_gpu_job"] = out["gpu_job"] = any(os.path.basename(s) == "gpu_job.sh" for s in scripts)
        out["violations"] += bad
        out["comfy"] |= comfy
        out["sms_blocking"] += blocking
        seen = set()
        for sc in scripts[:6]:
            if sc in seen or os.path.basename(sc) == "gpu_job.sh":
                continue
            seen.add(sc)
            body = _scan_script(sc, cwd)
            if body is None:
                continue
            b2, _s2, c2, k2 = _shell_findings(body)
            out["violations"] += [f"{v} (in {sc})" for v in b2]
            out["comfy"] |= c2
            out["sms_blocking"] += [f"{k} (in {sc})" for k in k2]
            out["allow_gpu"] |= "SMS_ALLOW_GPU=1" in body
    elif tool_name == "execute_code":
        code = str(args.get("code") or "")
        for rx, why in ((_RX_SYSTEMCTL, "systemctl on the retired llama-server unit"),
                        (_RX_PKILL, "pkill/killall on the engine"),
                        (_RX_KILL_PGREP, "kill on pgrep'd llama-server PIDs"),
                        (_RX_LAUNCH, "launching llama-server by hand")):
            if rx.search(code):
                out["violations"].append(f"{why} (in execute_code)")
        out["comfy"] = bool(_RX_COMFY.search(code))
        out["gpu_job"] = "gpu_job.sh" in code
        out["allow_gpu"] = "SMS_ALLOW_GPU=1" in code
        m = _RX_SMS_BLOCKING.search(code)
        if m:
            out["sms_blocking"].append(m.group(0).strip()[:60])
    return out


def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return v is True or (isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0)


def notifies(args: dict) -> bool:
    """Will Hermes tell the agent when this background terminal command ends?
    `notify=true` (or the older `notify_on_complete=true`) and `heartbeat=N` do.
    `notify=[patterns]` does not: it only fires on matching output lines."""
    if _truthy(args.get("notify")) or _truthy(args.get("notify_on_complete")):
        return True
    try:
        return int(args.get("heartbeat") or 0) > 0
    except (TypeError, ValueError):
        return False


def _audit(cache_dir: Path, msg: str) -> None:
    try:
        with open(cache_dir / "guard.log", "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except OSError:
        pass


def make_hook(tools):
    """pre_tool_call hook bound to the plugin's tools module."""
    def pre_tool_call(tool_name: str = "", args: dict | None = None, **_kw):
        try:
            if tool_name not in ("terminal", "execute_code"):
                return None
            if not tools.get_cfg().get("guard_tools", True):
                return None
            found = inspect(tool_name, args or {})
            if tool_name == "terminal" and found["runs_gpu_job"] and not found["violations"]:
                how = ("Launch it with the terminal tool, background=true and notify=true (exactly "
                       "`true`, not a list of patterns: only `true` sends the notification when the job "
                       "ends), and nothing appended after the job command.")
                after = ("After launching it reply in one short sentence and END your turn: make no "
                         "further tool or model calls until the completion notification arrives, then "
                         "deliver the result (attach the file of the `SAVED=` line).")
                if not _truthy((args or {}).get("background")):
                    _audit(tools.cache_dir(), "BLOCK terminal: gpu_job.sh in the foreground")
                    return {"action": "block", "message":
                            "[smart-model-swap guard] Blocked: gpu_job.sh must run in the BACKGROUND. "
                            "It unloads or swaps the model that runs this conversation. "
                            f"{how} {after}"}
                if not notifies(args or {}):
                    _audit(tools.cache_dir(), "BLOCK terminal: gpu_job.sh without a completion notification")
                    return {"action": "block", "message":
                            "[smart-model-swap guard] Blocked: this would run gpu_job.sh without a "
                            "completion notification, so the result would never be delivered to the user. "
                            f"{how} {after}"}
            lms = [b[4:] for b in found["sms_blocking"] if b.startswith("lms:")]
            if lms and tools.active_engine_id() == "lmstudio":
                why = "; ".join(dict.fromkeys(lms))
                _audit(tools.cache_dir(), f"BLOCK {tool_name}: {why}")
                return {"action": "block", "message":
                        f"[smart-model-swap guard] Blocked: {why}. LM Studio is the active engine and "
                        f"SmartModelProvider owns which model is loaded. {skill_hint(tools)}"}
            blocking = list(dict.fromkeys(b for b in found["sms_blocking"] if not b.startswith("lms:")))
            if blocking and not found["violations"] and not (
                    tool_name == "terminal" and _truthy((args or {}).get("background")) and notifies(args or {})):
                _audit(tools.cache_dir(), f"BLOCK {tool_name}: blocking `{'; '.join(blocking)}` in the foreground")
                return {"action": "block", "message":
                        f"[smart-model-swap guard] Blocked: `{'; '.join(blocking)}` waits until a model swap "
                        f"ends, which can take up to ~{tools.swap_budget_s() // 60} min. A foreground command "
                        "is killed by the tool timeout, and that kill used to take the model load with it. "
                        "Instead: call sms_restore / sms_swap as the LAST tool call of the turn and tell the "
                        "user the model is switching (the next turn runs on it); or run "
                        "`hermes sms restore` (no --wait) and then `hermes sms wait` with background=true and "
                        "notify=true. After a timeout never re-run the swap: `hermes sms status` shows it is "
                        f"still going. ({_SMS_BLOCKING} are the blocking forms.)"}
            if found["violations"]:
                why = "; ".join(dict.fromkeys(found["violations"]))
                _audit(tools.cache_dir(), f"BLOCK {tool_name}: {why}")
                return {"action": "block", "message":
                        f"[smart-model-swap guard] Blocked: {why}. The local LLM is managed by "
                        f"SmartModelProvider; stopping/starting it by hand breaks swaps and can leave "
                        f"the agent without a model. {skill_hint(tools)}"}
            if found["comfy"] and not (found["gpu_job"] or found["allow_gpu"]):
                busy = tools.busy_state()
                current = tools.match_model(tools.llm_health(timeout=1.5))
                if current or busy:
                    what = f"a model swap is in flight ({busy})" if busy else f"'{current}' holds the GPU"
                    _audit(tools.cache_dir(), f"BLOCK {tool_name}: ComfyUI launch while {what}")
                    return {"action": "block", "message":
                            f"[smart-model-swap guard] Blocked: launching ComfyUI by hand while {what}. "
                            f"Run the job through gpu_job.sh: it moves the LLM out of the way through SMP, "
                            f"runs the job and leaves a model serving afterwards (a ComfyUI left open by "
                            f"hand also keeps the primary from coming back). If `sms_plan` "
                            f"(single_gpu=true) says the task already fits, re-run the same command "
                            f"prefixed with SMS_ALLOW_GPU=1. {skill_hint(tools)}"}
            return None
        except Exception:  # noqa: BLE001 - a guard must never break a tool call
            return None
    pre_tool_call.__name__ = "pre_tool_call"
    return pre_tool_call
