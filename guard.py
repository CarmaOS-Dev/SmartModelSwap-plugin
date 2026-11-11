"""pre_tool_call guard: keep the agent from managing the LLM behind SMP's back.

SmartModelProvider owns the engine port. The agent (and scripts it writes)
used to stop the brain with `systemctl --user stop llama-server` / `pkill`
and bring it back with the old unit — which leaves SMP with an "external"
server, races the runner and, on failure, no brain at all. This hook vetoes
those commands (the veto message becomes the tool result, so the agent learns
the right path) and launches of ComfyUI while the primary model holds the
GPU, unless they go through gpu_job.sh or carry SMS_ALLOW_GPU=1.

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
    return ("Use sms_swap / sms_restore, `hermes sms stop|start`, or for GPU work "
            f"`bash {tools.gpu_job_script()} <small-model|none> -- <job>` "
            f"(frees VRAM through SMP and always restores). Rules: {tools.SKILL_REF}.")

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


def _shell_findings(text: str, cwd: str | None = None) -> tuple[list[str], list[str], bool]:
    """(llm_violations, scripts_to_scan, launches_comfy) for one shell command text.
    Follows `cd` so relative script paths resolve (scripts come back absolute when possible)."""
    bad: list[str] = []
    scripts: list[str] = []
    comfy = False

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
                b2, s2, c2 = _shell_findings(args[1], cwd)
                bad += b2
                scripts += s2
                comfy |= c2
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
    return bad, scripts, comfy


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
    """{'violations': [...], 'comfy': bool, 'gpu_job': bool, 'allow_gpu': bool}"""
    out = {"violations": [], "comfy": False, "gpu_job": False, "allow_gpu": False}
    if tool_name == "terminal":
        text = str(args.get("command") or "")
        cwd = args.get("workdir") or args.get("cwd")
        out["gpu_job"] = "gpu_job.sh" in text
        out["allow_gpu"] = "SMS_ALLOW_GPU=1" in text
        if "hermes sms" in text and not re.search(r"systemctl|pkill|killall|llama-server\.sh", text):
            return out
        bad, scripts, comfy = _shell_findings(text, cwd)
        out["violations"] += bad
        out["comfy"] |= comfy
        seen = set()
        for sc in scripts[:6]:
            if sc in seen or os.path.basename(sc) == "gpu_job.sh":
                continue
            seen.add(sc)
            body = _scan_script(sc, cwd)
            if body is None:
                continue
            b2, _s2, c2 = _shell_findings(body)
            out["violations"] += [f"{v} (in {sc})" for v in b2]
            out["comfy"] |= c2
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
    return out


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
            if (tool_name == "terminal" and found["gpu_job"] and not (args or {}).get("background")
                    and not found["violations"]):
                _audit(tools.cache_dir(), "BLOCK terminal: gpu_job.sh in the foreground")
                return {"action": "block", "message":
                        "[smart-model-swap guard] Blocked: gpu_job.sh must run in the BACKGROUND "
                        "(terminal background=true, notify_on_complete=true). It unloads or swaps the "
                        "model that runs this conversation, so after launching it reply in one short "
                        "sentence and END your turn: make no further tool or model calls until the "
                        "completion notification arrives (the primary model is back by then)."}
            if found["violations"]:
                why = "; ".join(dict.fromkeys(found["violations"]))
                _audit(tools.cache_dir(), f"BLOCK {tool_name}: {why}")
                return {"action": "block", "message":
                        f"[smart-model-swap guard] Blocked: {why}. The local LLM is managed by "
                        f"SmartModelProvider; stopping/starting it by hand breaks swaps and can leave "
                        f"the agent without a model. {skill_hint(tools)}"}
            if found["comfy"] and not (found["gpu_job"] or found["allow_gpu"]):
                primary = tools.get_cfg().get("primary_model")
                current = tools.match_model(tools.llm_health(timeout=1.5))
                if current and tools.same_name(current, primary):
                    _audit(tools.cache_dir(), f"BLOCK {tool_name}: ComfyUI launch while '{current}' is loaded")
                    return {"action": "block", "message":
                            f"[smart-model-swap guard] Blocked: launching ComfyUI while the primary model "
                            f"'{current}' holds the GPU. Run the job through gpu_job.sh (frees VRAM via SMP "
                            f"and restores the primary afterwards). If `sms_plan` (single_gpu=true) says the "
                            f"task already fits, re-run the same command prefixed with SMS_ALLOW_GPU=1. "
                            f"{skill_hint(tools)}"}
            return None
        except Exception:  # noqa: BLE001 - a guard must never break a tool call
            return None
    pre_tool_call.__name__ = "pre_tool_call"
    return pre_tool_call
