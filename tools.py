"""Smart-Model-Swap core logic. Standard library only.

SmartModelProvider (package ``smp``) owns the inference engines: it builds
each model's launch line, spawns the engine process itself and stops it by
signal - no systemd unit involved. Flow of a swap:
  detect the server on the engine's port -> stop it (signal to its process
  group) -> settle VRAM -> start target -> health check + identity check.
Swaps run in a detached runner process (runner.py) so the agent's turn can
end before the brain is cut; a grace countdown plus a request drain give the
agent time to reply.
"""
from __future__ import annotations

import base64
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path

try:  # loaded as hermes_plugins.smart_model_swap.tools (gateway / dashboard)
    from .smp import cmdline as smp_cmdline
    from .smp import engines as smp_engines
    from .smp import registry as smp_registry
    from .smp import i18n as smp_i18n
except ImportError:  # standalone: runner.py, CLI tests
    from smp import cmdline as smp_cmdline  # type: ignore[no-redef]
    from smp import engines as smp_engines  # type: ignore[no-redef]
    from smp import registry as smp_registry  # type: ignore[no-redef]
    from smp import i18n as smp_i18n  # type: ignore[no-redef]

PLUGIN_DIR = Path(__file__).resolve().parent
PLUGIN_ID = "smart-model-swap"

DEFAULTS: dict = {
    "nvidia_smi": "/usr/lib/wsl/lib/nvidia-smi",
    "primary_model": "signal",
    "models_file": "",
    "grace_s": 10,
    "drain_timeout_s": 120,
    "drain_max_s": 600,
    "drain_quiet_s": 8,
    "vram_settle_mib": 300,
    "vram_settle_timeout_s": 90,
    "comfy_url": "http://127.0.0.1:8188",
    "inject_status": True,
    "autostart_primary": True,
    "gpu_wait_s": 180,
    "release_wait_s": 120,
    "guard_tools": True,
    "block_when_unloaded": True,
    "idle_restore_min": 15,
    "after_gpu_job": "small",
    "gpu_job_mode": "auto",
    "auto_recover": True,
    "isolate_scope": True,
    "comfy_free_on_idle": True,
    # Legacy (<= 0.2): read once to migrate models.json to v2, then unused.
    "llama_bin": "",
    "base_url": "",
    "systemd_unit": "",
}

BUSY_STATES = ("grace", "swapping", "restoring", "starting")
BOOL_KEYS = ("inject_status", "autostart_primary", "isolate_scope", "auto_recover",
             "guard_tools", "block_when_unloaded", "comfy_free_on_idle")
RUNNER_SCOPE_PREFIX = "sms-runner-"
LLM_SCOPE_PREFIX = "sms-llm-"
# A runner is spawned detached and registers its own pid once it holds the
# swap lock; until then a fresh busy state counts as in flight.
SPAWN_GRACE_S = 30

_CFG: dict = dict(DEFAULTS)
PLUGIN_CTX = None  # the PluginContext, when register(ctx) ran in this process


def _as_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def apply_cfg(cfg: dict) -> None:
    """Merge resolved config (from ctx.get_config or runner argv) over defaults."""
    global _CFG
    merged = dict(DEFAULTS)
    for k in DEFAULTS:
        v = cfg.get(k)
        if v not in (None, ""):
            merged[k] = v
    for k in BOOL_KEYS:
        merged[k] = _as_bool(merged[k])
    _CFG = merged


def get_cfg() -> dict:
    return dict(_CFG)


# ---------------------------------------------------------------- paths/state

def hermes_home() -> Path:
    """The active profile's Hermes home, so state, logs and the provider install
    follow it. The detached runner is a plain script without Hermes on its path:
    it gets the resolved home through HERMES_HOME (see child_env)."""
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except ImportError:
        return Path(os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes"))


_CHILD_ENV_KEEP = ("HOME", "PATH", "LANG", "LANGUAGE", "LD_LIBRARY_PATH", "TMPDIR", "USER",
                   "LOGNAME", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "PYTHONPATH",
                   "VIRTUAL_ENV", "WSL_INTEROP", "WSLENV", "WSL_DISTRO_NAME")
_CHILD_ENV_PREFIXES = ("LC_", "CUDA_", "GGML_", "SMS_")


def child_env() -> dict:
    """Environment for our own detached runner / watcher: an allow-list of the
    parent's (no provider API keys), with the resolved Hermes home pinned so
    both sides agree on where the state lives."""
    env = {k: v for k, v in os.environ.items()
           if k in _CHILD_ENV_KEEP or k.startswith(_CHILD_ENV_PREFIXES)}
    env["HERMES_HOME"] = str(hermes_home())
    return env


def cache_dir() -> Path:
    d = hermes_home() / "cache" / "smart-model-swap"
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_file() -> Path:
    return cache_dir() / "state.json"


def log_file() -> Path:
    return cache_dir() / "runner.log"


def model_log(name: str) -> Path:
    return cache_dir() / f"llama-{name}.log"


def read_state() -> dict:
    try:
        with open(state_file()) as f:
            st = json.load(f)
        return st if isinstance(st, dict) else {"state": "idle"}
    except FileNotFoundError:
        return {"state": "idle"}
    except (OSError, ValueError):
        return {"state": "idle", "_corrupt": True}


@contextmanager
def file_lock(name: str):
    """Blocking exclusive flock on cache/<name> (released by the kernel if the
    holder dies). Serialises read-modify-write of the small JSON files that the
    gateway, the CLI, the runner and the watcher all update."""
    fd = os.open(str(cache_dir() / name), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _atomic_json(path: Path, data) -> None:
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def update_state(fn) -> dict:
    """Locked read-modify-write: fn(state) mutates the dict in place. A state
    file that does not parse is kept aside (state.json.corrupt) instead of
    being silently replaced by a blank one."""
    with file_lock("state.lock"):
        st = read_state()
        if st.pop("_corrupt", False):
            try:
                os.replace(state_file(), state_file().with_name("state.json.corrupt"))
            except OSError:
                pass
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] state.json did not parse; "
                  "kept as state.json.corrupt", file=sys.stderr, flush=True)
        fn(st)
        st["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _atomic_json(state_file(), st)
        return st


def write_state(s: dict) -> None:
    """Merge `s` into the state (locked read-modify-write, atomic replace), so
    fields written by other parties - runner_pid, server, failures - survive."""
    update_state(lambda st: st.update(s))


@contextmanager
def swap_lock(timeout_s: float = 10.0):
    """fcntl guard: check+spawn window in the gateway, whole lifetime in the runner.

    The kernel drops the lock when the holder dies, so a crashed runner cannot
    wedge the state machine even if state.json still says 'swapping' (M4).
    Yields True when the lock was acquired.
    """
    path = cache_dir() / "swap.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    got = False
    try:
        deadline = time.time() + timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                got = True
                break
            except OSError:
                if time.time() >= deadline:
                    break
                time.sleep(0.2)
        yield got
    finally:
        if got:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)



# ------------------------------------------------------------ process helpers

def _run(cmd: list[str], timeout: float = 10.0, env: dict | None = None):
    """subprocess.run that returns None instead of raising (missing binary, timeout)."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except Exception:  # noqa: BLE001
        return None


def _proc_stat(pid) -> list[str]:
    """Fields of /proc/<pid>/stat after the command name: [state, ppid, ...]."""
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text()
    except (OSError, ValueError, TypeError):
        return []
    return raw.rsplit(")", 1)[-1].split()


def proc_start(pid) -> int | None:
    """Start time of a process (clock ticks since boot), to tell a recycled pid
    from the process we recorded."""
    stat = _proc_stat(pid)
    try:
        return int(stat[19])
    except (IndexError, ValueError):
        return None


def _pid_alive(pid, start=None) -> bool:
    """True for a live process. A zombie counts as dead (a long-lived parent may
    not have reaped it yet). With `start`, the process must also be the one
    recorded then (not a recycled pid)."""
    try:
        if int(pid) <= 0:
            return False
    except (TypeError, ValueError):
        return False
    stat = _proc_stat(pid)
    if not stat or stat[0] in ("Z", "X"):
        return False
    if start:
        cur = proc_start(pid)
        return cur is None or int(start) == cur
    return True


def _proc_argv(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except (OSError, ValueError):
        return []
    return [a.decode("utf-8", "replace") for a in raw.split(b"\0") if a]


def _proc_cgroup(pid: int) -> str:
    """cgroup path of a process (cgroup v2 '0::/path', else the last hierarchy)."""
    try:
        lines = Path(f"/proc/{int(pid)}/cgroup").read_text().splitlines()
    except (OSError, ValueError):
        return ""
    for line in lines:
        if line.startswith("0::"):
            return line[3:]
    return lines[-1].split(":", 2)[-1] if lines else ""


def runner_alive(st: dict) -> bool:
    """The runner of this state is running. A runner spawned less than
    SPAWN_GRACE_S ago that has not registered its pid yet counts as alive."""
    pid = st.get("runner_pid")
    if pid:
        return _pid_alive(pid, st.get("runner_start"))
    try:
        return time.time() - float(st.get("spawned_at") or 0) < SPAWN_GRACE_S
    except (TypeError, ValueError):
        return False


def busy_state() -> str | None:
    """The in-flight state, or None when idle or when the runner died mid-flight."""
    st = read_state()
    if st.get("state") in BUSY_STATES and runner_alive(st):
        return st.get("state")
    return None


def runner_died_midflight(st: dict | None = None) -> dict | None:
    """The state says a swap/restore/start is in flight but its runner is gone
    (killed with whatever spawned it, OOM, ...): {action, target} to relaunch."""
    st = read_state() if st is None else st
    if st.get("state") in BUSY_STATES and not runner_alive(st):
        return {"action": st.get("action"), "target": st.get("target"), "state": st.get("state")}
    return None


# ------------------------------------------------------------------- models

def models_file_path() -> Path:
    cfg = get_cfg()
    p = cfg.get("models_file") or ""
    if p:
        return Path(p).expanduser()
    return PLUGIN_DIR / "models.json"


def _legacy_unit_env() -> dict:
    """Environment= lines of the old llama-server user unit (migration only)."""
    unit = str(get_cfg().get("systemd_unit") or "llama-server")
    unit = unit if unit.endswith(".service") else f"{unit}.service"
    env: dict = {}
    try:
        text = (Path.home() / ".config" / "systemd" / "user" / unit).read_text()
    except OSError:
        return env
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Environment="):
            k, sep, v = line[len("Environment="):].strip().strip('"').partition("=")
            if sep and k and k != "LD_LIBRARY_PATH":  # LD_LIBRARY_PATH is defaulted anyway
                env[k] = v
    return env


def _legacy_ui() -> dict:
    try:
        return json.loads((cache_dir() / "ui.json").read_text())
    except Exception:  # noqa: BLE001
        return {}


@contextmanager
def _registry_file_lock():
    fd = os.open(str(cache_dir() / "models.lock"), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _read_models_raw() -> dict:
    p = models_file_path()
    if not p.exists():
        # fresh install: start from the bundled example (the live registry is not in git)
        example = PLUGIN_DIR / "models.example.json"
        if example.exists() and p == PLUGIN_DIR / "models.json":
            shutil.copy2(example, p)
    with open(p) as f:
        return json.load(f)


def load_models() -> dict:
    """The registry in v2 form. A v1 file is migrated once and rewritten
    (the original is kept next to it as <file>.bak-v1)."""
    data = _read_models_raw()
    if int(data.get("version") or 1) >= smp_registry.VERSION and isinstance(data.get("engines"), dict):
        out, notes = smp_registry.migrate(data)
        if notes:
            save_models(out)
        return out
    cfg = get_cfg()
    with _registry_file_lock():
        data = _read_models_raw()  # re-read under the lock: another process may have migrated
        legacy = {"llama_bin": cfg.get("llama_bin"), "base_url": cfg.get("base_url"),
                  "ui": _legacy_ui(), "env": _legacy_unit_env()}
        out, notes = smp_registry.migrate(data, legacy)
        if notes:
            p = models_file_path()
            bak = p.with_name(p.name + ".bak-v1")
            if not bak.exists() and int(data.get("version") or 1) < smp_registry.VERSION:
                shutil.copy2(p, bak)
            save_models(out)
    return out


def save_models(data: dict) -> None:
    """Atomic replace of the registry file (per-writer temp name)."""
    p = models_file_path()
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)


# Model names: letters (any case, accents ok), digits and basic signs.
# Excluded on purpose: "/" "\\" (names become log file names), ":" (Hermes
# provider:model separator) and whitespace (/swap and the CLI split on it).
MODEL_NAME_RE = re.compile(r"^[^\W_][\w.+()\[\],-]{0,63}$")
MODEL_NAME_HINT = ("nombre: letras (mayúsculas/minúsculas), dígitos y . _ - + ( ) [ ] , "
                   "— sin espacios, / ni :; empieza por letra o dígito (máx. 64)")


MODEL_NAME_HINT_EN = ("name: letters (upper/lower case), digits and . _ - + ( ) [ ] , "
                      "— no spaces, / or :; must start with a letter or digit (max. 64)")


def model_name_hint() -> str:
    return smp_i18n.t(MODEL_NAME_HINT, MODEL_NAME_HINT_EN)


def valid_model_name(name: str) -> bool:
    return bool(MODEL_NAME_RE.match(str(name or "")))


def same_name(a, b) -> bool:
    """Model names are unique case-insensitively ("Qwen" and "qwen" are the same)."""
    return a is not None and b is not None and str(a).casefold() == str(b).casefold()


def find_model(name: str, data: dict | None = None) -> dict | None:
    """Exact name first, then case-insensitive (so `/swap qwen` finds 'Qwen-3.5')."""
    data = load_models() if data is None else data
    models = data.get("models", [])
    for m in models:
        if m.get("name") == name:
            return m
    for m in models:
        if same_name(m.get("name"), name):
            return m
    return None


def available_models() -> list[str]:
    try:
        return [m["name"] for m in load_models().get("models", [])]
    except Exception:
        return []


# ------------------------------------------------------------------ engines

def active_engine_id(data: dict | None = None) -> str:
    data = load_models() if data is None else data
    eid = str(data.get("active_engine") or smp_engines.DEFAULT_ENGINE)
    return eid if eid in smp_engines.ENGINES else smp_engines.DEFAULT_ENGINE


def engine_for(eid: str | None = None, data: dict | None = None):
    """(Engine, normalized engine config). eid None = the active engine."""
    data = load_models() if data is None else data
    eid = eid or active_engine_id(data)
    eng = smp_engines.get(eid) or smp_engines.get(smp_engines.DEFAULT_ENGINE)
    return eng, eng.normalize((data.get("engines") or {}).get(eng.id))


def model_engine(entry: dict, data: dict | None = None):
    return engine_for(str(entry.get("engine") or smp_engines.DEFAULT_ENGINE), data)


def base_url(data: dict | None = None) -> str:
    """Endpoint of the active engine: the server every swap manages."""
    try:
        eng, ecfg = engine_for(None, data)
        return eng.endpoint(ecfg)
    except Exception:  # noqa: BLE001 - registry unreadable: last known / default
        return str(get_cfg().get("base_url") or "http://127.0.0.1:8080")


def engine_info(data: dict | None = None, with_version: bool = False) -> dict:
    eng, ecfg = engine_for(None, data)
    out = {"id": eng.id, "label": eng.label, "status": eng.status, "managed": eng.managed,
           "bin": eng.binary(ecfg), "bin_ok": eng.binary_ok(ecfg), "url": eng.endpoint(ecfg)}
    if with_version:
        out["version"] = eng.version(ecfg).get("version")
    return out


def _model_alias(entry: dict) -> str | None:
    args = [str(a) for a in (entry.get("args") or [])]
    for i, a in enumerate(args[:-1]):
        if a in ("-a", "--alias"):
            return args[i + 1]
    return None


# Sampling parameters exposed in the UI. Each maps to llama.cpp CLI flags:
# the canonical long form plus the short aliases it must recognize, since the
# registry's base args mix both. Order = display order in the pane.
PARAM_FLAGS: list[tuple[str, tuple[str, ...], str]] = [
    ("temperature", ("--temp", "-temp"), "0.7"),
    ("top_p", ("--top-p", "-top_p"), "0.95"),
    ("top_k", ("--top-k", "-top_k"), "40"),
    ("min_p", ("--min-p", "-min_p"), "0.05"),
    ("presence_penalty", ("--presence-penalty", "-presence-penalty", "-pp"), "0.0"),
    ("repeat_penalty", ("--repeat-penalty", "-repeat-penalty"), "1.1"),
]
PARAM_KEYS = [k for k, _, _ in PARAM_FLAGS]
_FLAGSET_BY_KEY = {k: frozenset(flags) for k, flags, _ in PARAM_FLAGS}
_CANON_BY_KEY = {k: flags[0] for k, flags, _ in PARAM_FLAGS}
_DEFAULT_BY_KEY = {k: v for k, _, v in PARAM_FLAGS}
_FLAG_TO_KEY = {f: k for k, flags, _ in PARAM_FLAGS for f in flags}


def _flag_value(args: list[str], flagset: frozenset) -> str | None:
    """Value of a CLI flag (any alias) inside an args list, or None if absent.
    Last occurrence wins, matching how the engine resolves duplicate flags."""
    val: str | None = None
    for i, a in enumerate(args[:-1]):
        if a in flagset:
            val = str(args[i + 1])
    return val


# Accepted range per param: (min, max, integer?). Values end up on the
# llama-server command line, so anything non-numeric must never get through.
PARAM_RANGES: dict[str, tuple[float, float, bool]] = {
    "temperature": (0.0, 5.0, False),
    "top_p": (0.0, 1.0, False),
    "top_k": (0, 1000, True),
    "min_p": (0.0, 1.0, False),
    "presence_penalty": (-2.0, 2.0, False),
    "repeat_penalty": (0.0, 3.0, False),
}


def param_specs() -> list[dict]:
    """Editable params for the UIs: key, canonical flag, default and range."""
    return [{"key": k, "flag": flags[0], "default": d, "min": PARAM_RANGES[k][0],
             "max": PARAM_RANGES[k][1], "integer": PARAM_RANGES[k][2]}
            for k, flags, d in PARAM_FLAGS]


def normalize_params(params: dict) -> tuple[dict, list[str]]:
    """Validate UI-supplied params. Returns (clean {key: str}, errors)."""
    clean: dict[str, str] = {}
    errors: list[str] = []
    for k, v in (params or {}).items():
        if k not in PARAM_RANGES:
            errors.append(f"{k}: unknown parameter")
            continue
        lo, hi, is_int = PARAM_RANGES[k]
        try:
            num = float(str(v).strip().replace(",", "."))
        except (TypeError, ValueError):
            errors.append(f"{k}: '{v}' is not a number")
            continue
        if num != num or num in (float("inf"), float("-inf")):
            errors.append(f"{k}: '{v}' is not a number")
        elif is_int and num != int(num):
            errors.append(f"{k}: must be an integer")
        elif not lo <= num <= hi:
            errors.append(f"{k}: {v} is outside {lo}..{hi}")
        else:
            clean[k] = str(int(num)) if is_int else repr(round(num, 4))
    return clean, errors


def base_params(entry: dict) -> dict:
    """Sampling params written in the entry's own args (defaults where absent)."""
    args = [str(a) for a in (entry.get("args") or [])]
    return {k: (_flag_value(args, _FLAGSET_BY_KEY[k]) or d) for k, _, d in PARAM_FLAGS}


def active_preset(entry: dict) -> str | None:
    """Name of the preset applied at load, or None (dangling names ignored)."""
    name = entry.get("active_preset")
    presets = entry.get("presets")
    return name if name and isinstance(presets, dict) and name in presets else None


def model_params(entry: dict) -> dict:
    """Effective sampling params: base args overridden by the active preset."""
    out = base_params(entry)
    name = active_preset(entry)
    if name:
        for k in PARAM_KEYS:
            v = entry["presets"][name].get(k) if isinstance(entry["presets"][name], dict) else None
            if v not in (None, ""):
                out[k] = str(v)
    return out


def _set_param_in_args(args: list[str], key: str, value) -> list[str]:
    """Replace the value of a sampling param (any alias) in an args list.

    Writes the canonical long form; if no occurrence exists it is appended at
    the end so it wins over any base value. Returns a new list.
    """
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(args):
        a = str(args[i])
        if a in _FLAGSET_BY_KEY[key] and i + 1 < len(args):
            if _FLAG_TO_KEY.get(a) == key:
                out.append(_CANON_BY_KEY[key])
                out.append(str(value))
                replaced = True
                i += 2
            else:  # belongs to another param
                out.append(a)
                i += 1
        else:
            out.append(a)
            i += 1
    if not replaced:
        out.extend([_CANON_BY_KEY[key], str(value)])
    return out


def apply_params_to_args(args: list[str], params: dict) -> list[str]:
    """Return args with each known sampling param set to its given value."""
    out = list(args)
    for k, v in params.items():
        if k in _CANON_BY_KEY and v not in (None, ""):
            out = _set_param_in_args(out, k, v)
    return out


def _serves(entry: dict, model_ids: list[str]) -> bool:
    """Does a /v1/models id list correspond to this registry entry?"""
    gguf = str(entry.get("gguf") or "")
    stem = os.path.basename(gguf)
    alias = _model_alias(entry)
    for mid in model_ids:
        if mid in (gguf, alias, entry.get("name")) or (stem and stem in mid):
            return True
    return False


CTX_FLAGS = ("-c", "--ctx-size")


def model_ctx(entry: dict) -> int | None:
    args = [str(a) for a in (entry.get("args") or [])]
    for i, a in enumerate(args[:-1]):
        if a in CTX_FLAGS:
            try:
                return int(args[i + 1])
            except ValueError:
                return None
    return None


def set_model_ctx(entry: dict, value: int) -> bool:
    """Set the context flag in a model's args (replace every occurrence, or
    append `-c value`). Returns True when something changed."""
    args = [str(a) for a in (entry.get("args") or [])]
    changed, found = False, False
    for i, a in enumerate(args[:-1]):
        if a in CTX_FLAGS:
            found = True
            if args[i + 1] != str(value):
                args[i + 1] = str(value)
                changed = True
    if not found:
        args += ["-c", str(value)]
        changed = True
    if changed:
        entry["args"] = args
    return changed


def shared_ctx(models: list, engine: str, exclude: str | None = None) -> int | None:
    """The context the other models of `engine` share (primary first, else the
    most common value)."""
    primary = get_cfg().get("primary_model")
    vals = []
    for m in models:
        if exclude is not None and same_name(m.get("name"), exclude):
            continue
        if (m.get("engine") or "llamacpp") != engine:
            continue
        c = model_ctx(m)
        if c is None:
            continue
        if same_name(m.get("name"), primary):
            return c
        vals.append(c)
    return max(set(vals), key=vals.count) if vals else None


def sync_ctx(models: list, source: dict, original_name: str | None = None) -> dict:
    """One context for every model of an engine.

    Hermes caches a single context length per model id and the provider serves
    every registry model under the configured one, so all models of an engine
    must share `-c`. The context of `source` (the model being saved) wins and is
    applied to the others; a source without `-c` inherits the shared one.
    Mutates `source` and `models`; returns {"ctx", "synced": [names], "inherited"}.
    """
    engine = source.get("engine") or "llamacpp"
    me = original_name or source.get("name")
    ctx = model_ctx(source)
    inherited = False
    if ctx is None:
        ctx = shared_ctx(models, engine, exclude=me)
        if ctx is None:
            return {"ctx": None, "synced": [], "inherited": False}
        set_model_ctx(source, ctx)
        inherited = True
    synced = []
    for m in models:
        if m is source or same_name(m.get("name"), me) or same_name(m.get("name"), source.get("name")):
            continue
        if (m.get("engine") or "llamacpp") != engine:
            continue
        if set_model_ctx(m, ctx):
            synced.append(m.get("name"))
    return {"ctx": ctx, "synced": synced, "inherited": inherited}



# ------------------------------------------------------- process isolation

_SYSTEMD_CACHE: dict = {}


def _user_systemd() -> bool:
    """Can we create transient user units (systemd-run --user)? Cached per process."""
    if "ok" not in _SYSTEMD_CACHE:
        ok = bool(shutil.which("systemctl") and shutil.which("systemd-run"))
        if ok:
            r = _run(["systemctl", "--user", "show", "-p", "Version"], timeout=5)
            ok = bool(r and r.returncode == 0)
        _SYSTEMD_CACHE["ok"] = ok
    return _SYSTEMD_CACHE["ok"]


def _isolated() -> bool:
    return bool(get_cfg().get("isolate_scope", True)) and _user_systemd()


def _unit_name(prefix: str, name: str, kind: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-") or "x"
    return f"{prefix}{safe}-{time.time_ns() // 1000}.{kind}"


def _scoped(cmd: list[str], prefix: str, name: str) -> tuple[list[str], str | None]:
    """Optionally run cmd in its own cgroup (transient user scope).

    Used for the engine process. It is NOT a service: nothing starts or stops
    through systemd. It only moves the process out of the cgroup of whoever
    spawned it, so neither a gateway restart (KillMode=control-group) nor the
    end of the runner's own service unit takes the loaded model with it. Off
    (isolate_scope: false) or without a user systemd, the process just gets
    its own session (setsid).
    """
    if not _isolated():
        return cmd, None
    unit = _unit_name(prefix, name, "scope")
    return (["systemd-run", "--user", "--scope", "--quiet", "--collect",
             "--unit", unit, "--"] + cmd, unit)


# Fresh, single-threaded interpreter that double-forks: the grandchild runs the
# command in its own session and is re-parented to the subreaper / init, so it
# is no longer a descendant of the caller when the intermediate exits.
_DOUBLE_FORK = r"""
import os, sys
log, cwd, argv = sys.argv[1], sys.argv[2], sys.argv[3:]
r, w = os.pipe()
if os.fork() == 0:
    os.setsid()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        nul = os.open(os.devnull, os.O_RDONLY)
        os.dup2(nul, 0); os.dup2(fd, 1); os.dup2(fd, 2)
        os.chdir(cwd)
        os.execvp(argv[0], argv)
    os.write(w, str(pid).encode())
    os._exit(0)
os.close(w)
os.wait()
print(os.read(r, 32).decode() or "0")
"""


def _detached(cmd: list[str], prefix: str, name: str, log_path: Path) -> dict:
    """Start a long-lived helper (runner, watcher) that is NOT a descendant of
    the caller: killing the caller's process tree - a terminal tool timing out
    on `hermes sms ... --wait`, a gpu_job.sh killed by its host - must never take
    a model load with it.

    With a user systemd: a transient *service* (its parent is systemd), output
    appended to log_path. Otherwise: a double fork. Returns {pid, unit}; the
    pid may be None for a service (the helper registers its own pid).
    """
    env = child_env()
    if _isolated():
        unit = _unit_name(prefix, name, "service")
        sd = ["systemd-run", "--user", "--quiet", "--collect", "--unit", unit,
              f"--working-directory={PLUGIN_DIR}",
              "-p", f"StandardOutput=append:{log_path}", "-p", f"StandardError=append:{log_path}"]
        sd += [f"--setenv={k}={v}" for k, v in sorted(env.items())]
        r = _run(sd + ["--"] + cmd, timeout=20, env=env)
        if r is not None and r.returncode == 0:
            pid = None
            q = _run(["systemctl", "--user", "show", "-p", "MainPID", "--value", unit], timeout=5)
            if q is not None and q.stdout.strip().isdigit() and int(q.stdout.strip()) > 0:
                pid = int(q.stdout.strip())
            return {"pid": pid, "unit": unit}
    r = subprocess.run([sys.executable, "-c", _DOUBLE_FORK, str(log_path), str(PLUGIN_DIR)] + cmd,
                       capture_output=True, text=True, timeout=20, env=env,
                       stdin=subprocess.DEVNULL)
    try:
        pid = int((r.stdout or "0").strip().splitlines()[-1]) or None
    except (ValueError, IndexError):
        pid = None
    return {"pid": pid, "unit": None}


# ---------------------------------------------------------- server detection

def managed_port(data: dict | None = None) -> int:
    u = urllib.parse.urlparse(base_url(data))
    return u.port or (443 if u.scheme == "https" else 80)


def _is_engine_proc(pid: int, data: dict | None = None) -> bool:
    """Is this process the active engine's server (by executable name)? Guards
    every stop path so a port held by some other service never gets it killed."""
    eng, ecfg = engine_for(None, data)
    return eng.is_engine_process(pid, ecfg)


def _listener_pid(port: int, data: dict | None = None) -> int | None:
    """PID listening on the managed TCP port (ss first, /proc scan as fallback)."""
    ss = shutil.which("ss")
    if ss:
        r = _run([ss, "-ltnpH", f"sport = :{port}"], timeout=5)
        if r is not None:
            m = re.search(r"pid=(\d+)", r.stdout)
            if m:
                return int(m.group(1))
    data = load_models() if data is None else data  # once, not once per /proc entry
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        pid = int(p.name)
        if not _is_engine_proc(pid, data):
            continue
        argv = _proc_argv(pid)
        flagged = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a in ("--port", "-port")]
        if flagged and flagged[-1] == str(port):
            return pid
    return None


def classify_pid(pid: int, data: dict | None = None) -> dict:
    """Who runs this server process?

    smp        started by SmartModelProvider (recorded in state.json, or living
               in one of our sms-llm-* scopes) -> signal its process group.
    external   an engine server SMP did not start (a terminal, an old unit)
               -> signal that one PID.
    foreign    the port is held by something that is not the engine - never touched.
    """
    info = {"mode": "external", "pid": pid, "unit": None}
    if not _is_engine_proc(pid, data):
        argv = _proc_argv(pid)
        info.update(mode="foreign", name=os.path.basename(argv[0]) if argv else "?")
        return info
    rec = read_state().get("server") or {}
    cg = _proc_cgroup(pid)
    leaf = cg.rstrip("/").rsplit("/", 1)[-1] if cg else ""
    if rec.get("pid") == pid or (leaf.startswith(LLM_SCOPE_PREFIX) and leaf.endswith(".scope")):
        info.update(mode="smp", unit=leaf if leaf.startswith(LLM_SCOPE_PREFIX) else rec.get("unit"),
                    model=rec.get("model") if rec.get("pid") == pid else None)
    return info


def detect_servers(data: dict | None = None) -> list[dict]:
    """Every managed server instance that a swap has to stop, de-duplicated.

    Sources: the listener on the engine's port, and whatever SMP started last
    (may still be loading, so not listening yet).
    """
    data = load_models() if data is None else data
    found: list[dict] = []

    def add(info: dict) -> None:
        if not any(f.get("pid") == info.get("pid") for f in found):
            found.append(info)

    port = managed_port(data)
    pid = _listener_pid(port, data)
    if pid:
        add({**classify_pid(pid, data), "via": "port", "port": port})

    rec = read_state().get("server") or {}
    rpid = rec.get("pid")
    if rpid and _pid_alive(rpid) and _is_engine_proc(int(rpid), data):
        add({**classify_pid(int(rpid), data), "via": "state"})
    return found


def describe_server(info: dict | None) -> str:
    if not info or info.get("mode") in (None, "none"):
        return "not running"
    mode = info["mode"]
    if mode == "smp":
        where = f", scope {info['unit']}" if info.get("unit") else ""
        return f"SmartModelProvider (pid {info.get('pid')}{where})"
    if mode == "foreign":
        return f"foreign process '{info.get('name')}' (pid {info.get('pid')})"
    return f"external engine process (pid {info.get('pid')})"


def detect_server() -> dict:
    """The primary detected server (for status/UI), or {'mode': 'none'}."""
    servers = detect_servers()
    return servers[0] if servers else {"mode": "none"}


def _signal(pid: int, sig: int, group: bool) -> None:
    if group:
        try:
            if os.getpgid(pid) == pid:  # we started it with its own session
                os.killpg(pid, sig)
                return
        except OSError:
            pass
    os.kill(pid, sig)


def _stop_one(info: dict) -> tuple[bool, str]:
    mode = info.get("mode")
    if mode == "foreign":
        return False, (f"port {info.get('port')} is held by '{info.get('name')}' "
                       f"(pid {info.get('pid')}), not the engine - not touching it")
    if mode not in ("smp", "external"):
        return False, f"unknown server mode {mode!r}"
    pid = int(info["pid"])
    group = mode == "smp"
    try:
        _signal(pid, signal.SIGINT, group)  # llama-server shuts down cleanly on SIGINT
    except ProcessLookupError:
        return True, f"pid {pid} already gone"
    except PermissionError:
        return False, f"no permission to signal pid {pid}"
    deadline = time.time() + 25
    while time.time() < deadline and _pid_alive(pid):
        time.sleep(0.5)
    if _pid_alive(pid):
        try:
            _signal(pid, signal.SIGKILL, group)
        except OSError:
            pass
        time.sleep(2)
        return (not _pid_alive(pid)), f"SIGINT ignored, SIGKILL pid {pid}"
    return True, f"SIGINT {'group ' if group else ''}pid {pid}"


def stop_server() -> dict:
    """Stop the managed server by signal, then settle VRAM.

    Returns {ok, actions, error?}. Never a machine-wide pkill: other engine
    instances (embeddings, tests) on other ports are left alone.
    """
    actions: list[str] = []
    for _attempt in (1, 2):  # second pass covers a wrapper that respawned it
        servers = detect_servers()
        if not servers:
            break
        for s in servers:
            _ok, msg = _stop_one(s)
            actions.append(f"{s.get('mode')}: {msg}")
        deadline = time.time() + 15
        while time.time() < deadline and [s for s in detect_servers() if s.get("mode") != "foreign"]:
            time.sleep(1)
        if all(s.get("mode") == "foreign" for s in detect_servers()):
            break
    remaining = detect_servers()
    if remaining:
        return {"ok": False, "actions": actions,
                "error": "server still running after stop: "
                         + "; ".join(describe_server(s) for s in remaining)}
    if not actions:
        actions.append("no managed server running")
    write_state({"server": None})
    actions.append("VRAM settled" if settle_vram() else "VRAM settle timeout (proceeding)")
    return {"ok": True, "actions": actions}


# --------------------------------------------------------------------- vram

def nvidia_smi_path() -> str | None:
    cfg = get_cfg()
    p = cfg.get("nvidia_smi") or ""
    if p and os.path.exists(p):
        return p
    return shutil.which("nvidia-smi")


def nvidia_env() -> dict:
    """Minimal environment for nvidia-smi (no provider keys from the host process)."""
    return smp_engines.minimal_env()


def get_vram() -> list[dict]:
    exe = nvidia_smi_path()
    if not exe:
        return []
    r = _run([exe, "--query-gpu=index,name,memory.total,memory.used",
              "--format=csv,noheader,nounits"], timeout=10, env=nvidia_env())
    if r is None:
        return []
    out = []
    try:
        for line in r.stdout.strip().splitlines():
            idx, name, total, used = [x.strip() for x in line.split(",")]
            out.append({
                "gpu": int(idx),
                "name": name,
                "total_mib": int(total),
                "used_mib": int(used),
                "free_mib": int(total) - int(used),
            })
    except ValueError:
        return []
    return out


def total_free_gb(vram: list[dict] | None = None) -> float:
    vram = get_vram() if vram is None else vram
    return sum(g["free_mib"] for g in vram) / 1024.0


def llm_vram_used_mib(pids: list[int]) -> int | None:
    """VRAM held by the given PIDs per nvidia-smi, or None when the driver
    does not report per-process usage (WSL prints N/A)."""
    exe = nvidia_smi_path()
    if not exe or not pids:
        return None
    r = _run([exe, "--query-compute-apps=pid,used_memory",
              "--format=csv,noheader,nounits"], timeout=10, env=nvidia_env())
    if r is None or r.returncode != 0:
        return None
    total, seen = 0, False
    for line in r.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            pid, used = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if pid in pids:
            total += used
            seen = True
    return total if seen else None


def settle_vram() -> bool:
    """Wait for VRAM to settle, ComfyUI-tolerant (M2).

    Succeeds when every GPU is below the threshold, or when no managed server
    is left and usage stopped moving between two consecutive reads: on a
    dual-GPU WSL rig ComfyUI (and the Windows desktop) pin VRAM permanently,
    so a hard threshold would burn the full timeout on every swap.
    """
    cfg = get_cfg()
    thr = int(cfg["vram_settle_mib"])
    deadline = time.time() + int(cfg["vram_settle_timeout_s"])
    prev = None
    while time.time() < deadline:
        cur = get_vram()
        if not cur:
            return True  # no nvidia-smi: nothing to wait on
        if all(g["used_mib"] <= thr for g in cur):
            return True
        if prev is not None and len(cur) == len(prev):
            flat = all(
                abs(cur[i]["used_mib"] - prev[i]["used_mib"]) <= 128
                for i in range(len(cur))
            )
            if flat and not detect_servers():
                return True
        prev = cur
        time.sleep(2)
    return False



# ---------------------------------------------------------------------- llm

def llm_health(timeout: float = 3.0) -> dict:
    eng, ecfg = engine_for()
    p = eng.probe(eng.endpoint(ecfg), timeout)
    return {"up": p["up"], "models": p["models"]}


def llm_ready() -> bool:
    """Model loaded and serving (llama.cpp: /health 200; 503 while loading)."""
    eng, ecfg = engine_for()
    return eng.ready(eng.endpoint(ecfg))


def requests_processing() -> int | None:
    """In-flight requests on the active engine, or None if it cannot tell."""
    eng, ecfg = engine_for()
    return eng.in_flight(eng.endpoint(ecfg))


def drain_requests(log=print) -> None:
    """Let in-flight requests finish so the agent's reply is not cut.

    Keeps waiting while generation makes progress (the engine's token counter
    moves), up to drain_max_s; cuts early only when nothing has moved for
    drain_timeout_s (a stuck request)."""
    cfg = get_cfg()
    stall = int(cfg.get("drain_timeout_s") or 0)
    cap = max(stall, int(cfg.get("drain_max_s") or 0))
    # A turn is a chain of requests with short gaps (tool calls, Hermes' own
    # follow-up calls): one idle read is not "the turn ended". The engine must
    # stay idle for drain_quiet_s in a row.
    quiet = max(0.0, float(cfg.get("drain_quiet_s") or 0))
    n = requests_processing()
    if n is None or stall <= 0 or (n <= 0 and quiet <= 0):
        return
    announce = f"(cut after {stall}s without progress, max {cap}s)"
    announced = n > 0
    if announced:
        log(f"draining {n} in-flight request(s) {announce}")
    t0 = last_move = time.time()
    quiet_since = None if n > 0 else t0
    last = llm_activity()
    blind = 0
    while True:
        now = time.time()
        n = requests_processing()
        if n is None:
            # One failed read can be a busy engine; three in a row means it is
            # gone and there is nothing left to protect.
            blind += 1
            if blind >= 3:
                if announced:
                    log("drain: the engine stopped answering")
                return
            time.sleep(1)
            continue
        blind = 0
        if n <= 0:
            quiet_since = now if quiet_since is None else quiet_since
            if now - quiet_since >= max(quiet, 1.0):
                if announced:
                    log(f"drained in {int(now - t0)}s")
                return
        else:
            quiet_since = None
            if not announced:
                log(f"a request started during the quiet window: draining {n} in-flight request(s) {announce}")
                announced = True
                last_move = now
        c = llm_activity()
        if c is not None and last is not None and c != last:
            last_move = now
        last = c if c is not None else last
        if n > 0 and now - last_move >= stall:
            log(f"drain: no progress for {stall}s - cutting with {n} request(s) in flight")
            return
        if now - t0 >= cap:
            log(f"drain: still generating after {cap}s - cutting with {n} request(s) in flight")
            return
        time.sleep(1)


def graceful_stop(log=print) -> dict:
    """Unload the LLM the polite way (what a swap does before its stop): grace
    countdown so the agent finishes its turn, drain in-flight requests, stop."""
    grace = int(get_cfg().get("grace_s") or 0)
    if grace > 0:
        log(f"grace: sleeping {grace}s before unloading")
        time.sleep(grace)
    drain_requests(log)
    return stop_server()


def llm_activity() -> float | None:
    """Monotonic work counter of the active engine (tokens processed), or None."""
    eng, ecfg = engine_for()
    return eng.activity(eng.endpoint(ecfg))


# ------------------------------------------------------------ idle watcher
#
# After a swap to a non-primary model, a detached watcher restores the
# primary once the engine has been idle for `idle_restore_min` minutes (no
# tokens processed, no gpu_job.sh running, no other GPU users) and the release
# gate passes. One watcher at a time (pid file); it never holds the swap lock.

def gpu_job_marker() -> Path:
    return cache_dir() / "gpu_job.active"


def gpu_job_active() -> bool:
    try:
        return _pid_alive(gpu_job_marker().read_text().split()[0])
    except (OSError, IndexError):
        return False


def gpu_job_info() -> dict | None:
    """The running gpu_job.sh, for the UIs: {pid, mode, job, since, elapsed_s}.
    mode is 'none' (LLM unloaded for the job) or the model kept during it."""
    try:
        marker = gpu_job_marker()
        fields = marker.read_text().split()
        pid = int(fields[0])
        if not _pid_alive(pid):
            return None
        since = marker.stat().st_mtime
    except (OSError, IndexError, ValueError):
        return None
    applied = fields[1] if len(fields) > 1 else ""  # none | keep | the model kept (gpu_job.sh)
    argv = _proc_argv(pid)
    mode, job = "", ""
    for i, a in enumerate(argv):
        if os.path.basename(a) == "gpu_job.sh" and i + 1 < len(argv):
            mode = argv[i + 1]
            rest = argv[i + 2:]
            if rest and rest[0] == "--":
                rest = rest[1:]
            job = " ".join(os.path.basename(x) if "/" in x else x for x in rest)[:120]
            break
    if applied and applied not in GPU_JOB_MODES[:1]:  # not the unresolved "auto"
        mode = applied
    return {"pid": pid, "mode": mode, "job": job,
            "since": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(since)),
            "elapsed_s": int(time.time() - since)}


def idle_watch_file() -> Path:
    return cache_dir() / "idle_watch.json"


def watch_lock(timeout_s: float = 0.0) -> int | None:
    """Take the watcher singleton lock (idle_watch.lock), polling up to
    timeout_s. Returns the fd to keep open for the watcher's lifetime, or None
    when another watcher holds it. The kernel drops it if the watcher dies."""
    fd = os.open(str(cache_dir() / "idle_watch.lock"), os.O_CREAT | os.O_RDWR, 0o600)
    deadline = time.time() + max(0.0, timeout_s)
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError:
            if time.time() >= deadline:
                os.close(fd)
                return None
            time.sleep(0.5)


def idle_watch_status() -> dict | None:
    """The running watcher ({pid, target, ...}), or None. Alive = it holds the
    singleton lock (no pid-file race, no recycled pid)."""
    fd = watch_lock(0.0)
    if fd is not None:
        os.close(fd)  # nobody holds it
        return None
    try:
        info = json.loads(idle_watch_file().read_text())
        return info if isinstance(info, dict) else {}
    except (OSError, ValueError):
        return {"pid": None}


FAILURE_WINDOW_S = 30 * 60


def note_failure(name: str, why: str = "") -> None:
    """Remember that `name` failed to load or died, so the automatic fallbacks
    do not pick it again for a while (no crash loops). Explicit swaps ignore it."""
    now = time.time()

    def add(st: dict) -> None:
        hist = [f for f in (st.get("failures") or [])
                if isinstance(f, dict) and now - float(f.get("at") or 0) <= FAILURE_WINDOW_S]
        hist.append({"model": name, "at": now, "why": str(why)[:200]})
        st["failures"] = hist[-10:]
    update_state(add)


def recent_failures(window_s: float = FAILURE_WINDOW_S) -> set:
    now = time.time()
    return {str(f.get("model")) for f in (read_state().get("failures") or [])
            if isinstance(f, dict) and now - float(f.get("at") or 0) <= window_s}


def crashed_model() -> str | None:
    """Name of the model whose server died on its own: the state still records
    a running server (a deliberate stop clears it) but its process is gone."""
    srv = read_state().get("server") or {}
    if srv.get("model") and srv.get("pid") and not _pid_alive(srv.get("pid"), srv.get("start")):
        return str(srv["model"])
    return None


def stopped_by_signal(name: str) -> bool:
    """True when the engine log of `name` ends with a clean shutdown: the server
    was told to stop (SIGINT/SIGTERM from outside SMP), it did not crash."""
    return any("cleaning up before exit" in ln for ln in _log_tail(model_log(name), 4))


def server_unresponsive() -> str | None:
    """Name of the model whose server process is alive but does not answer
    (overloaded host, hung engine). Not the same as stopped or crashed."""
    srv = read_state().get("server") or {}
    if srv.get("model") and srv.get("pid") and _pid_alive(srv.get("pid"), srv.get("start")):
        return str(srv["model"])
    return None


def recovery_candidates(failed: str | None, prefer: list | None = None) -> list[str]:
    """Models to try, in order, when `failed` could not be loaded or died:
    `prefer` first, then the primary, then the rest from small to big. Skips
    `failed` and everything that failed in the last FAILURE_WINDOW_S."""
    bad = recent_failures() | ({failed} if failed else set())
    models = sorted(load_models().get("models", []),
                    key=lambda m: float(m.get("min_free_vram_gb") or 0))
    primary = find_model(str(get_cfg().get("primary_model") or ""))
    order = [n for n in (prefer or []) if n]
    order += ([primary["name"]] if primary else []) + [m["name"] for m in models]
    return [n for n in dict.fromkeys(order) if n not in bad and find_model(n)]


def spawn_idle_watch(target: str) -> dict:
    """Start the watcher for `target` unless disabled or one is already running.
    It restores the primary when idle (non-primary models) and swaps to another
    model if the one serving dies (auto_recover)."""
    cfg = get_cfg()
    idle_on = (float(cfg.get("idle_restore_min") or 0) > 0
               and not same_name(target, cfg.get("primary_model")))
    if not idle_on and not cfg.get("auto_recover", True):
        return {"ok": True, "skipped": "nothing to watch (primary, or idle restore off) and auto_recover is off"}
    cur = idle_watch_status()
    if cur:
        return {"ok": True, "skipped": f"watcher already running (pid {cur.get('pid')})"}
    b64 = base64.b64encode(json.dumps(cfg).encode()).decode()
    rotate_log(log_file())
    out = _detached([sys.executable, str(PLUGIN_DIR / "runner.py"), str(state_file()),
                     "idle_watch", target, b64], RUNNER_SCOPE_PREFIX, "idle", log_file())
    return {"ok": True, "pid": out["pid"], "unit": out["unit"], "target": target}


def ensure_idle_watch() -> dict:
    """Gateway start-up: a non-primary model is serving but nobody watches it."""
    current = match_model(llm_health(timeout=2.0))
    if not current or busy_state():
        return {"ok": True, "skipped": "nothing loaded or busy"}
    return spawn_idle_watch(current)


def match_model(health: dict) -> str | None:
    if not health.get("up"):
        return None
    ids = health.get("models", [])
    models = load_models().get("models", [])
    for m in models:  # exact: gguf path / file name / alias / name
        if _serves(m, ids):
            return m["name"]
    for m in models:  # loose: registry name appears in the id
        if any(m.get("name") and str(m["name"]).lower() in mid.lower() for mid in ids):
            return m["name"]
    return None


_comfy_cache: dict = {"at": 0.0, "url": None, "data": None}


def comfy_status(max_age: float = 0.0) -> dict:
    """ComfyUI /system_stats. max_age > 0 allows a cached answer (status polling)."""
    cfg = get_cfg()
    if (max_age > 0 and _comfy_cache["data"] is not None and _comfy_cache["url"] == cfg["comfy_url"]
            and time.time() - _comfy_cache["at"] < max_age):
        return _comfy_cache["data"]
    res = _comfy_probe(cfg)
    _comfy_cache.update(at=time.time(), url=cfg["comfy_url"], data=res)
    return res


def _comfy_probe(cfg: dict) -> dict:
    """{up, vram, running, pending, busy}. busy = its queue has work; an open
    but idle ComfyUI is not GPU work (it may still cache models in VRAM)."""
    base = cfg["comfy_url"].rstrip("/")
    try:
        with urllib.request.urlopen(base + "/system_stats", timeout=2) as r:
            data = json.load(r)
        gib = 1024 ** 3
        vram = []
        for i, d in enumerate(data.get("devices", [])):
            item = {"name": d.get("name", f"GPU{i}"),
                    "free_gib": round(d.get("vram_free", 0) / gib, 1),
                    "total_gib": round(d.get("vram_total", 0) / gib, 1)}
            # vram_free/vram_total are device-wide (they include the LLM);
            # torch_vram_total is what ComfyUI's own process has reserved.
            if "torch_vram_total" in d:
                item["held_gib"] = round(d.get("torch_vram_total", 0) / gib, 1)
            vram.append(item)
    except Exception:  # noqa: BLE001
        return {"up": False, "vram": [], "running": 0, "pending": 0, "busy": False}
    running = pending = 0
    queue_known = False
    try:
        with urllib.request.urlopen(base + "/queue", timeout=2) as r:
            q = json.load(r)
        running, pending = len(q.get("queue_running") or []), len(q.get("queue_pending") or [])
        queue_known = True
    except Exception:  # noqa: BLE001 - unknown queue: assume it works (safe side)
        pass
    return {"up": True, "vram": vram, "running": running, "pending": pending,
            "busy": (running + pending) > 0 or not queue_known}


def comfy_free() -> dict:
    """Ask an idle ComfyUI to unload its models and release cached VRAM
    (POST /free). The process keeps running; nothing is killed."""
    url = get_cfg()["comfy_url"].rstrip("/") + "/free"
    body = json.dumps({"unload_models": True, "free_memory": True}).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            ok = 200 <= r.status < 300
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    _comfy_cache["data"] = None
    return {"ok": ok}


def comfy_retained_gb(comfy: dict | None = None) -> float:
    """VRAM held by ComfyUI itself (its torch reservation), summed across GPUs."""
    c = comfy_status() if comfy is None else comfy
    if not c.get("up"):
        return 0.0
    total = 0.0
    for d in c.get("vram", []):
        if "held_gib" in d:
            total += d["held_gib"]
        else:  # old ComfyUI without torch_vram_total: device-wide usage (pessimistic)
            total += d.get("total_gib", 0) - d.get("free_gib", 0)
    return round(total, 1)


def vram_budget(vram: list[dict] | None = None) -> dict:
    """VRAM the target model can use once the current LLM is unloaded.

    Best: free now + what the running server holds (per-process nvidia-smi).
    WSL does not report per-process usage, so there: total - ComfyUI's own
    reservation. Nothing running: simply what is free now.
    """
    vram = get_vram() if vram is None else vram
    total = sum(g["total_mib"] for g in vram) / 1024.0
    free = sum(g["free_mib"] for g in vram) / 1024.0
    pids = [int(s["pid"]) for s in detect_servers() if s.get("pid") and s.get("mode") != "foreign"]
    if not pids:
        return {"gb": round(free, 1), "method": "free now (no managed server loaded)"}
    used = llm_vram_used_mib(pids)
    if used is not None:
        return {"gb": round(free + used / 1024.0, 1),
                "method": "free now + current LLM usage (nvidia-smi per-process)"}
    return {"gb": round(max(0.0, total - comfy_retained_gb()), 1),
            "method": "total - ComfyUI reservation (per-process usage unavailable)"}


def comfy_busy(max_age: float = 5.0) -> bool:
    """ComfyUI is up AND has work queued or running."""
    c = comfy_status(max_age=max_age)
    return bool(c.get("up") and c.get("busy"))


# ------------------------------------------------- release gate (per GPU)
#
# Before a model is loaded, every GPU it will occupy must have room for its
# share. What a model really takes per GPU is learned on each successful load
# (VRAM delta, recorded in vram_profile.json); until then it is estimated from
# min_free_vram_gb and the engine's split flags (-ts / --device / -sm).

RELEASE_MARGIN_GIB = {"measured": 0.1, "measured-other-ctx": 0.5, "estimated": 0.5, "total": 0.5}


def vram_profile_file() -> Path:
    return cache_dir() / "vram_profile.json"


def load_vram_profile() -> dict:
    try:
        return json.loads(vram_profile_file().read_text())
    except (OSError, ValueError):
        return {}


def profile_key(name: str, pin=None) -> str:
    """A model pinned to one GPU takes a different share than its normal
    multi-GPU layout: it gets its own profile entry."""
    return name if pin is None else f"{name}@gpu{int(pin)}"


def serving_entry(current: str | None, data: dict | None = None) -> dict | None:
    """Registry entry of the model serving now, carrying the GPU pin it was
    launched with (so the VRAM it gives back is credited to the right GPU)."""
    entry = find_model(current, data) if current else None
    if not entry:
        return None
    srv = read_state().get("server") or {}
    if srv.get("pin_gpu") is not None and same_name(srv.get("model"), entry["name"]) \
            and _pid_alive(srv.get("pid")):
        entry = dict(entry, pin_gpu=int(srv["pin_gpu"]))
    return entry


def record_vram_profile(name: str, before: list[dict], after: list[dict], pin=None) -> dict | None:
    """Per-GPU VRAM a model took on load (after - before). Skipped when the
    delta is implausibly small (another process freed VRAM meanwhile)."""
    b = {g["gpu"]: g["used_mib"] for g in before or []}
    per = {}
    for g in after or []:
        if g["gpu"] in b:
            per[str(g["gpu"])] = max(0, g["used_mib"] - b[g["gpu"]])
    if pin is not None:
        per = {k: v for k, v in per.items() if k == str(int(pin))}
    key = profile_key(name, pin)
    entry = find_model(name) or {}
    floor = 0.5 * float(entry.get("min_free_vram_gb", 0)) * 1024
    if not per or sum(per.values()) < max(512.0, floor):
        return None
    ctx = model_ctx(entry) if entry else None
    on = {int(pin)} if pin is not None else (
        layout_gpus(entry, [g["gpu"] for g in after or []]) if entry else None)
    with file_lock("vram_profile.lock"):
        prof = load_vram_profile()
        old = prof.get(key) or {}
        hist = [h for h in (old.get("history") or []) if isinstance(h, dict) and h.get("per_gpu_mib")]
        hist.append({"per_gpu_mib": per, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "ctx": ctx})
        hist = hist[-5:]
        med = _profile_median(hist, ctx, on) or per
        prof[key] = {"per_gpu_mib": med, "at": hist[-1]["at"], "ctx": ctx, "history": hist}
        _atomic_json(vram_profile_file(), prof)
    try:
        _raise_declared_min(name, sum(med.values()) / 1024.0)
    except Exception:  # noqa: BLE001 - the profile is what matters; the registry is cosmetic
        pass
    return prof[key]


def _raise_declared_min(name: str, measured_gib: float) -> float | None:
    """Keep the registry's min_free_vram_gb honest: raise it (never lower it)
    to what the model really took on its last load."""
    with _registry_file_lock():
        data = _read_models_raw()
        for m in data.get("models", []):
            if m.get("name") == name:
                declared = float(m.get("min_free_vram_gb") or 0)
                new = round(measured_gib + 0.05, 1)
                if new > declared + 0.25:
                    m["min_free_vram_gb"] = new
                    save_models(data)
                    return new
                return None
    return None


def layout_gpus(entry: dict, gpus: list[int], data: dict | None = None) -> set | None:
    """Physical GPUs the model's launch flags put it on, or None if unknown."""
    try:
        data = load_models() if data is None else data
        eng, ecfg = model_engine(entry, data)
        split = eng.gpu_split(ecfg, entry, launch_args(entry, data), gpus)
    except Exception:  # noqa: BLE001
        return None
    return set(split) if split else None


def _same_layout(hist: list, on: set | None) -> list:
    """Loads made on the same set of GPUs as the current flags. After the user
    moves a model (--device / -ts / CUDA_VISIBLE_DEVICES) the older deltas
    describe another placement and must not be mixed in."""
    if not on:
        return list(hist)
    return [h for h in hist
            if {int(g) for g, v in (h.get("per_gpu_mib") or {}).items() if v > 256} == on]


def _profile_median(hist: list, ctx: int | None, on: set | None = None) -> dict | None:
    """Per-GPU median over the loads made with context `ctx` (entries without a
    recorded ctx count as matching) on the GPU set `on`. Other processes growing
    or shrinking during a load skew a single delta by a few hundred MiB either way."""
    same = [h for h in _same_layout(hist, on) if h.get("ctx") in (None, ctx)]
    if not same:
        return None
    med = {}
    for g in same[-1]["per_gpu_mib"]:
        vals = sorted(h["per_gpu_mib"][g] for h in same if g in h["per_gpu_mib"])
        med[g] = vals[len(vals) // 2] if len(vals) % 2 else (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) // 2
    return med


def model_gpu_need(entry: dict, gpus: list[int], data: dict | None = None) -> dict:
    """{"per_gpu": {gpu: GiB}, "source": measured|estimated|total, "total_gib"}."""
    total = float(entry.get("min_free_vram_gb", 0))
    pin = entry.get("pin_gpu")
    if pin is not None:
        pin = int(pin)
        rec = load_vram_profile().get(profile_key(entry["name"], pin)) or {}
        med = _profile_median(rec.get("history") or [], model_ctx(entry))
        if med and str(pin) in med:
            gib = med[str(pin)] / 1024.0
            return {"per_gpu": {pin: gib}, "source": "measured", "total_gib": gib}
        # never loaded on this GPU alone yet: everything it takes when spread, on one GPU
        spread = model_gpu_need({k: v for k, v in entry.items() if k != "pin_gpu"}, gpus, data)
        return {"per_gpu": {pin: spread["total_gib"]}, "source": "estimated",
                "total_gib": spread["total_gib"]}
    rec = load_vram_profile().get(entry["name"]) or {}
    prof = rec.get("per_gpu_mib") or {}
    source = "measured"
    if rec.get("history"):
        cur_ctx = model_ctx(entry)
        on = layout_gpus(entry, gpus, data)
        med = _profile_median(rec["history"], cur_ctx, on)
        placed = _same_layout(rec["history"], on)
        if med:
            prof = med
        elif placed:
            prof = placed[-1]["per_gpu_mib"]
            source = "measured-other-ctx"  # context changed since: a guess until the next load
        else:
            prof = {}  # never loaded on these GPUs: estimate from the flags below
    if prof:
        per = {int(k): v / 1024.0 for k, v in prof.items() if int(k) in gpus}
        # A measurement taken with another context is a guess: never let it go
        # below the declared minimum. One taken with the current context is the
        # truth, and the declared minimum may be stale: it is auto-raised on
        # load and never lowered, so after a context reduction it would inflate
        # the need and block a model that fits.
        scale = 1.0
        if per and source != "measured":
            scale = max(1.0, total / max(0.001, sum(per.values())))
        if source == "measured" and per:
            total = sum(per.values())
        return {"per_gpu": {g: v * scale for g, v in per.items()}, "source": source,
                "total_gib": max(total, sum(per.values()))}
    try:
        data = load_models() if data is None else data
        eng, ecfg = model_engine(entry, data)
        split = eng.gpu_split(ecfg, entry, launch_args(entry, data), gpus)
    except Exception:  # noqa: BLE001
        split = None
    if split is None:
        return {"per_gpu": {}, "source": "total", "total_gib": total}
    return {"per_gpu": {g: total * f for g, f in split.items()}, "source": "estimated",
            "total_gib": total}


def llm_server_pids() -> list[int]:
    return [int(s["pid"]) for s in detect_servers() if s.get("pid") and s.get("mode") != "foreign"]


def gpu_holders(exclude_pids: list[int] | None = None) -> list[dict]:
    """Other heavy GPU users: ComfyUI (via its API) and, where the driver
    reports them, compute processes that are not the LLM server."""
    exclude = set(exclude_pids if exclude_pids is not None else llm_server_pids())
    out: list[dict] = []
    comfy = comfy_status(max_age=5.0)
    comfy_item = None
    if comfy.get("up"):
        comfy_item = {"kind": "comfyui", "name": "ComfyUI", "url": get_cfg()["comfy_url"],
                      "held_gib": comfy_retained_gb(comfy), "busy": bool(comfy.get("busy")),
                      "running": comfy.get("running", 0), "pending": comfy.get("pending", 0)}
        out.append(comfy_item)
    exe = nvidia_smi_path()
    if not exe:
        return out
    r_gpu = _run([exe, "--query-gpu=index,uuid", "--format=csv,noheader"], timeout=10, env=nvidia_env())
    r_app = _run([exe, "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                 timeout=10, env=nvidia_env())
    if r_app is None or r_app.returncode != 0:
        return out
    uuid_idx = {}
    for line in (r_gpu.stdout if r_gpu is not None else "").strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            uuid_idx[parts[1]] = int(parts[0])
    procs: dict[int, set] = {}
    for line in r_app.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        try:
            pid = int(parts[0])
        except (ValueError, IndexError):
            continue
        if pid in exclude:
            continue
        procs.setdefault(pid, set())
        if len(parts) > 1 and parts[1] in uuid_idx:
            procs[pid].add(uuid_idx[parts[1]])
    for pid, gset in sorted(procs.items()):
        argv = _proc_argv(pid)
        cmd = " ".join(argv)[:160]
        if comfy_item is not None and "main.py" in cmd:
            comfy_item.update(pid=pid, gpus=sorted(gset))
            continue
        out.append({"kind": "process", "pid": pid, "gpus": sorted(gset),
                    "cmd": cmd or "(not visible)"})
    return out


def check_release(name: str, wait_s: float = 0.0, poll_s: float = 5.0) -> dict:
    """CLI/script gate before loading `name`: credits the model that would be
    unloaded (the one serving now) and polls up to wait_s."""
    entry = find_model(str(name or ""))
    if not entry:
        return {"ok": False, "error": f"unknown model '{name}'", "available": available_models()}
    deadline = time.time() + max(0.0, wait_s)
    while True:
        busy = busy_state()
        if busy:  # mid-swap the VRAM numbers mean nothing: wait for it to end
            if time.time() >= deadline:
                return {"ok": False, "busy": busy, "model": entry["name"],
                        "summary": f"a swap is in flight ({busy}); VRAM not measured"}
            time.sleep(poll_s)
            continue
        current = match_model(llm_health(timeout=1.5))
        if current == entry["name"]:
            return {"ok": True, "model": entry["name"], "summary": f"'{current}' is already loaded"}
        rc = release_check(entry, unloading=serving_entry(current))
        rc["unloading"] = current
        if rc["ok"] or time.time() >= deadline:
            return rc
        time.sleep(poll_s)


def describe_holder(h: dict) -> str:
    if h.get("kind") == "comfyui":
        what = (f"working: {h.get('running', 0)} running, {h.get('pending', 0)} queued"
                if h.get("busy") else "idle")
        return f"ComfyUI ({h.get('url')}, {what}, holds ~{h.get('held_gib', 0)} GiB" + (
            f", pid {h['pid']})" if h.get("pid") else ")")
    return f"pid {h.get('pid')} on GPU{h.get('gpus') or '?'}: {h.get('cmd')}"


def release_check(entry: dict, unloading: dict | None = None,
                  vram: list[dict] | None = None) -> dict:
    """Is there room on every GPU the model will use?

    unloading: the model that will be stopped first (its share is credited
    back). None = judge the VRAM free right now (the runner, after the stop).
    """
    vram = get_vram() if vram is None else vram
    if not vram:
        return {"ok": True, "gpus": [], "holders": [], "source": "no-gpu-info",
                "summary": "nvidia-smi unavailable; not checked"}
    gpus = [g["gpu"] for g in vram]
    need = model_gpu_need(entry, gpus)
    credit = {}
    if unloading:
        cn = model_gpu_need(unloading, gpus)
        credit = cn["per_gpu"] or {}
        credit_total = cn["total_gib"] if not credit else sum(credit.values())
    else:
        credit_total = 0.0
    margin = RELEASE_MARGIN_GIB.get(need["source"], 0.5)
    rows, ok = [], True
    if need["per_gpu"]:
        for g in vram:
            n = need["per_gpu"].get(g["gpu"])
            if not n:
                continue
            free = g["free_mib"] / 1024.0 + credit.get(g["gpu"], 0.0)
            fits = free >= n + margin
            ok &= fits
            rows.append({"gpu": g["gpu"], "free_gib": round(free, 1), "need_gib": round(n, 1),
                         "ok": fits})
    else:
        free = sum(g["free_mib"] for g in vram) / 1024.0 + credit_total
        fits = free >= need["total_gib"] + margin
        ok = fits
        rows.append({"gpu": "all", "free_gib": round(free, 1),
                     "need_gib": round(need["total_gib"], 1), "ok": fits})
    holders = gpu_holders()
    short = ", ".join(f"GPU{r['gpu']} {r['free_gib']}/{r['need_gib']} GiB" + ("" if r["ok"] else " ✗")
                      for r in rows)
    summary = f"free/need: {short} [{need['source']}]"
    if holders:
        summary += "; other GPU users: " + "; ".join(describe_holder(h) for h in holders)
    return {"ok": ok, "model": entry["name"], "gpus": rows, "holders": holders,
            "source": need["source"], "margin_gib": margin, "summary": summary}



# ------------------------------------------------------------ process mgmt

def launch_args(entry: dict, data: dict | None = None) -> list[str]:
    """Engine common_args + the model's args, with the active preset applied."""
    eng, ecfg = model_engine(entry, data)
    args = [str(a) for a in (ecfg.get("common_args") or [])]
    args += [str(a) for a in (entry.get("args") or [])]
    name = active_preset(entry)
    if name:
        clean, _ = normalize_params({k: v for k, v in entry["presets"][name].items()
                                     if k in PARAM_RANGES})
        args = apply_params_to_args(args, clean)
    return eng.finalize_args(ecfg, entry, dedupe_switches(args))


def dedupe_switches(args: list[str]) -> list[str]:
    """Drop a repeated value-less switch (`--metrics` in the engine's
    common_args and again in the model's args): engines warn about it. Options
    that take a value are left alone, repeated or not."""
    is_flag = smp_cmdline.is_flag
    out: list[str] = []
    seen: set = set()
    for i, tok in enumerate(args):
        bare = is_flag(tok) and (i + 1 == len(args) or is_flag(args[i + 1]))
        if bare:
            if tok in seen:
                continue
            seen.add(tok)
        out.append(tok)
    return out


LOG_MAX_BYTES = 5 * 1024 * 1024


def rotate_log(path: Path, max_bytes: int = LOG_MAX_BYTES) -> None:
    """Keep one previous generation (`<name>.1`) once a log passes max_bytes.
    Called right before a log is opened for a new run; a process that still
    writes to the old file keeps its handle."""
    try:
        if path.stat().st_size > max_bytes:
            os.replace(path, path.with_name(path.name + ".1"))
    except OSError:
        pass


def build_cmd(entry: dict, data: dict | None = None) -> list[str]:
    data = load_models() if data is None else data
    eng, ecfg = model_engine(entry, data)
    return eng.build_cmd(ecfg, entry, launch_args(entry, data))


def _log_tail(path: Path, lines: int = 8) -> list[str]:
    try:
        with open(path, errors="replace") as f:
            return [ln.rstrip("\n")[:300] for ln in f.readlines()[-lines:]]
    except OSError:
        return []


def start_model(entry: dict) -> dict:
    """Spawn the model on its engine and wait until it is loaded AND is the
    model we asked for."""
    name = entry["name"]
    data = load_models()
    eng, ecfg = model_engine(entry, data)
    if not eng.managed:
        return {"ok": False, "started_by": "smp",
                "error": smp_i18n.t(f"{eng.label}: motor en construcción", f"{eng.label}: engine under construction")}
    try:
        argv = eng.build_cmd(ecfg, entry, launch_args(entry, data))
    except smp_engines.EngineUnavailable as e:
        return {"ok": False, "started_by": "smp", "error": str(e)}
    cmd, scope = _scoped(argv, LLM_SCOPE_PREFIX, name)
    logf = model_log(name)
    rotate_log(logf)
    vram_before = get_vram()
    with open(logf, "ab") as lf:
        lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} start {name} =====\n"
                 f"{smp_cmdline.join(argv)}\n".encode())
        lf.flush()
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=eng.env(ecfg, entry),
                                cwd=os.path.dirname(argv[0]) or None,
                                start_new_session=True)
    write_state({"server": {"mode": "smp", "engine": eng.id, "unit": scope,
                            "pid": proc.pid, "start": proc_start(proc.pid),
                            "model": name, "pin_gpu": entry.get("pin_gpu")}})
    res = wait_health(entry, "scope" if scope else "session", proc=proc)
    if res.get("ok"):
        try:
            prof = record_vram_profile(name, vram_before, get_vram(), pin=entry.get("pin_gpu"))
            if prof:
                res["vram_per_gpu_mib"] = prof["per_gpu_mib"]
        except Exception:  # noqa: BLE001 - measuring must never fail a good load
            pass
    return res


def wait_release(entry: dict, timeout_s: float, log=None, poll_s: float = 5.0) -> dict:
    """Runner gate: wait until other heavy processes have released enough VRAM
    on every GPU the model will use (the current LLM is already stopped)."""
    deadline = time.time() + max(0.0, float(timeout_s))
    announced = False
    while True:
        rc = release_check(entry)
        if rc["ok"]:
            if announced and log:
                log("  resources released: " + rc["summary"])
            return rc
        if time.time() >= deadline:
            return rc
        if not announced and log:
            log(f"  waiting up to {int(timeout_s)}s for other GPU processes to release VRAM: "
                + rc["summary"])
            announced = True
        time.sleep(poll_s)


def wait_health(entry: dict, started_by: str, proc=None) -> dict:
    timeout = int(entry.get("load_timeout_s", 120))
    name = entry["name"]
    logf = str(model_log(name))
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            return {"ok": False, "started_by": started_by, "log": logf,
                    "error": f"engine exited with code {proc.returncode} during load",
                    "log_tail": _log_tail(model_log(name))}
        if llm_ready():
            h = llm_health()
            if h.get("models") and not _serves(entry, h["models"]):
                return {"ok": False, "started_by": started_by,
                        "error": f"server is up but serving {h['models']}, expected '{name}' "
                                 f"({os.path.basename(str(entry.get('gguf', '')))})"}
            return {"ok": True, "current": name, "load_s": int(time.time() - t0),
                    "started_by": started_by}
        time.sleep(2)
    return {"ok": False, "error": f"health check timeout after {timeout}s",
            "started_by": started_by, "log": logf, "log_tail": _log_tail(model_log(name))}


# ------------------------------------------------------------------ helpers

SKILL_NAME = "smart-model-routing"
SKILL_REF = 'skill_view("smart-model-swap:smart-model-routing")'


def gpu_job_script() -> str:
    """Path of the bundled gpu_job.sh, with ~ for the home directory."""
    p = str(PLUGIN_DIR / "skills" / SKILL_NAME / "templates" / "gpu_job.sh")
    home = os.path.expanduser("~")
    return "~" + p[len(home):] if p.startswith(home + os.sep) else p


def jdump(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def validate_argv(entry: dict, data: dict | None = None) -> dict:
    """Parse-check the exact launch argv against the engine without loading a model."""
    data = load_models() if data is None else data
    eng, ecfg = model_engine(entry, data)
    res = eng.validate(ecfg, entry, launch_args(entry, data))
    if isinstance(res.get("cmd"), list):
        res["cmd"] = smp_cmdline.join(res["cmd"])
    return res


def preflight(cfg_entry: dict, need_extra_gb: float = 0.0, swap: bool = True) -> dict:
    """Validate a model before touching anything.

    When swap=True (the normal case) the VRAM fit is judged against the pool
    available AFTER the current model is unloaded, not the live free amount —
    otherwise a full brain would always report 'no room' for the small model.
    """
    checks = []
    data = load_models()
    eng, ecfg = model_engine(cfg_entry, data)
    checks.append({"check": "engine managed by SMP", "ok": eng.managed,
                   "detail": eng.label + ("" if eng.managed else smp_i18n.t(" (en construcción)", " (under construction)"))})
    active = active_engine_id(data)
    if eng.id != active:
        checks.append({"check": "model engine is the active engine", "ok": False,
                       "detail": f"model on '{eng.id}', active engine '{active}'"})
    checks.append({"check": "engine binary", "ok": eng.binary_ok(ecfg, cfg_entry),
                   "detail": eng.binary(ecfg, cfg_entry) or "(not configured)"})
    gguf = cfg_entry.get("gguf", "")
    checks.append({"check": "gguf exists", "ok": bool(gguf) and os.path.exists(gguf), "detail": gguf})
    mm = cfg_entry.get("mmproj")
    if mm:
        checks.append({"check": "mmproj exists", "ok": os.path.exists(mm), "detail": mm})
    if eng.managed:
        argv = validate_argv(cfg_entry, data)
        checks.append({"check": "argv accepted by binary", "ok": argv["ok"],
                       "detail": argv.get("error") or (str(argv.get("cmd", ""))[:120] + " ...")})
    foreign = [s for s in detect_servers(data) if s.get("mode") == "foreign"]
    if foreign:
        checks.append({"check": "engine port is free or held by the engine", "ok": False,
                       "detail": describe_server(foreign[0])})
    need = float(cfg_entry.get("min_free_vram_gb", 0)) + need_extra_gb
    if swap:
        budget = vram_budget()
        avail = budget["gb"]
        detail = f"available-after-unload {avail:.1f} GiB >= need {need:.1f} GiB [{budget['method']}]"
    else:
        avail = total_free_gb()
        detail = f"free {avail:.1f} GiB >= need {need:.1f} GiB"
    checks.append({"check": "VRAM fit", "ok": avail >= need, "detail": detail})
    current = match_model(llm_health(timeout=1.5)) if swap else None
    rc = release_check(cfg_entry, unloading=serving_entry(current))
    checks.append({"check": "heavy GPU processes released (VRAM per GPU)", "ok": rc["ok"],
                   "detail": rc["summary"]})
    return {"ok": all(c["ok"] for c in checks), "checks": checks,
            "available_gb": round(avail, 1)}


def spawn_runner(action: str, target: str, grace: int | None = None, pin=None) -> dict:
    """Detached runner: grace sleep -> drain -> stop -> start -> state update."""
    cfg = get_cfg()
    b64 = base64.b64encode(json.dumps(cfg).encode()).decode()
    grace = int(cfg["grace_s"]) if grace is None else int(grace)
    write_state({
        "state": "starting" if action == "start" else "grace",
        "action": action,
        "target": target,
        "grace_until": time.time() + grace,
        "pin_gpu": None if pin is None else int(pin),
        "runner_pid": None,
        "runner_start": None,
        "runner_unit": None,
        "spawned_at": time.time(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    rotate_log(log_file())
    with open(log_file(), "ab") as lf:
        lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} spawn {action} {target} =====\n".encode())
    try:
        out = _detached([sys.executable, str(PLUGIN_DIR / "runner.py"), str(state_file()),
                         action, target, b64], RUNNER_SCOPE_PREFIX, action, log_file())
    except Exception as e:  # noqa: BLE001 - never leave a phantom "grace" behind
        write_state({"state": "failed", "spawned_at": None,
                     "last_result": {"action": action, "target": target, "ok": False,
                                     "error": f"could not start the runner: {type(e).__name__}: {e}",
                                     "at": time.strftime("%Y-%m-%dT%H:%M:%S")}})
        return {"ok": False, "error": f"could not start the runner: {type(e).__name__}: {e}"}
    write_state({"runner_unit": out["unit"]})
    note = ("Loading in the background." if action == "start" else
            f"Model will be cut in ~{grace}s (after in-flight requests drain) "
            "and loaded in the background. Finish the current turn.")
    return {"ok": True, "state": "starting" if action == "start" else "grace", "grace_s": grace,
            "target": target, "runner_pid": out["pid"], "runner_unit": out["unit"],
            "budget_s": swap_budget_s(find_model(target) if action == "swap" else None),
            "note": note}


def autostart_primary(replace: bool = False) -> dict:
    """Load the primary model (gateway start-up).

    Replaces the old llama-server systemd unit: no-op when disabled, when the
    primary is already serving, or when a swap is in flight. Non-blocking:
    the runner waits for the GPU and the model file (gpu_wait_s) and loads it
    in the background.

    replace=True (the gateway): start-up always ends on the primary, so a
    registered non-primary model left serving (e.g. across a gateway restart)
    is swapped for it through a normal restore (grace + drain + release gate,
    falling back to what was serving). Left alone: a running GPU job or
    ComfyUI (the idle watcher restores later), a server that is still
    loading, and anything on the port that is not a registered model.
    replace=False (`hermes sms start`): only when nothing is serving.
    """
    cfg = get_cfg()
    if not cfg.get("autostart_primary", True):
        return {"ok": True, "skipped": "autostart_primary is off"}
    name = str(cfg.get("primary_model") or "")
    data = load_models()
    entry = find_model(name, data)
    if not entry:
        return {"ok": False, "skipped": f"primary model '{name}' not in registry"}
    eng, _ = model_engine(entry, data)
    if not eng.managed:
        return {"ok": True, "skipped": f"{eng.label} is under construction"}
    with swap_lock(timeout_s=2.0) as got:
        if not got:
            return {"ok": True, "skipped": "swap lock held"}
        if busy_state():
            return {"ok": True, "skipped": f"busy ({busy_state()})"}
        servers = detect_servers(data)
        if not servers:
            return spawn_runner("start", name, grace=0)
    running = "; ".join(describe_server(s) for s in servers)
    current = match_model(llm_health(timeout=2.0))
    if not replace or not current or same_name(current, name):
        return {"ok": True, "skipped": "already running: " + running}
    if gpu_job_active() or comfy_busy():
        return {"ok": True, "skipped": f"'{current}' is serving and a GPU job / ComfyUI is running: "
                                       "the primary comes back through the idle restore"}
    out = json.loads(h_restore({}))  # takes the swap lock itself
    if out.get("ok"):
        out["note"] = f"'{current}' was serving at start-up -> restoring the primary '{name}'"
    else:
        out["note"] = f"'{current}' stays: restore of '{name}' refused: {out.get('error')}"
    return out


# ----------------------------------------------------------------- handlers

def sms_available() -> bool:
    """check_fn: hide the tools when the active engine's binary is missing."""
    try:
        eng, ecfg = engine_for()
        return eng.binary_ok(ecfg)
    except Exception:  # noqa: BLE001
        return False


def swap_view(st: dict | None = None) -> dict:
    """Swap state for status/UI. A busy state whose runner is dead is reported
    as 'failed' so the UI and the agent are never locked out by a crash."""
    st = read_state() if st is None else st
    state = st.get("state", "idle")
    alive = runner_alive(st)
    stale = state in BUSY_STATES and not alive
    return {
        "state": "failed" if stale else state,
        "stale": stale,
        "action": st.get("action"),
        "target": st.get("target"),
        "runner_pid": st.get("runner_pid"),
        "runner_unit": st.get("runner_unit"),
        "runner_alive": alive,
        "grace_until": st.get("grace_until"),
        "updated_at": st.get("updated_at"),
        "last_result": st.get("last_result"),
    }


def h_status(args: dict, **kwargs) -> str:
    try:
        cfg = get_cfg()
        data = load_models()
        health = llm_health()
        vram = get_vram()
        comfy = comfy_status(max_age=10.0)
        server = detect_server()
        models = []
        for m in data.get("models", []):
            item = {"name": m["name"], "role": m.get("role", ""), "engine": m.get("engine"),
                    "min_free_vram_gb": m.get("min_free_vram_gb"),
                    "ctx": model_ctx(m), "gguf": m.get("gguf")}
            if active_preset(m):
                item["preset"] = active_preset(m)
            if args.get("include_cmd"):
                try:
                    item["cmd"] = smp_cmdline.join(build_cmd(m, data))
                except smp_engines.EngineUnavailable as e:
                    item["cmd"] = f"({e})"
            models.append(item)
        return jdump({
            "ok": True,
            "swap": swap_view(),
            "engine": engine_info(data),
            "llm": {"up": health.get("up"), "current": match_model(health),
                    "models": health.get("models", [])},
            "server": {**server, "summary": describe_server(server)},
            "vram": vram,
            "total_free_gb": round(total_free_gb(vram), 1),
            "comfyui": comfy,
            "gpu_holders": [describe_holder(h) for h in gpu_holders()],
            "gpu_job": gpu_job_info(),
            "watcher": idle_watch_status(),
            "failures": [{"model": f.get("model"), "why": f.get("why"),
                          "at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(float(f.get("at") or 0)))}
                         for f in (read_state().get("failures") or [])
                         if isinstance(f, dict) and time.time() - float(f.get("at") or 0) <= FAILURE_WINDOW_S],
            "swap_budget_s": swap_budget_s(),
            "primary_model": cfg["primary_model"],
            "models": models,
        })
    except Exception as e:  # noqa: BLE001
        return jdump({"ok": False, "error": f"{type(e).__name__}: {e}"})


def h_plan(args: dict, **kwargs) -> str:
    """Which model to keep loaded so another GPU task (ComfyUI, …) fits.

    Uses what each model really takes per GPU (measured on load, see
    vram_profile.json) instead of the registry's declared minimum, and by
    default assumes the task runs on ONE GPU (ComfyUI --cuda-device N), so the
    room that counts is the best single GPU, not the sum of both.
    """
    try:
        need = float(args.get("needed_free_gb") or 0)
        single = bool(args.get("single_gpu", True))
        vram = get_vram()
        gpus = [g["gpu"] for g in vram]
        free_now = {g["gpu"]: g["free_mib"] / 1024.0 for g in vram}
        total = sum(g["total_mib"] for g in vram) / 1024.0
        comfy = comfy_status()
        data = load_models()
        current = match_model(llm_health())
        cur_entry = serving_entry(current, data)

        def per_gpu_need(entry: dict) -> tuple[dict, dict]:
            n = model_gpu_need(entry, gpus, data)
            per = dict(n["per_gpu"])
            if not per and gpus:  # unknown split: assume an even spread
                per = {g: n["total_gib"] / len(gpus) for g in gpus}
            return per, n

        cur_need = per_gpu_need(cur_entry)[0] if cur_entry else {}
        base = {g: free_now[g] + cur_need.get(g, 0.0) for g in gpus}  # pool with no LLM loaded

        def room(free_after: dict) -> tuple[float, int | None]:
            if not free_after:
                return 0.0, None
            if single:
                best = max(free_after, key=lambda g: free_after[g])
                return free_after[best], best
            return sum(free_after.values()), None

        rows = []
        for m in data.get("models", []):
            per, n = per_gpu_need(m)
            if cur_entry is not None and m["name"] == current:
                free_after, loadable = dict(free_now), True
            else:
                margin = RELEASE_MARGIN_GIB.get(n["source"], 0.5)
                free_after = {g: base[g] - per.get(g, 0.0) for g in gpus}
                loadable = all(base[g] >= per.get(g, 0.0) + margin for g in gpus)
            r, best = room(free_after)
            rows.append({
                "name": m["name"], "role": m.get("role", ""),
                "need_gib": round(sum(per.values()), 1), "need_source": n["source"],
                "need_per_gpu_gib": {str(g): round(v, 1) for g, v in per.items()},
                "free_after_per_gpu_gib": {str(g): round(v, 1) for g, v in free_after.items()},
                "loadable": loadable, "task_room_gib": round(r, 1), "task_gpu": best,
                "fits_with_budget": loadable and r >= need,
                "headroom_gib": round(r - need, 1),
                "min_free_vram_gb": m.get("min_free_vram_gb"),
            })
        r_none, best_none = room(base)
        fitting = sorted([x for x in rows if x["fits_with_budget"]], key=lambda x: x["need_gib"])
        cur_row = next((x for x in rows if x["name"] == current), None)
        where = (lambda x: f" (run the task on GPU {x['task_gpu']})" if single and x.get("task_gpu") is not None else "")
        if cur_row and cur_row["fits_with_budget"]:
            rec, action = f"stay on '{current}'{where(cur_row)}: the budget is already free", "stay"
        elif fitting:
            pick = fitting[-1]
            rec, action = (f"swap to '{pick['name']}' (largest model that still leaves the budget free)"
                           f"{where(pick)}", "swap")
        elif r_none >= need:
            rec, action = (f"no model fits next to the task: unload the LLM during it "
                           f"(gpu_job.sh none){' — task on GPU %s' % best_none if single else ''}", "unload")
        else:
            rec, action = (f"the task does not fit even with the LLM unloaded "
                           f"({r_none:.1f} GiB {'on the best GPU' if single else 'in total'} < {need:.1f})",
                           "impossible")
        return jdump({
            "ok": True, "task": args.get("task"), "needed_free_gb": need, "single_gpu": single,
            "total_vram_gb": round(total, 1), "free_gb": round(sum(free_now.values()), 1),
            "free_per_gpu_gib": {str(g): round(v, 1) for g, v in free_now.items()},
            "available_after_unload_gb": round(sum(base.values()), 1),
            "unloaded_task_room_gib": round(r_none, 1),
            "comfyui_up": comfy.get("up"), "comfyui_vram": comfy.get("vram", []),
            "current_model": current, "models": rows,
            "recommendation": rec, "action": action,
            "target": (fitting[-1]["name"] if action == "swap" else
                       current if action == "stay" else None),
            # what `gpu_job.sh auto` will do on this machine
            "gpu_job_auto": job_layout(),
        })
    except Exception as e:  # noqa: BLE001
        return jdump({"ok": False, "error": f"{type(e).__name__}: {e}"})

def _launch(action: str, entry: dict, dry: bool, extra: dict, force: bool = False) -> str:
    """Shared tail of swap/restore: preflight, busy check, lock, spawn.

    A swap/restore to the model that is already serving is a no-op unless
    `force` (a reload costs minutes without any LLM for every session).
    """
    name = entry["name"]
    pin = entry.get("pin_gpu")
    if not dry and not force and not busy_state():
        current = match_model(llm_health(timeout=2.0))
        serving = serving_entry(current) or {}
        same_place = serving.get("pin_gpu") == pin or (
            pin is not None and layout_gpus(serving, [g["gpu"] for g in get_vram()]) == {int(pin)})
        if current == name and same_place:
            return jdump({"ok": True, "skipped": f"'{name}' is already loaded; nothing to do",
                          "current": name, "server": describe_server(detect_server())})
    pf = preflight(entry)
    if dry:
        busy = busy_state()
        return jdump({"ok": pf["ok"] and not busy, "dry_run": True, "model": name,
                      "preflight": pf, "busy": busy,
                      "server": describe_server(detect_server()),
                      "engine": entry.get("engine"), **extra})
    if not pf["ok"]:
        return jdump({"ok": False, "error": "preflight failed", "model": name, "preflight": pf})
    with swap_lock() as got:  # C3+M4: serialize check+spawn; the runner holds it
        if not got:
            return jdump({"ok": False, "busy": True,
                          "error": "another swap/restore holds the lock; retry in a few seconds"})
        busy = busy_state()
        if busy:
            return jdump({"ok": False, "busy": True, "error": f"swap already in progress (state: {busy})"})
        # Nothing is serving: there is no reply to protect, so skip the grace.
        grace = None if llm_health(timeout=2.0).get("up") else 0
        return jdump(spawn_runner(action, name, grace, pin))


def swap_budget_s(entry: dict | None = None) -> int:
    """Worst-case seconds one swap/restore to `entry` (default: the primary)
    can take: grace + drain cap + stop (SIGINT wait, SIGKILL, port release) +
    VRAM settle + release gate + the model's load timeout + a margin. Used as
    the default for every --wait / --timeout so a wait never gives up on a
    swap that is still legitimately running."""
    cfg = get_cfg()
    entry = entry or find_model(str(cfg.get("primary_model") or "")) or {}

    def num(v, default):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)
    return int(num(cfg.get("grace_s"), 10) + num(cfg.get("drain_max_s"), 600) + 45
               + num(cfg.get("vram_settle_timeout_s"), 90) + num(cfg.get("release_wait_s"), 120)
               + num(entry.get("load_timeout_s"), 120) + 60)


def wait_idle(timeout_s: float | None = None, target: str | None = None, poll_s: float = 3.0,
              since: str | None = None) -> dict:
    """Block until no swap/restore/start is in flight, then report the outcome.

    For scripts (`hermes sms swap X --wait`, gpu_job.sh): the tool handlers
    return as soon as the runner is spawned, this waits for it. ok = the last
    result succeeded and the engine serves a registered model (and that model
    is `target`, when given). timeout_s None/0 = swap_budget_s(target).
    since: ISO time of the launch; an older last_result is not this run's.
    """
    if not timeout_s:
        timeout_s = swap_budget_s(find_model(target) if target else None)
    t0 = time.monotonic()
    while busy_state():
        if time.monotonic() - t0 > timeout_s:
            st = read_state()
            return {"ok": False, "error": f"timeout after {int(timeout_s)}s (the swap goes on in "
                                          "the background; `hermes sms wait` waits for it again)",
                    "state": st.get("state"), "target": st.get("target")}
        time.sleep(poll_s)
    st = read_state()
    lr = st.get("last_result") or {}
    stale = bool(since) and str(lr.get("at") or "") < str(since)
    current = match_model(llm_health(timeout=3.0))
    ok = (bool(lr.get("ok")) and not stale and current is not None
          and (target is None or same_name(current, target)))
    out = {"ok": ok, "current": current, "waited_s": int(time.monotonic() - t0),
           "last_result": lr}
    if stale:
        out["error"] = "the runner ended without reporting a result (killed?); see runner.log"
        return out
    if not ok:
        out["error"] = (lr.get("error") or
                        (f"loaded '{current}', expected '{target}'" if current else "no model is serving"))
    return out


GPU_JOB_MODES = ("auto", "none", "nano", "multi", "keep")


def gpu_job_mode() -> str:
    v = str(get_cfg().get("gpu_job_mode") or "auto").strip().lower()
    return v if v in GPU_JOB_MODES else "auto"


def job_layout(task_gpu=None, model: str | None = None, mode: str | None = None,
               vram: list[dict] | None = None) -> dict:
    """Where the LLM and a GPU job (ComfyUI …) go — what `gpu_job.sh auto` does.

    `mode` (default: the gpu_job_mode setting, picked in the UI):
      auto   best for this machine: `multi` with 2+ GPUs, else `none`.
      none   unload the LLM during the job (all the VRAM goes to the job).
      nano   keep the small model, in its normal layout, next to the job.
      multi  small model whole on ONE GPU, another GPU left to the job.
      keep   leave the loaded model alone (light jobs that fit beside it).
    A mode that cannot be honoured degrades to `none` and says why.

    Returns `mode`: what the script must do — "small" (swap to `model`, pinned
    to `llm_gpu` when set), "none" (unload) or "keep" (touch nothing) — and
    `applied`: the named mode that ends up in effect.
    """
    requested = str(mode or gpu_job_mode()).strip().lower()
    if requested not in GPU_JOB_MODES:
        return {"ok": False, "error": f"unknown mode '{requested}'", "modes": list(GPU_JOB_MODES)}
    vram = get_vram() if vram is None else vram
    gpus = [g["gpu"] for g in vram]
    data = load_models()
    current = match_model(llm_health(timeout=2.0))
    cur = serving_entry(current, data)
    credit = (model_gpu_need(cur, gpus, data)["per_gpu"] or {}) if cur else {}
    base = {g["gpu"]: g["free_mib"] / 1024.0 + credit.get(g["gpu"], 0.0) for g in vram}
    if task_gpu is not None and str(task_gpu) != "":
        task_gpu = int(task_gpu)
        if task_gpu not in gpus:
            return {"ok": False, "error": f"no GPU {task_gpu} on this machine"}
    else:
        task_gpu = max(gpus, key=lambda g: (base[g], -g)) if gpus else None
    out = {"ok": True, "requested": requested, "applied": "none", "mode": "none", "model": None,
           "llm_gpu": None, "task_gpu": task_gpu, "gpus": len(gpus), "current": current,
           "free_without_llm_gib": {str(g): round(v, 1) for g, v in base.items()}}

    def done(applied: str, script_mode: str, reason: str, **kw) -> dict:
        out.update(applied=applied, mode=script_mode, reason=reason, **kw)
        out["degraded"] = requested not in ("auto", applied)
        return out

    if requested == "none":
        return done("none", "none", "the LLM is unloaded during the job")
    if requested == "keep":
        return done("keep", "keep", f"'{current}' stays loaded; the job runs beside it" if current
                    else "nothing is loaded; the job runs as is", model=current)

    models = sorted(data.get("models", []), key=lambda m: float(m.get("min_free_vram_gb") or 0))
    small = find_model(str(model), data) if model else (models[0] if models else None)
    if not small:
        return done("none", "none", "no model in the registry: the LLM is unloaded during the job")
    name = small["name"]

    if requested == "nano":
        rc = release_check(small, unloading=cur, vram=vram)
        if current == name or rc["ok"]:
            return done("nano", "small", f"'{name}' stays loaded in its normal layout during the job",
                        model=name)
        return done("none", "none", f"'{name}' does not fit now ({rc['summary'].split(';')[0]}): "
                                    "the LLM is unloaded during the job")

    # multi / auto
    if len(gpus) < 2:
        return done("none", "none", "single GPU: the LLM is unloaded during the job")
    for g in sorted((g for g in gpus if g != task_gpu), key=lambda g: (-base[g], g)):
        need = model_gpu_need(dict(small, pin_gpu=g), gpus, data)
        margin = RELEASE_MARGIN_GIB.get(need["source"], 0.5)
        if base[g] >= need["total_gib"] + margin:
            return done("multi", "small", f"'{name}' whole on GPU {g}; GPU {task_gpu} is left to the job",
                        model=name, llm_gpu=g, llm_need_gib=round(need["total_gib"], 1))
    return done("none", "none", f"'{name}' does not fit on one GPU next to the job: "
                                "the LLM is unloaded during it")


def after_job_policy() -> str:
    v = str(get_cfg().get("after_gpu_job") or "small").strip().lower()
    return v if v in ("small", "primary") else "small"


def after_job(prefer: str | None = None, timeout_s: float | None = None) -> dict:
    """What gpu_job.sh runs when the GPU job ends: make sure the agent has a
    brain to deliver the result with.

    after_gpu_job = "small" (default): answer with the small model. A
    registered model that is already serving stays (no reload at all);
    when the LLM was unloaded for the job, the smallest model that fits is
    loaded (`prefer` first). The primary comes back later: on demand
    (sms_restore, when deep work arrives) or through the idle watcher.

    after_gpu_job = "primary": reload the primary right away (the old
    behaviour); if its VRAM is not released, fall back to the small path
    instead of leaving the agent without a model.
    """
    cfg = get_cfg()
    primary = str(cfg.get("primary_model") or "")
    policy = after_job_policy()
    out: dict = {"policy": policy}
    if busy_state():
        # a swap is still running (e.g. the job's own FREE step outlived its
        # wait): never measure VRAM or launch in the middle of it
        out["waited_for_swap"] = wait_idle(timeout_s)
    current = match_model(llm_health(timeout=3.0))

    def launch(name: str) -> dict:
        def go() -> dict:
            return json.loads(h_restore({}) if same_name(name, primary) else h_swap({"model": name}))
        res = go()
        if res.get("busy"):  # another runner holds the lock: wait for it, then retry once
            wait_idle(timeout_s)
            if same_name(match_model(llm_health(timeout=3.0)), name):
                return {"ok": True, "current": name}
            res = go()
        if res.get("ok") and not res.get("skipped"):
            res = wait_idle(timeout_s, name, since=time.strftime("%Y-%m-%dT%H:%M:%S"))
        return res

    if policy == "primary" and not same_name(current, primary):
        rc = check_release(primary, float(cfg.get("release_wait_s") or 0))
        if rc.get("ok"):
            res = launch(primary)
            if res.get("ok"):
                return {**out, "ok": True, "current": primary, "loaded": primary}
            out["primary_error"] = res.get("error")
        else:
            out["primary_blocked"] = rc.get("summary")
        current = match_model(llm_health(timeout=3.0))

    if current:
        if not same_name(current, primary):
            ensure_idle_watch()
        return {**out, "ok": True, "current": current, "loaded": None,
                "note": f"'{current}' is already serving; nothing reloaded"}

    models = sorted(load_models().get("models", []),
                    key=lambda m: float(m.get("min_free_vram_gb") or 0))
    names = [m["name"] for m in models]
    pref = find_model(str(prefer)) if prefer else None
    if pref:
        names = [pref["name"]] + [n for n in names if n != pref["name"]]
    errors = {}
    for name in names:
        rc = release_check(find_model(name))
        if not rc["ok"]:
            errors[name] = rc["summary"]
            continue
        res = launch(name)
        if res.get("ok"):
            return {**out, "ok": True, "current": name, "loaded": name}
        errors[name] = res.get("error")
        break  # a load was attempted and failed: do not chain multi-minute retries
    return {**out, "ok": False, "current": match_model(llm_health(timeout=3.0)),
            "error": "no model could be loaded after the GPU job", "tried": errors}


def _cmd_extra(entry: dict) -> dict:
    try:
        return {"cmd": smp_cmdline.join(build_cmd(entry))}
    except smp_engines.EngineUnavailable as e:
        return {"cmd": None, "engine_error": str(e)}


def h_swap(args: dict, **kwargs) -> str:
    try:
        name = str(args.get("model") or "").strip()
        entry = find_model(name)
        if not entry:
            return jdump({"ok": False, "error": f"unknown model '{name}'",
                          "available": available_models()})
        gpu = args.get("gpu")
        if gpu is not None and str(gpu) != "":
            gpu = int(gpu)
            if gpu not in [g["gpu"] for g in get_vram()]:
                return jdump({"ok": False, "error": f"no GPU {gpu} on this machine"})
            entry = dict(entry, pin_gpu=gpu)
        return _launch("swap", entry, bool(args.get("dry_run", False)), _cmd_extra(entry),
                       force=bool(args.get("force", False)))
    except Exception as e:  # noqa: BLE001
        return jdump({"ok": False, "error": f"{type(e).__name__}: {e}"})


def h_restore(args: dict, **kwargs) -> str:
    try:
        cfg = get_cfg()
        entry = find_model(cfg["primary_model"])
        if not entry:
            return jdump({"ok": False,
                          "error": f"primary model '{cfg['primary_model']}' not in registry",
                          "available": available_models()})
        return _launch("restore", entry, bool(args.get("dry_run", False)), _cmd_extra(entry),
                       force=bool(args.get("force", False)))
    except Exception as e:  # noqa: BLE001
        return jdump({"ok": False, "error": f"{type(e).__name__}: {e}"})


# ------------------------------------------------------------ slash + cli

def cmd_swap(raw: str) -> str:
    parts = (raw or "").split()
    if not parts or parts[0] in ("status", "st"):
        return h_status({})
    if parts[0] == "plan":
        try:
            need = float(parts[1]) if len(parts) > 1 else 0.0
        except ValueError:
            return smp_i18n.t("Uso: /swap plan [gb]", "Usage: /swap plan [gb]")
        return h_plan({"needed_free_gb": need})
    if parts[0] == "restore":
        return h_restore({"dry_run": "dry" in parts[1:]})
    if parts[0] == "swap":
        if len(parts) < 2:
            return smp_i18n.t("Uso: /swap <modelo> [dry] · /swap restore [dry] · /swap plan [gb] · /swap status",
                              "Usage: /swap <model> [dry] · /swap restore [dry] · /swap plan [gb] · /swap status")
        return h_swap({"model": parts[1], "dry_run": "dry" in parts[2:]})
    # Assume the first token is a model name
    return h_swap({"model": parts[0], "dry_run": "dry" in parts[1:]})


def setup_sms(parser) -> None:
    sub = parser.add_subparsers(dest="sms_cmd", required=True)
    sub.add_parser("status", help="Show swap state, engine, VRAM, current model, ComfyUI")
    p = sub.add_parser("plan", help="Recommend a model for a VRAM budget")
    p.add_argument("--needed-free-gb", type=float, default=0)
    p.add_argument("--task", default="")
    p.add_argument("--any-gpu", action="store_true",
                   help="The task can spread over all GPUs (default: it runs on one GPU, like ComfyUI)")
    p = sub.add_parser("swap", help="Hot-swap to a model (background, with grace)")
    p.add_argument("model")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="Reload even if the model is already serving")
    p.add_argument("--gpu", type=int, default=None, help="Load the model whole on this GPU (nvidia-smi index), leaving the others free")
    p.add_argument("--wait", action="store_true", help="Block until the swap finishes; exit 0 only if the target is serving")
    p.add_argument("--timeout", type=float, default=0,
                   help="Max seconds for --wait (default: the worst case of this swap, from the settings)")
    p = sub.add_parser("restore", help="Restore the primary model (background, with grace)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="Reload even if the model is already serving")
    p.add_argument("--wait", action="store_true", help="Block until the restore finishes; exit 0 only if the primary is serving")
    p.add_argument("--timeout", type=float, default=0,
                   help="Max seconds for --wait (default: the worst case of this swap, from the settings)")
    p = sub.add_parser("check", help="Release gate: may this model load now? (other GPU processes released, VRAM per GPU)")
    p.add_argument("model", nargs="?", default=None, help="Model to check (default: the primary)")
    p.add_argument("--wait", type=float, default=0, help="Poll up to N seconds until it passes")
    p = sub.add_parser("wait", help="Block until no swap/restore/start is in flight, then print the result")
    p.add_argument("--timeout", type=float, default=0, help="Max seconds (default: worst case of a swap)")
    p.add_argument("--target", default=None, help="Exit 0 only if this model is the one serving")
    p = sub.add_parser("job-layout", help="Where the LLM and a GPU job go: small model on one GPU + job on another (2+ GPUs), or no LLM (1 GPU); used by gpu_job.sh auto")
    p.add_argument("--task-gpu", type=int, default=None, help="GPU the job will use (default: the one with most room)")
    p.add_argument("--model", default=None, help="Model to keep during the job (default: the smallest)")
    p.add_argument("--mode", default=None, choices=list(GPU_JOB_MODES), help="Mode to apply (default: the gpu_job_mode setting chosen in the UI)")
    p.add_argument("--plain", action="store_true", help="One line for scripts: <small|none|keep> <model|-> <llm_gpu|-> <task_gpu|-> <applied mode> <budget_s> <cache_dir> <comfy_url> <nvidia_smi|->")
    p = sub.add_parser("after-job", help="End of a GPU job: keep/load the model that will answer (setting after_gpu_job); used by gpu_job.sh")
    p.add_argument("--prefer", default=None, help="Model to load first if nothing is serving")
    p.add_argument("--timeout", type=float, default=0, help="Max seconds per wait (default: worst case of a swap)")
    sub.add_parser("start", help="Load the primary model now if nothing is serving (what the gateway does at start-up)")
    p = sub.add_parser("stop", help="Stop the model served by the active engine")
    p.add_argument("--graceful", action="store_true",
                   help="Grace countdown + drain in-flight requests first (like a swap); used by gpu_job.sh none")
    p = sub.add_parser("cmd", help="Print the exact launch command of a model")
    p.add_argument("model")
    p = sub.add_parser("provider", help="SmartModelProvider: expose the registry models as a Hermes provider")
    p.add_argument("action", choices=["status", "install", "remove"])


def cli_sms(args) -> int:
    a = vars(args)
    cmd = a.get("sms_cmd")
    if cmd == "status":
        out = h_status({})
    elif cmd == "plan":
        out = h_plan({"needed_free_gb": a.get("needed_free_gb", 0),
                      "task": a.get("task", ""), "single_gpu": not a.get("any_gpu")})
    elif cmd in ("swap", "restore"):
        if cmd == "swap":
            out = h_swap({"model": a.get("model"), "dry_run": a.get("dry_run", False),
                          "force": a.get("force", False), "gpu": a.get("gpu")})
            target = a.get("model")
        else:
            out = h_restore({"dry_run": a.get("dry_run", False), "force": a.get("force", False)})
            target = get_cfg().get("primary_model")
        if a.get("wait") and not a.get("dry_run"):
            timeout = float(a.get("timeout") or 0) or None
            launched = json.loads(out)
            if launched.get("busy"):
                # another swap is running: wait for it, then ask again once
                print(out, flush=True)
                wait_idle(timeout)
                out = (h_swap({"model": a.get("model"), "force": a.get("force", False), "gpu": a.get("gpu")})
                       if cmd == "swap" else h_restore({"force": a.get("force", False)}))
                launched = json.loads(out)
            if launched.get("ok") and not launched.get("skipped"):
                since = time.strftime("%Y-%m-%dT%H:%M:%S")
                print(out, flush=True)  # the launch receipt, then the outcome
                out = jdump({"action": cmd, **wait_idle(timeout, target, since=since)})
    elif cmd == "check":
        out = jdump(check_release(a.get("model") or get_cfg().get("primary_model"),
                                  float(a.get("wait") or 0)))
    elif cmd == "wait":
        out = jdump(wait_idle(float(a.get("timeout") or 0) or None, a.get("target") or None))
    elif cmd == "job-layout":
        lay = job_layout(a.get("task_gpu"), a.get("model"), a.get("mode"))
        if a.get("plain") and lay.get("ok"):
            dash = lambda v: "-" if v is None else str(v)  # noqa: E731
            budget = swap_budget_s(find_model(lay["model"]) if lay.get("model") else None)
            print(lay["mode"], dash(lay["model"]), dash(lay["llm_gpu"]), dash(lay["task_gpu"]),
                  lay["applied"], budget, cache_dir(), get_cfg()["comfy_url"], dash(nvidia_smi_path()))
            return 0
        out = jdump(lay)
    elif cmd == "after-job":
        out = jdump(after_job(a.get("prefer") or None, float(a.get("timeout") or 0) or None))
    elif cmd == "start":
        out = jdump(autostart_primary())
    elif cmd == "stop":
        with swap_lock() as got:
            if not got or busy_state():
                out = jdump({"ok": False, "error": "a swap is in progress"})
            else:
                logs: list[str] = []
                res = (graceful_stop(lambda m: (logs.append(m), print(m, file=sys.stderr, flush=True)))
                       if a.get("graceful") else stop_server())
                if res.get("ok"):  # a deliberate stop: the watcher must not "recover" it
                    write_state({"state": "idle", "current": None, "spawned_at": None,
                                 "last_stop": {"at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                               "graceful": bool(a.get("graceful"))}})
                out = jdump({**res, **({"log": logs} if logs else {})})
    elif cmd == "cmd":
        entry = find_model(str(a.get("model") or ""))
        out = jdump({"ok": bool(entry), **(_cmd_extra(entry) if entry else
                                           {"error": "unknown model", "available": available_models()})})
    elif cmd == "provider":
        try:
            from . import provider_sync
        except ImportError:
            import provider_sync  # type: ignore[no-redef]
        fn = {"install": provider_sync.install_provider, "remove": provider_sync.remove_provider,
              "status": lambda: {"ok": True, **provider_sync.provider_status()}}[a.get("action")]
        out = jdump(fn())
    else:
        out = '{"ok": false, "error": "unknown subcommand"}'
    print(out)
    try:
        return 0 if json.loads(out).get("ok") else 1
    except Exception:
        return 1
