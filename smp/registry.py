"""Model registry schema (models.json v2) and the v1 -> v2 migration.

v1 (<= 0.2.x): top-level host/port/common_args, llama_bin + base_url in the
plugin settings, per-model ``start_mode`` (systemd | nohup), other servers in
cache/ui.json. v2: everything engine-related under ``engines.<id>`` and each
model names its ``engine``. Pure functions: callers do the I/O.
"""
from __future__ import annotations

import copy
import urllib.parse

from . import engines as _engines
from .cmdline import strip_managed

VERSION = 2
_UI_KEYS = {"llama.cpp": "llamacpp", "llamacpp": "llamacpp", "ollama": "ollama",
            "vllm": "vllm", "lmstudio": "lmstudio"}


def _flag(args: list, names: tuple) -> str | None:
    val = None
    for i, a in enumerate(args[:-1]):
        if a in names:
            val = str(args[i + 1])
    return val


def migrate(data: dict, legacy: dict | None = None) -> tuple[dict, list[str]]:
    """Return (v2 registry, notes). ``notes`` is empty when nothing changed.

    legacy: {"llama_bin", "base_url", "ui": ui.json dict, "env": {K: V}} -
    values the v1 layout kept outside models.json.
    """
    legacy = legacy or {}
    out = copy.deepcopy(data if isinstance(data, dict) else {})
    notes: list[str] = []
    engines = out.get("engines") if isinstance(out.get("engines"), dict) else {}

    if int(out.get("version") or 1) < VERSION:
        common = [str(a) for a in (out.pop("common_args", None) or [])]
        host = _flag(common, ("--host", "-host")) or out.get("host")
        port = _flag(common, ("--port", "-port")) or out.get("port")
        if legacy.get("base_url") and not (host and port):
            u = urllib.parse.urlparse(str(legacy["base_url"]))
            host, port = host or u.hostname, port or u.port
        common, _ = strip_managed(common)
        out.pop("host", None)
        out.pop("port", None)
        llama = dict(engines.get("llamacpp") or {})
        llama.setdefault("bin", str(legacy.get("llama_bin") or ""))
        llama.setdefault("host", str(host or "127.0.0.1"))
        llama.setdefault("port", int(port or 8080))
        llama.setdefault("common_args", common)
        llama.setdefault("env", dict(legacy.get("env") or {}))
        engines["llamacpp"] = llama
        notes.append("common_args/host/port moved to engines.llamacpp")

        ui = legacy.get("ui") or {}
        for key, srv in (ui.get("servers") or {}).items():
            eid = _UI_KEYS.get(key)
            if eid and eid != "llamacpp" and isinstance(srv, dict):
                e = dict(engines.get(eid) or {})
                e.setdefault("host", str(srv.get("host") or "127.0.0.1"))
                if srv.get("port"):
                    e.setdefault("port", int(srv["port"]))
                engines[eid] = e
        active = _UI_KEYS.get(str(ui.get("active_server") or ""), "llamacpp")
        eng = _engines.get(active)
        out["active_engine"] = active if (eng and eng.managed) else "llamacpp"

        for m in out.get("models") or []:
            if not isinstance(m, dict):
                continue
            if m.pop("start_mode", None) is not None:
                notes.append(f"{m.get('name')}: start_mode removed (SMP launches the engine)")
            m.setdefault("engine", "llamacpp")
            args, dropped = strip_managed([str(a) for a in (m.get("args") or [])])
            if dropped:
                notes.append(f"{m.get('name')}: {', '.join(dropped)} removed from args (set by the engine)")
            m["args"] = args
        out["version"] = VERSION

    # v2 invariants (also repairs hand-edited files)
    for eid, eng in _engines.ENGINES.items():
        cur = engines.get(eid)
        if not isinstance(cur, dict):
            engines[eid] = eng.defaults()
            if data.get("version") == VERSION:
                notes.append(f"engines.{eid} added")
        else:
            for k, v in eng.defaults().items():
                if k not in cur or (k == "bin" and not cur.get("bin") and eid == "llamacpp"):
                    cur[k] = v
    out["engines"] = engines
    if out.get("active_engine") not in _engines.ENGINES:
        out["active_engine"] = _engines.DEFAULT_ENGINE
    for m in out.get("models") or []:
        if isinstance(m, dict) and m.get("engine") not in _engines.ENGINES:
            m["engine"] = _engines.DEFAULT_ENGINE
    return out, notes
