"""Engine contract for SmartModelProvider.

An engine knows how to turn a registry entry into a launch argv, how to tell
whether its server is up / busy / serving which model, and how to recognise
its own processes. Spawning, stopping (signals to the process group), VRAM
settle and the swap state machine are shared and live in ``tools.py``; an
engine only contributes argv, env and probes.

Engine config (``models.json`` -> ``engines.<id>``):
    {"bin": str, "host": str, "port": int, "env": {K: V}, "common_args": [str]}
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from ..i18n import t


class EngineUnavailable(RuntimeError):
    """The engine cannot launch models (not configured, or under construction)."""


def http_json(url: str, timeout: float = 2.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def proc_argv(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except (OSError, ValueError):
        return []
    return [a.decode("utf-8", "replace") for a in raw.split(b"\0") if a]


class Engine:
    id = ""
    label = ""
    status = "stable"        # "stable" | "wip" (shown as "under construction")
    managed = True           # SMP can start/stop/swap models on it
    default_port = 8080
    exe_names: tuple[str, ...] = ()
    note = ""

    # ------------------------------------------------------------ config
    def defaults(self) -> dict:
        return {"bin": "", "host": "127.0.0.1", "port": self.default_port, "env": {}, "common_args": []}

    def normalize(self, ecfg: Optional[dict]) -> dict:
        out = self.defaults()
        for k, v in (ecfg or {}).items():
            if v is not None:
                out[k] = v
        try:
            out["port"] = int(out.get("port") or self.default_port)
        except (TypeError, ValueError):
            out["port"] = self.default_port
        if not isinstance(out.get("env"), dict):
            out["env"] = {}
        if not isinstance(out.get("common_args"), list):
            out["common_args"] = []
        return out

    def endpoint(self, ecfg: dict) -> str:
        e = self.normalize(ecfg)
        return f"http://{e.get('host') or '127.0.0.1'}:{int(e['port'])}"

    def binary(self, ecfg: dict, entry: Optional[dict] = None) -> str:
        """Per-model override first, else the engine's. A directory resolves to
        the engine executable inside it (Launcher's per-model llama.cpp dir)."""
        raw = str((entry or {}).get("bin") or "").strip() or str(self.normalize(ecfg).get("bin") or "").strip()
        p = os.path.expanduser(raw)
        if p and os.path.isdir(p):
            for name in self.exe_names:
                cand = os.path.join(p, name)
                if os.path.isfile(cand):
                    return cand
        return p

    def binary_ok(self, ecfg: dict, entry: Optional[dict] = None) -> bool:
        b = self.binary(ecfg, entry)
        return bool(b) and os.path.isfile(b) and os.access(b, os.X_OK)

    def env(self, ecfg: dict, entry: Optional[dict] = None) -> dict:
        env = dict(os.environ)
        env.setdefault("LD_LIBRARY_PATH", "/usr/lib/wsl/lib")
        for src in (self.normalize(ecfg).get("env") or {}, (entry or {}).get("env") or {}):
            if isinstance(src, dict):
                env.update({str(k): str(v) for k, v in src.items()})
        return env

    def version(self, ecfg: dict, entry: Optional[dict] = None) -> dict:
        b = self.binary(ecfg, entry)
        if not self.binary_ok(ecfg, entry):
            return {"ok": False, "error": t(f"binario no encontrado o no ejecutable: {b or '(vacío)'}",
                                            f"binary not found or not executable: {b or '(empty)'}")}
        try:
            r = subprocess.run([b, "--version"], capture_output=True, text=True, timeout=15,
                               env=self.env(ecfg, entry))
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        text = f"{r.stdout}\n{r.stderr}".strip()
        line = next((ln.strip() for ln in text.splitlines() if "version" in ln.lower()),
                    text.splitlines()[0] if text else "")
        return {"ok": bool(line), "version": line, "bin": b}

    # ------------------------------------------------------------ launch
    def build_cmd(self, ecfg: dict, entry: dict, args: list[str]) -> list[str]:
        raise EngineUnavailable(t(f"{self.label}: motor en construcción",
                                  f"{self.label}: engine under construction"))

    def validate(self, ecfg: dict, entry: dict, args: list[str]) -> dict:
        return {"ok": False, "error": t(f"{self.label}: motor en construcción",
                                        f"{self.label}: engine under construction")}

    def catalog(self, ecfg: dict, entry: Optional[dict], cache_dir: Path) -> dict:
        return {"groups": [], "version": "", "help_available": False, "count": 0}

    def help_info(self, ecfg: dict, entry: Optional[dict], cache_dir: Path):
        return None

    def gpu_split(self, ecfg: dict, entry: dict, args: list[str],
                  gpus: list[int]) -> Optional[dict]:
        """{physical GPU index: fraction of the model's VRAM}, or None if unknown."""
        return None

    # ------------------------------------------------------------ probes
    def probe(self, url: str, timeout: float = 1.5) -> dict:
        """OpenAI-compatible /v1/models (llama.cpp, vLLM, Ollama, LM Studio)."""
        u = url.rstrip("/") + "/v1/models"
        try:
            data = http_json(u, timeout)
            return {"up": True, "url": u, "models": [m.get("id", "") for m in data.get("data", [])][:50]}
        except Exception:  # noqa: BLE001
            return {"up": False, "url": u, "models": []}

    def ready(self, url: str) -> bool:
        return bool(self.probe(url, 3.0).get("up"))

    def in_flight(self, url: str) -> Optional[int]:
        return None

    def activity(self, url: str) -> Optional[float]:
        """A counter that grows while the engine works (tokens processed), or None."""
        return None

    def is_engine_process(self, pid: int, ecfg: dict) -> bool:
        names = set(self.exe_names)
        b = os.path.basename(self.binary(ecfg))
        if b:
            names.add(b)
        argv = proc_argv(pid)
        if any(os.path.basename(a) in names for a in argv[:2]):
            return True
        try:
            return os.path.basename(os.readlink(f"/proc/{int(pid)}/exe")) in names
        except (OSError, ValueError):
            return False

    def describe(self, ecfg: dict) -> dict:
        e = self.normalize(ecfg)
        return {"id": self.id, "label": self.label, "status": self.status, "managed": self.managed,
                "note": self.note_text(), "url": self.endpoint(e), **e}

    def note_text(self) -> str:
        """`note` may be a (es, en) pair."""
        n = self.note
        return t(*n) if isinstance(n, tuple) else n
