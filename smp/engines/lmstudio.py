"""LM Studio engine (beta): a resident server that loads and unloads models.

SMP does not spawn or signal LM Studio. It drives the `lms` CLI:
  load    lms load <model_key> --identifier <name> -c <ctx> [args] -y
  unload  lms unload <name>          (or --all when nothing is known)
  loaded  GET /api/v0/models         -> entries with state == "loaded"
and checks a launch line with `lms load ... --estimate-only` (nothing loads).

Registry entry: `model_key` is the LM Studio model key (`lms ls`); `args` are
`lms load` options (`-c`, `--gpu max|off|0..1`, `--parallel N`, `--ttl S`,
speculative-decoding flags). The model is served under its registry name
(`--identifier`), so /v1/models, Hermes and the swap identity check all see
the same id. Works with LM Studio on Windows from WSL (lms.exe via interop).

Limits (beta): LM Studio exposes no in-flight / token counters, so the drain
before a swap relies on the grace countdown plus a quiet window since the
agent's last LLM call, and the idle restore counts the agent's turns.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from typing import Optional

from ..i18n import t
from .base import Engine, http_json


def _run(cmd: list[str], timeout: float, env: dict):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env,
                              stdin=subprocess.DEVNULL)
    except Exception as e:  # noqa: BLE001
        return e


class LmStudioEngine(Engine):
    id = "lmstudio"
    label = "LM Studio"
    status = "beta"
    managed = True
    lifecycle = "daemon"
    file_based = False
    default_port = 1234
    exe_names = ("lms", "lms.exe")
    note = ("Beta: SMP carga y descarga modelos con `lms` (no lanza procesos). Sin contador de "
            "peticiones: el drenaje usa la cuenta atrás y la pausa desde la última llamada.",
            "Beta: SMP loads and unloads models with `lms` (it spawns no process). No request "
            "counter: the drain uses the grace countdown and the quiet time since the last call.")

    # ------------------------------------------------------------ config
    def binary(self, ecfg: dict, entry: Optional[dict] = None) -> str:
        b = super().binary(ecfg, entry)
        if b:
            return b
        for name in self.exe_names:
            p = shutil.which(name)
            if p:
                return p
        for cand in [os.path.expanduser("~/.lmstudio/bin/lms")] + sorted(
                glob.glob("/mnt/c/Users/*/.lmstudio/bin/lms.exe")):
            if os.path.isfile(cand):
                return cand
        return ""

    def model_key(self, entry: dict) -> str:
        return str(entry.get("model_key") or entry.get("gguf") or "").strip()

    def build_cmd(self, ecfg: dict, entry: dict, args: list[str]) -> list[str]:
        """The `lms load` line (what `hermes sms cmd` prints and load() runs)."""
        return ([self.binary(ecfg, entry), "load", self.model_key(entry),
                 "--identifier", str(entry.get("name") or "")]
                + [str(a) for a in args] + ["-y"])

    # ------------------------------------------------------------ lifecycle
    def ensure_server(self, ecfg: dict) -> dict:
        if super().probe(self.endpoint(ecfg), 2.0).get("up"):
            return {"ok": True}
        port = str(self.normalize(ecfg)["port"])
        r = _run([self.binary(ecfg), "server", "start", "--port", port], 60, self.env(ecfg))
        if isinstance(r, Exception) or r.returncode != 0:
            return {"ok": False, "error": t("no se pudo arrancar el servidor de LM Studio: ",
                                            "could not start the LM Studio server: ")
                    + (str(r) if isinstance(r, Exception) else (r.stderr or r.stdout).strip()[-300:])}
        return {"ok": True, "started": True}

    def _exact_key_error(self, ecfg: dict, entry: dict) -> Optional[str]:
        """`lms load` matches keys loosely and would load ANOTHER model for an
        unknown key: only an exact, known key is accepted."""
        key = self.model_key(entry)
        if not key:
            return t("falta model_key (la clave de LM Studio, ver `lms ls`)",
                     "model_key is missing (the LM Studio key, see `lms ls`)")
        known = self.known_model(ecfg, entry)
        if known is None:
            return t("LM Studio no responde: no se puede comprobar la clave",
                     "LM Studio does not answer: the key cannot be checked")
        if not known:
            return t(f"LM Studio no tiene un modelo con la clave exacta '{key}' (ver `lms ls`)",
                     f"LM Studio has no model with the exact key '{key}' (see `lms ls`)")
        return None

    def load(self, ecfg: dict, entry: dict, args: list[str], timeout: float) -> dict:
        srv = self.ensure_server(ecfg)
        if not srv["ok"]:
            return srv
        bad = self._exact_key_error(ecfg, entry)
        if bad:
            return {"ok": False, "error": bad}
        cmd = self.build_cmd(ecfg, entry, args)
        r = _run(cmd, max(30.0, float(timeout)), self.env(ecfg, entry))
        if isinstance(r, Exception):
            return {"ok": False, "error": f"{type(r).__name__}: {r}", "cmd": cmd}
        out = f"{r.stdout}\n{r.stderr}".strip()
        if r.returncode != 0:
            return {"ok": False, "error": (out.splitlines() or ["lms load failed"])[-1][:300],
                    "cmd": cmd, "log_tail": out.splitlines()[-10:]}
        if str(entry.get("name")) not in self.loaded(ecfg):
            return {"ok": False, "error": t("`lms load` terminó pero el modelo no aparece cargado",
                                            "`lms load` returned but the model is not loaded"),
                    "cmd": cmd, "log_tail": out.splitlines()[-10:]}
        return {"ok": True, "cmd": cmd}

    def unload(self, ecfg: dict, entry: Optional[dict] = None) -> dict:
        ids = self.loaded(ecfg)
        if not ids:
            return {"ok": True, "actions": ["LM Studio: nothing loaded"]}
        targets = [str(entry["name"])] if entry and str(entry.get("name")) in ids else ids
        actions, ok = [], True
        for ident in targets:
            r = _run([self.binary(ecfg), "unload", ident], 120, self.env(ecfg))
            good = not isinstance(r, Exception) and r.returncode == 0
            ok = ok and good
            actions.append(f"LM Studio: unload '{ident}'" + ("" if good else " FAILED"))
        return {"ok": ok and not (set(targets) & set(self.loaded(ecfg))), "actions": actions}

    def _api_models(self, ecfg_or_url) -> Optional[list[dict]]:
        url = ecfg_or_url if isinstance(ecfg_or_url, str) else self.endpoint(ecfg_or_url)
        try:
            return list(http_json(url.rstrip("/") + "/api/v0/models", 3.0).get("data") or [])
        except Exception:  # noqa: BLE001
            return None

    def loaded(self, ecfg: dict) -> list[str]:
        return [str(m.get("id")) for m in (self._api_models(ecfg) or []) if m.get("state") == "loaded"]

    def known_model(self, ecfg: dict, entry: dict) -> Optional[bool]:
        models = self._api_models(ecfg)
        if models is None:
            return None
        key = self.model_key(entry)
        return any(str(m.get("id")) == key for m in models)

    # ------------------------------------------------------------ checks
    def validate(self, ecfg: dict, entry: dict, args: list[str]) -> dict:
        """`lms load ... --estimate-only`: parses the options and estimates the
        memory without loading anything."""
        cmd = self.build_cmd(ecfg, entry, args)
        if not self.binary_ok(ecfg, entry):
            return {"ok": False, "error": t(f"`lms` no encontrado: {cmd[0] or '(vacío)'}",
                                            f"`lms` not found: {cmd[0] or '(empty)'}"), "cmd": cmd}
        bad = self._exact_key_error(ecfg, entry)
        if bad:
            return {"ok": False, "error": bad, "cmd": cmd}
        r = _run([a for a in cmd if a != "-y"] + ["--estimate-only", "-y"], 60, self.env(ecfg, entry))
        if isinstance(r, Exception):
            return {"ok": False, "error": f"{type(r).__name__}: {r}", "cmd": cmd}
        out = f"{r.stdout}\n{r.stderr}".strip()
        if r.returncode != 0 or "error" in out.lower():
            return {"ok": False, "error": (out.splitlines() or ["lms load --estimate-only failed"])[-1][:300],
                    "cmd": cmd}
        est = next((ln.split(":", 1)[1].strip() for ln in out.splitlines()
                    if ln.lower().startswith("estimated gpu memory")), "")
        picked = next((ln.split(":", 1)[1].strip() for ln in out.splitlines()
                       if ln.startswith("Model:")), "")
        if picked and picked != self.model_key(entry):
            return {"ok": False, "cmd": cmd, "error": t(
                f"LM Studio resolvió la clave a otro modelo: {picked}",
                f"LM Studio resolved the key to another model: {picked}")}
        return {"ok": True, "cmd": cmd, "estimate": est}

    def version(self, ecfg: dict, entry: Optional[dict] = None) -> dict:
        if not self.binary_ok(ecfg, entry):
            return {"ok": False, "error": t("`lms` no encontrado", "`lms` not found")}
        r = _run([self.binary(ecfg, entry), "server", "status", "--json"], 20, self.env(ecfg, entry))
        if isinstance(r, Exception) or r.returncode != 0:
            return {"ok": True, "version": "lms", "bin": self.binary(ecfg, entry)}
        try:
            st = json.loads(r.stdout)
            return {"ok": True, "version": f"lms (server {'on :' + str(st.get('port')) if st.get('running') else 'off'})",
                    "bin": self.binary(ecfg, entry)}
        except ValueError:
            return {"ok": True, "version": "lms", "bin": self.binary(ecfg, entry)}

    # ------------------------------------------------------------ probes
    def probe(self, url: str, timeout: float = 1.5) -> dict:
        """Up = the server answers; models = only the LOADED ones (/v1/models
        lists every downloaded model, which would fool the identity check)."""
        models = self._api_models(url)
        if models is None:
            return {"up": False, "url": url, "models": []}
        ids = [str(m.get("id")) for m in models if m.get("state") == "loaded"]
        return {"up": bool(ids), "server_up": True, "url": url, "models": ids[:50]}

    def ready(self, url: str) -> bool:
        return bool(self.probe(url, 3.0).get("up"))

    def is_engine_process(self, pid: int, ecfg: dict) -> bool:
        return False  # a Windows / desktop process: never signalled by SMP
