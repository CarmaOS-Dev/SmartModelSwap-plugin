"""llama.cpp (llama-server): the managed engine.

One model per process: SMP spawns ``llama-server -m <gguf> --host H --port P
<common_args> <model args>``, probes ``/health`` (503 while loading), drains via
``/metrics`` and checks identity via ``/v1/models``.
"""
from __future__ import annotations

import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from .. import llama_help
from ..cmdline import args_to_rows
from ..i18n import t
from .base import Engine


class LlamaCppEngine(Engine):
    id = "llamacpp"
    label = "llama.cpp"
    status = "stable"
    managed = True
    default_port = 8080
    exe_names = ("llama-server",)
    note = ("Gestionado por SmartModelProvider: arranque, parada y swap completos.",
            "Managed by SmartModelProvider: full start, stop and swap.")

    def defaults(self) -> dict:
        return {"bin": os.path.expanduser("~/llama.cpp/build/bin/llama-server"), "host": "127.0.0.1",
                "port": self.default_port, "env": {}, "common_args": []}

    def _endpoint_args(self, ecfg: dict) -> list[str]:
        e = self.normalize(ecfg)
        return ["--host", str(e.get("host") or "127.0.0.1"), "--port", str(int(e["port"]))]

    def build_cmd(self, ecfg: dict, entry: dict, args: list[str]) -> list[str]:
        """``args`` = common_args + model args (+ preset), already merged."""
        return ([self.binary(ecfg, entry), "-m", str(entry.get("gguf") or "")]
                + self._endpoint_args(ecfg) + [str(a) for a in args])

    def gpu_split(self, ecfg: dict, entry: dict, args: list[str],
                  gpus: list[int]) -> Optional[dict]:
        """How llama.cpp spreads the model over physical GPUs, from its flags.

        CUDA_VISIBLE_DEVICES remaps indices; --device picks among the visible
        ones (CUDAk); -sm none puts everything on -mg; -ts weights the rest.
        Returns {} for a CPU-only launch (-ngl 0 / --device none).
        """
        def val(*names):
            out = None
            for i, a in enumerate(args):
                if a in names and i + 1 < len(args):
                    out = str(args[i + 1])
                for n in names:
                    if n.startswith("--") and a.startswith(n + "="):
                        out = a.split("=", 1)[1]
            return out

        if val("-ngl", "--gpu-layers", "--n-gpu-layers") in ("0",):
            return {}
        visible = list(gpus)
        cvd = self.env(ecfg, entry).get("CUDA_VISIBLE_DEVICES")
        if cvd is not None:
            try:
                visible = [int(x) for x in cvd.replace(" ", "").split(",") if x != ""]
            except ValueError:
                return None  # UUID-style lists: unknown mapping
        visible = [g for g in visible if g in gpus] if gpus else visible
        dev = val("-dev", "--device")
        if dev:
            if dev.strip().lower() == "none":
                return {}
            picked = []
            for d in dev.split(","):
                d = d.strip().upper()
                if not d.startswith("CUDA"):
                    return None
                try:
                    k = int(d[4:])
                except ValueError:
                    return None
                if k < len(visible):
                    picked.append(visible[k])
            visible = picked
        if not visible:
            return None
        if (val("-sm", "--split-mode") or "").lower() == "none":
            try:
                mg = int(val("-mg", "--main-gpu") or 0)
            except ValueError:
                mg = 0
            return {visible[min(mg, len(visible) - 1)]: 1.0}
        ts = val("-ts", "--tensor-split")
        weights = [1.0] * len(visible)
        if ts:
            try:
                w = [float(x) for x in ts.replace("/", ",").split(",") if x.strip() != ""]
                if w and sum(w) > 0:
                    weights = (w + [0.0] * len(visible))[:len(visible)]
            except ValueError:
                pass
        tot = sum(weights) or 1.0
        return {g: wt / tot for g, wt in zip(visible, weights) if wt > 0}

    def validate(self, ecfg: dict, entry: dict, args: list[str]) -> dict:
        """Parse-check the exact argv without loading a model: the argv plus
        --version makes llama-server's parser exit immediately (newer builds
        reject flag spellings older ones accepted, which used to surface only
        after a full model load)."""
        cmd = [self.binary(ecfg, entry)] + self._endpoint_args(ecfg) + [str(a) for a in args] + ["--version"]
        if not self.binary_ok(ecfg, entry):
            return {"ok": False, "error": t(f"binario no encontrado: {cmd[0] or '(vacío)'}",
                                            f"binary not found: {cmd[0] or '(empty)'}"), "cmd": cmd}
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15, env=self.env(ecfg, entry))
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "cmd": cmd}
        out = f"{r.stdout}\n{r.stderr}".strip()
        bad = [ln.strip() for ln in out.splitlines()
               if "invalid argument" in ln or ln.strip().lower().startswith("error")]
        if bad:
            return {"ok": False, "error": bad[0], "cmd": cmd}
        if "version:" not in out:
            return {"ok": False, "error": t(f"sin salida de versión (rc={r.returncode})",
                                            f"no version output (rc={r.returncode})"), "cmd": cmd}
        return {"ok": True, "cmd": cmd}

    def help_info(self, ecfg: dict, entry: Optional[dict], cache_dir: Path):
        return llama_help.get_help(self.binary(ecfg, entry), self.env(ecfg, entry), cache_dir)

    def catalog(self, ecfg: dict, entry: Optional[dict], cache_dir: Path) -> dict:
        return llama_help.catalog(self.binary(ecfg, entry), self.env(ecfg, entry), cache_dir)

    def rows(self, args: list, info) -> list[dict]:
        return args_to_rows(args, info.takes_value if info is not None else None)

    # ------------------------------------------------------------ probes
    def ready(self, url: str) -> bool:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=3) as r:
                return r.status == 200
        except urllib.error.HTTPError as e:
            if e.code == 404:  # very old builds: no /health
                return bool(self.probe(url, 3.0).get("up"))
            return False  # 503 while the model loads
        except Exception:  # noqa: BLE001
            return False

    def activity(self, url: str) -> Optional[float]:
        """prompt + predicted token counters from /metrics: any change = the engine worked."""
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=3) as r:
                text = r.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return None
        total, seen = 0.0, False
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            name = line.split(" ", 1)[0]
            if name in ("llamacpp:prompt_tokens_total", "llamacpp:tokens_predicted_total"):
                try:
                    total += float(line.split()[-1])
                    seen = True
                except (IndexError, ValueError):
                    pass
        return total if seen else None

    def in_flight(self, url: str) -> Optional[int]:
        """'llamacpp:requests_processing N' from /metrics (needs --metrics)."""
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=3) as r:
                text = r.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return None
        for line in text.splitlines():
            if line.startswith("#") or "requests_processing" not in line:
                continue
            try:
                return int(float(line.split()[-1]))
            except (IndexError, ValueError):
                return None
        return None
