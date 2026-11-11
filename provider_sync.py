"""Install / update / remove SmartModelProvider, the model-provider half of
this plugin.

Hermes loads model providers from $HERMES_HOME/plugins/model-providers/<name>/
and a plugin directory can only be of one kind, so this half lives in its own
directory there. It is written ONLY when the user asks (UI button or
`hermes sms provider install`); once installed, the plugin keeps its
catalog.json in step with the model registry. Directories this module did not
create (no marker file) are never touched.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

try:
    from . import tools
except ImportError:  # standalone (runner / tests / dashboard backend)
    import tools  # type: ignore[no-redef]

PROVIDER_ID = "smart-model-provider"
DISPLAY_NAME = "SmartModelProvider"
MARKER = ".sms-managed.json"
SRC = Path(__file__).resolve().parent / "provider"


def provider_dir() -> Path:
    return tools.hermes_home() / "plugins" / "model-providers" / PROVIDER_ID


def _plugin_version() -> str:
    try:
        text = (tools.PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8")
        m = re.search(r'^version:\s*["\']?([^"\'\s]+)', text, re.M)
        return m.group(1) if m else "0"
    except OSError:
        return "0"


def _catalog() -> dict:
    cfg = tools.get_cfg()
    data = tools.load_models()
    models = []
    for m in data.get("models", []):
        if not m.get("name"):
            continue
        models.append({"name": m["name"], "ctx": tools.model_ctx(m),
                       "description": m.get("description", ""),
                       "engine": m.get("engine") or "llamacpp",
                       "vision": bool(m.get("mmproj"))})
    return {"owner": "smart-model-swap", "owner_dir": str(tools.PLUGIN_DIR),
            "engine": tools.active_engine_id(data),
            "base_url": tools.base_url(data),
            "primary": str((tools.find_model(str(cfg.get("primary_model") or ""), data) or {}).get("name")
                           or cfg.get("primary_model") or ""), "models": models}


def _wanted() -> dict[str, str]:
    """File name -> content the provider directory should hold."""
    version = _plugin_version()
    return {
        "__init__.py": (SRC / "__init__.py.tmpl").read_text(encoding="utf-8"),
        "plugin.yaml": (SRC / "plugin.yaml.tmpl").read_text(encoding="utf-8").replace("{version}", version),
        "catalog.json": json.dumps(_catalog(), indent=2, ensure_ascii=False) + "\n",
        MARKER: json.dumps({"owner": "smart-model-swap", "version": version}, indent=2) + "\n",
    }


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def is_ours(d: Path | None = None) -> bool:
    d = d or provider_dir()
    try:
        return json.loads((d / MARKER).read_text(encoding="utf-8")).get("owner") == "smart-model-swap"
    except Exception:  # noqa: BLE001
        return False


def provider_status() -> dict:
    d = provider_dir()
    exists = d.is_dir()
    installed = exists and is_ours(d)
    out = {"id": PROVIDER_ID, "display_name": DISPLAY_NAME, "path": str(d),
           "installed": installed, "foreign": exists and not installed, "in_sync": False,
           "models": []}
    if installed:
        try:
            want = _wanted()
            out["in_sync"] = all(_read(d / n) == c for n, c in want.items())
            cat = json.loads(_read(d / "catalog.json") or "{}")
            out["models"] = [{"name": m.get("name"), "ctx": m.get("ctx")} for m in cat.get("models", [])]
        except Exception as e:  # noqa: BLE001
            out["error"] = f"{type(e).__name__}: {e}"
    return out


def _write(d: Path, want: dict[str, str]) -> list[str]:
    changed = []
    for name, content in want.items():
        if _read(d / name) == content:
            continue  # leave mtimes alone: Hermes fingerprints this directory
        tmp = d / f".{name}.{os.getpid()}.tmp"
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, d / name)
        changed.append(name)
    return changed


def install_provider() -> dict:
    """Create (or repair) the provider directory. Only called on user request."""
    d = provider_dir()
    if d.exists() and not is_ours(d):
        return {"ok": False, "error": f"{d} already exists and was not created by smart-model-swap; "
                                      "it was left untouched"}
    try:
        want = _wanted()
        d.mkdir(parents=True, exist_ok=True)
        changed = _write(d, want)
    except OSError as e:
        return {"ok": False, "error": f"could not write {d}: {e}"}
    return {"ok": True, "changed": changed, **provider_status()}


def refresh_provider() -> dict:
    """Bring an INSTALLED provider up to date. Never installs."""
    d = provider_dir()
    if not (d.is_dir() and is_ours(d)):
        return {"ok": True, "installed": False, "changed": []}
    try:
        return {"ok": True, "installed": True, "changed": _write(d, _wanted())}
    except Exception as e:  # noqa: BLE001 - never break the caller over this
        return {"ok": False, "installed": True, "error": f"{type(e).__name__}: {e}"}


def remove_provider() -> dict:
    d = provider_dir()
    if not d.exists():
        return {"ok": True, "removed": False, **provider_status()}
    if not is_ours(d):
        return {"ok": False, "error": f"{d} was not created by smart-model-swap; it was left untouched"}
    try:
        shutil.rmtree(d)
    except OSError as e:
        return {"ok": False, "error": f"could not remove {d}: {e}"}
    return {"ok": True, "removed": True, **provider_status()}
