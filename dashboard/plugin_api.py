"""Smart-Model-Swap — UI backend (Hermes Desktop + Dashboard).

Mounted by the gateway at ``/api/plugins/smart-model-swap/``. Exposes the FastAPI
``router`` the app discovers via ``dashboard/manifest.json``. The hot-mount fallback
splices it in on first request — no gateway restart needed.

Design: thin endpoints over the core (``tools.py`` + the SmartModelProvider
package ``smp``) so the UI, the agent tools and the CLI all mutate the same
single source of truth: ``models.json`` (engines + models), ``state.json`` and
the plugin settings. The UI only keeps its slots in ``cache/smart-model-swap/ui.json``.

Endpoints
  GET    /state              full snapshot: swap state, engines, LLM, VRAM, models, slots
  GET    /logs               tail of the runner log
  POST   /slots              set primary / secondary slot (picking the other slot's model exchanges them)
  GET    /engines            engines with config, status (stable / wip), version and health
  POST   /engines/update     edit an engine: bin, host, port, env, common_args (wip: host/port only)
  POST   /engines/select     choose the active engine (managed engines only)
  POST   /engines/test       run the engine binary with --version
  GET    /params/catalog     parameter catalogue for a model / engine (DB + the binary's --help)
  GET    /models/detail      one model as editor rows + its launch command
  POST   /models             upsert a model (full editor: engine, args, env, bin, mmproj, ...)
  POST   /models/preview     launch argv + parse-check + flag fixes for a draft, nothing saved
  POST   /models/parse_cmd   import a launch line (llama-server -m x.gguf ...) into editor fields
  POST   /models/duplicate   copy a model under a new name
  POST   /models/delete      remove a model from the registry
  POST   /params/save        save sampling params: into the model's base args, or into a named preset
  POST   /params/preset/select  choose the preset applied at next load ("" = base args)
  POST   /params/preset/delete  remove a named preset from a model
  POST   /provider/install   install SmartModelProvider's model-provider half - user request only
  POST   /provider/remove    remove it again
  POST   /swap               hot-swap to a model (default: secondary slot)
  POST   /restore            restore the primary model
  GET    /health             re-probe every engine endpoint now (bypasses the cache)

Swap and restore accept ``dry_run`` to preflight without touching anything.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional, Union

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

PLUGIN_ROOT = Path(__file__).resolve().parent.parent

# C2: resolve the core module through the plugin namespace when the gateway
# has registered it (loader: hermes_plugins.smart_model_swap). A bare
# `import tools` inside the gateway resolves to hermes-agent's own `tools/`
# package and silently serves the wrong module. Fall back to the sys.path
# form only when run standalone (e.g. direct TestClient outside the loader).
try:
    from hermes_plugins.smart_model_swap import provider_sync, tools  # noqa: E402
    from hermes_plugins.smart_model_swap.smp import cmdline, engines, i18n, llama_help  # noqa: E402
except ImportError:
    if str(PLUGIN_ROOT) not in sys.path:
        sys.path.insert(0, str(PLUGIN_ROOT))
    import tools  # noqa: E402
    import provider_sync  # noqa: E402
    from smp import cmdline, engines, i18n, llama_help  # noqa: E402

PLUGIN_ID = "smart-model-swap"
t = i18n.t


async def _request_lang(lang: Optional[str] = None) -> None:
    """Both UIs send ?lang=<host locale>; messages follow it (default: Spanish).
    Async so the ContextVar is set in the request task and copied into the
    threadpool that runs the sync endpoints."""
    i18n.set_lang(lang)


router = APIRouter(dependencies=[Depends(_request_lang)])

_HOST_RE = re.compile(r"^[A-Za-z0-9.:-]{1,253}$")

# --------------------------------------------------------------------- helpers


def resolve_cfg() -> dict:
    """Re-resolve plugin settings (same resolution as __init__.register) and apply.

    Runs per request so edits made from the settings form or this UI are picked up
    without a plugin reload.
    """
    resolved: dict = {}
    try:
        from hermes_cli.config import load_config_readonly  # documented read path
        entries = ((load_config_readonly() or {}).get("plugins") or {}).get("entries") or {}
        entry = entries.get(PLUGIN_ID) or {}
        for k in list(tools.DEFAULTS):
            v = (entry.get("settings") or {}).get(k)
            if v is None:
                v = (entry.get("config") or {}).get(k)
            if v not in (None, ""):
                resolved[k] = v
    except Exception:  # noqa: BLE001
        # Config unreadable right now: keep what register() resolved instead of
        # silently resetting every setting to its default.
        return tools.get_cfg()
    tools.apply_cfg(resolved)
    return tools.get_cfg()


_SETTING_TYPES = {"bool": (bool,), "int": (int,), "float": (int, float), "string": (str,), "str": (str,)}


def _schema_type(key: str) -> Optional[str]:
    """The `type` this plugin's plugin.yaml config_schema declares for `key`."""
    try:
        import yaml  # bundled with Hermes
        schema = (yaml.safe_load((PLUGIN_ROOT / "plugin.yaml").read_text()) or {}).get("config_schema") or {}
        spec = schema.get(key)
        return str((spec or {}).get("type") or "string") if spec is not None else None
    except Exception:  # noqa: BLE001
        return None


def set_setting(key: str, value: Any) -> dict:
    """Write one plugin setting (`plugins.entries.<id>.settings.<key>`).

    Public surface only (catalog rule 9). `ctx.set_config` is the plugin API and
    is used whenever `register(ctx)` ran in this process. The dashboard process
    may import this module without ever registering the plugin; then there is no
    context, and `hermes_cli.config.save_config(merge_existing=True)` merges the
    one key into the on-disk config under the config lock, like `hermes config set`.
    """
    kind = _schema_type(key)
    if key not in tools.DEFAULTS or kind is None:
        return {"ok": False, "error": t(f"ajuste desconocido: '{key}'", f"unknown setting: '{key}'")}
    expected = _SETTING_TYPES.get(kind, (str,))
    if not isinstance(value, expected) or (isinstance(value, bool) and bool not in expected):
        return {"ok": False, "error": t(f"'{key}' debe ser {kind}", f"'{key}' must be {kind}")}
    try:
        from hermes_cli.config import is_managed, load_config_readonly, save_config
        if is_managed():
            return {"ok": False, "error": t("instalación gestionada: los ajustes no se pueden cambiar",
                                            "managed install: settings cannot be changed")}
        ctx = getattr(tools, "PLUGIN_CTX", None)
        via = "ctx"
        if ctx is not None and hasattr(ctx, "set_config"):
            ctx.set_config(key, value)
        else:
            via = "config"
            save_config({"plugins": {"entries": {PLUGIN_ID: {"settings": {key: value}}}}}, merge_existing=True)
        entry = (((load_config_readonly() or {}).get("plugins") or {}).get("entries") or {}).get(PLUGIN_ID) or {}
        if (entry.get("settings") or {}).get(key) != value:
            return {"ok": False, "error": t(f"no se pudo guardar '{key}' (¿clave administrada?)",
                                            f"could not save '{key}' (administrator-managed key?)")}
        return {"ok": True, "via": via, "written": [key]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def ui_file() -> Path:
    return tools.cache_dir() / "ui.json"


def read_ui() -> dict:
    """UI-only state: the primary / secondary slots. (Up to 0.2 this file also
    held the servers; they moved to models.json -> engines.)"""
    try:
        data = json.loads(ui_file().read_text())
    except Exception:  # noqa: BLE001
        data = {}
    cfg = resolve_cfg()
    slots = data.get("slots") if isinstance(data.get("slots"), dict) else {}
    slots["primary"] = cfg["primary_model"]  # single source of truth: the plugin setting
    slots.setdefault("secondary", "")
    return {"slots": slots}


def write_ui(data: dict) -> None:
    p = ui_file()
    tmp = p.with_name(f"ui.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"slots": data.get("slots") or {}}, indent=2, ensure_ascii=False))
    os.replace(tmp, p)


_HEALTH_TTL_S = 15.0
_health_cache: dict = {"key": None, "at": 0.0, "data": {}}
_health_lock = threading.Lock()


def health_all(reg: dict, force: bool = False) -> dict:
    """Probe every engine endpoint in parallel. Cached for a few seconds: the
    pane and the status-bar chip poll /state continuously and four HTTP probes
    per poll (1.5 s timeout each when a server is down) add up."""
    targets = {}
    for eid, eng in engines.ENGINES.items():
        targets[eid] = (eng, eng.endpoint((reg.get("engines") or {}).get(eid) or {}))
    key = json.dumps({k: v[1] for k, v in targets.items()}, sort_keys=True)
    with _health_lock:
        fresh = time.time() - _health_cache["at"] < _HEALTH_TTL_S
        if not force and fresh and _health_cache["key"] == key:
            return dict(_health_cache["data"])
    out: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(targets)) as ex:
        futs = {ex.submit(eng.probe, url): eid for eid, (eng, url) in targets.items()}
        for fut in concurrent.futures.as_completed(futs):
            out[futs[fut]] = fut.result()
    with _health_lock:
        _health_cache.update(key=key, at=time.time(), data=out)
    return dict(out)


def load_registry() -> dict:
    try:
        return tools.load_models()
    except Exception:  # noqa: BLE001
        return {"version": 2, "engines": {}, "models": []}


def save_registry(data: dict) -> None:
    tools.save_models(data)
    _refresh_provider()


def _refresh_provider() -> None:
    """Update an installed SmartModelProvider after a registry/settings change.
    Never installs it and never fails the request."""
    try:
        provider_sync.refresh_provider()
    except Exception:  # noqa: BLE001
        pass


def _provider_status() -> dict:
    try:
        return provider_sync.provider_status()
    except Exception as e:  # noqa: BLE001
        return {"installed": False, "error": f"{type(e).__name__}: {e}"}


def _clean_presets(entry: dict) -> dict:
    out = {}
    presets = entry.get("presets")
    for n, p in (presets.items() if isinstance(presets, dict) else []):
        if isinstance(p, dict):
            out[str(n)] = {k: str(v) for k, v in p.items() if k in tools.PARAM_KEYS}
    return out


_registry_lock = threading.Lock()  # read-modify-write of models.json from the UI


def _mutate_registry(fn) -> dict:
    """Run fn(registry) under the thread + file locks and persist unless it
    returns a dict with ok False. fn returns the response dict."""
    load_registry()  # migrates a v1 file first (that path takes the file lock itself)
    with _registry_lock, tools._registry_file_lock():
        reg = load_registry()
        out = fn(reg)
        if out.get("ok", True):
            save_registry(reg)
        return out


def _edit_model(name: str, fn) -> dict:
    """Apply fn(entry) to one registry entry and persist. fn returns an error
    string to abort without writing."""
    def run(reg: dict) -> dict:
        for entry in reg.get("models", []):
            if tools.same_name(entry.get("name"), name):
                err = fn(entry)
                if err:
                    return {"ok": False, "error": err}
                return {"ok": True, "model": name,
                        "params": tools.model_params(entry),
                        "base_params": tools.base_params(entry),
                        "active_preset": tools.active_preset(entry),
                        "presets": _clean_presets(entry)}
        return {"ok": False, "error": t(f"modelo desconocido: '{name}'", f"unknown model: '{name}'")}
    return _mutate_registry(run)


def _preset_name(raw: Optional[str]) -> tuple[str, Optional[str]]:
    name = " ".join(str(raw or "").split())
    if len(name) > 40:
        return name, t("el nombre del preset admite 40 caracteres como máximo", "preset names are limited to 40 characters")
    if any(ord(c) < 32 for c in name):
        return name, t("nombre de preset inválido", "invalid preset name")
    return name, None


def _help_for(entry: Optional[dict], reg: dict, eid: Optional[str] = None):
    eng, ecfg = tools.engine_for(eid or (entry or {}).get("engine"), reg)
    try:
        return eng, ecfg, eng.help_info(ecfg, entry, tools.cache_dir())
    except Exception:  # noqa: BLE001
        return eng, ecfg, None


def _rows(args: list, info) -> list[dict]:
    return cmdline.args_to_rows([str(a) for a in (args or [])], info.takes_value if info else None)


def _path_exists(p: Any) -> bool:
    return bool(p) and Path(str(p)).expanduser().exists()


def models_list(reg: dict) -> list:
    out = []
    for m in reg.get("models", []):
        out.append({
            "name": m.get("name"),
            "engine": m.get("engine") or engines.DEFAULT_ENGINE,
            "role": m.get("role", ""),
            "description": m.get("description", ""),
            "gguf": m.get("gguf", ""),
            "mmproj": m.get("mmproj", ""),
            "bin": m.get("bin", ""),
            "env": m.get("env") or {},
            "args": [str(a) for a in (m.get("args") or [])],
            "min_free_vram_gb": m.get("min_free_vram_gb"),
            "load_timeout_s": m.get("load_timeout_s"),
            "ctx": tools.model_ctx(m),
            "exists": _path_exists(m.get("gguf")),
            "params": tools.model_params(m),          # effective at next load
            "base_params": tools.base_params(m),      # the model's own args
            "active_preset": tools.active_preset(m),
            "presets": _clean_presets(m),
        })
    return out


def engines_list(reg: dict, with_version: bool = False) -> list:
    out = []
    for eid, eng in engines.ENGINES.items():
        ecfg = eng.normalize((reg.get("engines") or {}).get(eid))
        item = eng.describe(ecfg)
        item["bin_ok"] = eng.binary_ok(ecfg) if eng.managed else None
        item["active"] = eid == tools.active_engine_id(reg)
        item["models"] = [m.get("name") for m in reg.get("models", []) if (m.get("engine") or "llamacpp") == eid]
        if with_version and eng.managed:
            item["version"] = eng.version(ecfg)
        out.append(item)
    return out


def tail_log(lines: int = 40) -> list:
    try:
        p = tools.log_file()
        with open(p) as f:
            return [ln.rstrip("\n") for ln in f.readlines()[-lines:]]
    except Exception:  # noqa: BLE001
        return []


# ------------------------------------------------------------------- request bodies

class SlotBody(BaseModel):
    slot: str = Field(..., pattern="^(primary|secondary)$")
    model: str


class EngineBody(BaseModel):
    engine: str


class EngineUpdateBody(BaseModel):
    engine: str
    bin: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = Field(None, ge=1, le=65535)
    env: Optional[Union[str, dict]] = None
    common_args: Optional[Union[str, list]] = None


class EngineTestBody(BaseModel):
    engine: str = "llamacpp"
    bin: Optional[str] = None


class Row(BaseModel):
    key: str
    value: Optional[str] = None


class ModelDraft(BaseModel):
    name: str
    original_name: Optional[str] = None   # set when renaming an existing model
    engine: str = "llamacpp"
    gguf: str = ""
    mmproj: Optional[str] = ""
    bin: Optional[str] = ""
    env: Optional[Union[str, dict]] = None
    rows: Optional[list[Row]] = None      # editor rows (preferred)
    args: Optional[list] = None           # or a flat argv list
    role: str = ""
    description: str = ""
    min_free_vram_gb: float = Field(8.0, ge=0, le=1024)
    load_timeout_s: int = Field(240, ge=10, le=7200)


class ParseCmdBody(BaseModel):
    cmdline: str = Field(..., max_length=20000)


class DuplicateBody(BaseModel):
    name: str
    new_name: str


class DeleteModelBody(BaseModel):
    name: str


class GpuJobModeBody(BaseModel):
    mode: str


class SwapBody(BaseModel):
    model: Optional[str] = None
    dry_run: bool = False


class SaveParamsBody(BaseModel):
    model: str
    params: dict = Field(default_factory=dict)
    # None / "" -> write the model's base args; a name -> store as that preset
    # and make it the one applied at next load.
    preset: Optional[str] = None


class PresetBody(BaseModel):
    model: str
    preset: str = ""


# ------------------------------------------------------------- draft handling

def _sync_mmproj(args: list[str], mmproj: str) -> tuple[list[str], str]:
    """The mmproj field and a --mmproj flag in args describe the same thing:
    the field wins when set; otherwise a flag in args fills the field."""
    rows = cmdline.args_to_rows(args)
    flag_val = next((r["value"] for r in rows if r["key"] in cmdline.MMPROJ_FLAGS and r["value"]), "")
    if mmproj:
        rows = [r for r in rows if r["key"] not in cmdline.MMPROJ_FLAGS]
        rows.insert(0, {"key": "--mmproj", "value": mmproj})
        return cmdline.rows_to_args(rows), mmproj
    return args, flag_val


def _draft_entry(body: ModelDraft, reg: dict) -> tuple[Optional[dict], list[str], list[str]]:
    """Validate a draft into a registry entry. Returns (entry|None, errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    name = body.name.strip()
    if not tools.valid_model_name(name):
        errors.append(tools.model_name_hint())
    eng = engines.get(body.engine)
    if eng is None:
        errors.append(t(f"motor desconocido: '{body.engine}'", f"unknown engine: '{body.engine}'"))
    elif not eng.managed:
        errors.append(t(f"{eng.label}: motor en construcción, todavía no admite modelos", f"{eng.label}: engine under construction, models are not supported yet"))
    gguf = str(Path(body.gguf.strip()).expanduser()) if body.gguf.strip() else ""
    if not gguf:
        errors.append(t("falta la ruta del modelo (GGUF)", "the model path (GGUF) is missing"))
    elif not Path(gguf).exists():
        errors.append(t(f"el GGUF no existe: {gguf}", f"GGUF not found: {gguf}"))
    mmproj = str(Path(str(body.mmproj).strip()).expanduser()) if (body.mmproj or "").strip() else ""
    if mmproj and not Path(mmproj).exists():
        errors.append(t(f"el mmproj no existe: {mmproj}", f"mmproj not found: {mmproj}"))
    binp = (body.bin or "").strip()
    if binp and not Path(binp).expanduser().exists():
        errors.append(t(f"el binario/directorio no existe: {binp}", f"binary/directory not found: {binp}"))
    env, env_err = cmdline.parse_env(body.env if body.env is not None else {})
    errors += env_err
    if body.rows is not None:
        rows = [r.model_dump() for r in body.rows if r.key.strip()]
        bad = [r["key"] for r in rows if not cmdline.is_flag(r["key"].strip())]
        if bad:
            errors.append(t(f"cada parámetro debe empezar por '-': {', '.join(bad[:5])}", f"every parameter must start with '-': {', '.join(bad[:5])}"))
        args = cmdline.rows_to_args(rows)
    else:
        args = [str(a) for a in (body.args or [])]
    args, dropped = cmdline.strip_managed(args)
    if dropped:
        warnings.append(t(f"{', '.join(sorted(set(dropped)))}: los pone el motor (GGUF / host / puerto); no se guardan", f"{', '.join(sorted(set(dropped)))}: set by the engine (GGUF / host / port); not saved"))
    args, mmproj = _sync_mmproj(args, mmproj)
    if errors:
        return None, errors, warnings
    entry = {"name": name, "engine": eng.id, "role": body.role.strip(),
             "description": body.description.strip(), "gguf": gguf,
             "min_free_vram_gb": float(body.min_free_vram_gb),
             "load_timeout_s": int(body.load_timeout_s), "args": args}
    if mmproj:
        entry["mmproj"] = mmproj
    if binp:
        entry["bin"] = binp
    if env:
        entry["env"] = env
    return entry, errors, warnings


def _preview(entry: dict, reg: dict) -> dict:
    eng, ecfg, info = _help_for(entry, reg)
    try:
        argv = eng.build_cmd(ecfg, entry, tools.launch_args(entry, reg))
    except engines.EngineUnavailable as e:
        return {"ok": False, "error": str(e)}
    rows = _rows(entry.get("args"), info)
    fixed, notes = llama_help.fix_rows(rows, info)
    common_rows = _rows(ecfg.get("common_args"), info)
    val = eng.validate(ecfg, entry, tools.launch_args(entry, reg))
    if isinstance(val.get("cmd"), list):
        val["cmd"] = cmdline.join(val["cmd"])
    return {"ok": True, "argv": argv, "cmdline": cmdline.join(argv), "validate": val,
            "rows": rows, "common_rows": common_rows,
            "unsupported": llama_help.unsupported(rows + common_rows, info),
            "fix_notes": notes, "fixed_rows": fixed if notes else None,
            "build": getattr(info, "version", "") if info else ""}


def _gpu_job_modes() -> list:
    """The modes of the UI selector, localized (id, label, description)."""
    return [
        {"id": "auto", "label": "Auto",
         "description": t("Elige el mejor modo según tu equipo: Multi si hay 2 o más GPUs, None si solo hay una.",
                          "Picks the best mode for your machine: Multi with 2+ GPUs, None with a single one.")},
        {"id": "none", "label": "None",
         "description": t("Descarga el modelo para dejar toda la VRAM al trabajo. Hermes queda sin modelo hasta que termina.",
                          "Unloads the model so the job gets all the VRAM. Hermes has no model until it ends.")},
        {"id": "nano", "label": "Nano",
         "description": t("Carga el modelo pequeño para que Hermes no se quede sin modelo mientras el trabajo corre en segundo plano.",
                          "Loads the small model so Hermes keeps a model while the job runs in the background.")},
        {"id": "multi", "label": "Multi",
         "description": t("Carga el modelo pequeño entero en una GPU y deja la otra libre para el trabajo pesado.",
                          "Loads the small model whole on one GPU and leaves the other free for the heavy job.")},
        {"id": "keep", "label": "Keep",
         "description": t("No toca el modelo cargado: para trabajos ligeros que caben junto a él.",
                          "Leaves the loaded model alone: for light jobs that fit beside it.")},
    ]


def _gpu_job_view(cfg: dict, status: dict) -> dict:
    """Selected mode + what it means on this machine right now."""
    mode = tools.gpu_job_mode()
    vram = status.get("vram") if isinstance(status.get("vram"), list) else None
    try:
        lay = tools.job_layout(vram=vram or None)
    except Exception as e:  # noqa: BLE001
        lay = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    summary = ""
    if lay.get("ok"):
        ap, name, task = lay.get("applied"), lay.get("model"), lay.get("task_gpu")
        on_task = t(f" Trabajo en GPU {task}.", f" Job on GPU {task}.") if task is not None and lay.get("gpus", 0) > 1 else ""
        if ap == "multi":
            summary = t(f"'{name}' entero en GPU {lay.get('llm_gpu')}.", f"'{name}' whole on GPU {lay.get('llm_gpu')}.") + on_task
        elif ap == "nano":
            summary = t(f"'{name}' sigue cargado durante el trabajo.", f"'{name}' stays loaded during the job.") + on_task
        elif ap == "keep":
            summary = (t(f"'{name}' se queda cargado; el trabajo corre a su lado.", f"'{name}' stays loaded; the job runs beside it.")
                       if name else t("No hay modelo cargado; el trabajo corre tal cual.", "No model is loaded; the job runs as is."))
        else:
            summary = t("Sin modelo durante el trabajo; al terminar se carga uno para responder.",
                        "No model during the job; one is loaded to answer when it ends.") + on_task
        if lay.get("degraded"):
            summary = t(f"'{lay.get('requested')}' no es posible ahora ({lay.get('reason')}). ",
                        f"'{lay.get('requested')}' is not possible now ({lay.get('reason')}). ") + summary
    return {"mode": mode, "modes": _gpu_job_modes(), "layout": lay, "summary": summary,
            "after": tools.after_job_policy()}


# ------------------------------------------------------------------------ endpoints

@router.get("/state")
def state() -> dict:
    cfg = resolve_cfg()
    ui = read_ui()
    reg = load_registry()
    status = json.loads(tools.h_status({}))
    return {
        "ok": True,
        "status": status,
        "config": {"base_url": tools.base_url(reg), "primary_model": cfg["primary_model"],
                   "autostart_primary": cfg.get("autostart_primary")},
        "gpu_job_modes": _gpu_job_view(cfg, status),
        "ui": ui,
        "active_engine": tools.active_engine_id(reg),
        "engines": engines_list(reg),
        "models": models_list(reg),
        "param_specs": tools.param_specs(),
        "provider": _provider_status(),
        "health": health_all(reg),
        "log_tail": tail_log(30),
    }


@router.get("/logs")
def logs() -> dict:
    return {"ok": True, "lines": tail_log(200)}


@router.post("/slots")
def set_slots(body: SlotBody) -> dict:
    name = body.model.strip()
    if not tools.find_model(name):
        return {"ok": False, "error": t(f"modelo desconocido: '{name}'", f"unknown model: '{name}'"),
                "available": tools.available_models()}
    ui = read_ui()
    slots = ui["slots"]
    other = "secondary" if body.slot == "primary" else "primary"
    previous = slots.get(body.slot) or ""
    new = {body.slot: name}
    if slots.get(other) == name:
        # the model already sits in the other slot: exchange the two
        new[other] = previous if previous != name else ""
    if new.get("primary") == "":
        return {"ok": False, "error": t("el primario no puede quedar vacío; elige antes otro primario", "the primary cannot be empty; pick another primary first")}
    setting = None
    if "primary" in new and new["primary"] != slots.get("primary"):
        setting = set_setting("primary_model", new["primary"])
        if not setting.get("ok"):
            return {"ok": False, "error": t(f"no se pudo guardar primary_model: {setting.get('error')}", f"could not save primary_model: {setting.get('error')}")}
        resolve_cfg()
    slots.update(new)
    write_ui(ui)
    if setting:
        _refresh_provider()
    return {"ok": True, "slot": body.slot, "model": name, "slots": slots,
            "exchanged": other in new, "setting": setting}


# ------------------------------------------------------------------ engines

@router.get("/engines")
def get_engines() -> dict:
    reg = load_registry()
    return {"ok": True, "active_engine": tools.active_engine_id(reg),
            "engines": engines_list(reg, with_version=True), "health": health_all(reg, force=True)}


@router.post("/engines/update")
def update_engine(body: EngineUpdateBody) -> dict:
    eng = engines.get(body.engine)
    if eng is None:
        return {"ok": False, "error": t(f"motor desconocido: '{body.engine}'", f"unknown engine: '{body.engine}'")}
    patch: dict = {}
    errors: list[str] = []
    warnings: list[str] = []
    if body.host is not None:
        host = body.host.strip() or "127.0.0.1"
        if not _HOST_RE.match(host):
            errors.append(t(f"host inválido: '{host}'", f"invalid host: '{host}'"))
        patch["host"] = host
    if body.port is not None:
        patch["port"] = int(body.port)
    if eng.managed:
        if body.bin is not None:
            b = str(Path(body.bin.strip()).expanduser()) if body.bin.strip() else ""
            probe = eng.normalize({"bin": b})
            if not b:
                errors.append(t("el binario del motor es obligatorio", "the engine binary is required"))
            elif not eng.binary_ok(probe):
                errors.append(t(f"no es un ejecutable del motor: {b}", f"not an engine executable: {b}"))
            patch["bin"] = b
        if body.env is not None:
            env, env_err = cmdline.parse_env(body.env)
            errors += env_err
            patch["env"] = env
        if body.common_args is not None:
            raw = (cmdline.tokenize(body.common_args) if isinstance(body.common_args, str)
                   else [str(a) for a in body.common_args])
            args, dropped = cmdline.strip_managed(raw)
            if dropped:
                warnings.append(t(f"{', '.join(sorted(set(dropped)))}: se configuran como host/puerto/GGUF, no como args", f"{', '.join(sorted(set(dropped)))}: configure them as host/port/GGUF, not as args"))
            patch["common_args"] = args
    elif any(v is not None for v in (body.bin, body.env, body.common_args)):
        warnings.append(t(f"{eng.label} está en construcción: solo se guardan host y puerto", f"{eng.label} is under construction: only host and port are saved"))
    if errors:
        return {"ok": False, "error": "; ".join(errors)}

    def run(reg: dict) -> dict:
        cur = dict((reg.get("engines") or {}).get(eng.id) or eng.defaults())
        old_url = eng.endpoint(cur)
        cur.update(patch)
        reg.setdefault("engines", {})[eng.id] = cur
        out = {"ok": True, "engine": eng.describe(eng.normalize(cur)), "warnings": warnings}
        if eng.endpoint(cur) != old_url and tools.active_engine_id(reg) == eng.id and tools.detect_servers():
            out["warnings"].append(t("el modelo cargado sigue en el puerto anterior hasta el próximo swap/restore", "the loaded model stays on the previous port until the next swap/restore"))
        return out

    return _mutate_registry(run)


@router.post("/engines/select")
def select_engine(body: EngineBody) -> dict:
    eng = engines.get(body.engine)
    if eng is None:
        return {"ok": False, "error": t(f"motor desconocido: '{body.engine}'", f"unknown engine: '{body.engine}'")}
    if not eng.managed:
        return {"ok": False, "error": t(f"{eng.label}: motor en construcción (solo detección)", f"{eng.label}: engine under construction (detection only)")}
    if tools.busy_state():
        return {"ok": False, "error": "hay un swap en curso"}

    def run(reg: dict) -> dict:
        reg["active_engine"] = eng.id
        return {"ok": True, "active_engine": eng.id, "base_url": tools.base_url(reg)}

    return _mutate_registry(run)


@router.post("/engines/test")
def test_engine(body: EngineTestBody) -> dict:
    eng = engines.get(body.engine)
    if eng is None:
        return {"ok": False, "error": t(f"motor desconocido: '{body.engine}'", f"unknown engine: '{body.engine}'")}
    if not eng.managed:
        return {"ok": False, "error": t(f"{eng.label}: motor en construcción", f"{eng.label}: engine under construction")}
    reg = load_registry()
    ecfg = eng.normalize((reg.get("engines") or {}).get(eng.id))
    if body.bin and body.bin.strip():
        ecfg = {**ecfg, "bin": str(Path(body.bin.strip()).expanduser())}
    return eng.version(ecfg)


# ------------------------------------------------------------- parameters

@router.get("/params/catalog")
def params_catalog(model: str = "", engine: str = "") -> dict:
    reg = load_registry()
    entry = tools.find_model(model, reg) if model else None
    eng, ecfg = tools.engine_for(engine or (entry or {}).get("engine"), reg)
    if not eng.managed:
        return {"ok": True, "engine": eng.id, "groups": [], "count": 0,
                "note": t(f"{eng.label}: motor en construcción", f"{eng.label}: engine under construction")}
    cat = eng.catalog(ecfg, entry, tools.cache_dir())
    return {"ok": True, "engine": eng.id, "bin": eng.binary(ecfg, entry), **cat}


# ------------------------------------------------------------------ models

@router.get("/models/detail")
def model_detail(name: str) -> dict:
    reg = load_registry()
    entry = tools.find_model(name, reg)
    if not entry:
        return {"ok": False, "error": t(f"modelo desconocido: '{name}'", f"unknown model: '{name}'")}
    item = next(m for m in models_list(reg) if m["name"] == entry["name"])
    return {"ok": True, "model": item, **_preview(entry, reg)}


@router.post("/models")
def upsert_model(body: ModelDraft) -> dict:
    def run(reg: dict) -> dict:
        entry, errors, warnings = _draft_entry(body, reg)
        if errors:
            return {"ok": False, "error": "; ".join(errors), "warnings": warnings}
        models = reg.setdefault("models", [])
        original = (body.original_name or entry["name"]).strip()
        idx = next((i for i, m in enumerate(models) if m.get("name") == original), None)
        if idx is None:
            idx = next((i for i, m in enumerate(models) if tools.same_name(m.get("name"), original)), None)
        # names are unique case-insensitively; renaming "qwen" -> "Qwen" is the same model
        clash = next((i for i, m in enumerate(models) if tools.same_name(m.get("name"), entry["name"])), None)
        if clash is not None and clash != idx:
            return {"ok": False, "error": t(f"ya existe un modelo '{models[clash].get('name')}' (los nombres no distinguen mayúsculas)",
                                            f"a model named '{models[clash].get('name')}' already exists (names are case-insensitive)")}
        if idx is None:
            models.append(entry)
            merged = entry
        else:
            old = models[idx]
            merged = {**entry}
            for k in ("presets", "active_preset"):  # editor does not carry these
                if k in old:
                    merged[k] = old[k]
            models[idx] = merged
        cs = tools.sync_ctx(models, merged, original)
        return {"ok": True, "name": entry["name"], "renamed_from": original if original != entry["name"] else None,
                "model": merged, "warnings": warnings, "ctx_sync": cs, "notes": _ctx_notes(cs)}

    out = _mutate_registry(run)
    if out.get("ok"):
        _refresh_provider()  # the catalog carries each model's context
    renamed = out.get("renamed_from")
    if out.get("ok") and renamed:
        cfg = resolve_cfg()
        if tools.same_name(cfg.get("primary_model"), renamed):
            out["setting"] = set_setting("primary_model", out["name"])
            resolve_cfg()
        ui = read_ui()
        if tools.same_name(ui["slots"].get("secondary"), renamed):
            ui["slots"]["secondary"] = out["name"]
            write_ui(ui)
        _refresh_provider()
    return out


@router.post("/models/preview")
def preview_model(body: ModelDraft) -> dict:
    reg = load_registry()
    entry, errors, warnings = _draft_entry(body, reg)
    if errors:
        return {"ok": False, "error": "; ".join(errors), "warnings": warnings}
    old = tools.find_model((body.original_name or body.name).strip(), reg) or {}
    for k in ("presets", "active_preset"):
        if k in old:
            entry[k] = old[k]
    # what saving would do to the shared context (on copies: nothing is written)
    others = json.loads(json.dumps(reg.get("models", [])))
    cs = tools.sync_ctx(others, entry, (body.original_name or body.name).strip())
    return {**_preview(entry, reg), "warnings": warnings, "ctx_sync": cs, "notes": _ctx_notes(cs)}


def _ctx_notes(cs: dict) -> list[str]:
    notes = []
    if cs.get("inherited"):
        notes.append(t(f"sin -c propio: se usa el contexto compartido ({cs['ctx']})",
                       f"no own -c: the shared context ({cs['ctx']}) is used"))
    if cs.get("synced"):
        names = ", ".join(cs["synced"])
        notes.append(t(f"contexto {cs['ctx']} aplicado también a: {names}",
                       f"context {cs['ctx']} also applied to: {names}"))
    return notes


@router.post("/models/parse_cmd")
def parse_cmd(body: ParseCmdBody) -> dict:
    text = body.cmdline.strip()
    if not text:
        return {"ok": False, "error": t("línea vacía", "empty line")}
    parsed = cmdline.parse_cmdline(text)
    reg = load_registry()
    _eng, _ecfg, info = _help_for(None, reg, "llamacpp")
    rows = _rows(parsed["args"], info)
    notes = []
    eng_cfg = (reg.get("engines") or {}).get("llamacpp") or {}
    if parsed.get("port") and str(parsed["port"]) != str(eng_cfg.get("port")):
        notes.append(t(f"la línea usa el puerto {parsed['port']}; el motor usa {eng_cfg.get('port')} (se ignora el de la línea)", f"the line uses port {parsed['port']}; the engine uses {eng_cfg.get('port')} (the line's port is ignored)"))
    common = set((r["key"], r["value"]) for r in _rows(eng_cfg.get("common_args"), info))
    dup = [r for r in rows if (r["key"], r["value"]) in common]
    if dup:
        rows = [r for r in rows if (r["key"], r["value"]) not in common]
        notes.append(t(f"{len(dup)} parámetro(s) ya están en los args comunes del motor; se omiten", f"{len(dup)} parameter(s) already in the engine's common args; skipped"))
    b = str(Path(parsed["bin"]).expanduser()) if parsed["bin"] else ""
    engine_bin = str(eng_cfg.get("bin") or "")
    if not b or not Path(b).exists() or os.path.realpath(b) == os.path.realpath(engine_bin or "/nonexistent"):
        b = ""  # same as the engine's, a shell variable, or not on this machine: keep the engine's
    return {"ok": True, "gguf": parsed["gguf"], "mmproj": parsed["mmproj"], "bin": b,
            "env": parsed["env"], "rows": rows, "notes": notes}


@router.post("/models/duplicate")
def duplicate_model(body: DuplicateBody) -> dict:
    new = body.new_name.strip()
    if not tools.valid_model_name(new):
        return {"ok": False, "error": tools.model_name_hint()}

    def run(reg: dict) -> dict:
        models = reg.setdefault("models", [])
        twin = next((m for m in models if tools.same_name(m.get("name"), new)), None)
        if twin:
            return {"ok": False, "error": t(f"ya existe un modelo '{twin.get('name')}' (los nombres no distinguen mayúsculas)",
                                            f"a model named '{twin.get('name')}' already exists (names are case-insensitive)")}
        src = tools.find_model(body.name.strip(), reg)
        if not src:
            return {"ok": False, "error": t(f"modelo desconocido: '{body.name}'", f"unknown model: '{body.name}'")}
        copy = json.loads(json.dumps(src))
        copy["name"] = new
        copy["role"] = ""
        models.append(copy)
        return {"ok": True, "name": new}

    return _mutate_registry(run)


@router.post("/models/delete")
def delete_model(body: DeleteModelBody) -> dict:
    found = tools.find_model(body.name.strip())
    name = found["name"] if found else body.name.strip()
    if tools.same_name(name, resolve_cfg().get("primary_model")):
        return {"ok": False, "error": t("es el modelo primario; elige antes otro primario", "it is the primary model; pick another primary first")}

    def run(reg: dict) -> dict:
        before = len(reg.get("models", []))
        reg["models"] = [m for m in reg.get("models", []) if m.get("name") != name]
        if len(reg["models"]) == before:
            return {"ok": False, "error": t(f"modelo no encontrado: '{name}'", f"model not found: '{name}'")}
        return {"ok": True, "name": name}

    out = _mutate_registry(run)
    if out.get("ok"):
        ui = read_ui()
        if tools.same_name(ui["slots"].get("secondary"), name):
            ui["slots"]["secondary"] = ""
            write_ui(ui)
    return out


@router.post("/params/save")
def save_params(body: SaveParamsBody) -> dict:
    clean, errors = tools.normalize_params(body.params or {})
    if errors:
        return {"ok": False, "error": "; ".join(errors), "specs": tools.param_specs()}
    if not clean:
        return {"ok": False, "error": t("sin valores que guardar", "nothing to save")}
    preset, err = _preset_name(body.preset)
    if err:
        return {"ok": False, "error": err}

    def edit(entry: dict) -> Optional[str]:
        if preset:
            presets = entry.get("presets") if isinstance(entry.get("presets"), dict) else {}
            # a new preset starts from the values currently in effect
            seed = presets.get(preset) if isinstance(presets.get(preset), dict) else tools.model_params(entry)
            presets[preset] = {**{k: str(v) for k, v in seed.items() if k in tools.PARAM_KEYS}, **clean}
            entry["presets"] = presets
            entry["active_preset"] = preset
        else:
            entry["args"] = tools.apply_params_to_args(
                [str(a) for a in (entry.get("args") or [])], clean)
        return None

    out = _edit_model(body.model.strip(), edit)
    if out.get("ok"):
        out["saved_to"] = preset or "base"
    return out


@router.post("/params/preset/select")
def select_preset(body: PresetBody) -> dict:
    preset, err = _preset_name(body.preset)
    if err:
        return {"ok": False, "error": err}

    def edit(entry: dict) -> Optional[str]:
        if not preset:
            entry.pop("active_preset", None)  # base args, no override
            return None
        if preset not in (entry.get("presets") or {}):
            return t(f"preset no encontrado: '{preset}'", f"preset not found: '{preset}'")
        entry["active_preset"] = preset
        return None

    return _edit_model(body.model.strip(), edit)


@router.post("/params/preset/delete")
def delete_preset(body: PresetBody) -> dict:
    preset, err = _preset_name(body.preset)
    if err or not preset:
        return {"ok": False, "error": err or t("falta el nombre del preset", "the preset name is missing")}

    def edit(entry: dict) -> Optional[str]:
        presets = entry.get("presets") or {}
        if preset not in presets:
            return t(f"preset no encontrado: '{preset}'", f"preset not found: '{preset}'")
        del presets[preset]
        if entry.get("active_preset") == preset:
            entry.pop("active_preset", None)
        if presets:
            entry["presets"] = presets
        else:
            entry.pop("presets", None)
        return None

    out = _edit_model(body.model.strip(), edit)
    if out.get("ok"):
        out["deleted"] = preset
    return out


@router.post("/provider/install")
def provider_install() -> dict:
    resolve_cfg()
    return provider_sync.install_provider()


@router.post("/provider/remove")
def provider_remove() -> dict:
    return provider_sync.remove_provider()


@router.post("/gpu_job/mode")
def set_gpu_job_mode(body: GpuJobModeBody) -> dict:
    """The UI selector: what `gpu_job.sh auto` does with the LLM during a GPU job."""
    mode = body.mode.strip().lower()
    if mode not in tools.GPU_JOB_MODES:
        return {"ok": False, "error": t(f"modo desconocido: '{mode}'", f"unknown mode: '{mode}'"),
                "modes": list(tools.GPU_JOB_MODES)}
    res = set_setting("gpu_job_mode", mode)
    if not res.get("ok"):
        return {"ok": False, "error": t(f"no se pudo guardar el modo: {res.get('error')}",
                                        f"could not save the mode: {res.get('error')}")}
    cfg = resolve_cfg()
    return {"ok": True, "mode": mode, **_gpu_job_view(cfg, json.loads(tools.h_status({})))}


@router.post("/swap")
def swap(body: SwapBody) -> dict:
    resolve_cfg()
    reg = load_registry()
    eng, _ = tools.engine_for(None, reg)
    if not eng.managed:
        return {"ok": False, "error": t(f"el motor activo ('{eng.label}') está en construcción", f"the active engine ('{eng.label}') is under construction")}
    name = (body.model or read_ui()["slots"].get("secondary") or "").strip()
    if not name:
        return {"ok": False, "error": t("sin modelo: el slot secundario está vacío", "no model: the secondary slot is empty")}
    return json.loads(tools.h_swap({"model": name, "dry_run": bool(body.dry_run)}))


@router.post("/restore")
def restore(body: SwapBody = SwapBody()) -> dict:
    resolve_cfg()
    return json.loads(tools.h_restore({"dry_run": bool(body.dry_run)}))


@router.get("/health")
def health() -> dict:
    reg = load_registry()
    return {"ok": True, "active_engine": tools.active_engine_id(reg), "health": health_all(reg, force=True)}
