"""Introspection of the installed llama.cpp build (``llama-server --help``).

Ported from Llamacpp-Model-Launcher ``core/llama_cli.py`` (MIT, (c) 2025
Kaspur2012). llama.cpp renames and removes flags every few weeks, so the
parameter catalogue = the static DB (``paramdb_llamacpp.json``, ported from the
Launcher) checked against the binary's own --help, plus the flags the build has
that the DB does not know. Parsed help is cached on disk keyed by binary path +
mtime, so a rebuild is picked up automatically.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from .i18n import t

HERE = Path(__file__).resolve().parent
DB_FILE = HERE / "paramdb_llamacpp.json"

_mem_cache: dict = {}

# ------------------------------------------------------------------ parsing

_FLAG_TOKEN = re.compile(r"(?:^|[\s,])(-{1,2}[A-Za-z][A-Za-z0-9_.+-]*)")
_SECTION = re.compile(r"^-{3,}\s*(.+?)\s*-{3,}\s*$")


def _split_flags_and_desc(line: str) -> tuple[str, str]:
    normalized = re.sub(r",\s+", ", ", line.rstrip())
    parts = re.split(r"\s{2,}", normalized, maxsplit=1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def _extract_placeholder(flag_part: str, flags: list[str]) -> str:
    last = flags[-1]
    idx = flag_part.rfind(last)
    tail = flag_part[idx + len(last):].strip() if idx >= 0 else ""
    return tail.lstrip(",").strip()


def _extract_options(placeholder: str, description: str) -> list[str]:
    for pattern in (r"^\{([^}]+)\}$", r"^\[([^\]]+)\]$", r"^<([^>]+)>$"):
        m = re.match(pattern, placeholder)
        if m:
            sep = "," if "," in m.group(1) else "|"
            opts = [o.strip() for o in m.group(1).split(sep)]
            if all(re.match(r"^[A-Za-z0-9_+.-]+$", o) for o in opts) and len(opts) > 1:
                if not any(o.replace(".", "").isdigit() and "..." in m.group(1) for o in opts):
                    return opts
    m = re.search(r"allowed values:\s*([A-Za-z0-9_,\s]+?)(?:\(|$)", description)
    if m:
        return [o.strip() for o in m.group(1).split(",") if o.strip()]
    if placeholder in ("MODE", "TYPE"):
        bullets = re.findall(r"(?:^|\s)-\s([a-z][a-z0-9+_-]*):", description)
        if len(bullets) > 1:
            return bullets
    if (placeholder and "," in placeholder and " " not in placeholder and "..." not in placeholder
            and not placeholder.isupper()):
        return [o for o in placeholder.split(",") if o]
    return []


def parse_help(text: str) -> list[dict]:
    """--help text -> [{flags, placeholder, description, env, options, removed,
    deprecated, section}]."""
    entries: list[dict] = []
    cur: Optional[dict] = None
    section = ""
    for raw in (text or "").splitlines():
        line = raw.rstrip()
        sec = _SECTION.match(line.strip())
        if sec:
            section, cur = sec.group(1), None
            continue
        if line.startswith("-") and not line.startswith("---"):
            flag_part, desc = _split_flags_and_desc(line)
            flags: list[str] = []
            for tok in _FLAG_TOKEN.findall(flag_part):
                if tok not in flags:
                    flags.append(tok)
            if not flags:
                cur = None
                continue
            cur = {"flags": flags, "placeholder": _extract_placeholder(flag_part, flags),
                   "description": desc.strip(), "env": "", "section": section}
            entries.append(cur)
        elif cur is not None and line.strip():
            t = line.strip()
            env = re.match(r"^\(env:\s*([A-Z0-9_]+)\)$", t)
            if env:
                cur["env"] = env.group(1)
            else:
                cur["description"] = (cur["description"] + " " + t).strip()
    for e in entries:
        low = e["description"].lower()
        e["removed"] = "argument has been removed" in low
        e["deprecated"] = "(deprecated)" in low
        e["options"] = _extract_options(e["placeholder"], e["description"])
    return entries


class HelpInfo:
    """Parsed --help of one binary."""

    def __init__(self, entries: list[dict], version: str = ""):
        self.entries = entries
        self.version = version
        self.flag_map: dict[str, dict] = {}
        for e in entries:
            for f in e["flags"]:
                self.flag_map[f] = e

    def entry(self, flag: str) -> Optional[dict]:
        return self.flag_map.get(flag)

    def known(self, flag: str) -> bool:
        return flag in self.flag_map

    def removed(self, flag: str) -> bool:
        e = self.flag_map.get(flag)
        return bool(e and e["removed"])

    def supports(self, flag: str) -> bool:
        e = self.flag_map.get(flag)
        return bool(e and not e["removed"])

    def takes_value(self, flag: str) -> Optional[bool]:
        e = self.flag_map.get(flag)
        if e is None:
            return None
        ph = e["placeholder"]
        if not ph:
            return False
        return None if ph.startswith("[") else True  # [optional] value: let the argv decide


# ------------------------------------------------------------------ discovery

def _run(binary: str, args: list[str], env: Optional[dict]) -> str:
    try:
        r = subprocess.run([binary] + args, capture_output=True, text=True, errors="ignore",
                           timeout=30, env=env, cwd=os.path.dirname(binary) or None)
        return (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return ""


def get_help(binary: str, env: Optional[dict] = None, cache_dir: Optional[Path] = None) -> Optional[HelpInfo]:
    """Cached HelpInfo for a binary, or None when it cannot be queried."""
    if not binary or not os.path.isfile(binary):
        return None
    key = f"{os.path.realpath(binary)}:{os.path.getmtime(binary)}"
    if key in _mem_cache:
        return _mem_cache[key]
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"help-{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            info = HelpInfo(data["entries"], data.get("version", ""))
            _mem_cache[key] = info
            return info
        except Exception:  # noqa: BLE001 - no cache yet / corrupt: re-query
            pass
    entries = parse_help(_run(binary, ["--help"], env))
    if not entries:
        return None
    version = ""
    m = re.search(r"version:\s*(.+)", _run(binary, ["--version"], env))
    if m:
        version = m.group(1).strip()
    info = HelpInfo(entries, version)
    _mem_cache[key] = info
    if path is not None:
        try:
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps({"key": key, "version": version, "entries": entries}),
                           encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass
    return info


# ------------------------------------------------------------------ catalogue

_NOT_BROWSABLE = {"-h", "--help", "--completion-bash", "--version", "--list-devices", "-cl",
                  "--cache-list", "-m", "--model", "--host", "--port", "--usage"}


def load_db() -> list[dict]:
    try:
        return json.loads(DB_FILE.read_text(encoding="utf-8")).get("groups", [])
    except Exception:  # noqa: BLE001
        return []


def _extra_rows(info: HelpInfo, known: set[str]) -> list[dict]:
    rows = []
    for e in info.entries:
        if e["removed"] or e["deprecated"] or set(e["flags"]) & (known | _NOT_BROWSABLE):
            continue
        flag = min(e["flags"], key=len)
        if not e["placeholder"]:
            neg = [f for f in e["flags"] if f.startswith(("--no-", "-no-"))]
            pos = [f for f in e["flags"] if f not in neg]
            default_on = "default: enabled" in e["description"].lower()
            pool = neg if (default_on and neg) else (pos or e["flags"])
            flag, ptype = min(pool, key=len), "checkbox"
        elif e["options"] and not e["placeholder"].startswith(("MiB", "TOOL")):
            ptype = "select"
        else:
            ptype = "text"
        desc = e["description"]
        if e["env"]:
            desc += f" Environment variable: {e['env']}"
        if e["placeholder"] and ptype == "text":
            desc = f"Value: {e['placeholder']}. " + desc
        rows.append({"name": flag.lstrip("-").replace("-", " ").title(), "id": flag.lstrip("-"),
                     "type": ptype, "default": "" if ptype != "checkbox" else False,
                     "options": e["options"], "description": desc, "prefix": flag,
                     "section": e["section"]})
    return rows


def catalog(binary: str, env: Optional[dict] = None, cache_dir: Optional[Path] = None) -> dict:
    """Parameter catalogue for the editor: DB groups, each param flagged
    ``status`` = ok | removed | unknown (not in this build) | unchecked (no
    --help available), plus a 'Detectado en tu build' group."""
    info = get_help(binary, env, cache_dir)
    groups = []
    known: set[str] = set()
    for g in load_db():
        params = []
        for p in g.get("parameters", []):
            prefix = p.get("prefix") or ""
            if not prefix or prefix in _NOT_BROWSABLE:
                continue
            known.add(prefix)
            e = info.entry(prefix) if info else None
            if e:
                known.update(e["flags"])
            status = ("unchecked" if info is None else
                      "removed" if (e and e["removed"]) else
                      "ok" if e else "unknown")
            params.append({**p, "status": status, "env": (e or {}).get("env", "")})
        if params:
            groups.append({"name": g.get("name", ""), "parameters": params})
    if info is not None:
        extra = _extra_rows(info, known)
        if extra:
            for p in extra:
                p["status"] = "ok"
            groups.append({"name": t("🔎 Detectado en tu build", "🔎 Detected in your build"), "parameters": extra})
    return {"groups": groups, "version": info.version if info else "",
            "help_available": info is not None,
            "count": sum(len(g["parameters"]) for g in groups)}


# ------------------------------------------------------------------ lint / fixes

_BOOL_VALUE_FLAGS = {
    "--context-shift": ("--context-shift", "--no-context-shift"),
    "-cb": ("-cb", "-nocb"), "--cont-batching": ("-cb", "-nocb"),
    "--slots": ("--slots", "--no-slots"),
    "--jinja": ("--jinja", "--no-jinja"),
    "--warmup": ("--warmup", "--no-warmup"),
}
_TRUE = {"enabled", "true", "on", "1", "yes"}
_FALSE = {"disabled", "false", "off", "0", "no"}
_RENAMES = {"--draft": "--spec-draft-n-max", "--draft-n": "--spec-draft-n-max",
            "--draft-max": "--spec-draft-n-max", "--draft-min": "--spec-draft-n-min",
            "--draft-n-min": "--spec-draft-n-min"}
_ALIAS_UPGRADES = {"--webui": "--ui", "--no-webui": "--no-ui"}
_LOAD_MODE_FLAGS = ("--no-mmap", "--mmap", "--mlock")
_REPEATABLE = {"-ot", "--override-tensor", "-otd", "--override-tensor-draft", "--override-kv",
               "--lora", "--lora-scaled", "--control-vector", "--control-vector-scaled", "-l",
               "--logit-bias", "--dry-sequence-breaker", "-r", "--reverse-prompt"}


def fix_rows(rows: list[dict], info: Optional[HelpInfo]) -> tuple[list[dict], list[str]]:
    """Migrate flags the build rejects and collapse duplicate aliases.

    Returns (rows, notes). Same rules as the Launcher's migrate_parameters +
    dedupe_aliases; migrations only apply when the build is known to reject
    the old flag, so older builds keep working.
    """
    notes: list[str] = []
    rows = [dict(r) for r in rows]

    def rejected(flag: str) -> bool:
        return info is not None and (not info.known(flag) or info.removed(flag))

    keys = [r["key"] for r in rows]
    if any(f in keys for f in _LOAD_MODE_FLAGS) and any(rejected(f) for f in _LOAD_MODE_FLAGS if f in keys):
        no_mmap, mlock = "--no-mmap" in keys, "--mlock" in keys
        mode = ("mlock" if no_mmap else "mmap+mlock") if mlock else ("none" if no_mmap else "mmap")
        has_lm = "-lm" in keys or "--load-mode" in keys
        notes.append(f"{', '.join(f for f in _LOAD_MODE_FLAGS if f in keys)} → "
                     + ("eliminado (ya hay --load-mode)" if has_lm else f"-lm {mode}"))
        out, inserted = [], has_lm
        for r in rows:
            if r["key"] in _LOAD_MODE_FLAGS:
                if not inserted:
                    out.append({"key": "-lm", "value": mode})
                    inserted = True
                continue
            out.append(r)
        rows = out

    out = []
    for r in rows:
        k, v = r["key"], r.get("value")
        if k in _BOOL_VALUE_FLAGS and v is not None and str(v).strip().lower() in (_TRUE | _FALSE):
            on, off = _BOOL_VALUE_FLAGS[k]
            nk = on if str(v).strip().lower() in _TRUE else off
            notes.append(f"'{k} {v}' → switch '{nk}'")
            out.append({"key": nk, "value": None})
            continue
        if k in _RENAMES and rejected(k):
            notes.append(f"'{k}' renombrado a '{_RENAMES[k]}'")
            out.append({"key": _RENAMES[k], "value": v})
            continue
        if k in _ALIAS_UPGRADES and info is not None and info.supports(_ALIAS_UPGRADES[k]):
            notes.append(f"alias obsoleto '{k}' → '{_ALIAS_UPGRADES[k]}'")
            out.append({"key": _ALIAS_UPGRADES[k], "value": v})
            continue
        if info is not None and info.removed(k):
            notes.append(f"'{k}' eliminado: ya no existe en este build")
            continue
        out.append(r)
    rows = out

    if info is not None:
        def collapsible(e):
            return e is not None and bool(e["placeholder"]) and not (set(e["flags"]) & _REPEATABLE)
        last: dict[int, int] = {}
        for i, r in enumerate(rows):
            e = info.entry(r["key"])
            if collapsible(e):
                last[id(e)] = i
        out = []
        for i, r in enumerate(rows):
            e = info.entry(r["key"])
            if collapsible(e) and last.get(id(e)) != i:
                w = rows[last[id(e)]]
                notes.append(f"'{r['key']} {r.get('value')}' duplicado; gana '{w['key']} {w.get('value')}'")
                continue
            out.append(r)
        rows = out
    return rows, notes


def unsupported(rows: list[dict], info: Optional[HelpInfo]) -> list[str]:
    if info is None:
        return []
    return [r["key"] for r in rows
            if r["key"].startswith("-") and (not info.known(r["key"]) or info.removed(r["key"]))]
