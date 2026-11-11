"""Command-line parsing for the parameter editor.

Tokenizer ported from Llamacpp-Model-Launcher ``core/command_builder.py``
(MIT, (c) 2025 Kaspur2012): unlike shlex it keeps backslashes literal (Windows
paths) and understands both ``\\"`` and ``""`` inside double quotes, so JSON
values such as ``--chat-template-kwargs "{\\"k\\": true}"`` survive a round
trip. Differences from the original: negative numbers (``--seed -1``) are
values, not flags, and the result is split into the registry's fields.
"""
from __future__ import annotations

import os
import re
import shlex
from typing import Callable, Optional

from .i18n import t as _t

# Flags the engine injects itself: never stored in a model's args.
MODEL_FLAGS = ("-m", "--model")
ENDPOINT_FLAGS = ("--host", "-host", "--port", "-port")
MMPROJ_FLAGS = ("-mm", "--mmproj")

_NUMBER = re.compile(r"^-?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def tokenize(text: str) -> list[str]:
    """Split a command string into tokens (see module docstring for quoting)."""
    tokens: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace() or (ch == "\\" and i + 1 < n and text[i + 1] == "\n"):
            i += 2 if ch == "\\" else 1  # also skip shell line continuations
            continue
        if ch == '"':
            i += 1
            buf = []
            while i < n:
                c = text[i]
                if c == "\\" and i + 1 < n and text[i + 1] == '"':
                    buf.append('"')
                    i += 2
                elif c == '"' and i + 1 < n and text[i + 1] == '"':
                    buf.append('"')
                    i += 2
                elif c == '"':
                    i += 1
                    break
                else:
                    buf.append(c)
                    i += 1
            tokens.append("".join(buf))
        elif ch == "'":
            i += 1
            buf = []
            while i < n and text[i] != "'":
                buf.append(text[i])
                i += 1
            i += 1
            tokens.append("".join(buf))
        else:
            buf = []
            while i < n and not text[i].isspace():
                buf.append(text[i])
                i += 1
            tokens.append("".join(buf))
    return tokens


def is_flag(token: str) -> bool:
    """A token is a flag when it starts with '-' and is not a (negative) number."""
    return token.startswith("-") and len(token) > 1 and not _NUMBER.match(token)


def args_to_rows(args: list, takes_value: Optional[Callable[[str], Optional[bool]]] = None) -> list[dict]:
    """Flat argv list -> [{key, value}] rows (value None = switch).

    ``takes_value(flag)`` may answer True/False from the binary's --help; when
    it returns None (unknown flag) the next token decides: a non-flag token is
    the value.
    """
    rows: list[dict] = []
    toks = [str(a) for a in (args or [])]
    i = 0
    while i < len(toks):
        tok = toks[i]
        if not is_flag(tok):
            i += 1  # stray positional: dropped, like the Launcher
            continue
        known = takes_value(tok) if takes_value else None
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if known is False or nxt is None or (known is None and is_flag(nxt)):
            rows.append({"key": tok, "value": None})
            i += 1
        else:
            rows.append({"key": tok, "value": nxt})
            i += 2
    return rows


def rows_to_args(rows: list) -> list[str]:
    out: list[str] = []
    for r in rows or []:
        key = str((r or {}).get("key") or "").strip()
        if not key:
            continue
        out.append(key)
        val = r.get("value")
        if val is not None:
            out.append(str(val))
    return out


def join(argv: list) -> str:
    return shlex.join([str(a) for a in argv])


def parse_env(text_or_map) -> tuple[dict, list[str]]:
    """'KEY=value' lines (or a dict) -> (env dict, errors). Accepts 'set'/'export'."""
    env: dict[str, str] = {}
    errors: list[str] = []
    if isinstance(text_or_map, dict):
        items = [(str(k), "" if v is None else str(v)) for k, v in text_or_map.items()]
    else:
        items = []
        for raw in str(text_or_map or "").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            for pre in ("set ", "export "):
                if line.lower().startswith(pre):
                    line = line[len(pre):].strip()
            k, sep, v = line.partition("=")
            if not sep:
                errors.append(f"'{raw.strip()}': falta '='")
                continue
            items.append((k.strip(), v.strip().strip('"').strip("'")))
    for k, v in items:
        if not _ENV_KEY.match(k):
            errors.append(_t(f"clave de entorno inválida: '{k}'", f"invalid environment key: '{k}'"))
        else:
            env[k] = v
    return env, errors


def parse_cmdline(text: str) -> dict:
    """Import a full launch line (``llama-server -m x.gguf ...``).

    Returns {bin, gguf, mmproj, args, env, host, port}. Leading ``KEY=value``
    tokens (``CUDA_VISIBLE_DEVICES=0 llama-server ...``) become env vars;
    -m/--model, --mmproj and --host/--port are lifted out of args because the
    registry stores them as fields / engine settings.
    """
    # `exec` and `nohup` wrappers from shell scripts are noise here.
    toks = tokenize(text)
    env: dict[str, str] = {}
    while toks and ("=" in toks[0] and _ENV_KEY.match(toks[0].split("=", 1)[0])
                    or toks[0] in ("exec", "nohup", "env")):
        t = toks.pop(0)
        if "=" in t:
            k, v = t.split("=", 1)
            env[k] = v
    out = {"bin": "", "gguf": "", "mmproj": "", "args": [], "env": env, "host": None, "port": None}
    if toks and not is_flag(toks[0]):
        out["bin"] = toks.pop(0)
    rows = args_to_rows(toks)
    kept = []
    for r in rows:
        k, v = r["key"], r["value"]
        if k in MODEL_FLAGS and v:
            out["gguf"] = v
        elif k in MMPROJ_FLAGS and v:
            out["mmproj"] = v
            kept.append(r)  # --mmproj stays in args: it is a real launch flag
        elif k in ("--host", "-host") and v:
            out["host"] = v
        elif k in ("--port", "-port") and v:
            out["port"] = v
        else:
            kept.append(r)
    out["args"] = rows_to_args(kept)
    return out


def strip_managed(args: list) -> tuple[list[str], list[str]]:
    """Drop -m/--model and --host/--port from a model's args (the engine sets
    them). Returns (clean args, dropped flags)."""
    rows = args_to_rows(args)
    kept, dropped = [], []
    for r in rows:
        if r["key"] in MODEL_FLAGS + ENDPOINT_FLAGS:
            dropped.append(r["key"])
        else:
            kept.append(r)
    return rows_to_args(kept), dropped


def looks_like_path(value: str) -> bool:
    v = str(value or "").strip("\"'")
    return v.startswith(("/", "~")) or bool(re.match(r"^[A-Za-z]:[\\/]", v)) or (
        os.path.splitext(v)[1].lower() in (".gguf", ".jinja", ".json", ".txt", ".bin"))
