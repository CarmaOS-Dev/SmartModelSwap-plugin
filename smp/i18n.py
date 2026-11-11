"""UI language for user-facing messages (Spanish / English).

The dashboard API sets the language per request from its `?lang=` query
parameter (both UIs send the host's locale). Everywhere else (agent tools,
hooks, CLI) the language follows Hermes' own `display.language` setting:
`es*` -> Spanish, anything else (default `en`) -> English. Messages are written
inline as ``t("es", "en")`` so the two versions stay next to each other.
"""
from __future__ import annotations

import time
from contextvars import ContextVar

DEFAULT_LANG = "en"
_LANG: ContextVar[str | None] = ContextVar("sms_lang", default=None)
_HOST_CACHE: dict = {"at": 0.0, "lang": DEFAULT_LANG}
_HOST_TTL_S = 60.0


def host_lang() -> str:
    """Hermes' `display.language` (public config read), cached for a minute."""
    now = time.time()
    if now - _HOST_CACHE["at"] < _HOST_TTL_S:
        return _HOST_CACHE["lang"]
    lang = DEFAULT_LANG
    try:
        from hermes_cli.config import load_config_readonly
        raw = ((load_config_readonly() or {}).get("display") or {}).get("language")
        if raw:
            lang = norm(str(raw))
    except Exception:  # noqa: BLE001 - outside Hermes (tests, standalone runner): English
        pass
    _HOST_CACHE.update(at=now, lang=lang)
    return lang


def norm(lang: str | None) -> str:
    """'es', 'es-MX', 'es_ES' -> 'es'; anything else -> 'en'; empty -> the host's."""
    s = str(lang or "").strip().lower()
    if not s:
        return host_lang()
    return "es" if s == "es" or s.startswith(("es-", "es_")) else "en"


def set_lang(lang: str | None) -> str:
    v = norm(lang)
    _LANG.set(v)
    return v


def get_lang() -> str:
    return _LANG.get() or host_lang()


def t(es: str, en: str) -> str:
    return es if get_lang() == "es" else en
