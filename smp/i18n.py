"""UI language for user-facing messages (Spanish / English).

The dashboard API sets the language per request from its `?lang=` query
parameter (both UIs send the host's locale); everything else (agent tools,
CLI) keeps the default. Messages are written inline as ``t("es", "en")`` so
the two versions stay next to each other.
"""
from __future__ import annotations

from contextvars import ContextVar

DEFAULT_LANG = "es"
_LANG: ContextVar[str] = ContextVar("sms_lang", default=DEFAULT_LANG)


def norm(lang: str | None) -> str:
    """'es', 'es-MX', 'es_ES' -> 'es'; anything else -> 'en'; empty -> default."""
    s = str(lang or "").strip().lower()
    if not s:
        return DEFAULT_LANG
    return "es" if s == "es" or s.startswith(("es-", "es_")) else "en"


def set_lang(lang: str | None) -> str:
    v = norm(lang)
    _LANG.set(v)
    return v


def get_lang() -> str:
    return _LANG.get()


def t(es: str, en: str) -> str:
    return en if _LANG.get() == "en" else es
