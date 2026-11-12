"""Engine registry for SmartModelProvider."""
from __future__ import annotations

from typing import Optional

from .base import Engine, EngineUnavailable, minimal_env
from .llamacpp import LlamaCppEngine
from .wip import LmStudioEngine, OllamaEngine, VllmEngine

ENGINES: dict[str, Engine] = {e.id: e for e in (
    LlamaCppEngine(), OllamaEngine(), VllmEngine(), LmStudioEngine())}
DEFAULT_ENGINE = "llamacpp"


def get(engine_id: Optional[str]) -> Optional[Engine]:
    return ENGINES.get(str(engine_id or DEFAULT_ENGINE))


__all__ = ["ENGINES", "DEFAULT_ENGINE", "Engine", "EngineUnavailable", "get"]
