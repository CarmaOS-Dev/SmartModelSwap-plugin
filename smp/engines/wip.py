"""Engines under construction: detection only (health + model list).

They are wired into the registry and the UI so their endpoint can be
configured now, but SMP does not start, stop or swap models on them yet:
build_cmd/validate raise or report "motor en construcción".

Planned management (not implemented):
  * Ollama   - ``ollama serve`` if down; load/unload via /api/generate keep_alive,
               per-model options (num_ctx, num_gpu, sampling).
  * vLLM     - ``vllm serve <model> --port P <args>``, one model per process
               (same lifecycle as llama.cpp).
  * LM Studio - ``lms server start`` + ``lms load/unload`` (context length, GPU offload).
"""
from __future__ import annotations

from .base import Engine


class _Wip(Engine):
    status = "wip"
    managed = False
    note = ("En construcción: solo detección (salud + lista de modelos).",
            "Under construction: detection only (health + model list).")


class OllamaEngine(_Wip):
    id = "ollama"
    label = "Ollama"
    default_port = 11434
    exe_names = ("ollama",)


class VllmEngine(_Wip):
    id = "vllm"
    label = "vLLM"
    default_port = 8000
    exe_names = ("vllm",)


class LmStudioEngine(_Wip):
    id = "lmstudio"
    label = "LM Studio"
    default_port = 1234
    exe_names = ("lms",)
