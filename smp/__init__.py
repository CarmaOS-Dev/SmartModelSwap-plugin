"""SmartModelProvider (SMP): the engine manager of smart-model-swap.

SMP configures and launches the inference engine the user picked (the engine
binary must already be installed; SMP only needs its path and endpoint) and
manages the models that run on it - no systemd unit involved.

Subpackages / modules:
  engines/    engine contract + llama.cpp (managed) + Ollama/vLLM/LM Studio (wip)
  cmdline     launch-line tokenizer and args <-> editor rows
  llama_help  llama-server --help introspection: parameter catalogue, flag fixes
  registry    models.json v2 schema and the v1 migration
  i18n        es/en user-facing messages (per-request language from the UIs)
"""
from . import cmdline, engines, i18n, llama_help, registry  # noqa: F401

__all__ = ["cmdline", "engines", "i18n", "llama_help", "registry"]
