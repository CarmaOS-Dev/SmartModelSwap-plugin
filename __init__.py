"""Smart-Model-Swap: hot-swap local LLM models by task.

Entry point. Registers 4 tools, 1 slash command (/swap), 1 CLI command
(hermes sms) and hooks: pre_llm_call (status / voice / keyword hints,
hints.py), pre_tool_call (guard.py: no manual llama-server management) and
pre_transcription (voice detection). Handlers return JSON and never raise.

SmartModelProvider (package ``smp``) launches the inference engine itself; in
the gateway process it also autostarts the primary model when nothing is
serving (this replaced the old llama-server systemd unit).
"""
from __future__ import annotations

import logging
import sys

from . import guard, hints, provider_sync, schemas, tools

log = logging.getLogger(__name__)


def _status_context(**kwargs):
    """pre_llm_call: status / voice / keyword hints (see hints.py). Silent in the
    normal case; never raises."""
    try:
        return hints.build_context(tools, **kwargs)
    except Exception:  # noqa: BLE001 - a hook must never break the turn
        return None


def _is_gateway_process() -> bool:
    """`hermes gateway run` (the long-lived gateway), not the CLI or `hermes serve`."""
    try:
        from gateway.status import looks_like_gateway_command_line
        # the real command line (sys.argv[0] is rewritten by the launcher)
        return looks_like_gateway_command_line(" ".join(sys.orig_argv))
    except (ImportError, AttributeError):  # older host
        argv = [a.lower() for a in sys.argv[1:]]
        return "gateway" in argv and "run" in argv


def register(ctx) -> None:
    # Resolve config once and seed the module-level store used by handlers
    # and (later) by the detached runner via base64 argv.
    resolved = {}
    for key in list(tools.DEFAULTS):
        try:
            val = ctx.get_config(key, tools.DEFAULTS[key])
        except Exception:  # noqa: BLE001 - get_config may not accept a default
            val = tools.DEFAULTS[key]
        if val not in (None, ""):
            resolved[key] = val
    tools.apply_cfg(resolved)
    tools.PLUGIN_CTX = ctx  # the dashboard backend writes settings through it when loaded here

    # SmartModelProvider (the model-provider half): keep an INSTALLED copy in
    # step with the registry. Installing it is the user's call (UI / CLI).
    try:
        provider_sync.refresh_provider()
    except Exception:  # noqa: BLE001 - never block plugin load over this
        log.debug("SmartModelProvider refresh skipped", exc_info=True)

    # Autostart the primary model - gateway process only (the CLI, `hermes
    # serve` and the dashboard backend also load plugins and must not).
    try:
        if _is_gateway_process():
            res = tools.autostart_primary()
            log.info("smart-model-swap autostart: %s", res.get("skipped") or res.get("note") or res)
            # a smaller model left serving across a gateway restart still gets its idle watcher
            res = tools.ensure_idle_watch()
            log.info("smart-model-swap idle watcher: %s", res.get("skipped") or res)
    except Exception:  # noqa: BLE001 - never block plugin load over this
        log.warning("smart-model-swap autostart failed", exc_info=True)

    def _refresh() -> None:
        """Re-read settings: the UI can change primary_model/base_url at any time."""
        fresh = {}
        for key in list(tools.DEFAULTS):
            try:
                val = ctx.get_config(key, tools.DEFAULTS[key])
            except Exception:  # noqa: BLE001
                return
            if val not in (None, ""):
                fresh[key] = val
        tools.apply_cfg(fresh)

    def _fresh(fn):
        def wrapped(*args, **kwargs):
            try:
                _refresh()
            except Exception:  # noqa: BLE001
                pass
            return fn(*args, **kwargs)
        wrapped.__name__ = getattr(fn, "__name__", "handler")
        return wrapped

    check = tools.sms_available

    ctx.register_tool(
        "sms_status", "smart_model_swap", schemas.STATUS_SCHEMA, _fresh(tools.h_status),
        check_fn=check, emoji="📊",
        description="VRAM, current model and Smart-Model-Swap state",
    )
    ctx.register_tool(
        "sms_plan", "smart_model_swap", schemas.PLAN_SCHEMA, _fresh(tools.h_plan),
        check_fn=check, emoji="🧠",
        description="Recommend a model for a VRAM budget",
    )
    ctx.register_tool(
        "sms_swap", "smart_model_swap", schemas.SWAP_SCHEMA, _fresh(tools.h_swap),
        check_fn=check, emoji="🔄",
        description="Hot-swap to the target model (background, with grace)",
    )
    ctx.register_tool(
        "sms_restore", "smart_model_swap", schemas.RESTORE_SCHEMA, _fresh(tools.h_restore),
        check_fn=check, emoji="🏠",
        description="Restore the primary model (the agent's main brain)",
    )

    ctx.register_command(
        "swap", _fresh(tools.cmd_swap),
        description="Smart-Model-Swap: /swap <model> [dry] · /swap restore [dry] · /swap plan [gb] · /swap status",
        args_hint="[modelo|restore|plan|status] [dry]",
    )

    ctx.register_hook("pre_llm_call", _fresh(_status_context))
    ctx.register_hook("pre_tool_call", guard.make_hook(tools))
    ctx.register_hook("pre_transcription", hints.note_voice)

    # Bundled skills (read-only, namespaced: smart-model-swap:<name>).
    try:
        from pathlib import Path
        skills_dir = Path(__file__).parent / "skills"
        for child in sorted(skills_dir.iterdir()) if skills_dir.is_dir() else []:
            skill_md = child / "SKILL.md"
            if child.is_dir() and skill_md.exists():
                ctx.register_skill(child.name, skill_md)
    except Exception:  # noqa: BLE001 - never block plugin load over a skill
        log.debug("smart-model-swap skills not registered", exc_info=True)

    ctx.register_cli_command(
        "sms",
        help="Smart-Model-Swap: status/plan/swap/restore local models",
        setup_fn=tools.setup_sms,
        handler_fn=tools.cli_sms,
    )
