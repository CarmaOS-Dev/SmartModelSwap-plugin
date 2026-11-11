"""Tool schemas for Smart-Model-Swap. Strict JSON-Schema (additionalProperties: false)."""

STATUS_SCHEMA = {
    "name": "sms_status",
    "description": (
        "Live status of the local model swap system: swap state machine, the active "
        "inference engine (SmartModelProvider), current LLM model, who runs the server "
        "(SmartModelProvider / external process), "
        "per-GPU VRAM usage, ComfyUI status, and the last swap result. Use it to check "
        "which local model is loaded, whether a swap finished or failed, or how much "
        "VRAM is free before a GPU-heavy task."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "include_cmd": {
                "type": "boolean",
                "description": "Also return the exact start command for each registered model.",
                "default": False,
            },
        },
        "required": [],
        "additionalProperties": False,
    },
}

PLAN_SCHEMA = {
    "name": "sms_plan",
    "description": (
        "Recommend which model to run given a VRAM budget for other work "
        "(e.g. ComfyUI). Returns free VRAM, per-model fit analysis and a recommendation. "
        "Call it BEFORE starting image/video generation, TTS/voice or any other task "
        "that needs GPU memory, passing the VRAM that task needs, to decide whether a "
        "swap to a smaller model is required. Big-vs-small rules (GPU jobs, deep "
        "analysis, quick voice chat): skill smart-model-routing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "needed_free_gb": {
                "type": "number",
                "description": "VRAM (GiB) that must remain free for other apps (ComfyUI workflows, etc). 0 = just report fit.",
                "default": 0,
            },
            "task": {
                "type": "string",
                "description": "Optional free-text description of the task (context only).",
            },
            "single_gpu": {
                "type": "boolean",
                "description": "The task runs on ONE GPU (ComfyUI --cuda-device N, default true): the room "
                               "that counts is the best single GPU. false = it can spread over all GPUs.",
                "default": True,
            },
        },
        "required": [],
        "additionalProperties": False,
    },
}

SWAP_SCHEMA = {
    "name": "sms_swap",
    "description": (
        "Hot-swap the local LLM to the given model. Use it to free VRAM before a "
        "GPU-heavy task (image/video generation in ComfyUI, voice) by switching to a "
        "small model, or to load a bigger model for analysis and coding. Runs in the "
        "background: after a grace countdown (default 10s) it waits for in-flight "
        "requests, stops the current model, settles VRAM, and loads the target. Use "
        "dry_run=true to validate without touching anything. The model answering this "
        "conversation is replaced: call this as the LAST tool of the turn, then give a "
        "short final reply. The next turn runs on the new model (check sms_status). "
        "Every swap leaves ALL sessions without an LLM for 2-4 min: never swap for a "
        "single message. Rules and the blocking CLI (`hermes sms swap X --wait`): "
        "skill smart-model-routing. Never start/stop llama-server by hand or via systemctl. "
        "After a swap to a smaller model, the primary is restored automatically once the "
        "engine has been idle for a while (setting idle_restore_min)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "description": "Target model name from the model registry (e.g. 'qwen', 'signal').",
            },
            "dry_run": {
                "type": "boolean",
                "description": "Validate files, VRAM fit and runner liveness without starting anything. Default false.",
                "default": False,
            },
        },
        "required": ["model"],
        "additionalProperties": False,
    },
}

RESTORE_SCHEMA = {
    "name": "sms_restore",
    "description": (
        "Restore the primary model (the big agent brain) after a swap. Call it once "
        "the GPU-heavy task that needed the VRAM has finished, or when the user asks "
        "for analysis, coding or other work the small model handles poorly. "
        "Refused (without cutting the current model) while other heavy GPU processes "
        "such as ComfyUI still hold the VRAM the primary needs: check first with "
        "`hermes sms check` and close finished GPU jobs. "
        "Backgrounded with the same grace countdown and request drain; call it as the "
        "LAST tool of the turn. Use dry_run=true to validate first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "dry_run": {
                "type": "boolean",
                "description": "Validate without starting anything. Default false.",
                "default": False,
            },
        },
        "required": [],
        "additionalProperties": False,
    },
}
