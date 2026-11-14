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
        "analysis, quick voice chat): load skill_view(\"smart-model-swap:smart-model-routing\")."
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
        "Every swap leaves ALL sessions without an LLM for a few minutes (longer if a reply "
        "is still being generated): never swap for a single message. Never wait for it with "
        "`hermes sms swap X --wait` in a foreground terminal (a timeout kills it). Rules: "
        "skill_view(\"smart-model-swap:smart-model-routing\"). Never start/stop llama-server by hand or via systemctl. "
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
            "gpu": {
                "type": "integer",
                "description": "Optional: load the model whole on this GPU (nvidia-smi index), leaving the other GPUs free for a GPU job. Omit for the model's normal layout.",
            },
            "dry_run": {
                "type": "boolean",
                "description": "Validate files, VRAM fit and runner liveness without starting anything. Default false.",
                "default": False,
            },
            "force": {
                "type": "boolean",
                "description": "Reload the model even if it is already serving (e.g. after editing its launch flags). Default false.",
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
        "Restore the primary model (the big agent brain). Call it when the user asks for "
        "analysis, coding or other deep work the small model handles poorly, or asks for "
        "the primary. The end of a GPU job alone is NOT a reason: the small model delivers "
        "the result and the primary comes back by itself once the engine is idle. "
        "Refused (without cutting the current model) while other heavy GPU processes "
        "such as a working ComfyUI still hold the VRAM the primary needs (`hermes sms check` "
        "tells). Backgrounded with the same grace countdown and request drain: call it as the "
        "LAST tool of the turn and tell the user the next turn runs on the primary. Never "
        "`hermes sms restore --wait` in a foreground terminal. Use dry_run=true to validate first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "dry_run": {
                "type": "boolean",
                "description": "Validate without starting anything. Default false.",
                "default": False,
            },
            "force": {
                "type": "boolean",
                "description": "Reload the primary even if it is already serving. Default false.",
                "default": False,
            },
        },
        "required": [],
        "additionalProperties": False,
    },
}
