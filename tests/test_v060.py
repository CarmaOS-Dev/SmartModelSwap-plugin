"""0.6.0: daemon-engine lifecycle (LM Studio), turn-based activity, the single
fallback loop and shared helpers. Nothing real is loaded: LM Studio is faked."""
from __future__ import annotations

import time

import _env  # noqa: F401
import guard
import runner
import tools
from _env import FakeClock, SandboxCase
from smp.engines.lmstudio import LmStudioEngine


class FakeLms(LmStudioEngine):
    """LM Studio with the CLI and the REST API replaced by a dict."""

    def __init__(self):
        self.catalog_keys = {"qwen-key", "big-key"}
        self.loaded_ids: list[str] = []
        self.calls: list[str] = []

    def binary_ok(self, ecfg, entry=None):
        return True

    def ensure_server(self, ecfg):
        return {"ok": True}

    def _api_models(self, ecfg_or_url):
        return ([{"id": k, "state": "not-loaded", "type": "llm"} for k in sorted(self.catalog_keys)]
                + [{"id": i, "state": "loaded", "type": "llm"} for i in self.loaded_ids])

    def load(self, ecfg, entry, args, timeout):
        bad = self._exact_key_error(ecfg, entry)
        if bad:
            return {"ok": False, "error": bad}
        self.calls.append("load " + entry["name"])
        self.loaded_ids.append(entry["name"])
        return {"ok": True, "cmd": self.build_cmd(ecfg, entry, args)}

    def unload(self, ecfg, entry=None):
        self.calls.append("unload")
        self.loaded_ids = []
        return {"ok": True, "actions": ["LM Studio: unload"]}


MODELS = {"version": 2, "active_engine": "lmstudio", "engines": {"lmstudio": {"port": 1234}},
          "models": [{"name": "small", "engine": "lmstudio", "model_key": "qwen-key", "args": ["-c", "4096"],
                      "min_free_vram_gb": 6},
                     {"name": "big", "engine": "lmstudio", "model_key": "big-key", "args": ["-c", "4096"],
                      "min_free_vram_gb": 14}]}


class Daemon(SandboxCase):
    def setUp(self) -> None:
        super().setUp()
        self.eng = FakeLms()
        self.patch(tools, "load_models", lambda: MODELS)
        self.patch(tools, "engine_for", lambda e=None, data=None: (self.eng, {}))
        self.patch(tools, "model_engine", lambda entry, data=None: (self.eng, {}))
        self.patch(tools, "get_vram", lambda: [])
        self.patch(tools, "settle_vram", lambda: True)

    def test_start_and_stop_go_through_the_engine(self):
        res = tools.start_model(tools.find_model("small"))
        self.assertTrue(res["ok"], res)
        self.assertEqual(tools.read_state()["server"]["mode"], "daemon")
        self.assertEqual(tools.detect_servers()[0]["models"], ["small"])
        self.assertIn("LM Studio (loaded: small)", tools.describe_server(tools.detect_server()))
        out = tools.stop_server()
        self.assertTrue(out["ok"])
        self.assertEqual(self.eng.calls, ["load small", "unload"])
        self.assertIsNone(tools.read_state()["server"])

    def test_unknown_key_never_loads(self):
        res = tools.start_model(dict(tools.find_model("small"), model_key="nope"))
        self.assertFalse(res["ok"])
        self.assertIn("exact key", res["error"])
        self.assertEqual(self.eng.calls, [])

    def test_unloaded_behind_our_back_is_a_crash(self):
        tools.start_model(tools.find_model("small"))
        self.assertIsNone(tools.crashed_model())
        self.eng.loaded_ids = []   # LM Studio UI / TTL unloaded it
        self.assertEqual(tools.crashed_model(), "small")
        self.assertIsNone(tools.server_unresponsive())

    def test_probe_lists_only_loaded_models(self):
        p = self.eng.probe("http://x")
        self.assertEqual((p["up"], p["models"]), (False, []))
        self.eng.loaded_ids = ["big"]
        self.assertEqual(self.eng.probe("http://x")["models"], ["big"])

    def test_preflight_checks_the_key_not_a_file(self):
        self.patch(tools, "validate_argv", lambda entry, data=None: {"ok": True, "cmd": "lms load"})
        self.patch(tools, "llm_health", lambda timeout=3.0: {"up": False, "models": []})
        self.patch(tools, "vram_budget", lambda vram=None: {"gb": 99.0, "method": "test"})
        pf = tools.preflight(tools.find_model("small"))
        names = [c["check"] for c in pf["checks"]]
        self.assertNotIn("gguf exists", names)
        self.assertIn("LM Studio knows the model key", names)


class TurnDrain(SandboxCase):
    def test_waits_for_the_agent_to_go_quiet(self):
        eng = FakeLms()
        self.patch(tools, "engine_for", lambda e=None, data=None: (eng, {}))
        clock = FakeClock(time.time())
        self.patch(tools.time, "time", clock.time)
        self.patch(tools.time, "sleep", clock.sleep)
        marks = {"t": clock.now}
        self.patch(tools, "last_turn_at", lambda: marks["t"])
        said = []
        tools.drain_requests(said.append)  # LM Studio: no in-flight counter
        self.assertGreaterEqual(clock.now - marks["t"], tools.TURN_QUIET_MIN_S)
        self.assertTrue(any("no request counter" in s for s in said))

    def test_llama_without_counter_does_not_wait(self):
        self.patch(tools, "requests_processing", lambda: None)
        clock = FakeClock()
        self.patch(tools.time, "sleep", clock.sleep)
        tools.drain_requests(lambda m: None)
        self.assertEqual(clock.sleeps, 0)

    def test_note_turn_moves_the_marker(self):
        self.assertIsNone(tools.last_turn_at())
        tools.note_turn()
        self.assertIsNotNone(tools.last_turn_at())


class SharedHelpers(SandboxCase):
    def test_models_by_size_and_smallest(self):
        data = {"models": [{"name": "b", "min_free_vram_gb": 14}, {"name": "a", "min_free_vram_gb": 6},
                           {"name": "c"}]}
        self.assertEqual([m["name"] for m in tools.models_by_size(data)], ["c", "a", "b"])
        self.assertEqual(tools.smallest_model(data)["name"], "c")

    def test_release_margin(self):
        self.assertEqual(tools.release_margin("measured"), 0.1)
        self.assertEqual(tools.release_margin(None), 0.5)

    def test_load_first_skips_what_does_not_fit_and_notes_failures(self):
        models = {"a": {"name": "a"}, "b": {"name": "b"}, "c": {"name": "c"}}
        self.patch(tools, "find_model", lambda n, data=None: models.get(n))
        self.patch(tools, "release_check", lambda e, unloading=None, vram=None:
                   {"ok": e["name"] != "a", "summary": "no room"})
        self.patch(tools, "start_model", lambda e: {"ok": e["name"] == "c", "error": "boom"})
        self.patch(tools, "stop_server", lambda: {"ok": True})
        log = []
        self.assertEqual(runner._load_first(["a", "b", "c"], "x", log.append, "auto-recover"), "c")
        self.assertEqual(tools.recent_failures(), {"b"})
        self.assertTrue(any("'a' does not fit" in s for s in log))


class GuardLms(SandboxCase):
    def setUp(self) -> None:
        super().setUp()
        self.hook = guard.make_hook(tools)
        self.patch(tools, "llm_health", lambda timeout=1.5: {"up": False})
        self.patch(tools, "match_model", lambda h: None)
        self.patch(tools, "busy_state", lambda: None)

    def call(self, cmd):
        return self.hook(tool_name="terminal", args={"command": cmd})

    def test_manual_lms_load_blocked_only_when_lmstudio_is_active(self):
        self.patch(tools, "active_engine_id", lambda data=None: "lmstudio")
        self.assertEqual(self.call("lms load qwen-key -y")["action"], "block")
        self.assertEqual(self.call("lms.exe unload --all")["action"], "block")
        self.assertIsNone(self.call("lms load qwen-key --estimate-only"))
        self.assertIsNone(self.call("lms ps --json"))
        self.patch(tools, "active_engine_id", lambda data=None: "llamacpp")
        self.assertIsNone(self.call("lms load qwen-key -y"))


if __name__ == "__main__":
    import unittest
    unittest.main()
