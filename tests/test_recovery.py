"""Watcher and recovery helpers: slow engine, stop by signal, failures, logs, argv."""
from __future__ import annotations

import os
import time

import _env  # noqa: F401
import runner
import tools
from _env import FakeClock, SandboxCase


class Helpers(SandboxCase):
    def test_dedupe_switches_drops_only_repeated_bare_switches(self):
        args = ["-ngl", "99", "--jinja", "--metrics", "--temp", "-1", "--lora", "a.gguf",
                "--lora", "b.gguf", "--metrics", "-c", "77824", "--jinja"]
        self.assertEqual(tools.dedupe_switches(args),
                         ["-ngl", "99", "--jinja", "--metrics", "--temp", "-1", "--lora", "a.gguf",
                          "--lora", "b.gguf", "-c", "77824"])

    def test_dedupe_switches_keeps_a_clean_argv(self):
        args = ["-c", "4096", "--metrics", "-ts", "0.5,0.5", "--no-warmup"]
        self.assertEqual(tools.dedupe_switches(args), args)

    def test_rotate_log_keeps_one_generation(self):
        log = tools.cache_dir() / "runner.log"
        log.write_bytes(b"x" * 100)
        tools.rotate_log(log, max_bytes=1000)
        self.assertTrue(log.exists())
        tools.rotate_log(log, max_bytes=50)
        self.assertFalse(log.exists())
        self.assertEqual((tools.cache_dir() / "runner.log.1").stat().st_size, 100)
        tools.rotate_log(log, max_bytes=50)  # a missing file is not an error

    def test_stopped_by_signal_reads_the_engine_log_tail(self):
        tools.model_log("signal").write_text(
            "srv  llama_server: listening on http://127.0.0.1:8080\n"
            "srv    operator(): operator(): cleaning up before exit...\n")
        tools.model_log("qwen").write_text("ggml_abort: GGML_BACKEND_SPLIT_AXIS_UNKNOWN\n")
        self.assertTrue(tools.stopped_by_signal("signal"))
        self.assertFalse(tools.stopped_by_signal("qwen"))
        self.assertFalse(tools.stopped_by_signal("never-started"))

    def test_note_failure_drops_expired_records(self):
        old = time.time() - tools.FAILURE_WINDOW_S - 60
        tools.write_state({"failures": [{"model": "signal", "at": old, "why": "server died"}]})
        tools.note_failure("qwen", "load failed")
        self.assertEqual([f["model"] for f in tools.read_state()["failures"]], ["qwen"])
        self.assertEqual(tools.recent_failures(), {"qwen"})

    def test_server_unresponsive_needs_a_live_pid(self):
        self.assertIsNone(tools.server_unresponsive())
        tools.write_state({"server": {"model": "signal", "pid": os.getpid()}})
        self.assertEqual(tools.server_unresponsive(), "signal")
        self.assertIsNone(tools.crashed_model())
        tools.write_state({"server": {"model": "signal", "pid": 2 ** 22 + 12345}})
        self.assertIsNone(tools.server_unresponsive())
        self.assertEqual(tools.crashed_model(), "signal")


class Stop(Exception):
    pass


class Watcher(SandboxCase):
    """runner.idle_watch with the engine faked: `healths` is one entry per poll."""

    def watch(self, healths, *, state_server, log_text="", failures=()):
        tools.apply_cfg({"primary_model": "signal"})
        tools.write_state({"server": state_server, "failures": list(failures)})
        if log_text:
            tools.model_log("signal").write_text(log_text)
        clock = FakeClock()
        seq = list(healths)

        def health(timeout=3.0):
            if not seq:
                raise Stop()
            return seq.pop(0)
        launched = []
        self.patch(tools.time, "time", clock.time)
        self.patch(tools.time, "sleep", clock.sleep)
        self.patch(tools, "llm_health", health)
        self.patch(tools, "match_model", lambda h: h.get("model") if h.get("up") else None)
        self.patch(tools, "llm_activity", lambda: 0)
        self.patch(tools, "busy_state", lambda: None)
        self.patch(tools, "gpu_job_active", lambda: False)
        self.patch(tools, "recovery_candidates", lambda dead, prefer=None: ["qwen"])
        self.patch(tools, "h_restore", lambda a: launched.append("restore") or '{"ok": true}')
        self.patch(tools, "h_swap", lambda a: launched.append("swap " + a["model"]) or '{"ok": true}')
        said = []
        self.patch(runner, "print", lambda *a, **k: said.append(" ".join(map(str, a))), )
        try:
            rc = runner.idle_watch("signal")
        except Stop:
            rc = "still watching"
        return rc, said, launched

    UP = {"up": True, "model": "signal"}
    DOWN = {"up": False}

    def test_slow_engine_is_not_a_stop(self):
        # 2026-11-12 20:52: /health timed out under memory pressure, the pid was alive.
        rc, said, launched = self.watch([self.DOWN] * 6 + [self.UP],
                                        state_server={"model": "signal", "pid": os.getpid()})
        self.assertEqual(rc, "still watching")
        self.assertEqual(launched, [])
        self.assertFalse(any("stopped on purpose" in s for s in said))
        self.assertTrue(any("does not answer /health" in s for s in said))
        self.assertTrue(any("answers again" in s for s in said))

    def test_one_missed_probe_is_silent(self):
        rc, said, _ = self.watch([self.DOWN, self.UP],
                                 state_server={"model": "signal", "pid": os.getpid()})
        self.assertEqual(rc, "still watching")
        self.assertFalse(any("does not answer" in s or "answers again" in s for s in said))

    def test_deliberate_stop_ends_the_watcher(self):
        rc, said, launched = self.watch([self.DOWN], state_server=None)
        self.assertEqual(rc, 0)
        self.assertEqual(launched, [])
        self.assertTrue(any("stopped on purpose" in s for s in said))

    def test_primary_stopped_by_a_signal_is_loaded_again(self):
        # 2026-11-12 01:12: clean shutdown 11 s after loading, the agent spent 53 min on the small model.
        rc, said, launched = self.watch(
            [self.DOWN, self.DOWN], state_server={"model": "signal", "pid": 2 ** 22 + 12345},
            log_text="srv    operator(): operator(): cleaning up before exit...\n")
        self.assertEqual((rc, launched), (0, ["restore"]))
        self.assertTrue(any("stopped by a signal" in s for s in said))
        self.assertEqual(tools.recent_failures(), {"signal"})

    def test_second_stop_in_the_window_goes_to_another_model(self):
        rc, _, launched = self.watch(
            [self.DOWN, self.DOWN], state_server={"model": "signal", "pid": 2 ** 22 + 12345},
            log_text="srv    operator(): operator(): cleaning up before exit...\n",
            failures=[{"model": "signal", "at": FakeClock().now - 60, "why": "stopped by a signal"}])
        self.assertEqual((rc, launched), (0, ["swap qwen"]))

    def test_crash_goes_to_another_model(self):
        rc, said, launched = self.watch(
            [self.DOWN, self.DOWN], state_server={"model": "signal", "pid": 2 ** 22 + 12345},
            log_text="ggml_abort: out of memory\n")
        self.assertEqual((rc, launched), (0, ["swap qwen"]))
        self.assertTrue(any("died on its own" in s for s in said))


if __name__ == "__main__":
    import unittest
    unittest.main()
