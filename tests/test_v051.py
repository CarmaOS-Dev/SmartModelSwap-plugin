"""0.5.1: detached runners, interrupted swaps, idle ComfyUI, blocking CLI in the
foreground, locked state, swap budget. No real model is loaded or stopped."""
from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time

import _env  # noqa: F401
import guard
import runner
import tools
from _env import FakeClock, SandboxCase


class Detached(SandboxCase):
    def test_service_when_user_systemd(self):
        calls = []

        def run(cmd, timeout=10.0, env=None):
            calls.append((cmd, env))
            out = "4242\n" if cmd[:2] == ["systemctl", "--user"] else ""
            return type("R", (), {"returncode": 0, "stdout": out})()
        self.patch(tools, "_isolated", lambda: True)
        self.patch(tools, "_run", run)
        os.environ["OPENAI_API_KEY"] = "sk-test-secret"
        self.addCleanup(os.environ.pop, "OPENAI_API_KEY", None)
        out = tools._detached(["python3", "runner.py", "x"], "sms-runner-", "restore",
                              tools.log_file())
        sd = calls[0][0]
        self.assertEqual(sd[:2], ["systemd-run", "--user"])
        self.assertNotIn("--scope", sd)  # a service: its parent is systemd, not the caller
        self.assertTrue(out["unit"].startswith("sms-runner-restore-") and out["unit"].endswith(".service"))
        self.assertIn(f"StandardOutput=append:{tools.log_file()}", sd)
        self.assertIn(f"--setenv=HERMES_HOME={self.home}", sd)
        self.assertFalse(any("sk-test-secret" in a for a in sd))
        self.assertEqual(sd[sd.index("--") + 1:], ["python3", "runner.py", "x"])
        self.assertEqual(out["pid"], 4242)

    def test_double_fork_is_not_our_child(self):
        self.patch(tools, "_isolated", lambda: False)
        out = tools._detached([sys.executable, "-c", "import time; time.sleep(3)"],
                              "sms-runner-", "t", tools.cache_dir() / "t.log")
        pid = out["pid"]
        self.assertIsNone(out["unit"])
        self.assertTrue(pid and tools._pid_alive(pid))
        ppid = int(tools._proc_stat(pid)[1])
        self.assertNotEqual(ppid, os.getpid())  # re-parented: a tree kill of ours misses it
        os.kill(pid, 9)

    def test_child_env_drops_secrets(self):
        os.environ["ANTHROPIC_API_KEY"] = "x"
        self.addCleanup(os.environ.pop, "ANTHROPIC_API_KEY", None)
        env = tools.child_env()
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertEqual(env["HERMES_HOME"], str(self.home))
        self.assertIn("PATH", env)


class RunnerLiveness(SandboxCase):
    def test_spawn_grace_then_stale(self):
        clock = FakeClock(time.time())
        self.patch(tools.time, "time", clock.time)
        tools.write_state({"state": "grace", "action": "restore", "target": "signal",
                           "runner_pid": None, "spawned_at": clock.now})
        self.assertEqual(tools.busy_state(), "grace")
        self.assertIsNone(tools.runner_died_midflight())
        clock.now += tools.SPAWN_GRACE_S + 1
        self.assertIsNone(tools.busy_state())
        self.assertEqual(tools.runner_died_midflight()["action"], "restore")
        self.assertTrue(tools.swap_view()["stale"])

    def test_recycled_pid_is_not_the_runner(self):
        tools.write_state({"state": "restoring", "runner_pid": os.getpid(),
                           "runner_start": (tools.proc_start(os.getpid()) or 0) + 1})
        self.assertIsNone(tools.busy_state())


class ResumeLost(SandboxCase):
    def setUp(self) -> None:
        super().setUp()
        tools.apply_cfg({"primary_model": "signal"})
        self.launched = []
        self.patch(tools, "llm_health", lambda timeout=3.0: {"up": False})
        self.patch(tools, "match_model", lambda h: None)
        self.patch(tools, "h_restore", lambda a: self.launched.append("restore") or '{"ok": true}')
        self.patch(tools, "h_swap", lambda a: self.launched.append("swap " + a["model"]) or '{"ok": true}')
        self.said = []

    def test_killed_restore_is_resumed_without_a_failure(self):
        # 2026-11-13 14:04: the terminal timeout killed `hermes sms restore --wait`.
        tools.write_state({"state": "restoring", "action": "restore", "target": "signal",
                           "runner_pid": 2 ** 22 + 7, "server": None})
        lost = tools.runner_died_midflight()
        self.assertTrue(runner._resume_lost(lost, "signal", time.time(), self.said.append))
        self.assertEqual(self.launched, ["restore"])
        self.assertEqual(tools.recent_failures(), set())
        self.assertTrue(any("not a model failure" in s.lower() for s in self.said))

    def test_resumed_only_once(self):
        tools.write_state({"state": "swapping", "action": "swap", "target": "qwen",
                           "runner_pid": 2 ** 22 + 7, "resumed": {"target": "qwen", "at": time.time()}})
        self.assertFalse(runner._resume_lost(tools.runner_died_midflight(), "signal", time.time(),
                                             self.said.append))
        self.assertEqual(self.launched, [])
        self.assertEqual(tools.read_state()["state"], "failed")

    def test_model_came_up_anyway(self):
        self.patch(tools, "llm_health", lambda timeout=3.0: {"up": True})
        self.patch(tools, "match_model", lambda h: "signal")
        tools.write_state({"state": "restoring", "action": "restore", "target": "signal",
                           "runner_pid": 2 ** 22 + 7})
        self.assertFalse(runner._resume_lost(tools.runner_died_midflight(), "signal", time.time(),
                                             self.said.append))
        self.assertEqual(self.launched, [])
        st = tools.read_state()
        self.assertEqual((st["state"], st["current"]), ("idle", "signal"))


class Stop(Exception):
    pass


class IdleComfy(SandboxCase):
    """Idle restore on the small model with ComfyUI open."""

    def run_watch(self, comfy_seq, fits_after_free=True):
        tools.apply_cfg({"primary_model": "signal", "idle_restore_min": 1})
        clock = FakeClock()
        polls = {"n": 0}

        def health(timeout=3.0):
            polls["n"] += 1
            if polls["n"] > 40:
                raise Stop()
            return {"up": True, "model": "qwen"}
        seq = list(comfy_seq)

        def comfy(max_age=0.0):
            return seq.pop(0) if len(seq) > 1 else seq[0]
        freed = []
        fits = {"ok": False}

        def check(name, wait_s=0.0, poll_s=5.0):
            return {"ok": fits["ok"], "summary": "GPU0 7.0/14.5 GiB"}

        def free():
            freed.append(1)
            fits["ok"] = fits_after_free
            return {"ok": True}
        launched = []
        for mod in (tools.time, runner.time):
            self.patch(mod, "time", clock.time)
            self.patch(mod, "sleep", clock.sleep)
        self.patch(tools, "llm_health", health)
        self.patch(tools, "match_model", lambda h: h.get("model") if h.get("up") else None)
        self.patch(tools, "llm_activity", lambda: 0)
        self.patch(tools, "requests_processing", lambda: 0)
        self.patch(tools, "busy_state", lambda: None)
        self.patch(tools, "gpu_job_active", lambda: False)
        self.patch(tools, "comfy_status", comfy)
        self.patch(tools, "check_release", check)
        self.patch(tools, "comfy_free", free)
        self.patch(tools, "h_restore", lambda a: launched.append("restore") or '{"ok": true}')
        said = []
        self.patch(runner, "print", lambda *a, **k: said.append(" ".join(map(str, a))))
        try:
            rc = runner.idle_watch("qwen")
        except Stop:
            rc = "still watching"
        return rc, said, launched, freed

    IDLE = {"up": True, "busy": False, "running": 0, "pending": 0}
    BUSY = {"up": True, "busy": True, "running": 1, "pending": 0}

    def test_idle_comfy_is_freed_and_primary_restored(self):
        rc, said, launched, freed = self.run_watch([self.IDLE])
        self.assertEqual((rc, launched, len(freed)), (0, ["restore"], 1))
        self.assertTrue(any("/free" in s for s in said))

    def test_busy_comfy_holds_the_clock(self):
        rc, said, launched, freed = self.run_watch([self.BUSY])
        self.assertEqual((rc, launched, freed), ("still watching", [], []))
        self.assertTrue(any("ComfyUI is working" in s for s in said))

    def test_free_disabled(self):
        tools.apply_cfg({"comfy_free_on_idle": False})
        self.patch(tools, "apply_cfg", lambda c: None)  # keep it off inside run_watch
        tools._CFG.update(primary_model="signal", idle_restore_min=1, comfy_free_on_idle=False)
        rc, _, launched, freed = self.run_watch([self.IDLE])
        self.assertEqual((launched, freed), ([], []))


class GuardBlocking(SandboxCase):
    def setUp(self) -> None:
        super().setUp()
        self.hook = guard.make_hook(tools)
        self.patch(tools, "llm_health", lambda timeout=1.5: {"up": True})
        self.patch(tools, "match_model", lambda h: "qwen")
        self.patch(tools, "busy_state", lambda: None)

    def call(self, **args):
        return self.hook(tool_name="terminal", args=args)

    def test_foreground_restore_wait_is_blocked(self):
        res = self.call(command="hermes sms check && hermes sms restore --wait")
        self.assertEqual(res["action"], "block")
        self.assertIn("sms_restore", res["message"])

    def test_blocking_forms(self):
        for cmd in ("hermes sms swap qwen --wait", "hermes sms wait --timeout 170",
                    "hermes sms after-job", "hermes sms stop --graceful"):
            self.assertEqual(self.call(command=cmd)["action"], "block", cmd)

    def test_background_with_notify_is_allowed(self):
        self.assertIsNone(self.call(command="hermes sms restore --wait", background=True, notify=True))

    def test_safe_sms_commands_pass(self):
        for cmd in ("hermes sms status", "hermes sms check", "hermes sms restore",
                    "hermes sms restore --wait --dry-run", "hermes sms job-layout --plain"):
            self.assertIsNone(self.call(command=cmd), cmd)

    def test_sms_mention_no_longer_hides_a_kill(self):
        res = self.call(command="echo hermes sms; pkill -f llama-server")
        self.assertEqual(res["action"], "block")

    def test_comfy_blocked_with_the_small_model(self):
        res = self.call(command="cd ~/ComfyUI && venv/bin/python main.py --listen", background=True)
        self.assertEqual(res["action"], "block")
        self.assertIn("'qwen' holds the GPU", res["message"])

    def test_comfy_inside_gpu_job_passes(self):
        cmd = ("bash ~/.hermes/plugins/smart-model-swap/skills/smart-model-routing/templates/gpu_job.sh "
               "auto -- bash -c 'cd ~/ComfyUI && python main.py --listen'")
        self.assertIsNone(self.call(command=cmd, background=True, notify=True))

    def test_mentioning_gpu_job_does_not_allow_comfy(self):
        res = self.call(command="cat gpu_job.sh; cd ~/ComfyUI && python main.py --listen", background=True)
        self.assertEqual(res["action"], "block")


def _bump(home: str, n: int) -> None:
    os.environ["HERMES_HOME"] = home
    sys.path.insert(0, str(_env.PLUGIN_DIR))
    import tools as t  # noqa: PLC0415
    for _ in range(n):
        t.update_state(lambda st: st.update(count=int(st.get("count") or 0) + 1))


class StateLock(SandboxCase):
    def test_concurrent_writers_lose_nothing(self):
        ctx = multiprocessing.get_context("fork")
        self.patch(tools, "hermes_home", tools.hermes_home)  # children use HERMES_HOME
        procs = [ctx.Process(target=_bump, args=(str(self.home), 50)) for _ in range(4)]
        os.environ["HERMES_HOME"] = str(self.home)
        for p in procs:
            p.start()
        for p in procs:
            p.join(30)
        os.environ.pop("HERMES_HOME", None)
        st = json.loads((self.home / "cache" / "smart-model-swap" / "state.json").read_text())
        self.assertEqual(st["count"], 200)

    def test_corrupt_state_is_kept_aside(self):
        tools.state_file().write_text("{not json")
        tools.write_state({"state": "idle"})
        self.assertTrue((tools.cache_dir() / "state.json.corrupt").exists())
        self.assertEqual(tools.read_state()["state"], "idle")


class Budget(SandboxCase):
    def test_budget_covers_the_worst_case(self):
        tools.apply_cfg({})
        b = tools.swap_budget_s({"load_timeout_s": 420})
        self.assertEqual(b, 10 + 600 + 45 + 90 + 120 + 420 + 60)
        self.assertGreater(b, 900)  # the old fixed --timeout was too short

    def test_wait_idle_ignores_an_old_result(self):
        self.patch(tools, "busy_state", lambda: None)
        self.patch(tools, "llm_health", lambda timeout=3.0: {"up": False})
        tools.write_state({"last_result": {"ok": True, "at": "2026-11-13T10:00:00"}})
        out = tools.wait_idle(5, "signal", since="2026-11-13T14:01:20")
        self.assertFalse(out["ok"])
        self.assertIn("without reporting", out["error"])


if __name__ == "__main__":
    import unittest
    unittest.main()
