"""Guard: gpu_job.sh must run in the background AND with a completion notice."""
from __future__ import annotations

import _env  # noqa: F401  (sets sys.path)
import guard
import tools
from _env import SandboxCase

JOB = ("bash ~/.hermes/plugins/smart-model-swap/skills/smart-model-routing/templates/gpu_job.sh "
       "auto -- bash /tmp/my_job.sh")


class GuardGpuJob(SandboxCase):
    def setUp(self) -> None:
        super().setUp()
        self.hook = guard.make_hook(tools)

    def call(self, **args):
        return self.hook(tool_name="terminal", args=args)

    def guard_log(self) -> str:
        return (tools.cache_dir() / "guard.log").read_text()

    def test_foreground_is_blocked(self):
        res = self.call(command=JOB)
        self.assertEqual(res["action"], "block")
        self.assertIn("BACKGROUND", res["message"])
        self.assertIn("notify=true", res["message"])
        self.assertIn("in the foreground", self.guard_log())

    def test_background_with_patterns_is_blocked(self):
        # What the agent did on 2026-11-12 14:01: the job ended and nobody was told.
        res = self.call(command=JOB, background=True, notify=["^SAVED=|^RESULT job_rc"])
        self.assertEqual(res["action"], "block")
        self.assertIn("completion notification", res["message"])
        self.assertIn("without a completion notification", self.guard_log())

    def test_background_without_notify_is_blocked(self):
        self.assertEqual(self.call(command=JOB, background=True)["action"], "block")

    def test_background_with_notify_true_passes(self):
        self.assertIsNone(self.call(command=JOB, background=True, notify=True))

    def test_legacy_alias_and_heartbeat_pass(self):
        self.assertIsNone(self.call(command=JOB, background=True, notify_on_complete=True))
        self.assertIsNone(self.call(command=JOB, background=True, heartbeat=120))

    def test_reading_the_template_is_not_running_it(self):
        self.assertIsNone(self.call(command="ls -la templates/; head -40 templates/gpu_job.sh"))
        self.assertIsNone(self.call(command="cat ~/x/gpu_job.sh | grep MODE"))

    def test_env_prefix_and_direct_execution_count_as_running(self):
        self.assertEqual(self.call(command="TASK_GPU=0 bash /x/gpu_job.sh none -- bash j.sh")["action"], "block")
        self.assertEqual(self.call(command="/x/gpu_job.sh auto -- bash j.sh")["action"], "block")

    def test_hermes_sms_next_to_the_job_is_still_checked(self):
        res = self.call(command="hermes sms check && bash /x/gpu_job.sh auto -- bash j.sh")
        self.assertEqual(res["action"], "block")

    def test_guard_off_blocks_nothing(self):
        tools.apply_cfg({"guard_tools": False})
        self.assertIsNone(self.call(command=JOB))


class GuardNotifies(SandboxCase):
    def test_notifies(self):
        self.assertTrue(guard.notifies({"notify": True}))
        self.assertTrue(guard.notifies({"notify": "true"}))
        self.assertTrue(guard.notifies({"notify_on_complete": True}))
        self.assertTrue(guard.notifies({"heartbeat": 60}))
        self.assertFalse(guard.notifies({}))
        self.assertFalse(guard.notifies({"notify": False}))
        self.assertFalse(guard.notifies({"notify": ["^SAVED="]}))
        self.assertFalse(guard.notifies({"heartbeat": 0}))
        self.assertFalse(guard.notifies({"heartbeat": "soon"}))


class GuardEngine(SandboxCase):
    """The pre-existing rules keep working."""

    def test_pkill_is_blocked_and_reads_are_not(self):
        hook = guard.make_hook(tools)
        self.assertEqual(hook(tool_name="terminal", args={"command": "pkill -f llama-server"})["action"], "block")
        self.assertIsNone(hook(tool_name="terminal", args={"command": "ps aux | grep llama-server"}))
        self.assertIsNone(hook(tool_name="read_file", args={"path": "/x/gpu_job.sh"}))


if __name__ == "__main__":
    import unittest
    unittest.main()
