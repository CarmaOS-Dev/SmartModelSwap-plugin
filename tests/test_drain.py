"""Drain: the model is cut only after the engine stayed idle for drain_quiet_s."""
from __future__ import annotations

import _env  # noqa: F401
import tools
from _env import FakeClock, SandboxCase


class Drain(SandboxCase):
    def run_drain(self, reads, cfg=None, activity=None):
        """`reads`: successive values of requests_processing(); the last repeats."""
        tools.apply_cfg(cfg or {})
        clock = FakeClock()
        self.patch(tools.time, "time", clock.time)
        self.patch(tools.time, "sleep", clock.sleep)
        seq = list(reads)
        seen = []

        def in_flight():
            v = seq.pop(0) if len(seq) > 1 else seq[0]
            seen.append((clock.now, v))
            return v
        acts = list(activity or [0])
        self.patch(tools, "requests_processing", in_flight)
        self.patch(tools, "llm_activity", lambda: acts.pop(0) if len(acts) > 1 else acts[0])
        lines = []
        t0 = clock.now
        tools.drain_requests(lines.append)
        return clock.now - t0, lines, seen

    def test_idle_engine_waits_the_quiet_window_in_silence(self):
        waited, lines, _ = self.run_drain([0])
        self.assertEqual(waited, 8)
        self.assertEqual(lines, [])

    def test_request_starting_inside_the_quiet_window_is_drained(self):
        # 2026-11-12 13:50: idle at the first read, a request 3 s later.
        reads = [0, 0, 0, 1, 1, 1, 1, 1, 0]
        waited, lines, seen = self.run_drain(reads, activity=list(range(100)))
        self.assertTrue(any("started during the quiet window" in ln for ln in lines))
        self.assertTrue(lines[-1].startswith("drained in"))
        last_busy = max(t for t, v in seen if v)
        self.assertGreaterEqual(seen[-1][0] - last_busy, 8)

    def test_gap_between_two_requests_of_a_turn_does_not_end_the_drain(self):
        reads = [1, 1, 0, 0, 0, 1, 1, 0]          # a 3 s tool-call gap
        _, lines, seen = self.run_drain(reads, activity=list(range(100)))
        self.assertEqual(sum(ln.startswith("drained in") for ln in lines), 1)
        last_busy = max(t for t, v in seen if v)
        self.assertGreaterEqual(seen[-1][0] - last_busy, 8)

    def test_quiet_zero_keeps_the_old_behaviour(self):
        waited, lines, _ = self.run_drain([0], cfg={"drain_quiet_s": 0})
        self.assertEqual((waited, lines), (0, []))

    def test_engine_down_returns_at_once(self):
        waited, lines, _ = self.run_drain([None])
        self.assertEqual((waited, lines), (0, []))

    def test_one_failed_read_does_not_cut_a_live_request(self):
        reads = [1, None, 1, 1, 0]
        _, lines, _ = self.run_drain(reads, activity=list(range(100)))
        self.assertFalse(any("stopped answering" in ln for ln in lines))
        self.assertTrue(lines[-1].startswith("drained in"))

    def test_engine_gone_mid_drain_stops_waiting(self):
        _, lines, _ = self.run_drain([1, None])
        self.assertIn("stopped answering", lines[-1])

    def test_stalled_request_is_cut_after_the_stall_timeout(self):
        waited, lines, _ = self.run_drain([1], cfg={"drain_timeout_s": 30, "drain_max_s": 600})
        self.assertIn("no progress for 30s", lines[-1])
        self.assertEqual(waited, 30)

    def test_long_generation_is_cut_at_the_cap(self):
        waited, lines, _ = self.run_drain([1], cfg={"drain_timeout_s": 30, "drain_max_s": 90},
                                          activity=list(range(1000)))
        self.assertIn("still generating after 90s", lines[-1])
        self.assertEqual(waited, 90)

    def test_drain_disabled(self):
        waited, lines, _ = self.run_drain([1], cfg={"drain_timeout_s": 0})
        self.assertEqual((waited, lines), (0, []))


if __name__ == "__main__":
    import unittest
    unittest.main()
