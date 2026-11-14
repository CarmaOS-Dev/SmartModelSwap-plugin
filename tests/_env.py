"""Shared test set-up: import the plugin modules the way runner.py does (flat,
from the plugin directory) and keep every test inside a throw-away Hermes home.
No test loads, stops or signals a real model.

Run:  python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PLUGIN_DIR = Path(__file__).resolve().parent.parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

import tools  # noqa: E402


class SandboxCase(unittest.TestCase):
    """tools.* reads and writes under a temporary home, never the real cache."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.patch(tools, "hermes_home", lambda: self.home)
        tools.apply_cfg({})
        self.addCleanup(tools.apply_cfg, {})

    def patch(self, obj, name, value):
        p = mock.patch.object(obj, name, value)
        p.start()
        self.addCleanup(p.stop)
        return value


class FakeClock:
    """time.time / time.sleep pair that advances only when the code sleeps."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start
        self.sleeps = 0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        self.now += float(seconds)
