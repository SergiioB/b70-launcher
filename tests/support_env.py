"""Shared test environment for the b70-launcher suite.

launcher.py resolves all of its state paths at IMPORT time:
  DATA = XDG_STATE_HOME/b70-launcher   (logs, overrides, usage, remote overlay)
and starts a daemon thread that fetches B70_UPDATE_URL over the network.

Every test module MUST import this module before importing launcher. It:
  - points HOME and XDG_STATE_HOME at a fresh temp dir so ~ expansion,
    state writes, and scan defaults never touch the real user profile;
  - points B70_UPDATE_URL at a file:// path that fails instantly, so the
    update check opens no socket;
  - inserts the repo root on sys.path so `import launcher` works.
"""
import atexit
import copy
import os
import shutil
import socket
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # the b70-launcher checkout
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TMPROOT = Path(tempfile.mkdtemp(prefix="b70-launcher-tests-")).resolve()
FAKE_HOME = TMPROOT / "home"
FAKE_STATE = TMPROOT / "xdg-state"
EMPTY_BIN = TMPROOT / "empty-bin"  # PATH mask for subprocesses: hides docker/sudo/pkill
for d in (FAKE_HOME, FAKE_STATE, EMPTY_BIN):
    d.mkdir(parents=True, exist_ok=True)

os.environ["HOME"] = str(FAKE_HOME)
os.environ["XDG_STATE_HOME"] = str(FAKE_STATE)
os.environ["B70_UPDATE_URL"] = f"file://{TMPROOT}/no-such-version.json"
os.environ.pop("DOCKER_HOST", None)
os.environ.pop("DOCKER_CONTEXT", None)

atexit.register(shutil.rmtree, TMPROOT, True)

import launcher  # noqa: E402  (env above must be set first)


def free_port():
    """An emphemeral high port, verified bindable, never 8765 or <1024."""
    while True:
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", 0))
            p = s.getsockname()[1]
        finally:
            s.close()
        if p >= 1024 and p != 8765:
            return p


def write_file(path, data=b"\0" * 256):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def server_env(home, state, extra=None):
    """Minimal env for the real server subprocess (test_http.py).

    PATH points at an empty dir so shutil.which() finds no docker, sudo,
    lspci, pkill, fuser, xdg-open or terminal emulators inside the child —
    every external-command path in launcher.py degrades to a safe no-op.
    """
    env = {
        "HOME": str(home),
        "XDG_STATE_HOME": str(state),
        "B70_UPDATE_URL": f"file://{TMPROOT}/no-such-version.json",
        "PATH": str(EMPTY_BIN),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    if extra:
        env.update(extra)
    return env


class LauncherStateCase(unittest.TestCase):
    """Snapshot/restore every mutable launcher global a test might touch.

    RECIPES, SETTINGS, SCAN, RECIPE_REMOTE and USER_RECIPE_OVERRIDES are
    deep-copied per test; RUNNING/DOWNLOADS shallow-copied; state files the
    code writes under DATA are removed. Tests may mutate freely.
    """

    _SNAP = ("SETTINGS", "RECIPES", "SCAN", "RECIPE_REMOTE",
             "USER_RECIPE_OVERRIDES", "UPDATE_INFO")

    def setUp(self):
        self._saved = {k: copy.deepcopy(getattr(launcher, k)) for k in self._SNAP}
        self._running = dict(launcher.RUNNING)
        self._downloads = dict(launcher.DOWNLOADS)

    def tearDown(self):
        for k, v in self._saved.items():
            g = getattr(launcher, k)
            g.clear()
            g.update(v)
        launcher.RUNNING.clear()
        launcher.RUNNING.update(self._running)
        launcher.DOWNLOADS.clear()
        launcher.DOWNLOADS.update(self._downloads)
        for p in (launcher.OVERRIDE_PATH, launcher.REMOTE_RECIPES_PATH,
                  launcher.USAGE_PATH, launcher.STATE_PATH):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
