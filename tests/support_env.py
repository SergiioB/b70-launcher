"""Offline, isolated state for package and real-HTTP tests."""
import atexit
import copy
import os
import shutil
import socket
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TMPROOT = Path(tempfile.mkdtemp(prefix='b70-launcher-tests-')).resolve()
FAKE_HOME = TMPROOT / 'home'
FAKE_STATE = TMPROOT / 'xdg-state'
EMPTY_BIN = TMPROOT / 'empty-bin'
for folder in (FAKE_HOME, FAKE_STATE, EMPTY_BIN):
    folder.mkdir(parents=True)
os.environ['HOME'] = str(FAKE_HOME)
os.environ['XDG_STATE_HOME'] = str(FAKE_STATE)
os.environ['B70_UPDATE_URL'] = f'file://{TMPROOT}/no-such-version.json'
os.environ.pop('DOCKER_HOST', None)
os.environ.pop('DOCKER_CONTEXT', None)
atexit.register(shutil.rmtree, TMPROOT, True)
from b70launcher import catalog, downloads, hardware, paths, runtime, scan
catalog.load()
paths.init_state(7570)


def free_port():
    with socket.socket() as server:
        server.bind(('127.0.0.1', 0))
        return server.getsockname()[1]


def write_file(path, data=b'\0' * 256):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def server_env(home, state, extra=None):
    env = {'HOME': str(home), 'XDG_STATE_HOME': str(state),
           'B70_UPDATE_URL': f'file://{TMPROOT}/no-such-version.json',
           'PATH': str(EMPTY_BIN), 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8'}
    env.update(extra or {})
    return env


class LauncherStateCase(unittest.TestCase):
    _STATE = ((catalog, 'SETTINGS'), (catalog, 'RECIPES'), (catalog, 'RECIPE_REMOTE'),
              (catalog, 'USER_RECIPE_OVERRIDES'), (catalog, 'UPDATE_INFO'), (scan, 'SCAN'),
              (hardware, 'PREFLIGHT_CACHE'), (hardware, 'POWER_PREV'), (hardware, 'VRAM_CACHE'))

    def setUp(self):
        self._saved = [(module, name, copy.deepcopy(getattr(module, name))) for module, name in self._STATE]
        self._running = dict(runtime.RUNNING)
        self._downloads = dict(downloads.DOWNLOADS)
        catalog.SETTINGS['sync_clients'] = False

    def tearDown(self):
        for module, name, saved in self._saved:
            value = getattr(module, name)
            value.clear()
            value.update(saved)
        runtime.RUNNING.clear()
        runtime.RUNNING.update(self._running)
        downloads.DOWNLOADS.clear()
        downloads.DOWNLOADS.update(self._downloads)
        for path in (paths.OVERRIDE_PATH, paths.REMOTE_RECIPES_PATH, paths.USAGE_PATH, paths.STATE_PATH):
            path.unlink(missing_ok=True)
