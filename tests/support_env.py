"""Offline, isolated state for package and real-HTTP tests."""
import atexit
import copy
import os
import shutil
import socket
import sys
import tempfile
import unittest
from unittest import mock
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


def _nondisplay_preflight(real):
    """Deterministic display state for build tests: the dev host's GPU 0 may
    drive the desktop, which would cap gmu/ctx in recipe-default assertions.
    Only the default /sys probe is normalized — a fake sysfs_root or injected
    clients (the display-detection tests) pass through untouched."""
    def wrapper(*args, **kwargs):
        if kwargs.get('sysfs_root', '/sys') != '/sys' or kwargs.get('clients') is not None:
            return real(*args, **kwargs)
        data = real(*args, **kwargs)
        for dev in data.get('devices') or ():
            dev['is_display'] = False
            dev['desktop_vram_mib'] = 0.0
        for card in (data.get('profile') or {}).get('cards') or ():
            card['is_display'] = False
            card['desktop_vram_mib'] = 0.0
        return data
    return wrapper


class LauncherStateCase(unittest.TestCase):
    _STATE = ((catalog, 'SETTINGS'), (catalog, 'RECIPES'), (catalog, 'RECIPE_REMOTE'),
              (catalog, 'USER_RECIPE_OVERRIDES'), (catalog, 'UPDATE_INFO'), (scan, 'SCAN'),
              (hardware, 'PREFLIGHT_CACHE'), (hardware, 'POWER_PREV'), (hardware, 'VRAM_CACHE'))

    def setUp(self):
        self._saved = [(module, name, copy.deepcopy(getattr(module, name))) for module, name in self._STATE]
        self._running = dict(runtime.RUNNING)
        self._downloads = dict(downloads.DOWNLOADS)
        patcher = mock.patch.object(
            hardware, 'hardware_preflight',
            _nondisplay_preflight(hardware.hardware_preflight))
        patcher.start()
        self.addCleanup(patcher.stop)
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
        for path in (paths.OVERRIDE_PATH, paths.REMOTE_RECIPES_PATH, paths.USAGE_PATH, paths.STATE_PATH,
                     runtime.LOAD_HISTORY_PATH):
            path.unlink(missing_ok=True)
        runtime._LOAD_HISTORY = None
