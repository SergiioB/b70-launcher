#!/usr/bin/env python3
"""Native app-window spawn helper for b70-launcher.

Kept separate so the launcher's request handlers and the window process
management stay independent. Called only on Linux; open_window() in
launcher.py falls back to a browser app window when this cannot run.

Uses os.posix_spawn with a fixed argument list (no shell, no interpolation).
"""
import os
import shutil
import time
from pathlib import Path

# gi availability check without a probe process: the WebKitGTK typelibs
# must exist for the system python3 the child will use
_GI_DIRS = ("/usr/lib/x86_64-linux-gnu/girepository-1.0",
            "/usr/lib64/girepository-1.0", "/usr/lib/girepository-1.0")


class ChildProcess:
    """Minimal Popen-like handle around a posix_spawn'd pid."""

    def __init__(self, pid):
        self.pid = pid
        self._rc = None

    def poll(self):
        if self._rc is None:
            try:
                pid, status = os.waitpid(self.pid, os.WNOHANG)
                if pid == self.pid:
                    self._rc = os.waitstatus_to_exitcode(status)
            except ChildProcessError:
                self._rc = 0  # reaped elsewhere
        return self._rc

    def terminate(self):
        try:
            os.kill(self.pid, 15)
        except ProcessLookupError:
            pass

    def kill(self):
        try:
            os.kill(self.pid, 9)
        except ProcessLookupError:
            pass

    def wait(self, timeout=3.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.poll() is not None:
                return self._rc
            time.sleep(0.1)
        return self._rc


def _gtk_typelibs_present():
    for gdir in _GI_DIRS:
        g = Path(gdir)
        if (g / "Gtk-3.0.typelib").is_file() and (g / "WebKit2-4.1.typelib").is_file():
            return True
    return False


def open_app_window(script, url, port, title):
    """Spawn webwindow.py with the system python3. Returns a ChildProcess,
    or None when the native window is unavailable (caller opens a browser)."""
    py = shutil.which("python3") or shutil.which("python")
    script = Path(script)
    if not (py and script.is_file() and _gtk_typelibs_present()):
        return None
    env = dict(os.environ)
    # a frozen launcher poisons these for the system-python child
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("PYTHONPATH", None)
    # XWayland: keeps the app window visible to standard tools (wmctrl,
    # xdotool, ffmpeg x11grab) and identical on X11 sessions. GNOME exports
    # GDK_BACKEND=wayland globally, so this must override, not setdefault.
    env["GDK_BACKEND"] = "x11"
    argv = [py, str(script), url, str(port), title]
    pid = os.posix_spawn(py, argv, env)
    return ChildProcess(pid)
