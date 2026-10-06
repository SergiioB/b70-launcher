#!/usr/bin/env python3
"""Native app-window spawn helper for b70-launcher.

Kept separate so the launcher's request handlers and the window process
management stay independent. Called only on Linux; open_window() in
launcher.py falls back to a browser app window when this cannot run.

Direct run (smoke test against a running launcher):
    python3 appwindow.py --port 7570            # native window, browser fallback
    python3 appwindow.py --browser --port 7570  # force the browser path
    python3 appwindow.py --no-open --port 7570  # print the URL, open nothing

Uses os.posix_spawn with a fixed argument list (no shell, no interpolation).
"""
import argparse
import os
import shutil
import time
import urllib.parse
import webbrowser
from pathlib import Path

# Fast prefilter only: webwindow.py re-probes properly and picks the first
# usable GTK + WebKitGTK pair (4.1, 4.0 or 6.0) for the system python3.
_GI_DIRS = ("/usr/lib/x86_64-linux-gnu/girepository-1.0",
            "/usr/lib/aarch64-linux-gnu/girepository-1.0",
            "/usr/lib64/girepository-1.0", "/usr/lib/girepository-1.0")
_KIT_COMBOS = (("Gtk-3.0", ("WebKit2-4.1", "WebKit2-4.0")),
               ("Gtk-4.0", ("WebKit-6.0",)))
# A child that dies within this window never showed a window: report the
# native path as unavailable so the caller falls back to a browser instead
# of the launcher's watchdog reading it as the user's quit gesture.
_SPAWN_GRACE = 1.0


class ChildProcess:
    """Minimal Popen-like handle around a posix_spawn'd pid."""

    def __init__(self, pid):
        self.pid = pid
        self._rc = None

    @property
    def returncode(self):
        return self._rc

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
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.poll() is not None:
                return self._rc
            time.sleep(0.1)
        raise TimeoutError(f"child {self.pid} did not exit within {timeout}s")


def _typelibs_present():
    for gdir in _GI_DIRS:
        g = Path(gdir)
        for gtk, kits in _KIT_COMBOS:
            if (g / f"{gtk}.typelib").is_file() and any(
                    (g / f"{k}.typelib").is_file() for k in kits):
                return True
    return False


def _display_reachable(env):
    # Cheap guess; the spawn-grace poll below is the real arbiter. A bare
    # wayland-* socket counts: libwayland uses the default socket name even
    # when WAYLAND_DISPLAY is not exported.
    if env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"):
        return True
    runtime = env.get("XDG_RUNTIME_DIR")
    return bool(runtime) and any(Path(runtime).glob("wayland-*"))


def open_app_window(script, url, port, title):
    """Spawn webwindow.py with the system python3. Returns a ChildProcess,
    or None when the native window is unavailable (caller opens a browser)."""
    py = shutil.which("python3") or shutil.which("python")
    script = Path(script)
    if not (py and script.is_file() and _typelibs_present()):
        return None
    env = dict(os.environ)
    if not _display_reachable(env):
        return None
    # a frozen launcher poisons these for the system-python child
    for var in ("LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME"):
        env.pop(var, None)
    # XWayland: keeps the app window visible to standard tools (wmctrl,
    # xdotool, ffmpeg x11grab) and identical on X11 sessions. GNOME exports
    # GDK_BACKEND=wayland globally, so this must override, not setdefault.
    # Forced only when X11 exists; a Wayland-only session has no X server.
    if env.get("DISPLAY"):
        env["GDK_BACKEND"] = "x11"
    argv = [py, str(script), url, str(port), title]
    try:
        pid = os.posix_spawn(py, argv, env)
    except OSError:
        return None
    child = ChildProcess(pid)
    deadline = time.monotonic() + _SPAWN_GRACE
    while time.monotonic() < deadline:
        if child.poll() is not None:  # crash-fast spawn -> degrade to browser
            return None
        time.sleep(0.05)
    return child


def _open_browser(url):
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Open the B70 Launcher UI in the native WebKitGTK "
                    "window, or in the system browser when unavailable.")
    ap.add_argument("--port", type=int, default=7570,
                    help="launcher port for http://127.0.0.1:PORT (default 7570)")
    ap.add_argument("--url", help="open this URL instead of http://127.0.0.1:PORT")
    ap.add_argument("--title", default="B70 Launcher", help="window title")
    ap.add_argument("--browser", action="store_true",
                    help="use the system browser even when the native window works")
    ap.add_argument("--no-open", action="store_true",
                    help="print the URL and exit without opening anything")
    args = ap.parse_args(argv)
    url = args.url or f"http://127.0.0.1:{args.port}"
    try:
        port = urllib.parse.urlparse(url).port or args.port
    except ValueError:
        port = args.port
    if args.no_open:
        print(url)
        return 0
    if not args.browser and os.environ.get("B70_LAUNCHER_WINDOW") != "browser":
        child = open_app_window(Path(__file__).with_name("webwindow.py"),
                                url, port, args.title)
        if child is not None:
            try:
                while child.poll() is None:
                    time.sleep(0.25)
            except KeyboardInterrupt:
                child.terminate()
                try:
                    child.wait()
                except TimeoutError:
                    child.kill()
            return child.returncode or 0
        print("app window unavailable; opened the browser instead")
    _open_browser(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
