#!/usr/bin/env python3
"""Native app window host for b70-launcher (WebKitGTK via system python3).

Spawned by launcher.py with: webwindow.py <url> <port> [title]
Closing this window is the app's quit gesture: the process exits and the
launcher's watchdog performs the graceful shutdown (usage stored, state kept).

Orphan safety: PR_SET_PDEATHSIG ends this process if the launcher dies, and
a 4s port probe quits when the launcher's socket is gone in case the death
signal could not be armed (non-Linux host, or parent already gone).
"""
import os
import signal
import socket
import sys
from pathlib import Path

WM_CLASS = "b70-launcher"   # StartupWMClass in packaging/b70-launcher.desktop
ICON_NAME = "b70-launcher"  # hicolor icons installed by packaging/install.sh
ICON_FILE = Path(__file__).resolve().parent / "web" / "assets" / "b70-launcher-256.png"
DEF_W, DEF_H = 1440, 900    # match launcher.py's browser --window-size
MIN_W, MIN_H = 880, 560
PROBE_EVERY_S, GONE_MISSES = 4, 3   # quit ~12s after the launcher's port dies


def _geom_file():
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "b70-launcher" / "window-geometry"


def _geom_load(win):
    try:
        parts = _geom_file().read_text().split()
        w, h = int(parts[0]), int(parts[1])
        if MIN_W <= w <= 8192 and MIN_H <= h <= 8192:
            win.set_default_size(w, h)
        if "max" in parts[2:]:
            win.maximize()
    except Exception:
        pass


def _geom_save(win):
    try:
        try:
            w, h = win.get_size()                       # Gtk3
        except AttributeError:
            w, h = win.get_width(), win.get_height()    # Gtk4
        if w < 200 or h < 200:
            return
        maxed = bool(win.is_maximized()) if hasattr(win, "is_maximized") else False
        f = _geom_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"{w} {h}{' max' if maxed else ''}\n")
    except Exception:
        pass


def _pdeathsig():
    """SIGTERM when the launcher (parent) dies, so no orphan windows survive."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:
        pass


def _trap_signals(win, quit_cb):
    """Save geometry and leave the main loop on SIGTERM/SIGINT. GLib signal
    sources dispatch while Gtk.main() blocks; a Python handler would not."""
    import gi
    from gi.repository import GLib

    def on_sig(*_a):
        _geom_save(win)
        quit_cb()
        return GLib.SOURCE_REMOVE

    try:
        gi.require_version("GLibUnix", "2.0")
        from gi.repository import GLibUnix
        add = lambda sig: GLibUnix.signal_add(GLib.PRIORITY_DEFAULT, sig, on_sig)
    except (ValueError, ImportError):
        add = lambda sig: GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, sig, on_sig)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            add(sig)
        except Exception:
            pass


def _server_probe(port, on_gone):
    """Call on_gone once the launcher port has refused a few connections in a
    row; covers orphans where pdeathsig could not help."""
    if not 1 <= port <= 65535:
        return
    from gi.repository import GLib
    state = {"misses": 0}

    def tick():
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1.5).close()
            state["misses"] = 0
        except OSError:
            state["misses"] += 1
            if state["misses"] >= GONE_MISSES:
                on_gone()
                return False
        return True

    GLib.timeout_add_seconds(PROBE_EVERY_S, tick)


def _pick_toolkit(gi):
    """Return (Gtk, Gdk, WebKit module, is_gtk4) for the first usable pair.
    WebKitGTK 4.1 and 4.0 are the Gtk3 API, 6.0 is the Gtk4 API. The webkit
    namespace is required first: importing it pulls in its Gtk, and a Gtk
    require_version pinned early would make the other GTK unusable."""
    if not hasattr(gi, "require_version"):      # python3-gi too old to steer
        raise SystemExit("python3-gi is too old (no require_version); "
                         "falling back to browser")
    combos = (("WebKit2", "4.1", "3.0", False),
              ("WebKit2", "4.0", "3.0", False),
              ("WebKit", "6.0", "4.0", True))
    pin = os.environ.get("B70_WEBKIT_API", "")
    if pin:
        combos = tuple(c for c in combos if c[1] == pin)
        if not combos:
            raise SystemExit(f"B70_WEBKIT_API={pin!r} is not a supported WebKitGTK API")
    for wk_ns, wk_v, gtk_v, gtk4 in combos:
        try:
            gi.require_version(wk_ns, wk_v)   # ValueError if the typelib is absent
            repo = __import__("gi.repository", fromlist=[wk_ns, "Gtk", "Gdk"])
            wk = getattr(repo, wk_ns)         # lazy import; pulls in its Gtk dep
            gi.require_version("Gtk", gtk_v)  # Gdk tracks the Gtk version
            gi.require_version("Gdk", gtk_v)
            return repo.Gtk, repo.Gdk, wk, gtk4
        except (ValueError, ImportError):
            continue
    raise SystemExit("no usable WebKitGTK found (want 4.1, 4.0 or 6.0); "
                     "falling back to browser")


def _tune_view(view, WebKit, title):
    try:
        view.get_context().set_cache_model(WebKit.CacheModel.DOCUMENT_VIEWER)
    except Exception:
        pass    # WebKitGTK 6.0 moved cache control to the network session
    settings = view.get_settings()
    ver = title.split()[-1] if title.split() and title.split()[-1][0].isdigit() else ""
    for setter, val in (("set_enable_developer_extras", False),  # release: no inspector
                        ("set_enable_javascript", True),          # the UI is fetch()-based
                        ("set_enable_smooth_scrolling", True)):
        try:
            getattr(settings, setter)(val)
        except AttributeError:
            pass
    try:
        settings.set_user_agent_with_application_details("B70 Launcher", ver)
    except Exception:
        pass
    try:
        view.connect("context-menu", lambda *_a: True)  # no browser right-click chrome
    except Exception:
        pass


def _is_reload(keyval, state, Gdk):
    return keyval == Gdk.KEY_F5 or (state & Gdk.ModifierType.CONTROL_MASK
                                    and keyval in (Gdk.KEY_r, Gdk.KEY_R))


def _run_gtk3(Gtk, Gdk, WebKit, url, port, title):
    from gi.repository import GLib
    GLib.set_prgname(WM_CLASS)
    Gdk.set_program_class(WM_CLASS)          # WM_CLASS matches StartupWMClass
    try:
        if Gtk.IconTheme.get_default().has_icon(ICON_NAME):
            Gtk.Window.set_default_icon_name(ICON_NAME)
        elif ICON_FILE.is_file():
            Gtk.Window.set_default_icon_from_file(str(ICON_FILE))
    except Exception:
        pass
    win = Gtk.Window(title=title)
    win.set_default_size(DEF_W, DEF_H)
    win.set_size_request(MIN_W, MIN_H)
    _geom_load(win)
    view = WebKit.WebView()
    _tune_view(view, WebKit, title)

    def on_key(_w, event):
        if _is_reload(event.keyval,
                      event.state & Gtk.accelerator_get_default_mod_mask(), Gdk):
            view.reload()
            return True
        return False

    def on_close(*_a):
        _geom_save(win)
        return False                        # let destroy run -> Gtk.main_quit

    win.connect("key-press-event", on_key)
    win.connect("delete-event", on_close)
    win.connect("destroy", Gtk.main_quit)  # closing the window must end Gtk.main()
    _trap_signals(win, Gtk.main_quit)
    view.load_uri(url)
    win.add(view)
    win.show_all()
    win.present()
    _server_probe(port, Gtk.main_quit)
    Gtk.main()


def _run_gtk4(Gtk, Gdk, WebKit, url, port, title):
    from gi.repository import GLib
    GLib.set_prgname(WM_CLASS)   # Wayland app_id falls back to the prgname
    Gtk.init()
    win = Gtk.Window(title=title)
    win.set_default_size(DEF_W, DEF_H)
    win.set_size_request(MIN_W, MIN_H)
    _geom_load(win)
    win.set_icon_name(ICON_NAME)
    view = WebKit.WebView()
    _tune_view(view, WebKit, title)
    loop = GLib.MainLoop()
    keys = Gtk.EventControllerKey()

    def on_key(_c, keyval, _keycode, state):
        if _is_reload(keyval, state, Gdk):
            view.reload()
            return True
        return False

    def on_close(*_a):
        _geom_save(win)
        loop.quit()
        return False

    keys.connect("key-pressed", on_key)
    win.add_controller(keys)
    win.connect("close-request", on_close)
    _trap_signals(win, loop.quit)
    win.set_child(view)
    view.load_uri(url)
    win.present()
    _server_probe(port, loop.quit)
    loop.run()


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("usage: webwindow.py <url> <port> [title]")
        return
    if len(sys.argv) < 3:
        raise SystemExit("usage: webwindow.py <url> <port> [title]")
    url = sys.argv[1]
    try:
        port = int(sys.argv[2])
    except ValueError:
        raise SystemExit("webwindow.py: <port> must be an integer")
    title = sys.argv[3] if len(sys.argv) > 3 else "B70 Launcher"
    try:
        import gi
    except ImportError:
        raise SystemExit("python3-gi is not installed; falling back to browser")
    Gtk, Gdk, WebKit, gtk4 = _pick_toolkit(gi)
    _pdeathsig()
    try:
        if gtk4:
            _run_gtk4(Gtk, Gdk, WebKit, url, port, title)
        else:
            _run_gtk3(Gtk, Gdk, WebKit, url, port, title)
    except Exception as exc:   # e.g. RuntimeError: Gtk could not be initialized
        raise SystemExit(f"webwindow.py: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    main()
