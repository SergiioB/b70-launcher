#!/usr/bin/env python3
"""Native app window host for b70-launcher (WebKitGTK via system python3).

Spawned by launcher.py with: webwindow.py <url> <port> <title>
Closing this window is the app's quit gesture: the process exits and the
launcher's watchdog performs the graceful shutdown (usage stored, state kept).
"""
import os
import sys
import urllib.request


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: webwindow.py <url> <port> [title]")
    url, port = sys.argv[1], int(sys.argv[2])
    title = sys.argv[3] if len(sys.argv) > 3 else "B70 Launcher"
    try:
        import gi
    except ImportError:
        raise SystemExit("python3-gi is not installed; falling back to browser")
    gi.require_version("Gtk", "3.0")
    gi.require_version("WebKit2", "4.1")
    from gi.repository import Gtk, WebKit2

    win = Gtk.Window(title=title)
    win.set_default_size(1440, 900)
    icon = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "web", "assets", "b70-launcher-128.png")
    if os.path.exists(icon):
        try:
            Gtk.Window.set_default_icon_from_file(icon)
        except Exception:
            pass

    view = WebKit2.WebView()
    settings = view.get_settings()
    settings.set_enable_developer_extras(False)
    view.load_uri(url)

    win.connect("destroy", Gtk.main_quit)  # closing the window must end Gtk.main()
    win.add(view)
    win.show_all()
    win.present()
    Gtk.main()


if __name__ == "__main__":
    main()
