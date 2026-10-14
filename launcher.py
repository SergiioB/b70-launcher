#!/usr/bin/env python3
"""Desktop entry point. Importing it never starts a daemon or writes state."""
import argparse
import json
import os
import signal
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http.server import ThreadingHTTPServer

from b70launcher.paths import VERSION


def main():
    from b70launcher import catalog, http_api, paths, runtime, scan
    catalog.load()
    parser = argparse.ArgumentParser(description='B70 model launcher')
    parser.add_argument('--port', type=int, default=catalog.SETTINGS.get('port', 7570))
    parser.add_argument('--no-open', action='store_true')
    parser.add_argument('--browser', action='store_true')
    parser.add_argument('--kill', '--force', action='store_true', help='Gracefully replace an authenticated launcher on this port')
    parser.add_argument('--version', action='version', version=VERSION)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    url = f'http://127.0.0.1:{args.port}'
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), http_api.Handler)
    except OSError as exc:
        token = paths.read_token(args.port)
        try:
            req = urllib.request.Request(url + '/api/state', headers={'X-Launcher-Token': token})
            with urllib.request.urlopen(req, timeout=2) as response:
                data = json.loads(response.read(4 << 20))
            if not data.get('version') or 'recipes' not in data:
                raise ValueError('not a launcher')
            if not args.kill:
                if not args.no_open:
                    request = urllib.request.Request(url + '/api/window', b'{}', {'X-Launcher-Token': token, 'Content-Type': 'application/json'})
                    with urllib.request.urlopen(request, timeout=2) as response:
                        target = json.loads(response.read())['url']
                    child = None
                    if not args.browser and os.environ.get('B70_LAUNCHER_WINDOW', 'auto') != 'browser':
                        import appwindow
                        child = appwindow.open_app_window(paths.HERE / 'webwindow.py', target, args.port,
                                                          f'B70 Launcher {data["version"]}')
                    if child is None:
                        webbrowser.open(target)
                    else:
                        # Keep the window's parent alive. Closing this extra window
                        # must not stop the already-running daemon or its engines.
                        child.wait()
                print(f'Launcher already running at {url}; existing instance left unchanged.')
                return
            request = urllib.request.Request(url + '/api/shutdown', b'{}', {'X-Launcher-Token': token, 'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=2):
                pass
            deadline = time.monotonic() + 5
            while True:
                try:
                    server = ThreadingHTTPServer(('127.0.0.1', args.port), http_api.Handler)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Existing launcher has not released its port.')
                    time.sleep(.1)
        except Exception as failure:
            raise SystemExit(f'Cannot use {url}: {exc}. No process was killed. Choose another port. ({failure})')
    http_api.API_TOKEN = paths.init_state(args.port)
    paths.prune_logs(protected=paths.persisted_log_paths())
    runtime.SERVER = server
    threading.Thread(target=runtime.adopt_containers, daemon=True).start()
    threading.Thread(target=runtime.monitor, daemon=True).start()
    threading.Thread(target=catalog.check_for_updates, daemon=True).start()
    scan.start_scan_async(force=True)
    print(f'b70-launcher {VERSION} on {url}', flush=True)
    signal.signal(signal.SIGINT, lambda *_: threading.Thread(target=runtime.graceful_shutdown, daemon=True).start())
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=runtime.graceful_shutdown, daemon=True).start())
    child = None
    if not args.no_open:
        target = url + '/?nonce=' + http_api.bootstrap_nonce()
        if not args.browser and os.environ.get('B70_LAUNCHER_WINDOW', 'auto') != 'browser':
            import appwindow
            child = appwindow.open_app_window(paths.HERE / 'webwindow.py', target, args.port, f'B70 Launcher {VERSION}')
        if child is None:
            webbrowser.open(target)
        else:
            def watch():
                child.wait()
                # Agents opened from the launcher reach their model through this
                # daemon's gateway, so it stays up while an engine is live and
                # exits once none is left and no window has polled for a minute.
                while not runtime.STOP_EVT.is_set():
                    runtime.servers_snapshot()
                    with runtime.LOCK:
                        live = any(e.get('status') in runtime.LIVE_STATES for e in runtime.RUNNING.values())
                    if not live and time.monotonic() - http_api.LAST_UI_POLL > 60:
                        break
                    if runtime.STOP_EVT.wait(10):
                        return
                runtime.graceful_shutdown()
            threading.Thread(target=watch, daemon=True).start()
    try:
        server.serve_forever()
    finally:
        runtime.STOP_EVT.set()
        from b70launcher import gateway
        gateway.flush()
        server.server_close()
        runtime.persist_state()
        if child is not None and child.poll() is None:
            child.terminate()


if __name__ == '__main__':
    main()
