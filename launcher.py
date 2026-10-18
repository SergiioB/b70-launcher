#!/usr/bin/env python3
"""Desktop entry point. Importing it never starts a daemon or writes state."""
import argparse
import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http.server import ThreadingHTTPServer

from b70launcher.paths import VERSION

_CLI_CMDS = {
    'serve', 'down', 'list', 'ls', 'recipes', 'show', 'launch', 'stop',
    'status', 'ps', 'monitor', 'logs', 'test', 'download', 'open', 'ui',
    'env', 'agent', 'usage', 'tokens', 'doctor', 'scan', 'recipes-update',
    'version', 'completion', 'help',
}


def main():
    if len(sys.argv) > 1 and sys.argv[1] in _CLI_CMDS:
        import cli
        raise SystemExit(cli.main(sys.argv[1:]))
    from b70launcher import catalog, http_api, paths, runtime, scan
    catalog.load()
    parser = argparse.ArgumentParser(description='B70 model launcher')
    parser.add_argument('--port', type=int, default=catalog.SETTINGS.get('port', 7570))
    parser.add_argument('--headless', '--no-open', dest='no_open', action='store_true',
                        help='Run in headless CLI mode without opening a desktop window')
    parser.add_argument('--browser', action='store_true')
    parser.add_argument('--cli', action='store_true', help='Open the interactive CLI recipe launcher')
    parser.add_argument('--fg', '--foreground', dest='fg', action='store_true', help='Run the HTTP daemon in the foreground')
    parser.add_argument('--kill', '--force', action='store_true', help='Gracefully replace an authenticated launcher on this port')
    parser.add_argument('--stop', '--down', action='store_true', help='Gracefully stop an authenticated launcher on this port and exit')
    parser.add_argument('--version', action='version', version=VERSION)
    has_display = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
    interactive_tty = sys.stdin.isatty() and sys.stdout.isatty()
    args, extra = parser.parse_known_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    url = f'http://127.0.0.1:{args.port}'
    if args.stop:
        if extra:
            parser.parse_args()
        token = paths.read_token(args.port)
        try:
            req = urllib.request.Request(url + '/api/state', headers={'X-Launcher-Token': token})
            with urllib.request.urlopen(req, timeout=2) as response:
                data = json.loads(response.read(4 << 20))
            if not data.get('version') or 'recipes' not in data:
                raise ValueError('not a launcher')
            req_sd = urllib.request.Request(url + '/api/shutdown', b'{}', {'X-Launcher-Token': token, 'Content-Type': 'application/json'})
            with urllib.request.urlopen(req_sd, timeout=2):
                pass
            print(f'Shutdown requested — launcher at {url} stopped.')
            return
        except Exception as exc:
            raise SystemExit(f'Cannot stop launcher at {url}: {exc}')
    if args.cli or extra or (interactive_tty and not args.fg and not args.browser and (not has_display or args.no_open)):
        import cli
        if args.kill:
            rc = cli.main(['--api', url, 'serve', '--kill'])
            if rc != 0:
                raise SystemExit(rc)
        raise SystemExit(cli.main(['--api', url, 'launch', *extra]))
    if extra:
        parser.parse_args()
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
                port_flag = f' --port {args.port}' if args.port != 7570 else ''
                api_flag = f' --api {url}' if args.port != 7570 else ''
                print(f'Launcher already running at {url}; existing instance left unchanged.')
                print(f'  • Interactive CLI recipes: b70-launcher --headless{port_flag}  (or b70 launch{api_flag})')
                print(f'  • Stop daemon:             b70 down{api_flag}    (or b70-launcher --stop{port_flag})')
                print(f'  • Replace / restart:       b70 serve --kill{api_flag}  (or b70-launcher --kill{port_flag})')
                print(f'  • Check status:            b70 status{api_flag}')
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
                # daemon's gateway: with no live engine it quits now, otherwise it
                # stays up and quits once none is left and no window has polled
                # /api/state for a minute.
                waited = False
                while not runtime.STOP_EVT.is_set():
                    runtime.servers_snapshot()
                    with runtime.LOCK:
                        live = any(e.get('status') in runtime.LIVE_STATES for e in runtime.RUNNING.values())
                    if not live and (not waited or time.monotonic() - http_api.LAST_UI_POLL > 60):
                        break
                    waited = True
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
