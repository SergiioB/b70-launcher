"""Loopback-only JSON API and authenticated desktop surface."""
import hmac
import json
import mimetypes
import re
import secrets
import shutil
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from . import builders, catalog, downloads, hardware, harness, paths, runtime, scan, selfupdate, telemetry
from .locks import LOCK

API_TOKEN = ''
BOOTSTRAPS = {}


def bootstrap_nonce():
    nonce = secrets.token_urlsafe(24)
    BOOTSTRAPS[nonce] = time.monotonic() + 30
    return nonce


def scan_response():
    sc = scan.scan_blocking()
    matches = {}
    for model in catalog.RECIPES.get('models', []):
        matches[model['id']] = {}
        for engine in model['recipes']:
            det = scan.detect(model, engine) or {}
            matches[model['id']][engine] = {'detected': bool(det), 'path': det.get('path'),
                                           'size_mib': det.get('size_mib'), 'ctx_native': det.get('ctx')}
    probe = Path(catalog.SETTINGS.get('models_dir') or '~').expanduser()
    while not probe.is_dir() and probe != probe.parent:
        probe = probe.parent
    try:
        free_gb = round(shutil.disk_usage(probe).free / 1073741824, 1)
    except OSError:
        free_gb = None
    return {'state': sc.get('state'), 'ts': sc.get('ts'), 'error': sc.get('error'),
            'roots': sc.get('roots', []), 'catalog': sc.get('catalog', []), 'matches': matches, 'free_gb': free_gb}


def state_response():
    running = runtime.servers_snapshot()
    return {'recipes': catalog.RECIPES, 'settings': catalog.SETTINGS, 'running': running,
            'version': paths.VERSION, 'is_win': paths.IS_WIN,
            'update': dict(catalog.UPDATE_INFO, install=selfupdate.status()),
            'recipe_notices': catalog.recipe_notices(), 'recipes_remote': catalog.RECIPE_REMOTE,
            'preflight': runtime.preflight(), 'scan': {'state': scan.SCAN.get('state'), 'ts': scan.SCAN.get('ts'), 'roots': scan.SCAN.get('roots', [])}}


def _settings(cfg):
    allowed = {'scan_dirs', 'models_dir', 'open_with', 'sync_clients'}
    if not cfg or set(cfg) - allowed:
        raise ValueError('Only model folders, download folder, default client and client sync may be changed here.')
    values = {}
    if 'scan_dirs' in cfg:
        roots = cfg['scan_dirs']
        if not isinstance(roots, list) or len(roots) > 8:
            raise ValueError('Use at most eight model folders.')
        cleaned = []
        for root in roots:
            if not isinstance(root, str) or not Path(root).expanduser().is_absolute():
                raise ValueError('Use absolute model folder paths.')
            p = Path(root).expanduser().resolve()
            if p in (Path('/'), Path.home().resolve()) or not p.is_dir():
                raise ValueError(f'Choose an existing model folder, not an entire home or filesystem: {root}')
            if str(p) not in cleaned:
                cleaned.append(str(p))
        values['scan_dirs'] = cleaned
    if 'models_dir' in cfg:
        raw = cfg['models_dir']
        if not isinstance(raw, str) or not Path(raw).expanduser().is_absolute():
            raise ValueError('Use an absolute download folder path.')
        p = Path(raw).expanduser().resolve()
        if not p.is_dir() or p in (Path('/'), Path.home().resolve()):
            raise ValueError('Choose an existing download folder, not your home or filesystem root.')
        values['models_dir'] = str(p)
    if 'open_with' in cfg:
        if cfg['open_with'] not in ('pi', 'omp', 'droid', 'none'):
            raise ValueError('Choose Pi, omp, Droid or no client.')
        values['open_with'] = cfg['open_with']
    if 'sync_clients' in cfg:
        if not isinstance(cfg['sync_clients'], bool):
            raise ValueError('sync_clients must be true or false.')
        values['sync_clients'] = cfg['sync_clients']
    with LOCK:
        catalog.SETTINGS.update(values)
        catalog.save_override(tuple(values))
    scan.start_scan_async(force=True)
    return dict(values, ok=True)


def _preview(cfg):
    cfg = dict(cfg, dry_run=True)
    built = builders.build(cfg)
    if not built.get('error'):
        built['harness_line'] = harness.harness_line(cfg, built)
    return built


def _launch(cfg):
    built = builders.build(cfg)
    if built.get('error'):
        return built, 400
    if cfg.get('dry_run'):
        runtime.launch(cfg, built)
        built['harness_line'] = harness.harness_line(cfg, built)
        built['id'] = built['rid']
        return built, 200
    hardware.CLIENT_CACHE["ts"] = 0
    hardware.PREFLIGHT_CACHE["ts"] = 0
    hardware.VRAM_CACHE.clear()
    check = runtime.preflight()
    category = 'native' if built['native'] else 'exl3' if built['kind'] == 'exl3' else 'docker'
    blockers = check['blockers_by_kind'][category]
    if blockers:
        return {'error': '; '.join(blockers)}, 409
    if not built['detected']:
        return {'error': 'Model not found. Download it or add its model folder.'}, 409
    devices = check['devices']
    for g in built['gpus']:
        if g >= len(devices) or not devices[g].get('accessible') or not devices[g].get('b70'):
            return {'error': 'Selected GPU is not an accessible Arc Pro B70.'}, 409
        if devices[g].get('busy'):
            return {'error': f'GPU {g} is in use: {devices[g].get("busy_by") or "unknown workload"}. Choose a free GPU.'}, 409
    try:
        rid = runtime.launch(cfg, built)
    except ValueError as exc:
        return {'error': str(exc), 'id': built['rid']}, 409
    return {'id': rid, 'status': 'starting', 'harness_line': harness.harness_line(cfg, built)}, 200


def _download(cfg):
    model = catalog.find_model(cfg.get('model_id') or cfg.get('model'))
    engine = cfg.get('engine')
    if not model or engine not in model['recipes']:
        raise ValueError('Choose a model and engine.')
    plan = downloads.plan_download(model, engine)
    if not cfg.get('confirm', False):
        return plan, 200
    if plan.get('error') or not plan['enough']:
        return plan, 409
    did, error = downloads.start_download(model, engine, plan)
    return {'id': did, 'error': error}, 400 if error else 200


def _harness(cfg):
    runtime.servers_snapshot()
    with LOCK:
        entry = runtime.RUNNING.get(cfg.get('id', ''))
        if not entry:
            entry = next((e for e in runtime.RUNNING.values() if e.get('port') == runtime.parse_port(cfg.get('port', 8000))), None)
        entry = dict(entry) if entry else None
    if not entry or entry.get('status') != 'running':
        raise ValueError('Choose a ready server to open a client.')
    cfg = dict(cfg, port=entry['port'])
    return {'line': harness.open_harness(cfg, entry), 'opened': True}


GET_ROUTES = {
    '/api/state': state_response,
    '/api/scan': scan_response,
    '/api/power': lambda: {'power': hardware.power_probe() if not paths.IS_WIN else []},
    '/api/servers': lambda: {'running': runtime.servers_snapshot(), 'scan': {'state': scan.SCAN.get('state'), 'ts': scan.SCAN.get('ts')}},
    '/api/downloads': downloads.snapshot,
    '/api/metrics': telemetry.metrics_snapshot,
    '/api/usage': telemetry.usage_summary,
}
POST_ROUTES = {
    '/api/build': _preview,
    '/api/preview': _preview,
    '/api/launch': _launch,
    '/api/download': _download,
    '/api/download/cancel': lambda cfg: {'ok': downloads.cancel_download(cfg.get('id', ''))},
    '/api/stop': lambda cfg: {'ok': runtime.stop_server(cfg.get('id', ''))},
    '/api/test_prompt': runtime.test_prompt,
    '/api/harness': _harness,
    '/api/recipes/update': lambda cfg: catalog.apply_recipe_update(cfg.get('model_id'), cfg.get('engine')),
    '/api/settings': _settings,
    '/api/update/install': selfupdate.start_install,
    '/api/update/restart': selfupdate.restart,
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def _send(self, body, code=200, ctype='application/json'):
        self.send_response(code)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(json.dumps(obj).encode(), code)

    def _host_ok(self):
        return self.headers.get('Host') in ('127.0.0.1', 'localhost', f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')

    def _token_ok(self, supplied):
        # bytes: compare_digest raises TypeError on non-ASCII str input
        return bool(supplied) and hmac.compare_digest(supplied.encode('utf-8', 'surrogateescape'), API_TOKEN.encode())

    def _auth(self):
        if self._token_ok(self.headers.get('X-Launcher-Token', '')):
            return True
        name = f'b70_token_{self.server.server_port}'
        match = re.search(r'(?:^|;\s*)' + name + r'=([^;]+)', self.headers.get('Cookie', ''))
        return bool(match and self._token_ok(match.group(1)))

    def do_GET(self):
        if not self._host_ok():
            self._json({'error': 'unauthorized request'}, 403)
            return
        parsed = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        nonce = query.get('nonce', [''])[0]
        if parsed.path == '/' and BOOTSTRAPS.pop(nonce, 0) > time.monotonic():
            self.send_response(302)
            self.send_header('Location', '/')
            self.send_header('Set-Cookie', f'b70_token_{self.server.server_port}={API_TOKEN}; Path=/; SameSite=Strict; HttpOnly')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if not self._auth():
            self._json({'error': 'unauthorized request'}, 403)
            return
        try:
            if parsed.path in ('/', '/index.html'):
                self._send((paths.HERE / 'web/index.html').read_bytes().replace(b'__API_TOKEN__', API_TOKEN.encode()), ctype='text/html; charset=utf-8')
            elif parsed.path == '/api/logs':
                if query.get('id', [''])[0] not in runtime.RUNNING:
                    self._json({'error': 'no such server'}, 404)
                    return
                offset = query.get('offset', [None])[0]
                self._json(runtime.read_logs(query.get('id', [''])[0], offset))
            elif parsed.path in GET_ROUTES:
                self._json(GET_ROUTES[parsed.path]())
            elif parsed.path.startswith('/assets/'):
                root = (paths.HERE / 'web/assets').resolve()
                file = (root / parsed.path[len('/assets/'):]).resolve()
                if root not in file.parents or not file.is_file():
                    self._json({'error': 'not found'}, 404)
                    return
                self._send(file.read_bytes(), ctype=mimetypes.guess_type(str(file))[0] or 'application/octet-stream')
            else:
                self._json({'error': 'not found'}, 404)
        except (ValueError, OSError) as exc:
            self._json({'error': str(exc)}, 400)

    def do_POST(self):
        port = self.server.server_port
        if (not self._host_ok() or not self._token_ok(self.headers.get('X-Launcher-Token', ''))
                or self.headers.get('Origin') not in (None, f'http://127.0.0.1:{port}', f'http://localhost:{port}')
                or self.headers.get('Content-Type', '').split(';')[0] != 'application/json'):
            self._json({'error': 'unauthorized request'}, 403)
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 <= size <= 65536:
                raise ValueError('request too large')
            cfg = json.loads(self.rfile.read(size) or b'{}')
            if not isinstance(cfg, dict):
                raise ValueError('expected JSON object')
        except ValueError:
            self._json({'error': 'invalid JSON body'}, 400)
            return
        path = urllib.parse.urlsplit(self.path).path
        try:
            if path == '/api/window':
                self._json({'url': f'http://127.0.0.1:{port}/?nonce={bootstrap_nonce()}'})
                return
            if path == '/api/shutdown':
                self._json({'ok': True})
                threading.Thread(target=runtime.graceful_shutdown, args=(bool(cfg.get('stop_engines')),), daemon=True).start()
                return
            fn = POST_ROUTES.get(path)
            if not fn:
                self._json({'error': 'not found'}, 404)
                return
            result = fn(cfg)
            if path == '/api/recipes/update' and isinstance(result, dict) and not result.get('ok'):
                result = (result, 400)
            obj, status = result if isinstance(result, tuple) else (result, 200)
            self._json(obj, status)
        except (ValueError, OSError) as exc:
            self._json({'error': str(exc)}, 400)
        except Exception as exc:
            self._json({'error': f'Launcher request failed: {exc}'}, 500)
