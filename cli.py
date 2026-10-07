#!/usr/bin/env python3
"""b70 — headless CLI for the B70 model launcher.

Stdlib-only client for launcher.py's localhost JSON API (docs/api.md).
Works with no display, no desktop, and no GUI stack — the same recipes the
windowed app launches, scriptable end to end.

Every command accepts --json for machine-readable output and honors NO_COLOR.
Exit codes: 0 ok · 1 operation error · 2 usage error · 3 daemon unreachable ·
130 interrupted.
"""

import argparse
import difflib
import http.client
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CLI_VERSION = "1.0.0"
API_TIMEOUT = 15

# ── styling ──────────────────────────────────────────────────────────────

_TTY = sys.stdout.isatty()
_NO_COLOR = bool(os.environ.get("NO_COLOR")) or not _TTY
_JSON = False  # set in main() once args are parsed


def _init_color(no_color):
    global _NO_COLOR, OK, WARN, ERR, DOT
    _NO_COLOR = _NO_COLOR or no_color
    if _NO_COLOR:
        OK, WARN, ERR, DOT = "✓", "!", "✗", "·"


def _c(code, s):
    if _NO_COLOR or not s:
        return s
    return f"\033[{code}m{s}\033[0m"


def bold(s):    return _c("1", s)
def dim(s):     return _c("2", s)
def cyan(s):    return _c("36", s)
def blue(s):    return _c("34", s)
def green(s):   return _c("32", s)
def yellow(s):  return _c("33", s)
def red(s):     return _c("31", s)
def magenta(s): return _c("35", s)


OK = green("✓")
WARN = yellow("!")
ERR = red("✗")
DOT = cyan("·")

BANNER = r"""
  ██████╗ ███████╗ ██████╗
  ██╔══██╗╚════██║██╔═████╗
  ██████╔╝    ██╔╝██║██╔██║
  ██╔══██╗   ██╔╝ ████╔╝██║
  ██████╔╝   ██║  ╚██████╔╝
  ╚═════╝    ╚═╝   ╚═════╝"""


def banner(sub=""):
    print(cyan(bold(BANNER)))
    tag = "headless model launcher for Intel Arc Pro B70"
    print(f"  {dim(tag)}{('  ' + dim('— ') + magenta(sub)) if sub else ''}")
    print()


# ── errors / exits ─────────────────────────────────────────────────────────


class CliError(Exception):
    code = 1


class DaemonDown(CliError):
    code = 3


class AuthError(CliError):
    """Daemon answered but rejected the token — never auto-restart over it."""
    code = 1


def die(msg, code=1, hint=None):
    if _JSON:
        jprint({"ok": False, "error": msg, "code": code})
    else:
        print(f"{ERR} {red(msg)}", file=sys.stderr)
        if hint:
            print(f"  {dim(hint)}", file=sys.stderr)
    raise SystemExit(code)


def _model_miss(query, cands):
    """Best error wording for an unresolved model query."""
    q = (query or "").lower()
    ambiguous = bool(cands) and any(
        m["id"].lower().startswith(q) or q in m["id"].lower()
        or q in m.get("name", "").lower() for m in cands)
    if ambiguous:
        return (f"ambiguous model {query!r} — be more specific",
                "candidates: " + ", ".join(m["id"] for m in cands[:8]))
    return (f"unknown model {query!r}",
            "try: " + ", ".join(m["id"] for m in cands[:8]))


def jprint(obj):
    print(json.dumps(obj, indent=2, sort_keys=False), flush=True)


# ── daemon client ─────────────────────────────────────────────────────────


def state_dir():
    xdg = os.environ.get("XDG_STATE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".local" / "state") / "b70-launcher"


def token_file():
    override = os.environ.get("B70_TOKEN_FILE")
    return Path(override) if override else state_dir() / "token"


def read_token():
    tok = os.environ.get("B70_TOKEN")
    if tok:
        return tok.strip()
    try:
        return token_file().read_text().strip()
    except OSError:
        return ""


class Client:
    """Token-authenticated JSON client for the launcher daemon."""

    def __init__(self, api=None, token=None, timeout=API_TIMEOUT):
        self.api = (api or os.environ.get("B70_API")
                    or "http://127.0.0.1:7570").rstrip("/")
        u = urllib.parse.urlparse(self.api)
        if u.scheme not in ("http", "https") or not u.hostname:
            die(f"invalid daemon address {self.api!r} — expected http://host:port", code=2)
        self.token = read_token() if token is None else token
        self.timeout = timeout

    @property
    def port(self):
        return int(urllib.parse.urlparse(self.api).port or 7570)

    def _req(self, method, path, obj=None, timeout=None):
        url = self.api + path
        data = None
        headers = {"X-Launcher-Token": self.token}
        if method == "POST":
            data = json.dumps(obj if obj is not None else {}).encode()
            headers["Content-Type"] = "application/json"
            headers["Origin"] = f"http://127.0.0.1:{self.port}"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read() or b"{}")
            except Exception:
                body = {}
            msg = body.get("error") or f"HTTP {e.code} from launcher"
            if e.code in (401, 403):
                if not self.token:
                    msg += " — no API token found; is the daemon running? (b70 serve)"
                else:
                    msg += " — daemon rejected the API token (check --token / B70_TOKEN)"
                raise AuthError(msg)
            raise CliError(msg)
        except ValueError as e:
            raise CliError(f"invalid daemon address {self.api!r} ({e})")
        except urllib.error.URLError as e:
            raise DaemonDown(f"launcher daemon unreachable at {self.api} ({e.reason})")
        except (http.client.HTTPException, ConnectionResetError, ConnectionAbortedError):
            # daemon closed the socket mid-request/shutdown — same as unreachable
            raise DaemonDown(f"launcher daemon at {self.api} went away")
        except (TimeoutError, socket.timeout):
            raise DaemonDown(f"launcher daemon at {self.api} timed out")

    def get(self, path, timeout=None):
        return self._req("GET", path, timeout=timeout)

    def post(self, path, obj=None, timeout=None):
        return self._req("POST", path, obj, timeout=timeout)

    def alive(self):
        """True only if the API answers. Auth failures propagate — a live
        daemon with a wrong token must NOT trigger autostart (that would
        kill and replace a healthy daemon)."""
        try:
            self.get("/api/state", timeout=3)
            return True
        except DaemonDown:
            return False


def find_launcher():
    """Locate launcher.py (or its wrapper) for auto-starting the daemon."""
    cand = os.environ.get("B70_LAUNCHER")
    if cand:
        return cand
    here = Path(__file__).resolve().parent / "launcher.py"
    if here.is_file():
        return str(here)
    installed = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) \
        / "b70-launcher" / "launcher.py"
    if installed.is_file():
        return str(installed)
    exe = shutil.which("b70-launcher")
    return exe


def start_daemon(port, quiet=False):
    """Spawn launcher.py --no-open detached; wait for its API to come up."""
    launcher = find_launcher()
    if not launcher:
        die("launcher daemon not running and launcher.py not found",
            code=3,
            hint="install via packaging/install.sh, or set B70_LAUNCHER=/path/to/launcher.py")
    logdir = state_dir() / "logs"
    try:
        logdir.mkdir(parents=True, exist_ok=True)
        logf = open(logdir / "daemon.log", "ab")
    except OSError:
        logf = open(os.devnull, "ab")
    argv = [launcher, "--no-open", "--port", str(port)]
    if launcher.endswith(".py"):
        argv = [sys.executable, "-u"] + argv  # -u: daemon.log gets live output
    try:
        subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=logf,
                         stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as exc:
        die(f"could not start launcher daemon: {exc}", code=3)
    if not quiet:
        print(f"{DOT} started launcher daemon (pid detached, log: {logdir}/daemon.log)")
    w = Wait("waiting for daemon api", quiet=quiet).start()
    deadline = time.time() + 30
    try:
        while time.time() < deadline:
            # readiness = the API actually answering — a stale token file or a
            # foreign squatter on the port must not count as "up"
            tok = read_token()
            if tok:
                try:
                    Client(api=f"http://127.0.0.1:{port}", token=tok) \
                        .get("/api/state", timeout=2)
                    w.done()
                    return True
                except CliError:
                    w.tick("launching")
            else:
                w.tick("waiting for token")
            time.sleep(0.3)
    finally:
        w.done()
    die("daemon started but did not become ready in 30s",
        code=3, hint=f"check {logdir}/daemon.log")


def daemon_client(gargs, quiet=False):
    """Client bound to a running daemon — auto-starts it unless disabled."""
    c = Client(api=gargs.api, token=gargs.token)
    if c.alive():
        return c
    if gargs.no_autostart:
        raise DaemonDown(f"launcher daemon not running at {c.api}")
    start_daemon(c.port, quiet=quiet or gargs.quiet)
    c = Client(api=gargs.api, token=gargs.token)  # fresh token
    if not c.alive():
        raise DaemonDown(f"launcher daemon not answering at {c.api}")
    return c


# ── catalog helpers ────────────────────────────────────────────────────────


def engines_by_id(recipes):
    return {e["id"]: e for e in recipes.get("engines", [])}


def pick_model(recipes, query):
    """Fuzzy-resolve a model id or name. Returns (model, candidates)."""
    models = recipes.get("models", [])
    if not query:
        return None, models
    q = query.lower().strip()
    for m in models:
        if m["id"].lower() == q:
            return m, models
    pref = [m for m in models if m["id"].lower().startswith(q)]
    if len(pref) == 1:
        return pref[0], models
    sub = [m for m in models
           if q in m["id"].lower() or q in m.get("name", "").lower()]
    if len(sub) == 1:
        return sub[0], models
    if pref or sub:
        return None, pref or sub
    # nothing matched — offer close names
    names = {m["id"]: m for m in models}
    close = difflib.get_close_matches(q, names.keys(), n=3, cutoff=0.5)
    return None, [names[n] for n in close] or models


def pick_engine(model, engine):
    """Resolve the recipe engine for a model: explicit, recommended, or first."""
    recs = model.get("recipes", {})
    if engine:
        if engine in recs:
            return engine
        die(f"model '{model['id']}' has no '{engine}' recipe",
            hint="available: " + ", ".join(recs.keys()))
    rec = model.get("recommended_engine")
    return rec if rec in recs else next(iter(recs), None)


def parse_gpus(val):
    """'0' | '1' | '0,1' | 'both' -> [0], [1], [0,1]."""
    if val is None:
        return None
    v = val.strip().lower()
    if v in ("both", "all", "dual"):
        return [0, 1]
    try:
        gpus = [int(x) for x in v.split(",") if x.strip() != ""]
    except ValueError:
        die(f"invalid --gpus value: {val!r} (use 0, 1, or 0,1)", code=2)
    if not gpus or any(g not in (0, 1) for g in gpus) or len(set(gpus)) != len(gpus):
        die(f"invalid --gpus value: {val!r} (only distinct indices 0 and 1)", code=2)
    return gpus


def default_gpus(recipe):
    """GPU selection a recipe implies when the user doesn't pass --gpus.

    Recipes that declare `gpus`/`tp` (dual-card tensor split / TP2) need those
    indices sent, or build() tensor-splits across a single device."""
    g = recipe.get("gpus")
    if isinstance(g, list) and g and all(isinstance(i, int) and i in (0, 1) for i in g):
        return sorted(set(g))
    tp = recipe.get("tp")
    if isinstance(tp, int) and tp > 1:
        return list(range(min(tp, 2)))
    if recipe.get("kind") == "vllm-tp2":
        return [0, 1]
    return [0]


def resolve_rid(client, query, allow_port=True):
    """Map a rid / model name / bare port to a tracked server entry."""
    state = client.get("/api/state")
    running = [e for e in state.get("running", [])
               if not str(e.get("status", "")).startswith(("stopped", "dry-run"))]
    if not running:
        return None, state
    if not query:
        if len(running) == 1:
            return running[0], state
        return running, state  # ambiguous — caller reports list
    q = str(query).strip()
    for e in running:
        if e.get("id") == q:
            return e, state
    if allow_port and q.isdigit():
        for e in running:
            if e.get("port") == int(q):
                return e, state
    matches = [e for e in running if q.lower() in (e.get("id") or "").lower()
               or q.lower() in (e.get("model") or "").lower()
               or q.lower() in (e.get("model_id") or "").lower()]
    if len(matches) == 1:
        return matches[0], state
    return matches or None, state


def rid_or_die(client, query):
    ent, state = resolve_rid(client, query)
    if isinstance(ent, list):
        die("multiple servers match — pick a server id",
            hint="running: " + ", ".join(e["id"] for e in ent))
    if ent is None:
        running = [e["id"] for e in state.get("running", [])
                   if e.get("status") in ("running", "starting", "running (adopted)")]
        if not query:
            if running:
                die("which engine? multiple are running",
                    hint="ids: " + ", ".join(running))
            die("no engines running",
                hint="launch one first: b70 launch <model>")
        hint = "running servers: " + ", ".join(running) if running \
            else "nothing is running — launch one first: b70 launch <model>"
        die(f"no server matches {query!r}", hint=hint)
    return ent


def engine_models(port, timeout=2):
    """Model ids served by a live engine endpoint (no token needed)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",
                                    timeout=timeout) as r:
            data = json.loads(r.read() or b"{}")
        return [m.get("id") for m in data.get("data", []) if m.get("id")]
    except Exception:
        return []


# ── output helpers ─────────────────────────────────────────────────────────


def human_gb(mib):
    try:
        return f"{float(mib) / 1024:.1f} GiB"
    except (TypeError, ValueError):
        return "-"


def human_ctx(n):
    try:
        n = int(n)
        return f"{n // 1024}K" if n % 1024 == 0 else str(n)
    except (TypeError, ValueError):
        return "-"


def fmt_dur(s):
    s = int(s or 0)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{s % 3600 // 60:02d}m"


def table(rows, headers, aligns=None):
    """Borderless padded table; rows are lists of (possibly styled) strings."""
    widths = [len(_strip(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(_strip(str(cell))))
    aligns = aligns or ["<"] * len(headers)
    out = ["  " + dim("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))]
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            cell = str(cell)
            pad = widths[i] - len(_strip(cell))
            cells.append((" " * pad + cell) if aligns[i] == ">" else (cell + " " * pad))
        out.append("  " + "  ".join(cells))
    return "\n".join(out)


_ansi_re = re.compile(r"\033\[[0-9;]*m")


def _strip(s):
    return _ansi_re.sub("", s)


SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Wait:
    """Animated braille spinner on a TTY (~90ms frames, drawn by a render
    thread so the animation stays smooth no matter how slow the poll is);
    a status line every 15s when piped to a file."""

    def __init__(self, label, quiet=False):
        self.label = label
        self.quiet = quiet
        self.note = ""
        self.t0 = time.time()
        self._stop = threading.Event()
        self._thr = None
        self._last_line = 0.0

    def start(self):
        if _TTY and not self.quiet:
            self._thr = threading.Thread(target=self._spin, daemon=True)
            self._thr.start()
        return self

    def _spin(self):
        i = 0
        while not self._stop.wait(0.09):
            line = (f"\r  {cyan(SPINNER[i % len(SPINNER)])} {self.label} "
                    f"{dim(fmt_dur(time.time() - self.t0))}  {dim(self.note)}   ")
            sys.stdout.write(line)
            sys.stdout.flush()
            i += 1

    def tick(self, note=""):
        """Update the detail text. Off-TTY, prints a progress line every 15s."""
        self.note = note
        if not _TTY and not self.quiet and time.time() - self._last_line > 15:
            self._last_line = time.time()
            print(f"  {DOT} {self.label} — {fmt_dur(time.time() - self.t0)} {note}")

    def done(self):
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=0.6)
            self._thr = None
        if _TTY:
            sys.stdout.write("\r" + " " * 88 + "\r")
            sys.stdout.flush()


def spin_while(label, fn):
    """Run fn() (a blocking call) under a spinner; returns fn()'s result
    and re-raises its exceptions. Off-TTY it's just the call."""
    if not _TTY:
        return fn()
    w = Wait(label).start()
    box = {}

    def run():
        try:
            box["v"] = fn()
        except Exception as exc:  # noqa: BLE001 — propagated below
            box["e"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    while t.is_alive():
        t.join(0.05)
    w.done()
    if "e" in box:
        raise box["e"]
    return box.get("v")


def vram_bar(used_gib, total_gib, width=10):
    """Compact VRAM gauge: [██████░░░░] colored by pressure."""
    frac = (used_gib / total_gib) if total_gib else 0.0
    frac = max(0.0, min(1.0, frac))
    fill = round(frac * width)
    bar = "█" * fill + "░" * (width - fill)
    paint = green if frac < 0.70 else yellow if frac < 0.90 else red
    return paint(bar)


def progress_bar(pct, width=28):
    pct = max(0.0, min(100.0, float(pct or 0)))
    fill = int(width * pct / 100)
    return f"[{'█' * fill}{'░' * (width - fill)}] {pct:5.1f}%"


# ── commands ───────────────────────────────────────────────────────────────


def cmd_serve(g, a):
    c = Client(api=g.api, token=g.token)
    if c.alive():
        st = c.get("/api/state")
        if g.json:
            jprint({"ok": True, "already_running": True, "api": c.api,
                    "version": st.get("version"), "token_file": str(token_file())})
        else:
            print(f"{OK} daemon already running at {cyan(c.api)} "
                  f"(b70-launcher {st.get('version', '?')})")
            print(f"  {dim('open it: b70 open    stop it: b70 down')}")
        return 0
    port = c.port
    if a.kill:
        subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
        subprocess.run(["pkill", "-f", "webwindow.py"], capture_output=True)
        time.sleep(0.5)
    if a.foreground:
        launcher = find_launcher()
        if not launcher:
            die("launcher.py not found", code=3,
                hint="install via packaging/install.sh or set B70_LAUNCHER")
        argv = [launcher, "--no-open", "--port", str(port)]
        if launcher.endswith(".py"):
            argv = [sys.executable, "-u"] + argv
        print(f"{DOT} daemon in foreground on {c.api} — Ctrl+C to stop", flush=True)
        os.execvp(argv[0], argv)
    start_daemon(port, quiet=g.quiet)
    c2 = Client(api=g.api, token=g.token)
    st = {}
    try:
        st = c2.get("/api/state")
    except CliError:
        pass
    if g.json:
        jprint({"ok": True, "api": c2.api, "version": st.get("version"),
                "token_file": str(token_file()), "daemon_log": str(state_dir() / "logs" / "daemon.log")})
    else:
        print(f"{OK} daemon up at {cyan(c2.api)}  (b70-launcher {st.get('version', '?')})")
        if not g.quiet:
            print(f"  {dim('ui:')} b70 open    {dim('launch:')} b70 launch <model>    {dim('down:')} b70 down")
    return 0


def cmd_down(g, a):
    try:
        c = Client(api=g.api, token=g.token)
        res = c.post("/api/shutdown", {"stop_engines": a.stop_engines})
    except DaemonDown:
        if g.json:
            jprint({"ok": True, "note": "daemon was not running"})
        else:
            print(f"{DOT} daemon not running at {Client(api=g.api).api} — nothing to stop")
        return 0
    if g.json:
        jprint({"ok": res.get("ok", True), "stop_engines": a.stop_engines})
    else:
        extra = " and engines" if a.stop_engines else ""
        print(f"{OK} shutdown requested{extra} — daemon exiting")
    deadline = time.time() + 10
    while time.time() < deadline and Client(api=g.api, token=g.token).alive():
        time.sleep(0.3)
    return 0


def _detect_map(client):
    try:
        return client.get("/api/scan").get("matches", {})
    except CliError:
        return {}


def cmd_list(g, a):
    c = daemon_client(g)
    st = c.get("/api/state")
    recipes = st.get("recipes", {})
    eng_meta = engines_by_id(recipes)
    matches = _detect_map(c)
    running = {e.get("id") for e in st.get("running", [])
               if e.get("status") in ("running", "starting", "running (adopted)")}
    only_eng = a.engine
    if g.json:
        models = recipes.get("models", [])
        if only_eng:
            models = [dict(m, recipes={only_eng: m["recipes"][only_eng]})
                      for m in models if only_eng in m.get("recipes", {})]
            if not models:
                die(f"no recipes match engine '{only_eng}'")
        jprint({"catalog_ver": recipes.get("catalog_ver"),
                "models": models,
                "detected": matches, "running": sorted(running)})
        return 0
    rows = []
    for m in recipes.get("models", []):
        for eng, rec in m.get("recipes", {}).items():
            if only_eng and eng != only_eng:
                continue
            det = matches.get(m["id"], {}).get(eng, {})
            detected = det.get("detected")
            mark = green("● detected") if detected else dim("○ —")
            best = yellow("★") if eng == m.get("recommended_engine") else " "
            rid_prefix = f"{m['id']}-{eng}-"
            live = red("▶ live") if any(r.startswith(rid_prefix) for r in running) else ""
            ctx = human_ctx(rec.get("ctx"))
            kind = rec.get("kind", eng)
            rows.append([f"{best} {m['id']}", m.get("name", ""), eng, kind,
                         ctx, mark, live])
    if not rows:
        die("no recipes match" + (f" engine '{only_eng}'" if only_eng else ""))
    print(bold(f"  B70 recipes  {dim('catalog ' + str(recipes.get('catalog_ver', '?')))}"))
    print(table(rows, ["MODEL", "NAME", "ENGINE", "KIND", "CTX", "ARTIFACT", ""]))
    print(f"\n  {dim('★ recommended engine   ● artifact on disk   ▶ running now')}")
    print(f"  {dim('launch one:')} b70 launch <model> [-e engine]   {dim('detail:')} b70 show <model>")
    return 0


def cmd_show(g, a):
    c = daemon_client(g)
    st = c.get("/api/state")
    recipes = st.get("recipes", {})
    eng_meta = engines_by_id(recipes)
    model, cands = pick_model(recipes, a.model)
    if model is None:
        msg, hint = _model_miss(a.model, cands)
        die(msg, hint=hint)
    if a.engine and a.engine not in model.get("recipes", {}):
        die(f"model '{model['id']}' has no '{a.engine}' recipe",
            hint="available: " + ", ".join(model.get("recipes", {})))
    matches = _detect_map(c)
    if g.json:
        jprint({"model": model, "detected": matches.get(model["id"], {})})
        return 0
    print(bold(f"  {model.get('name', model['id'])}  {dim(model['id'])}"))
    if model.get("badge"):
        print(f"  {yellow(model['badge'])}")
    for line in (model.get("blurb"), model.get("recommendation")):
        if line:
            print(f"  {dim(line)}")
    print()
    for eng, rec in model.get("recipes", {}).items():
        if a.engine and eng != a.engine:
            continue
        em = eng_meta.get(eng, {})
        det = matches.get(model["id"], {}).get(eng, {})
        star = yellow("★ recommended") if eng == model.get("recommended_engine") else ""
        dmark = green("detected: " + det["path"]) if det.get("detected") else dim("not detected")
        print(f"  {cyan(bold(eng))}  {dim(em.get('tagline', ''))} {star}")
        print(f"    kind {rec.get('kind', eng)} · ctx {human_ctx(rec.get('ctx'))}"
              + (f"/{human_ctx(rec['ctx_max'])} max" if rec.get("ctx_max") else "")
              + (f" · ~{rec['power']}W" if rec.get("power") else ""))
        if rec.get("ctx_note"):
            print(f"    {dim(rec['ctx_note'])}")
        if rec.get("perf"):
            print(f"    {dim(rec['perf'])}")
        print(f"    artifact: {dmark}")
        dl = rec.get("download") or {}
        if dl.get("repo") and not det.get("detected"):
            print(f"    {dim('fetch:')} b70 download {model['id']} -e {eng}")
        print(f"    {dim('launch:')} b70 launch {model['id']} -e {eng}")
        print()
    return 0


def _interactive_pick(state, matches):
    """TTY-only two-step picker: model, then engine. Returns (model_id, engine)."""
    models = state.get("recipes", {}).get("models", [])
    print(bold("  pick a model:"))
    for i, m in enumerate(models):
        star = yellow(" ★") if m.get("badge") else ""
        engs = []
        for eng in m.get("recipes", {}):
            det = matches.get(m["id"], {}).get(eng, {}).get("detected")
            engs.append(green(eng) if det else dim(eng))
        print(f"  {cyan('[' + str(i + 1) + ']')} {m.get('name', m['id'])}{star}"
              f"  {dim(m['id'])}  —  {' '.join(engs)}")
    try:
        raw = input(f"{bold('model')} [{dim('1')}]{bold(':')} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(130)
    idx = 0
    if raw:
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            idx = int(raw) - 1
        else:
            m, _ = pick_model(state.get("recipes", {}), raw)
            if not m:
                die(f"no model matches {raw!r}")
            idx = models.index(m)
    model = models[idx]
    engs = list(model.get("recipes", {}).keys())
    rec_eng = pick_engine(model, None)
    det_map = matches.get(model["id"], {})
    print(f"\n{bold('  pick an engine for')} {cyan(model.get('name', model['id']))}{bold(':')}")
    for i, e in enumerate(engs):
        marks = []
        if e == rec_eng:
            marks.append(yellow("recommended"))
        if det_map.get(e, {}).get("detected"):
            marks.append(green("detected"))
        print(f"  {cyan('[' + str(i + 1) + ']')} {e}"
              + (f"  {dim('· ' + ' · '.join(marks))}" if marks else ""))
    default = engs.index(rec_eng) if rec_eng in engs else 0
    try:
        raw = input(f"{bold('engine')} [{dim(str(default + 1))}]{bold(':')} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(130)
    eng = rec_eng
    if raw:
        if raw.isdigit() and 1 <= int(raw) <= len(engs):
            eng = engs[int(raw) - 1]
        elif raw in engs:
            eng = raw
        else:
            die(f"no engine {raw!r} for {model['id']} (have: {', '.join(engs)})")
    return model["id"], eng


def _print_plan(out, a):
    """Render a dry-run launch plan."""
    print(bold("  launch plan (dry run — nothing started)"))
    print(f"    id        {out.get('id')}")
    if out.get("detected") is not None:
        det = out.get("detected")
        print(f"    artifact  {green('detected: ' + str(out.get('detected_path'))) if det else yellow('not detected — recipe defaults in use')}")
    print(f"    endpoint  {cyan('http://127.0.0.1:' + str(a.port or 8000) + '/v1')}")
    if out.get("ctx"):
        print(f"    ctx       {out['ctx']:,}")
    for w in out.get("warnings") or []:
        print(f"    {WARN} {yellow(w)}")
    if out.get("env"):
        print(f"    env       {dim(' '.join(f'{k}={v}' for k, v in list(out['env'].items())[:6]))}")
    print(f"\n  {dim('$')} {out.get('cmd')}")
    if out.get("harness_line"):
        print(f"  {dim('client:')} {out['harness_line']}")


def _wait_ready(c, rid, timeout):
    """Poll /api/state until the engine's OpenAI endpoint answers.

    The daemon marks an entry 'running' only after GET /v1/models returns 200,
    so reaching that state == serving."""
    w = Wait(f"loading {rid}").start()
    deadline = time.time() + timeout
    note = ""
    try:
        while time.time() < deadline:
            st = c.get("/api/state")
            ent = next((e for e in st.get("running", []) if e.get("id") == rid), None)
            status = (ent or {}).get("status", "gone")
            note = (ent or {}).get("phase") or note
            if status == "running" or status == "running (adopted)":
                w.done()
                return ent or {}
            if status == "stopped" or status.startswith("exited"):
                w.done()
                raise CliError(f"engine exited during load (status: {status})")
            w.tick(note or "starting")
            time.sleep(1.2)
    except KeyboardInterrupt:
        w.done()
        print(f"\n{WARN} interrupted — {bold(rid)} is still loading in the background")
        print(f"  {dim('watch:')} b70 logs {rid} -f   {dim('status:')} b70 status   {dim('stop:')} b70 stop {rid}")
        raise SystemExit(130)
    w.done()
    raise CliError(f"timed out after {fmt_dur(timeout)} waiting for {rid}",
                   )


def cmd_launch(g, a):
    c = daemon_client(g)
    st = c.get("/api/state")
    recipes = st.get("recipes", {})
    matches = _detect_map(c)

    cfg = {}
    target = a.model
    custom_path = a.path
    if target and not custom_path and ("/" in target or target.startswith("~")):
        if Path(target).expanduser().exists():
            custom_path, target = target, None  # `b70 launch ./model.gguf`
        else:
            die(f"artifact not found on disk: {target!r}",
                hint="check the path (or use --path); to pick a recipe: b70 list")

    if custom_path:
        cfg["model_id"] = "__custom__"
        cfg["custom_path"] = custom_path
        engine = a.engine
        if not engine:
            p = Path(custom_path).expanduser()
            if p.is_file() and p.suffix.lower() == ".gguf":
                engine = "llamacpp"
            elif p.is_dir():
                try:
                    if any(f.name.startswith("openvino_language_model.") for f in p.iterdir()):
                        engine = "openvino"
                except OSError:
                    pass
                engine = engine or "vllm"
            else:
                engine = engine or "llamacpp"
        cfg["engine"] = engine
        model_disp = Path(custom_path).name
        recipe = {}
    else:
        if not target:
            if not sys.stdin.isatty():
                die("b70 launch needs a model id (or --path for a custom artifact)",
                    hint="see: b70 list")
            target, eng = _interactive_pick(st, matches)
            if not a.engine:
                a.engine = eng
        model, cands = pick_model(recipes, target)
        if model is None:
            msg, hint = _model_miss(target, cands)
            die(msg, hint=hint)
        engine = pick_engine(model, a.engine)
        recipe = model["recipes"].get(engine, {})
        cfg["model_id"] = model["id"]
        cfg["engine"] = engine
        model_disp = model.get("name", model["id"])

    for k, v in (("ctx", a.ctx), ("port", a.port), ("slots", a.slots),
                 ("kv", a.kv), ("power", a.power)):
        if v is not None:
            cfg[k] = v
    for name, val, lo, hi in (("port", a.port, 1, 65535),
                              ("slots", a.slots, 1, 128),
                              ("ctx", a.ctx, 512, 262144)):
        if val is not None and not lo <= val <= hi:
            die(f"--{name} {val} out of range ({lo}–{hi})", code=2)
    gpus = parse_gpus(a.gpus)
    if gpus is None:
        gpus = default_gpus(recipe)
    cfg["gpus"] = gpus
    if a.mtp is not None:
        cfg["mtp"] = a.mtp
    if a.use_docker is not None:
        cfg["use_docker"] = a.use_docker
    if a.extra:
        cfg["extra"] = a.extra
    if a.env:
        for item in a.env:
            if "=" not in item:
                die(f"-E expects NAME=value (got {item!r})", code=2)
        cfg["extra_env"] = "\n".join(a.env)
    port = int(cfg.get("port") or 8000)
    rid_guess = f"{cfg['model_id']}-{cfg['engine']}-{port}"

    # already serving? short-circuit before POST (which would spawn a terminal)
    if not a.dry_run:
        prev = next((e for e in st.get("running", [])
                     if e.get("id") == rid_guess
                     and e.get("status") in ("running", "starting", "running (adopted)")), None)
        if prev:
            if g.json:
                jprint({"id": rid_guess, "already_running": True,
                        "endpoint": prev.get("endpoint"), "status": prev.get("status")})
            else:
                print(f"{DOT} {bold(rid_guess)} is already {prev.get('status')}")
                print(f"  {dim('endpoint:')} {prev.get('endpoint')}   {dim('open:')} b70 open {port}")
            return 0

    if a.dry_run:
        cfg["dry_run"] = True

    try:
        out = spin_while("preparing launch",
                         lambda: c.post("/api/launch", cfg, timeout=60))
    except CliError as exc:
        if not g.json and "not detected" in str(exc).lower() and recipe.get("download"):
            print(f"{ERR} {red(str(exc))}", file=sys.stderr)
            print(f"  {dim('get the artifact first:')} b70 download {cfg['model_id']} -e {cfg['engine']}",
                  file=sys.stderr)
            return 1
        if g.json:
            jprint({"ok": False, "error": str(exc)})
            return 1
        die(str(exc))

    if out.get("error"):
        if g.json:
            jprint({"ok": False, **out})
            return 1
        die(out["error"])

    rid = out.get("id") or rid_guess
    if a.dry_run:
        if g.json:
            jprint(out)
        else:
            _print_plan(out, a)
        return 0

    if g.json:
        # keep JSON mode quiet — the object below is the contract
        pass
    else:
        print(f"{OK} launch accepted: {bold(rid)}")
        print(f"  {dim('model')} {model_disp} · {cfg['engine']} · ctx {int(cfg.get('ctx') or 0) or 'recipe'}"
              f" · gpu{','.join(map(str, gpus))}")
        if out.get("recipe_notice"):
            n = out["recipe_notice"]
            print(f"  {WARN} {yellow('newer recipe available: ' + str(n.get('remote_ver', '')))} — b70 recipes-update")
        if out.get("harness_line"):
            print(f"  {dim('client:')} {out['harness_line']}")

    if a.no_wait:
        if g.json:
            jprint({"id": rid, "status": "starting", "endpoint": f"http://127.0.0.1:{port}/v1"})
        else:
            print(f"  {dim('not waiting —')} b70 status {dim('to check,')} b70 logs {rid} -f {dim('to tail')}")
        return 0

    try:
        ent = _wait_ready(c, rid, a.timeout)
    except CliError as exc:
        if g.json:
            jprint({"id": rid, "ok": False, "error": str(exc)})
            return 1
        # attach log tail for diagnosis
        try:
            lines = (c.get(f"/api/logs?id={urllib.parse.quote(rid)}") or {}).get("lines", [])
        except CliError:
            lines = []
        print(f"{ERR} {red(str(exc))}", file=sys.stderr)
        if lines:
            print(dim("  ── last log lines ──"), file=sys.stderr)
            for ln in lines[-12:]:
                print("  " + dim(ln.rstrip()), file=sys.stderr)
        print(f"  {dim('the engine may still be loading — watch:')} b70 logs {rid} -f",
              file=sys.stderr)
        return 1

    endpoint = ent.get("endpoint") or f"http://127.0.0.1:{port}/v1"
    load_s = ent.get("load_s")
    served = ent.get("model") or model_disp
    if g.json:
        jprint({"id": rid, "ok": True, "endpoint": endpoint, "load_s": load_s,
                "port": port, "model": served})
    else:
        print(f"{OK} {green(bold('serving'))} {bold(rid)}"
              + (f"  {dim('ready in ' + fmt_dur(load_s))}" if load_s else ""))
        print(f"  {dim('endpoint')}  {cyan(endpoint)}")
        print(f"  {dim('export')}    OPENAI_BASE_URL={endpoint}")
        print(f"  {dim('         ')}  OPENAI_API_KEY=b70-local")
        print(f"  {dim('next')}      b70 test {rid} · b70 open {port} · b70 stop {rid}")
    if a.open:
        _open_url(f"http://127.0.0.1:{port}", g, label="engine")
    return 0


def cmd_stop(g, a):
    c = daemon_client(g)
    ids = []
    if a.all:
        st = c.get("/api/state")
        ids = [e["id"] for e in st.get("running", [])
               if e.get("status") in ("running", "starting", "running (adopted)")]
        if not ids:
            if g.json:
                jprint({"stopped": []})
            else:
                print(f"{DOT} nothing running")
            return 0
    else:
        for q in a.rids:
            ent = rid_or_die(c, q)
            ids.append(ent["id"])
        if not a.rids:
            ent = rid_or_die(c, None)
            ids = [ent["id"]]
    stopped, failed = [], []
    for rid in ids:
        try:
            res = c.post("/api/stop", {"id": rid})
            (stopped if res.get("ok") else failed).append(rid)
        except CliError:
            failed.append(rid)
    if g.json:
        jprint({"stopped": stopped, "failed": failed})
    else:
        for rid in stopped:
            print(f"{OK} stopped {bold(rid)}")
        for rid in failed:
            print(f"{ERR} could not stop {rid}", file=sys.stderr)
    return 0 if not failed else 1


def _gpu_rows(metrics):
    rows = []
    for p in metrics.get("power", []) or []:
        rows.append([
            str(p.get("index")),
            (p.get("name") or "GPU")[:26],
            f"{p['watts']:.0f}W" if p.get("watts") is not None else "-",
            f"{p['cap_w']}W" if p.get("cap_w") else "-",
            f"{p['temp_c']}°C" if p.get("temp_c") is not None else "-",
            f"{p['util_pct']}%" if p.get("util_pct") is not None else "-",
            (f"{vram_bar(p['vram_used_gb'], p['vram_total_gb'])} "
             f"{p['vram_used_gb']:.1f}/{p['vram_total_gb']:.0f} GiB"
             if p.get("vram_used_gb") is not None and p.get("vram_total_gb")
             else "-"),
        ])
    return rows


def _status_payload(c):
    return {"state": c.get("/api/state"),
            "metrics": c.get("/api/metrics", timeout=10),
            "downloads": c.get("/api/downloads")}


def _print_status(payload):
    st, metrics, dls = payload["state"], payload["metrics"], payload["downloads"]
    ver = st.get("version", "?")
    cat = st.get("recipes", {}).get("catalog_ver", "?")
    running = [e for e in st.get("running", [])
               if e.get("status") not in ("stopped", "dry-run")]
    print(bold(f"  b70-launcher {ver}  {dim('catalog ' + str(cat))}"))

    gpu_rows = _gpu_rows(metrics)
    if gpu_rows:
        print(table(gpu_rows, ["#", "GPU", "DRAW", "CAP", "TEMP", "CLK", "VRAM"],
                    aligns=["<", "<", ">", ">", ">", ">", ">"]))
    else:
        print(f"  {dim('no GPU telemetry')}")

    if running:
        rows = []
        for e in running:
            m = metrics.get(e["id"], {}) or {}
            status = e.get("status", "?")
            phase = e.get("phase")
            s_disp = green(status) if status == "running" \
                else yellow(status + (f" · {phase}" if phase else ""))
            load = fmt_dur(e["load_s"]) if e.get("load_s") else "-"
            toks = m.get("tok_s")
            rows.append([
                e["id"], str(e.get("port") or "-"), s_disp, load,
                f"{toks:.1f} t/s" if toks else "-",
                str(m.get("requests", e.get("requests", 0))),
                f"{m.get('tokens_in', e.get('tokens_in', 0))}/{m.get('tokens_out', e.get('tokens_out', 0))}",
            ])
        print()
        print(table(rows, ["SERVER", "PORT", "STATUS", "LOAD", "RATE", "REQS", "TOK IN/OUT"]))
    else:
        print(f"  {dim('no engines running —')} b70 launch <model>")

    active = {k: v for k, v in (dls or {}).items()
              if v.get("state") in ("queued", "resolving", "downloading")}
    for did, d in active.items():
        print(f"  {cyan('↓')} {did}  {progress_bar(d.get('pct', 0))}"
              f"  {dim(d.get('speed', ''))} {dim(d.get('eta', ''))}")


def _status_json(payload):
    """One stable machine envelope for both `status` and `monitor`."""
    st = payload["state"]
    running = [e for e in st.get("running", [])
               if e.get("status") not in ("stopped", "dry-run")]
    return {"version": st.get("version"),
            "catalog_ver": st.get("recipes", {}).get("catalog_ver"),
            "running": running,
            "metrics": payload["metrics"],
            "downloads": payload["downloads"]}


def cmd_status(g, a):
    c = daemon_client(g)
    payload = _status_payload(c)
    if g.json:
        jprint(_status_json(payload))
    else:
        _print_status(payload)
    return 0


def cmd_monitor(g, a):
    c = daemon_client(g)
    if a.once or (not _TTY and not a.follow):
        payload = _status_payload(c)
        (lambda p: jprint(_status_json(p)) if g.json else _print_status(p))(payload)
        return 0
    try:
        while True:
            payload = _status_payload(c)
            if not g.json:
                if _TTY:
                    print("\033[H\033[2J", end="")
                else:
                    print("── " + time.strftime("%H:%M:%S") + " " + "─" * 40)
                _print_status(payload)
                print(f"\n  {dim('refresh 1.5s — Ctrl+C to exit')}", flush=True)
            else:
                jprint(_status_json(payload))
            time.sleep(1.5)
    except KeyboardInterrupt:
        return 0


def cmd_logs(g, a):
    c = daemon_client(g)
    ent = rid_or_die(c, a.rid)
    rid = ent["id"]

    def fetch():
        try:
            return (c.get(f"/api/logs?id={urllib.parse.quote(rid)}") or {}).get("lines", [])
        except CliError:
            return []

    lines = fetch()
    if g.json:
        jprint({"id": rid, "lines": lines[-a.lines:]})
        return 0
    for ln in lines[-a.lines:]:
        print(ln.rstrip())
    if a.follow:
        print(f"{dim('── following ' + rid + ' (Ctrl+C to stop) ──')}")
        try:
            while True:
                time.sleep(1.5)
                new = fetch()
                if len(new) > len(lines):
                    for ln in new[len(lines):]:
                        print(ln.rstrip(), flush=True)
                lines = new
        except KeyboardInterrupt:
            pass
    return 0


def cmd_test(g, a):
    c = daemon_client(g)
    q = a.rid
    prompt_parts = []
    if q and str(q).isdigit():
        port = int(q)
        rid = None
    else:
        ent, state = resolve_rid(c, q)
        if isinstance(ent, list):
            die("several servers match — pick one",
                hint="running: " + ", ".join(e["id"] for e in ent))
        if ent is None:
            # unresolved selector: with exactly one engine up, the first
            # positional is more likely the prompt than a server id
            running = [e for e in state.get("running", [])
                       if e.get("status") in ("running", "starting",
                                              "running (adopted)")]
            if q and len(running) == 1:
                ent = running[0]
                prompt_parts.append(q)
            else:
                hint = "running servers: " + ", ".join(e["id"] for e in running) \
                    if running else "nothing is running — launch one first: b70 launch <model>"
                die(f"no server matches {q!r}", hint=hint)
        port = ent.get("port")
        rid = ent["id"]
    if a.prompt:
        prompt_parts.append(a.prompt)
    prompt = " ".join(prompt_parts) or \
        "Why is dual Intel Arc Pro B70 effective for local MoE inference?"
    try:
        out = spin_while(
            f"{rid or 'port ' + str(port)} thinking",
            lambda: c.post("/api/test_prompt", {"port": port, "prompt": prompt},
                           timeout=130))
    except CliError as exc:
        die(str(exc))
    if g.json:
        jprint(out)
        return 0 if out.get("ok") else 1
    if not out.get("ok"):
        print(f"{ERR} {red(out.get('error', 'request failed'))}", file=sys.stderr)
        if rid:
            print(f"  {dim('engine log:')} b70 logs {rid}", file=sys.stderr)
        return 1
    toks = out.get("tokens") or 0
    lat = out.get("latency_s") or 0
    rate = f"{toks / lat:.1f} tok/s approx" if lat else "-"
    print(f"{OK} {bold(rid or 'port ' + str(port))} answered in {lat}s ({toks} tok, {rate})")
    print(f"  {dim('model:')} {out.get('model')}")
    print()
    print(out.get("reply", ""))
    return 0


def cmd_download(g, a):
    c = daemon_client(g)
    st = c.get("/api/state")
    recipes = st.get("recipes", {})
    model, cands = pick_model(recipes, a.model)
    if model is None:
        msg, hint = _model_miss(a.model, cands)
        die(msg, hint=hint)
    engine = pick_engine(model, a.engine)
    rec = model["recipes"][engine]
    dl = rec.get("download") or {}
    if not dl.get("repo"):
        die(f"recipe {model['id']}:{engine} has no downloadable artifact",
            hint="point a scan root at a local copy instead: b70 scan --roots")
    try:
        out = c.post("/api/download", {"model_id": model["id"], "engine": engine})
    except CliError as exc:
        die(str(exc))
    if out.get("error"):
        die(out["error"])
    did = out.get("id")
    if g.json:
        jprint({"id": did, "state": "started"})
    else:
        print(f"{OK} download started: {bold(did)}  {dim(dl.get('quant') or '')}")
        print(f"  {dim('repo:')} {dl.get('repo')}")
    if a.no_wait:
        return 0
    w = Wait(f"downloading {did}").start()
    terminal = {"done", "error", "cancelled"}
    try:
        while True:
            d = (c.get("/api/downloads") or {}).get(did)
            if not d:
                w.done()
                die("download vanished from tracker")
            if d.get("state") in terminal:
                w.done()
                break
            note = f"{progress_bar(d.get('pct', 0))} {d.get('speed', '')} {d.get('eta', '')}"
            w.tick(note)
            time.sleep(1.0)
    except KeyboardInterrupt:
        w.done()
        print(f"\n{WARN} interrupted — download continues in the daemon ({did})")
        return 130
    if g.json:
        jprint(d)
        return 0 if d.get("state") == "done" else 1
    if d.get("state") == "done":
        print(f"{OK} {bold(did)} downloaded → {d.get('dest', 'scan roots')}")
        print(f"  {dim('launch it:')} b70 launch {model['id']} -e {engine}")
        return 0
    print(f"{ERR} download {d.get('state')}: {d.get('error') or ''}", file=sys.stderr)
    return 1


def _has_display():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _open_url(url, g, label="ui"):
    if a_print := getattr(g, "print_url", False):
        print(url)
        return True
    if _has_display():
        opener = shutil.which("xdg-open")
        try:
            if opener:
                subprocess.Popen([opener, url], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            else:
                import webbrowser
                webbrowser.open(url)
            print(f"{OK} opened {label} → {cyan(url)}")
            return True
        except OSError:
            pass
    print(f"{DOT} headless — no display detected; URL:")
    print(f"    {cyan(url)}")
    port = urllib.parse.urlparse(url).port
    print(f"  {dim('from another machine:')} ssh -L {port}:127.0.0.1:{port} <this-host>")
    return False


def cmd_open(g, a):
    c = daemon_client(g)
    target = (a.target or "ui").strip()
    if target.lower() in ("ui", "launcher", "web", "gui"):
        tok = c.token or read_token()
        url = f"{c.api}/?token={tok}"
        if g.json:
            # machine mode is side-effect-free — report the URL, opened=false
            jprint({"url": url, "opened": False})
        else:
            _open_url(url, g, label="launcher UI")
        return 0
    # engine endpoint by rid or port
    untracked = False
    if target.isdigit():
        port = int(target)
        if not 1 <= port <= 65535:
            die(f"port {port} out of range (1–65535)", code=2)
        try:
            st = c.get("/api/state")
            live = {e.get("port") for e in st.get("running", [])
                    if e.get("status") in ("running", "starting", "running (adopted)")}
            untracked = port not in live
        except CliError:
            pass
    else:
        ent = rid_or_die(c, target)
        port = ent.get("port")
    url = f"http://127.0.0.1:{port}"
    if g.json:
        jprint({"url": url, "port": port, "tracked": not untracked, "opened": False})
    else:
        if untracked:
            print(f"{WARN} {yellow(f'port {port} is not a tracked engine — a bare port may be something else')}")
        _open_url(url, g, label=f"engine :{port}")
    return 0


def cmd_env(g, a):
    c = daemon_client(g)
    ent = rid_or_die(c, a.rid)
    port = ent.get("port")
    models = engine_models(port)
    name = models[0] if models else (ent.get("model") or "default")
    if g.json:
        jprint({"OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
                "OPENAI_API_KEY": "b70-local", "OPENAI_MODEL_NAME": name})
    else:
        print(f"export OPENAI_BASE_URL=http://127.0.0.1:{port}/v1")
        print("export OPENAI_API_KEY=b70-local")
        print(f"export OPENAI_MODEL_NAME={shlex.quote(name)}")
        print(dim("# eval \"$(b70 env " + ent["id"] + ")\" to apply"), file=sys.stderr)
    return 0


def cmd_doctor(g, a):
    c = daemon_client(g)
    st = c.get("/api/state")
    pf = st.get("preflight", {})
    scan = st.get("scan", {})
    if g.json:
        jprint({"version": st.get("version"), "api": c.api,
                "preflight": pf, "scan": {"roots": scan.get("roots"), "state": scan.get("state")},
                "token_file": str(token_file())})
        return 0
    print(bold(f"  doctor — b70-launcher {st.get('version', '?')} @ {c.api}"))
    print(f"  {dim('token file')}  {token_file()}")
    print(f"  {dim('scan roots')}  {', '.join(scan.get('roots') or ['-'])}  {dim('(' + str(scan.get('state', '?')) + ')')}")
    cards = pf.get("profile", {}).get("cards", [])
    if cards:
        rows = [[d.get("render", "?"), "B70" if d.get("b70") else "other",
                 green("accessible") if d.get("accessible") else red("no access"),
                 human_gb(d.get("vram_total_mib")),
                 f"{d['power_cap_w']}W" if d.get("power_cap_w") else "-"]
                for d in cards]
        print()
        print(table(rows, ["RENDER", "CARD", "ACCESS", "VRAM", "CAP"]))
    for b in pf.get("blockers", []):
        print(f"  {ERR} {yellow(b)}")
    for n in pf.get("notes", []):
        print(f"  {DOT} {dim(n)}")
    default = pf.get("default") or {}
    if default.get("model_id"):
        print(f"  {dim('suggested first launch:')} b70 launch {default['model_id']} -e {default.get('engine')}")
    if not pf.get("blockers"):
        print(f"\n  {OK} no blockers — ready to launch")
    return 0


def cmd_scan(g, a):
    c = daemon_client(g)
    if a.roots:
        roots = [r for r in a.roots.split(",") if r.strip()]
        try:
            out = c.post("/api/settings", {"scan_dirs": roots})
        except CliError as exc:
            die(str(exc))
        if not out.get("ok"):
            die(out.get("error", "scan roots rejected"))
        print(f"{OK} scan roots updated: {', '.join(out.get('scan_dirs', []))}")
    body = spin_while("scanning artifact roots",
                      lambda: c.get("/api/scan", timeout=120))
    if g.json:
        jprint(body)
        return 0
    print(bold(f"  artifact scan  {dim('(' + str(body.get('state')) + ')')}"))
    print(f"  {dim('roots:')} {', '.join(body.get('roots') or [])}   {dim('free:')} {body.get('free_gb')} GiB")
    cat = body.get("catalog") or []
    if cat:
        rows = [[i.get("kind"), i.get("name")[:46], human_gb(i.get("size_mib")),
                 i.get("path")] for i in cat]
        print()
        print(table(rows, ["KIND", "NAME", "SIZE", "PATH"]))
    else:
        print(f"  {dim('no artifacts found under the scan roots')}")
    print(f"\n  {dim('matched recipes appear as ● detected in:')} b70 list")
    return 0


def cmd_version(g, a):
    info = {"cli": CLI_VERSION}
    try:
        st = Client(api=g.api, token=g.token).get("/api/state", timeout=3)
        info.update({"daemon": st.get("version"),
                     "catalog_ver": st.get("recipes", {}).get("catalog_ver")})
    except CliError:
        info["daemon"] = None
    if g.json:
        jprint(info)
    else:
        print(f"b70 cli {CLI_VERSION}  ·  daemon {info.get('daemon') or 'not running'}"
              + (f"  ·  catalog {info['catalog_ver']}" if info.get("catalog_ver") else ""))
    return 0


def cmd_recipes_update(g, a):
    c = daemon_client(g)
    body = {}
    if a.model:
        model_state = c.get("/api/state").get("recipes", {})
        model, cands = pick_model(model_state, a.model)
        if model is None:
            msg, hint = _model_miss(a.model, cands)
            die(msg, hint=hint)
        body["model_id"] = model["id"]
        if a.engine:
            body["engine"] = pick_engine(model, a.engine)
    try:
        out = spin_while("fetching recipe catalog",
                         lambda: c.post("/api/recipes/update", body, timeout=60))
    except CliError as exc:
        die(str(exc))
    if g.json:
        jprint(out)
        return 0 if out.get("ok") else 1
    if out.get("ok"):
        print(f"{OK} recipes updated → catalog {out.get('catalog_ver')}"
              f"  {dim(str(out.get('applied', 0)) + ' applied')}")
        return 0
    print(f"{ERR} {out.get('error', 'update failed')}", file=sys.stderr)
    return 1


def cmd_completion(g, a):
    if a.shell != "bash":
        die("only bash completion is bundled (zsh/fish: adapt from it)")
    print(r"""# bash completion for b70 — source this file or drop it in
# ~/.local/share/bash-completion/completions/b70
_b70() {
    local cur prev cmds
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    cmds="serve down list ls recipes show launch stop status ps monitor logs test \
download open ui env doctor scan version recipes-update completion help"
    case "$prev" in
        b70) COMPREPLY=( $(compgen -W "$cmds" -- "$cur") ) ;;
        launch|show|download)
            local models
            models=$(b70 list --json 2>/dev/null | python3 -c \
                'import json,sys; print(" ".join(m["id"] for m in json.load(sys.stdin)["models"]))' 2>/dev/null)
            COMPREPLY=( $(compgen -W "$models" -- "$cur") ) ;;
        stop|logs|test|env|open)
            local rids
            rids=$(b70 status --json 2>/dev/null | python3 -c \
                'import json,sys; print(" ".join(e["id"] for e in json.load(sys.stdin)["running"]))' 2>/dev/null)
            COMPREPLY=( $(compgen -W "$rids ui" -- "$cur") ) ;;
        -e|--engine) COMPREPLY=( $(compgen -W "openvino vllm llamacpp exl3" -- "$cur") ) ;;
        --kv) COMPREPLY=( $(compgen -W "q8_0 q5_0/q4_1 q8_0/q4_1 f16 fp8" -- "$cur") ) ;;
        --gpus) COMPREPLY=( $(compgen -W "0 1 0,1 both" -- "$cur") ) ;;
        *) COMPREPLY=( $(compgen -f -W "--json --help" -- "$cur") ) ;;
    esac
}
complete -F _b70 b70""")
    return 0


# ── argparse / entry ───────────────────────────────────────────────────────

EPILOG = """\
quick start
  b70 serve                      start the launcher daemon (auto-starts on demand anyway)
  b70 list                       models × engines, detected artifacts, live marks
  b70 launch qwen38-27b          launch the recommended recipe, wait for serving
  b70 launch qwen38-27b -e vllm --ctx 65536 --port 8080
  b70 launch ./model.gguf        serve any artifact from a scan root (auto-detect)
  b70 launch qwen36-35b -e vllm --dry-run     resolved command, nothing started
  b70 status · b70 monitor       servers, GPU draw/temp/VRAM, tok/s
  b70 test <rid> "prompt"        end-to-end smoke prompt through the API
  b70 open · b70 open 8000       UI or an engine endpoint (prints SSH hint headless)
  b70 stop <rid> · b70 down      stop an engine, or the whole daemon
every launch flag maps 1:1 to the recipe API: --ctx --port --slots --kv
--gpus 0|1|0,1 --power --mtp/--no-mtp --docker/--native --extra --env K=V
"""


def build_parser():
    # global flags also live on every subparser so `b70 launch x --json`
    # works the same as `b70 --json launch x`. Every shared flag needs
    # default=SUPPRESS — otherwise the subparser's default overwrites the
    # value the top-level parser already set.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--api", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    common.add_argument("--token", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="machine-readable output")
    common.add_argument("--no-color", action="store_true", default=argparse.SUPPRESS,
                        help=argparse.SUPPRESS)
    common.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS,
                        help=argparse.SUPPRESS)
    common.add_argument("--no-autostart", action="store_true",
                        default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    common.add_argument("-V", "--version", action="version",
                        version=f"b70 {CLI_VERSION}", help=argparse.SUPPRESS)

    p = argparse.ArgumentParser(
        prog="b70",
        description="headless CLI for the B70 model launcher — all recipes, no display needed",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", help="daemon base URL (default http://127.0.0.1:7570, env B70_API)")
    p.add_argument("--token", help="API token (env B70_TOKEN or state token file)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--no-color", action="store_true", help="disable ANSI color")
    p.add_argument("-q", "--quiet", action="store_true", help="minimal chatter")
    p.add_argument("--no-autostart", action="store_true",
                   help="fail instead of auto-starting the daemon")
    p.add_argument("-V", "--version", action="version",
                   version=f"b70 {CLI_VERSION}")
    sub = p.add_subparsers(dest="cmd", metavar="<command>", parser_class=argparse.ArgumentParser)

    def add(name, **kw):
        return sub.add_parser(name, parents=[common], **kw)

    s = add("serve", help="start the launcher daemon (idempotent)")
    s.add_argument("--fg", "--foreground", dest="foreground", action="store_true",
                   help="run the daemon in the foreground (tmux/systemd)")
    s.add_argument("--kill", action="store_true", help="kill whatever holds the port first")
    s.set_defaults(fn=cmd_serve)

    s = add("down", help="stop the launcher daemon")
    s.add_argument("--stop-engines", action="store_true",
                   help="also stop running model servers")
    s.set_defaults(fn=cmd_down)

    s = add("list", aliases=["ls", "recipes"],
            help="list models × engine recipes")
    s.add_argument("-e", "--engine", help="only this engine")
    s.set_defaults(fn=cmd_list)

    s = add("show", help="model + recipe detail")
    s.add_argument("model")
    s.add_argument("-e", "--engine")
    s.set_defaults(fn=cmd_show)

    s = add("launch", help="launch a recipe (or an interactive picker)")
    s.add_argument("model", nargs="?", help="model id/name, or a path to a .gguf/IR artifact")
    s.add_argument("-e", "--engine", help="openvino | vllm | llamacpp | exl3")
    s.add_argument("--path", help="custom artifact path (must be under a scan root)")
    s.add_argument("-c", "--ctx", type=int, help="context tokens (recipe default if unset)")
    s.add_argument("-p", "--port", type=int, help="endpoint port (default 8000)")
    s.add_argument("--gpus", help="0 | 1 | 0,1 | both (default: recipe topology or 0)")
    s.add_argument("--kv", help="KV cache dtype: q8_0 | q5_0/q4_1 | q8_0/q4_1 | f16 | fp8")
    s.add_argument("-n", "--slots", type=int, help="max parallel sequences")
    s.add_argument("--power", type=int, help="watts hint (advisory; launcher never sets caps)")
    mtp = s.add_mutually_exclusive_group()
    mtp.add_argument("--mtp", dest="mtp", action="store_true", default=None,
                     help="force MTP speculative decoding on")
    mtp.add_argument("--no-mtp", dest="mtp", action="store_false",
                     help="disable MTP speculative decoding")
    dk = s.add_mutually_exclusive_group()
    dk.add_argument("--docker", dest="use_docker", action="store_true", default=None,
                    help="force the container path")
    dk.add_argument("--native", dest="use_docker", action="store_false",
                    help="force the native llama_bin path")
    s.add_argument("--extra", help="extra raw engine flags, e.g. --extra '--verbose'")
    s.add_argument("-E", "--env", action="append",
                   help="extra env var NAME=value (repeatable)")
    s.add_argument("-d", "--dry-run", action="store_true",
                   help="print the resolved launch plan without starting anything")
    s.add_argument("--no-wait", action="store_true",
                   help="return immediately after the launch is accepted")
    s.add_argument("-t", "--timeout", type=int, default=900,
                   help="seconds to wait for the endpoint (default 900)")
    s.add_argument("--open", action="store_true",
                   help="open the endpoint in a browser once serving")
    s.set_defaults(fn=cmd_launch)

    s = add("stop", help="stop a running engine (default: the only one)")
    s.add_argument("rids", nargs="*", help="server id(s), model name, or port")
    s.add_argument("--all", action="store_true", help="stop every running engine")
    s.set_defaults(fn=cmd_stop)

    s = add("status", aliases=["ps"], help="daemon dashboard")
    s.set_defaults(fn=cmd_status)

    s = add("monitor", help="live GPU + engine monitor")
    s.add_argument("--once", action="store_true", help="print one snapshot and exit")
    s.add_argument("-f", "--follow", action="store_true",
                 help="keep streaming snapshots (default on a TTY)")
    s.set_defaults(fn=cmd_monitor)

    s = add("logs", help="tail an engine log")
    s.add_argument("rid", nargs="?", help="server id (default: the only running one)")
    s.add_argument("-n", "--lines", type=int, default=60)
    s.add_argument("-f", "--follow", action="store_true")
    s.set_defaults(fn=cmd_logs)

    s = add("test", help="send a smoke prompt to a running engine")
    s.add_argument("rid", nargs="?", help="server id or bare port (default: only running)")
    s.add_argument("prompt", nargs="?", help="prompt text")
    s.set_defaults(fn=cmd_test)

    s = add("download", help="fetch a recipe artifact from Hugging Face")
    s.add_argument("model")
    s.add_argument("-e", "--engine")
    s.add_argument("--no-wait", action="store_true")
    s.set_defaults(fn=cmd_download)

    s = add("open", help="open the UI or an engine endpoint")
    s.add_argument("target", nargs="?", help="ui (default) | server id | port")
    s.add_argument("--print", dest="print_url", action="store_true",
                   help="print the URL instead of opening")
    s.set_defaults(fn=cmd_open)

    s = add("ui", help="shortcut for `open ui` — the launcher web UI")
    s.add_argument("--print", dest="print_url", action="store_true",
                   help="print the URL instead of opening")
    s.set_defaults(fn=cmd_open, target="ui")

    s = add("env", help="print OpenAI client env vars for a server")
    s.add_argument("rid", nargs="?", help="server id (default: only running)")
    s.set_defaults(fn=cmd_env)

    s = add("doctor", help="preflight: GPUs, blockers, scan roots")
    s.set_defaults(fn=cmd_doctor)

    s = add("scan", help="artifact scan results")
    s.add_argument("--roots", help="set scan roots (comma-separated), then rescan")
    s.set_defaults(fn=cmd_scan)

    s = add("recipes-update", help="apply published recipe catalog updates")
    s.add_argument("model", nargs="?", help="limit to one model")
    s.add_argument("-e", "--engine")
    s.set_defaults(fn=cmd_recipes_update)

    s = add("version", help="CLI + daemon + catalog versions")
    s.set_defaults(fn=cmd_version)

    s = add("completion", help="print a shell completion script")
    s.add_argument("shell", nargs="?", default="bash", choices=["bash"])
    s.set_defaults(fn=cmd_completion)

    s = add("help", help="show this help")
    s.set_defaults(fn=lambda g, a: (p.print_help(), 0)[1])

    return p


def main(argv=None):
    global _JSON
    p = build_parser()
    args = p.parse_args(argv)
    _init_color(args.no_color)
    _JSON = bool(args.json)
    if not args.cmd:
        if _TTY and not args.json:
            banner()
        p.print_help()
        return 0
    try:
        return args.fn(args, args)
    except DaemonDown as exc:
        if args.json:
            jprint({"ok": False, "error": str(exc), "code": exc.code})
        else:
            print(f"{ERR} {red(str(exc))}", file=sys.stderr)
            print(f"  {dim('start it:')} b70 serve   {dim('or check:')} b70 doctor",
                  file=sys.stderr)
        return exc.code
    except CliError as exc:
        if args.json:
            jprint({"ok": False, "error": str(exc)})
        else:
            print(f"{ERR} {red(str(exc))}", file=sys.stderr)
        return exc.code
    except KeyboardInterrupt:
        print(f"\n{WARN} interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
        return 0


if __name__ == "__main__":
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # `b70 status | head` exits clean
    sys.exit(main())
