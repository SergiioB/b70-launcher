"""Paths, version and small IO primitives.

Importing this module never touches the filesystem (no mkdir, no token
write): ``init_state`` does that, called from the entry point once the
port is known (the token file is port-scoped).
"""
import json
import os
import secrets
import sys
import tempfile
from pathlib import Path

VERSION = "0.6.14"  # single authority — release.py asserts it matches

IS_WIN = os.name == "nt"

# Install root: repo checkout, installed app dir, or PyInstaller bundle.
HERE = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
if getattr(sys, "frozen", False) and not (HERE / "web" / "index.html").is_file():
    bundled = Path(getattr(sys, "_MEIPASS", HERE))
    if (bundled / "web" / "index.html").is_file():
        HERE = bundled

DATA = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "b70-launcher"
LOGDIR = DATA / "logs"
OVERRIDE_PATH = DATA / "settings-override.json"
USAGE_PATH = DATA / "usage-history.json"
STATE_PATH = DATA / "servers-state.json"
REMOTE_RECIPES_PATH = DATA / "recipes-remote.json"


def token_path(port):
    """One token file per daemon port, so a second daemon on --port never
    overwrites the first daemon's token (audit L10)."""
    return DATA / f"token-{int(port)}"


def read_token(port=None):
    """Token for a daemon port; falls back to the pre-0.6.0 shared file."""
    if port is not None:
        try:
            return token_path(port).read_text().strip()
        except OSError:
            pass
    try:
        return (DATA / "token").read_text().strip()
    except OSError:
        return ""


def write_token(port):
    tok = secrets.token_urlsafe(32)
    # same-box processes can read the token file (0600) to call the API, e.g.
    # scripts or `curl -H "X-Launcher-Token: $(cat .../token-7570)"`
    try:
        fd = os.open(token_path(port), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(fd, 0o600)  # an existing file keeps its old mode otherwise
        with os.fdopen(fd, "w") as stream:
            stream.write(tok)
    except OSError:
        pass
    return tok


def init_state(port):
    """Create the state dirs and this daemon's token. Called once from the
    entry point — never at import time."""
    DATA.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(DATA, 0o700)  # state (paths, cmdlines, usage) is private
    except OSError:
        pass
    LOGDIR.mkdir(exist_ok=True)
    return write_token(port)


def prune_logs(keep=20, protected=()):
    """Bound the log dir, but never delete a log still referenced by the
    persisted server state or a live entry (audit L1)."""
    protected = {str(Path(p)) for p in protected if p}
    try:
        logs = sorted(LOGDIR.glob("*.log"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return
    for old in logs[:-keep] if len(logs) > keep else []:
        if str(old) in protected:
            continue
        try:
            old.unlink()
        except OSError:
            pass


def persisted_log_paths():
    """Log paths referenced by servers-state.json (adoption lifeline)."""
    try:
        snap = json.loads(STATE_PATH.read_text())
    except Exception:
        return []
    if not isinstance(snap, dict):
        return []
    return [e.get("log") for e in snap.values()
            if isinstance(e, dict) and e.get("log")]


def atomic_write(path, text):
    """Crash-safe write: mkstemp in the target dir + os.replace. Concurrent
    writers serialize on STATE_LOCK at the call sites (audit M12)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
