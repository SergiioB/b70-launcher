"""In-app launcher update: the same steps as downloads/get.sh, run by the daemon.

1. Re-read version.json from the update URL.
2. Accept only a tarball next to version.json (same origin and folder).
3. Download it, check its SHA256 against version.json, extract it safely.
4. Run the archive's rootless packaging/install.sh (it replaces ~/.local/share/b70-launcher).
5. Restart: a detached shell starts ~/.local/bin/b70-launcher after this daemon exits.

Only an installed launcher updates itself; a source checkout is refused.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
import urllib.request
from pathlib import Path

from . import catalog, paths

MAX_TARBALL = 64 << 20
STATE = {"state": "idle", "version": "", "error": "", "log": ""}
_LOCK = threading.Lock()


def installed_dir():
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "b70-launcher"


def _launcher_bin():
    return Path.home() / ".local" / "bin" / "b70-launcher"


def _fetch(url, limit):
    req = urllib.request.Request(url, headers=catalog.UA)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read(limit + 1)
    if len(data) > limit:
        raise ValueError("download is larger than expected")
    return data


def _plan():
    doc = json.loads(_fetch(catalog.UPDATE_URL, 1 << 20).decode())
    version, url, digest = str(doc.get("version", "")), str(doc.get("tarball_url", "")), str(doc.get("sha256", ""))
    if not version or catalog._ver_key(version) <= catalog._ver_key(paths.VERSION):
        raise ValueError(f"No newer launcher than {paths.VERSION} is published.")
    base = catalog.UPDATE_URL.rsplit("/", 1)[0] + "/"
    if not url.startswith(base) or "/" in url[len(base):]:
        raise ValueError("version.json points to a tarball outside its own folder; refusing.")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("version.json has no valid SHA256.")
    return version, url, digest


def _install(version, url, digest):
    with tempfile.TemporaryDirectory(prefix="b70-update-") as tmp:
        data = _fetch(url, MAX_TARBALL)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("SHA256 mismatch: the download does not match version.json. Nothing was installed.")
        tgz = Path(tmp) / "launcher.tar.gz"
        tgz.write_bytes(data)
        with tarfile.open(tgz) as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(tmp, filter="data")  # rejects absolute paths, links out, devices
            else:  # Python < 3.9.17 / 3.10.12 / 3.11.4
                for m in tar.getmembers():
                    if m.name.startswith("/") or ".." in Path(m.name).parts or not (m.isfile() or m.isdir()):
                        raise ValueError(f"Unsafe archive member {m.name!r}; nothing was installed.")
                tar.extractall(tmp)
        src = next(Path(tmp).glob("b70-launcher-*-linux-source"), None)
        if not src or not (src / "packaging" / "install.sh").is_file():
            raise ValueError("Unexpected archive layout; nothing was installed.")
        res = subprocess.run(["sh", str(src / "packaging" / "install.sh")],
                             capture_output=True, text=True, timeout=300)
        log = (res.stdout + res.stderr)[-4000:]
        if res.returncode:
            raise RuntimeError(f"Installer failed (exit {res.returncode}). {log[-400:]}")
        return log


def _worker():
    try:
        version, url, digest = _plan()
        STATE.update(version=version)
        STATE.update(state="done", log=_install(version, url, digest))
    except Exception as exc:
        STATE.update(state="failed", error=str(exc))


def start_install(_cfg=None):
    """POST /api/update/install: start the update in the background."""
    if Path(paths.HERE).resolve() != installed_dir().resolve():
        raise ValueError(f"This launcher runs from {paths.HERE}, not the installed copy. "
                         "Update it with: curl -fsSL https://xecores.com/downloads/get.sh | sh")
    with _LOCK:
        if STATE["state"] == "running":
            return status()
        STATE.update(state="running", version="", error="", log="")
        threading.Thread(target=_worker, daemon=True).start()
    return status()


def status(_cfg=None):
    return {k: STATE[k] for k in ("state", "version", "error")}


def restart(_cfg=None):
    """POST /api/update/restart: start the new launcher once this one has exited."""
    if STATE["state"] != "done":
        raise ValueError("No installed update to restart into.")
    exe = _launcher_bin()
    if not exe.is_file():
        raise ValueError(f"{exe} is missing; start B70 Launcher from the menu.")
    # Same arguments (port, --no-open) as this daemon; start once this PID is gone,
    # otherwise the new launcher would find the port busy and exit.
    waiter = 'while kill -0 "$1" 2>/dev/null; do sleep 0.5; done; shift; exec "$@"'
    subprocess.Popen(["sh", "-c", waiter, "b70-restart", str(os.getpid()), str(exe), *sys.argv[1:]],
                     start_new_session=True, stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from . import runtime
    threading.Thread(target=runtime.graceful_shutdown, args=(False,), daemon=True).start()
    return {"ok": True}
