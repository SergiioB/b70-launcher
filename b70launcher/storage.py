"""Storage manager: what the launcher's models and engine images occupy, and safe removal.

Read side: ``snapshot()`` lists every model artifact the scanner found, every
partial download, and the Docker images the recipes pin, with exact sizes, the
recipes that use each one, when it was last launched, and per-filesystem free
space.

Write side: ``delete(ids)`` removes items by the opaque ids of the latest
snapshot. The API never accepts a path. An item is refused when

* a running server uses it,
* it is not on the deletable list (an image the launcher does not manage), or
* its resolved path is outside every scan root, is a root itself, or is a link.

Docker images are removed without ``--force``: Docker's own "in use" check is
the last guard.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from . import catalog, downloads, paths, scan
from .locks import LOCK
from .runtime import RUNNING, docker_cmd

EXL3_SOCK = "/run/b70-exl3-docker.sock"
CACHE_TTL = 30.0
PARTIAL_MIN_AGE = 15 * 60     # a .part touched in the last 15 min may be an active download
# runtime.py keeps "stopping" and "stop_failed" in its reserved set: a failed stop
# leaves the engine alive with the weights mapped, so its files are not free.
LIVE_STATES = ("starting", "loading", "ready", "running", "serving", "stopping", "stop_failed")
DOWNLOAD_STATES = ("queued", "resolving", "downloading")

_CACHE = {"ts": 0.0, "snap": None}
_CACHE_LOCK = threading.Lock()
_SCAN_LOCK = threading.Lock()    # one full disk walk at a time


# ---------------------------------------------------------------- helpers
def _item_id(kind, key):
    return kind + ":" + hashlib.sha1(key.encode()).hexdigest()[:10]


def tree_size(path):
    """(bytes, newest mtime) of a file or directory tree; links are not followed."""
    total, newest = 0, 0.0
    stack = [str(path)]
    while stack:
        cur = stack.pop()
        try:
            st = os.lstat(cur)
        except OSError:
            continue
        newest = max(newest, st.st_mtime)
        if os.path.islink(cur) or not os.path.isdir(cur):
            total += st.st_size
            continue
        try:
            with os.scandir(cur) as it:
                for e in it:
                    stack.append(e.path)
        except OSError:
            continue
    return total, newest


def _mount_of(path):
    """Mount point holding `path`, from /proc/mounts (longest prefix)."""
    best = "/"
    try:
        real = os.path.realpath(path)
        for line in Path("/proc/mounts").read_text().splitlines():
            mnt = line.split()[1].replace("\\040", " ")
            if (real == mnt or real.startswith(mnt.rstrip("/") + "/")) and len(mnt) >= len(best):
                best = mnt
    except OSError:
        pass
    return best


def _usage_probe(path):
    """disk_usage for the nearest existing, readable ancestor of `path`."""
    probe = Path(path)
    while True:
        try:
            u = shutil.disk_usage(probe)
            return probe, u
        except OSError:
            if probe == probe.parent:
                return None, None
            probe = probe.parent


def _roots():
    out = []
    for r in scan.scan_roots():
        try:
            out.append(Path(r).resolve())
        except OSError:
            continue
    return out


def _under(path, root):
    try:
        Path(path).resolve().relative_to(root)
        return True
    except (ValueError, OSError):
        return False


def _download_paths():
    """Destinations an active download is writing into right now."""
    with LOCK:
        live = [dict(e) for e in downloads.DOWNLOADS.values() if e.get("state") in DOWNLOAD_STATES]
    return [str(e[k]) for e in live for k in ("dest", "dest_dir") if e.get(k)]


def _running_paths():
    """Artifact paths of live servers, and the image ids/names their containers use."""
    paths_, images = [], set()
    with LOCK:
        live = [e for e in RUNNING.values() if e.get("status") in LIVE_STATES]
    for e in live:
        for k in ("detected_path", "model_path", "path"):
            if e.get(k):
                paths_.append(str(e[k]))
        if e.get("image"):
            images.add(str(e["image"]))
    return paths_, images, live


def _last_launched():
    """{(model name, engine): 'YYYY-MM-DD HH:MM:SS'} from the usage history."""
    out = {}
    try:
        data = json.loads(paths.USAGE_PATH.read_text())
    except (OSError, ValueError):
        return out
    for s in data.get("sessions", []):
        key = (s.get("model"), s.get("engine"))
        ended = s.get("ended") or s.get("started") or ""
        if ended > out.get(key, ""):
            out[key] = ended
    return out


# ---------------------------------------------------------------- docker
def _run(argv, timeout=12):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)


def _daemons():
    out = [("docker", "")]
    if os.path.exists(EXL3_SOCK):
        out.append(("exl3 docker", EXL3_SOCK))
    return out


_UNITS = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12,
          "KIB": 1024, "MIB": 1024 ** 2, "GIB": 1024 ** 3, "TIB": 1024 ** 4}


def parse_size(text):
    """'31.8GB' -> bytes (docker prints decimal units); 0 when unparseable."""
    m = re.match(r"^\s*([\d.]+)\s*([A-Za-z]+)\s*$", str(text or ""))
    if not m:
        return 0
    return int(float(m.group(1)) * _UNITS.get(m.group(2).upper(), 0))


def _parse_pin(ref):
    """'repo:tag@sha256:abc' -> (repo, digest hex or '')."""
    ref = str(ref or "")
    digest = ""
    if "@sha256:" in ref:
        ref, digest = ref.split("@sha256:", 1)
    repo = re.sub(r":[^/:]+$", "", ref)
    return repo, digest


def pinned_images():
    """{(repo, digest): [(model_id, model name, engine)]} from the recipe catalog."""
    pins = {}
    for model in catalog.RECIPES.get("models", []):
        for engine, rec in (model.get("recipes") or {}).items():
            img = rec.get("image")
            if not img:
                continue
            repo, digest = _parse_pin(img)
            pins.setdefault((repo, digest), []).append((model["id"], model.get("name") or model["id"], engine))
    return pins


def _docker_images(sock):
    """[{id, repo_tags, repo_digests, size, created}] for one daemon, or (None, error)."""
    rc, out, err = _run(docker_cmd(sock) + ["image", "ls", "-q", "--no-trunc"])
    if rc:
        return None, err or "docker not reachable"
    ids = sorted(set(out.split()))
    if not ids:
        return [], ""
    # `image ls` reports disk usage (what `docker images` prints); `inspect .Size` is the
    # compressed size on the containerd image store and understates it about 4x.
    rc, lsout, _ = _run(docker_cmd(sock) + ["image", "ls", "--no-trunc", "--format", "{{.ID}}|{{.Size}}"])
    disk = {}
    for line in lsout.splitlines():
        iid, _, human = line.partition("|")
        disk[iid] = parse_size(human)
    fmt = "{{.Id}}|{{.Size}}|{{json .RepoTags}}|{{json .RepoDigests}}|{{.Created}}"
    rc, out, err = _run(docker_cmd(sock) + ["image", "inspect", "--format", fmt] + ids, timeout=25)
    if rc:
        return None, err or "docker inspect failed"
    items = []
    for line in out.splitlines():
        parts = line.split("|", 4)
        if len(parts) < 5:
            continue
        try:
            items.append({"id": parts[0], "size": disk.get(parts[0]) or int(parts[1]),
                          "repo_tags": json.loads(parts[2]) or [],
                          "repo_digests": json.loads(parts[3]) or [],
                          "created": parts[4]})
        except ValueError:
            continue
    return items, ""


def _docker_in_use(sock):
    rc, out, _ = _run(docker_cmd(sock) + ["ps", "-a", "--no-trunc", "--format", "{{.Image}}|{{.ID}}"])
    if rc:
        return set()
    names = set()
    for line in out.splitlines():
        names.add(line.split("|", 1)[0])
    # `docker ps` prints a tag or a short id; resolve every name to a full image id
    ids = set()
    for n in names:
        rc, o, _ = _run(docker_cmd(sock) + ["image", "inspect", "--format", "{{.Id}}", n])
        if not rc and o.strip():
            ids.add(o.strip())
    return ids


def _docker_root(sock):
    rc, out, _ = _run(docker_cmd(sock) + ["info", "--format", "{{.DockerRootDir}}"])
    return out.strip() if not rc and out.strip() else ("/var/lib/docker" if not sock else "")


# ---------------------------------------------------------------- snapshot
_GENERIC_NAMES = {"container", "snapshots", "main", "model", "models", "weights", "checkpoint"}


def _display_name(name, path):
    """A folder called 'container' says nothing; prefix its parent folder."""
    if name.lower() in _GENERIC_NAMES:
        return f"{Path(path).parent.name}/{name}"
    return name


def _model_hint(name):
    n = name.lower()
    if n.startswith("mmproj"):
        return "Vision projector for a GGUF model; the model needs it to read images"
    if any(k in n for k in ("dflash", "draft", "-mtp")):
        return "Speculative-decoding draft model; recipes that draft need it"
    return "No recipe in this catalog uses it"


def _model_items(now):
    sc = scan.scan_blocking()
    recipes = catalog.RECIPES.get("models", [])
    owners = {}
    for model in recipes:
        for engine in model.get("recipes", {}):
            try:
                det, _reason = scan.detect_status(model, engine)
            except Exception:
                det = None
            if det and det.get("path"):
                owners.setdefault(os.path.realpath(det["path"]), []).append(
                    {"model_id": model["id"], "name": model.get("name") or model["id"], "engine": engine})
            try:
                draft = scan._find_draft_dir(model["recipes"][engine])
            except Exception:
                draft = None
            if draft:
                owners.setdefault(os.path.realpath(draft), []).append(
                    {"model_id": model["id"], "name": model.get("name") or model["id"], "engine": engine,
                     "role": "draft model"})
    last = _last_launched()
    run_paths, _imgs, _live = _running_paths()
    run_real = [os.path.realpath(p) for p in run_paths]
    dl_real = [os.path.realpath(p) for p in _download_paths()]
    items = []
    for c in sc.get("catalog", []):
        path = c.get("path")
        if not path:
            continue
        real = os.path.realpath(path)
        size, mtime = tree_size(path)
        used_by = owners.get(real, [])
        launched = max((last.get((u["name"], u["engine"]), "") for u in used_by), default="")
        busy = any(real == r or r.startswith(real + os.sep) or real.startswith(r + os.sep) for r in run_real)
        dl_busy = any(real == d or d.startswith(real + os.sep) or real.startswith(d + os.sep) for d in dl_real)
        busy = busy or dl_busy
        items.append({
            "id": _item_id("model", real), "kind": "model", "format": c.get("kind"),
            "name": _display_name(c.get("name") or "", path), "path": path,
            "size_bytes": size, "mtime": mtime,
            "used_by": used_by, "last_launched": launched or None,
            "in_use": busy, "deletable": not busy,
            "why_kept": ("a download is writing into it" if dl_busy
                         else "a running server is using it" if busy else None),
            "suggested": False,
            "hint": None if used_by else _model_hint(c.get("name") or ""),
        })
    # partial downloads: <file>.part inside a scan root
    seen = set()
    for root in _roots():
        stack = [(str(root), 0)]
        while stack:
            cur, depth = stack.pop()
            try:
                with os.scandir(cur) as it:
                    for e in it:
                        try:
                            if e.is_symlink():
                                continue
                            if e.is_dir(follow_symlinks=False):
                                if depth < 6 and not e.name.startswith(".") and e.name not in scan.SKIP_DIRS:
                                    stack.append((e.path, depth + 1))
                            elif e.name.endswith(".part") and e.path not in seen:
                                seen.add(e.path)
                                st = e.stat(follow_symlinks=False)
                                age = now - st.st_mtime
                                active = age < PARTIAL_MIN_AGE
                                items.append({
                                    "id": _item_id("partial", e.path), "kind": "partial", "format": "partial",
                                    "name": e.name, "path": e.path, "size_bytes": st.st_size, "mtime": st.st_mtime,
                                    "used_by": [], "last_launched": None, "in_use": active,
                                    "deletable": not active,
                                    "why_kept": "touched in the last 15 minutes, a download may be running" if active else None,
                                    "suggested": not active,
                                    "hint": "Unfinished download; a new download of this file restarts from zero"
                                            if not active else None})
                        except OSError:
                            continue
            except OSError:
                continue
    return items, sc


def _image_items(volumes_paths, docker_roots, on_disk):
    pins = pinned_images()
    pinned_repos = {r for (r, _d) in pins}
    pinned_digests = {d: v for (r, d), v in pins.items() if d}
    tag_only_pins = {r: v for (r, d), v in pins.items() if not d}
    _run_paths, run_imgs, _live = _running_paths()
    items, daemons = [], []
    for dname, sock in _daemons():
        imgs, err = _docker_images(sock)
        daemons.append({"name": dname, "socket": sock or "default", "ok": imgs is not None, "error": err or None})
        if imgs is None:
            continue
        in_use = _docker_in_use(sock)
        root = _docker_root(sock)
        if root:
            volumes_paths.append(root)
            docker_roots[dname] = root
        for im in imgs:
            digests = [d.split("@sha256:", 1)[1] for d in im["repo_digests"] if "@sha256:" in d]
            repos = {t.rsplit(":", 1)[0] for t in im["repo_tags"] if ":" in t} | \
                    {d.split("@", 1)[0] for d in im["repo_digests"]}
            current = [pinned_digests[d] for d in digests if d in pinned_digests]
            tag_repos = {t.rsplit(":", 1)[0] for t in im["repo_tags"] if ":" in t}
            for repo in tag_repos:
                if repo in tag_only_pins:   # a recipe pins this repo by name, with no digest
                    current.append(tag_only_pins[repo])
            managed = bool(current) or bool(repos & pinned_repos)
            used_by = [{"model_id": m, "name": n, "engine": e} for lst in current for (m, n, e) in lst]
            busy = im["id"] in in_use
            variant = managed and not current
            local_only = bool(tag_repos & set(tag_only_pins))   # no digest pin: built here, cannot be pulled
            idle = bool(current) and not local_only and not any(
                (u["model_id"], u["engine"]) in on_disk for u in used_by)
            label = (im["repo_tags"][0] if im["repo_tags"]
                     else (im["repo_digests"][0].split("@", 1)[0] + "@" + im["id"][7:19] if im["repo_digests"]
                           else im["id"][7:19]))
            hint = None
            if idle:
                names = ", ".join(sorted({f"{u['name']} ({u['engine']})" for u in used_by})[:3])
                hint = f"No downloaded model uses it; pulled again when you launch {names}"
            elif local_only:
                hint = "Built locally; the launcher cannot download it again"
            elif variant:
                hint = "Same repository as a pinned image, different build: an older pin or a local build"
            elif not managed:
                hint = ("Built locally from a launcher image; manage it with docker"
                        if repos & pinned_repos else "Not a launcher image; manage it with docker")
            items.append({
                "id": _item_id("image", dname + im["id"]), "kind": "image", "format": "docker image",
                "name": label, "path": f"{dname} · {im['id'][7:19]}", "size_bytes": im["size"],
                "mtime": None, "created": im["created"], "daemon": dname, "socket": sock,
                "image_id": im["id"], "managed": managed,
                "used_by": used_by, "last_launched": None, "in_use": busy,
                "deletable": managed and not busy,
                "why_kept": "a container is using it" if busy else (None if managed else "not a pinned launcher image"),
                "suggested": idle and not busy, "hint": hint,
            })
    return items, daemons


def snapshot(force=False):
    with _CACHE_LOCK:
        if not force and _CACHE["snap"] and time.time() - _CACHE["ts"] < CACHE_TTL:
            return _CACHE["snap"]
    asked = time.time()
    with _SCAN_LOCK:
        with _CACHE_LOCK:
            # a walk that finished after this call asked for one is the answer
            if _CACHE["snap"] and _CACHE["ts"] >= asked:
                return _CACHE["snap"]
        return _build()


def _build():
    now = time.time()
    vol_paths = [str(r) for r in _roots()]
    model_dir = Path(str(catalog.SETTINGS.get("models_dir") or "~")).expanduser()
    vol_paths.append(str(model_dir))
    models, sc = _model_items(now)
    docker_roots = {}
    on_disk = {(u["model_id"], u["engine"]) for m in models for u in m["used_by"]}
    images, daemons = _image_items(vol_paths, docker_roots, on_disk)
    items = models + images

    vols = {}
    for p in vol_paths:
        probe, u = _usage_probe(p)
        if not probe:
            continue
        dev = os.stat(probe).st_dev
        v = vols.setdefault(dev, {"mount": _mount_of(probe), "total_bytes": u.total, "used_bytes": u.used,
                                  "free_bytes": u.free, "paths": [], "managed_bytes": 0})
        if p not in v["paths"]:
            v["paths"].append(p)
    for it in items:
        base = it["path"] if it["kind"] != "image" else None
        if base is None:
            continue
        probe, _u = _usage_probe(base)
        if probe:
            v = vols.get(os.stat(probe).st_dev)
            if v:
                v["managed_bytes"] += it["size_bytes"]
                it["mount"] = v["mount"]
    # images live under their daemon's data root
    for it in images:
        probe, _u = _usage_probe(docker_roots.get(it["daemon"], ""))
        if probe:
            v = vols.get(os.stat(probe).st_dev)
            if v:
                v["managed_bytes"] += it["size_bytes"]
                it["mount"] = v["mount"]
    items.sort(key=lambda i: -i["size_bytes"])
    summary = {
        "models_bytes": sum(i["size_bytes"] for i in items if i["kind"] == "model"),
        "images_bytes": sum(i["size_bytes"] for i in items if i["kind"] == "image" and i["managed"]),
        "partial_bytes": sum(i["size_bytes"] for i in items if i["kind"] == "partial"),
        "suggested_bytes": sum(i["size_bytes"] for i in items if i["suggested"] and i["deletable"]),
        "suggested_count": sum(1 for i in items if i["suggested"] and i["deletable"]),
    }
    snap = {"ts": now, "volumes": sorted(vols.values(), key=lambda v: v["mount"]),
            "items": items, "daemons": daemons, "summary": summary,
            "roots": [str(r) for r in _roots()], "scan_state": sc.get("state")}
    with _CACHE_LOCK:
        _CACHE.update({"ts": time.time(), "snap": snap})   # finish time: bursts that asked earlier share it
    return snap


def invalidate():
    with _CACHE_LOCK:
        _CACHE.update({"ts": 0.0, "snap": None})


# ---------------------------------------------------------------- delete
def _delete_model(item):
    path = item["path"]
    real = Path(os.path.realpath(path))
    roots = _roots()
    if not any(_under(real, r) and real != r for r in roots):
        raise ValueError("outside the model folders; refusing to delete")
    if os.path.islink(path):
        raise ValueError("is a link; refusing to delete")
    if real.is_dir():
        shutil.rmtree(real)
    else:
        real.unlink()


def _delete_image(item):
    rc, _out, err = _run(docker_cmd(item["socket"]) + ["image", "rm", item["image_id"]], timeout=60)
    if rc:
        raise ValueError(err.splitlines()[-1] if err else "docker image rm failed")


def delete(ids, dry_run=False):
    """Remove the items with these ids. Returns {results, freed_bytes, snapshot}."""
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
        raise ValueError("ids must be a non-empty list of item ids")
    snap = snapshot(force=True)
    by_id = {i["id"]: i for i in snap["items"]}
    results, freed = [], 0
    for iid in ids:
        item = by_id.get(iid)
        if not item:
            results.append({"id": iid, "ok": False, "error": "not found; refresh the list"})
            continue
        row = {"id": iid, "name": item["name"], "kind": item["kind"], "size_bytes": item["size_bytes"]}
        if not item["deletable"]:
            results.append(dict(row, ok=False, error=item["why_kept"] or "not deletable"))
            continue
        if dry_run:
            results.append(dict(row, ok=True, dry_run=True))
            freed += item["size_bytes"]
            continue
        try:
            (_delete_image if item["kind"] == "image" else _delete_model)(item)
            results.append(dict(row, ok=True))
            freed += item["size_bytes"]
        except (OSError, ValueError) as exc:
            results.append(dict(row, ok=False, error=str(exc)))
    if not dry_run:
        invalidate()
        scan.start_scan_async(force=True)
    return {"results": results, "freed_bytes": freed, "dry_run": bool(dry_run),
            "snapshot": snapshot(force=True) if not dry_run else None}
