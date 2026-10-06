#!/usr/bin/env python3
"""Local UI for inspecting B70 hardware, approved model artifacts and launch recipes.

Python stdlib only. Linux GPU execution requires a verified single B70, driver,
Docker, model and explicit launch consent. Other Intel cards and Windows GPU
execution are not validated by this release.

0.4.0: the model scan runs in the background (window paints instantly), the UI
opens in a native WebKitGTK app window when available, token counts come from
the engine's own Prometheus /metrics endpoint, per-session token usage is stored
under the state directory on exit, and running engines are re-adopted after an
app restart instead of being orphaned.
0.4.4: EXL3 XPU engine (isolated dockerd on /run/b70-exl3-docker.sock, auto-started
on demand), Qwen3.8-27B vLLM route switched to the AutoRound W4A16 artifact with
the patched MTP4 recipe and prefix caching disabled, and the OpenVINO recipe
points at the on-disk INT4-OV artifact.
0.4.5: honesty + normie pass — real on-disk artifact sizes instead of fixed
labels, honest Not-found states with per-recipe verified Hugging Face download
blocks (copyable curl / hf download commands, extra_files sidecars like
mmproj), draft sidecars excluded from model detection, broken hub exports
never offered, topology-aware dual-B70 notes, recommended engine follows each
recipe's recommendation, preview no longer writes harness configs or starts
the EXL3 dockerd, port validation on test_prompt, GPU telemetry labeled
"GPU clock" (the xe driver exposes no busy% counter), and file-path inputs
carry full-path tooltips.
0.4.7: portability + recipe freshness — shipped defaults contain no
machine-specific paths (scan roots, EXL3 dockerd data root, recipe artifact
paths all resolve portably), recipe match/prefer fields accept lists,
draft_model resolves through the disk scan, and when the published recipe
catalog is newer than the shipped one the UI surfaces a per-recipe notice
("a better recipe is available") at selection and launch time, with an
explicit apply action that overlays the updated recipe from the state dir.
"""
import argparse
import glob
import hmac
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import stat
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import appwindow

VERSION = "0.4.8"

UPDATE_URL = os.environ.get("B70_UPDATE_URL", "https://xecores.com/downloads/version.json")
UPDATE_INFO = {"checked": False, "has_update": False, "latest_version": "", "message": "", "download_url": "https://xecores.com/match"}
# Remote recipe catalog state: filled by the same background check as
# UPDATE_INFO. entries maps "<model_id>:<engine>" -> {"ver", "note", "recommended"}.
RECIPE_REMOTE = {"checked": False, "catalog_ver": "", "recipes_url": "", "entries": {}}


def _url_ok(url):
    """Remote fetches stay on https; plain http only for localhost test rigs."""
    return url.startswith("https://") or url.startswith("http://127.0.0.1") \
        or url.startswith("http://localhost")


def check_for_updates():
    global UPDATE_INFO
    try:
        req = urllib.request.Request(UPDATE_URL, headers=UA)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read(1 << 20).decode())
            latest = data.get("version", "")
            if latest and _ver_key(latest) > _ver_key(VERSION):
                UPDATE_INFO = {
                    "checked": True,
                    "has_update": True,
                    "latest_version": latest,
                    "message": data.get("message", f"B70 Launcher v{latest} is available."),
                    "download_url": data.get("download_url", "https://xecores.com/match")
                }
            else:
                UPDATE_INFO["checked"] = True
            check_recipe_manifest(data)
    except Exception as exc:
        UPDATE_INFO["checked"] = True
        RECIPE_REMOTE["checked"] = True
        print(f"update check failed ({UPDATE_URL}): {exc}")


def check_recipe_manifest(version_doc):
    """Fetch the small per-recipe manifest next to version.json.

    The manifest is data-only (versions + notes); the full recipes document is
    fetched lazily when the user applies an update."""
    manifest_url = version_doc.get("recipes_manifest_url")
    if not manifest_url:
        manifest_url = UPDATE_URL.rsplit("/", 1)[0] + "/recipes-manifest.json"
    if not _url_ok(str(manifest_url)):
        RECIPE_REMOTE["checked"] = True
        return
    try:
        req = urllib.request.Request(str(manifest_url), headers=UA)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read(4 << 20).decode())
        entries = data.get("recipes")
        if isinstance(entries, dict):
            RECIPE_REMOTE.update({
                "catalog_ver": str(data.get("catalog_ver") or ""),
                "recipes_url": str(data.get("recipes_url") or ""),
                "entries": {k: v for k, v in entries.items() if isinstance(v, dict)},
            })
    except Exception:
        pass
    RECIPE_REMOTE["checked"] = True


HERE = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
if getattr(sys, "frozen", False) and not (HERE / "web" / "index.html").is_file():
    bundled = Path(getattr(sys, "_MEIPASS", HERE))
    if (bundled / "web" / "index.html").is_file():
        HERE = bundled
RECIPES = json.loads((HERE / "recipes.json").read_text())
SETTINGS = json.loads((HERE / "settings.json").read_text())
DATA = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "b70-launcher"
DATA.mkdir(parents=True, exist_ok=True)
try:
    os.chmod(DATA, 0o700)  # state (paths, cmdlines, usage) shouldn't be world-readable
except OSError:
    pass
LOGDIR = DATA / "logs"
LOGDIR.mkdir(exist_ok=True)
OVERRIDE_PATH = DATA / "settings-override.json"
USAGE_PATH = DATA / "usage-history.json"
STATE_PATH = DATA / "servers-state.json"
API_TOKEN = secrets.token_urlsafe(32)
# same-box processes can read the token file (0600) to call the API, e.g.
# scripts or `curl -H "X-Launcher-Token: $(cat ~/.local/state/b70-launcher/token)"`
try:
    (DATA / "token").write_text(API_TOKEN)
    os.chmod(DATA / "token", 0o600)
except OSError:
    pass

IS_WIN = os.name == "nt"
RUNNING = {}   # id -> server entry
DOWNLOADS = {}  # id -> download entry
SCAN = {"state": "idle", "items": {"gguf": {}, "snapshots": {}}, "catalog": [], "roots": [], "ts": 0, "error": None}
PREFLIGHT_CACHE = {"data": None, "ts": 0.0}
CONTAINER_CACHE = {"data": {}, "ts": 0.0}
LOCK = threading.RLock()  # reentrant: some paths record usage while tracked
USAGE_LOCK = threading.Lock()
SHUTDOWN = {"started": False}
WIN_CHILD = None  # native window child process, set by main()
SERVER = None     # ThreadingHTTPServer, set by main(); used by graceful_shutdown
STOP_EVT = threading.Event()
POWER_PREV = {}   # pci -> (ts, energy1_input uJ) for live watt estimation
TRANSIENT_PROCS = []  # fire-and-forget Popens (terminals, browsers) reaped on state polls
VRAM_CACHE = {}  # pci -> (ts, used_gb, total_gb); _vram_mm sudo-forks per GPU

# env var names a user-supplied extra_env must not set — they can redirect code
# loading or process startup inside the launched engine/container
_BLOCKED_ENV = re.compile(
    r"(?:LD_.*|BASH_ENV|ENV|SHELLOPTS|BASHOPTS|GLOBIGNORE|PROMPT_COMMAND|"
    r"PYTHON.*|PERL5.*|RUBYLIB|NODE_OPTIONS|IFS|PATH|HOME|USER|SHELL|TERM)$")

HF = "https://huggingface.co"
UA = {"User-Agent": f"b70-launcher/{VERSION}"}

# Start background update check (after UA exists — it runs immediately)
threading.Thread(target=check_for_updates, daemon=True).start()
WINDOW_TITLE = "B70 Launcher " + VERSION


# ---------------------------------------------------------------- settings

USER_RECIPE_OVERRIDES = {}


def _apply_recipe_overrides(ov_map):
    """Per-recipe user overrides: {"<model_id>": {"<engine>": {field: value}}}.

    Lets a host keep site-specific recipe fields (a native llama_bin path, a
    non-standard artifact location) out of the shipped recipes.json. These
    always win — re-applied after any remote recipe overlay."""
    if not isinstance(ov_map, dict):
        return
    for mid, per_eng in ov_map.items():
        if isinstance(per_eng, dict):
            USER_RECIPE_OVERRIDES.setdefault(mid, {}).update(per_eng)
    for m in RECIPES.get("models", []):
        per = ov_map.get(m.get("id"))
        if not isinstance(per, dict):
            continue
        for eng, fields in per.items():
            if isinstance(fields, dict) and eng in m.get("recipes", {}):
                m["recipes"][eng].update(fields)


def _merge_override():
    """User edits (scan roots etc.) live in the state dir, never in the install."""
    try:
        ov = json.loads(OVERRIDE_PATH.read_text())
    except Exception:
        return
    if isinstance(ov, dict):
        for k, v in ov.items():
            if k in ("scan_dirs", "models_dir", "ovms_repo", "llama_bin",
                     "exl3_data_root"):
                SETTINGS[k] = v
        _apply_recipe_overrides(ov.get("recipe_overrides"))


_merge_override()


# ---------------------------------------------------------------- remote recipes

REMOTE_RECIPES_PATH = DATA / "recipes-remote.json"

# model-level fields a published recipe update may set (never "id"/"recipes")
REMOTE_MODEL_FIELDS = ("recommended_engine", "recommendation", "badge", "blurb",
                       "subtitle", "tags", "chips", "brand")

# recipe fields a published update may tune. Exec-shaping keys (image,
# docker_sock, llama_bin, fixed_flags, serve_config, kind) and host-path keys
# (model_path, gguf, source_model) never come from the wire: they would let a
# catalog swap the binary/image, pick the docker socket, append container argv
# or remount host paths. Everything else merges over the local recipe.
REMOTE_RECIPE_FIELDS = ("ctx", "ctx_max", "ctx_note", "ctx_safe", "power", "dtype",
                        "kv", "download", "spec_tokens", "spec_p_min", "speculative",
                        "draft_device", "tensor_split", "split_mode",
                        "offload_tensors", "tool_parser", "reasoning_parser",
                        "cim_long_ctx", "search_name", "inner_port", "tp",
                        "perf", "topology", "recipe_ver")
_MODEL_ID_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_ENGINE_RE = re.compile(r"[A-Za-z0-9_-]{1,32}")
_DRAFT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.gguf", re.I)


def _sanitize_remote_recipe(r):
    """Strip a published recipe to fields allowed over the wire."""
    out = {k: r[k] for k in REMOTE_RECIPE_FIELDS if k in r}
    dm = r.get("draft_model")  # bare artifact names only — resolved via scan
    if dm is not None and _DRAFT_RE.fullmatch(str(dm)):
        out["draft_model"] = str(dm)
    return out


def _stamp_ver(recipe, fallback):
    if isinstance(recipe, dict) and not recipe.get("recipe_ver") and fallback:
        recipe["recipe_ver"] = fallback


def apply_recipe_doc(doc, catalog_ver=""):
    """Overlay one published recipes document onto the live RECIPES dict.

    Remote recipes are sanitized to REMOTE_RECIPE_FIELDS and merged over the
    local recipe, so the update channel cannot replace exec-shaping fields.
    Unknown models are added sanitized. Returns (applied_count, error)."""
    models = doc.get("models") if isinstance(doc, dict) else None
    if not isinstance(models, list):
        return 0, "remote document has no models list"
    applied = 0
    with LOCK:
        by_id = {m.get("id"): m for m in RECIPES.get("models", [])}
        for rm in models:
            if not isinstance(rm, dict) or not _MODEL_ID_RE.fullmatch(str(rm.get("id") or "")):
                continue
            local = by_id.get(rm["id"])
            rrecipes = rm.get("recipes") or {}
            if local is None:
                nm = {"id": rm["id"], "name": str(rm.get("name") or rm["id"]),
                      "recipes": {}}
                for f in REMOTE_MODEL_FIELDS:
                    if f in rm:
                        nm[f] = rm[f]
                for eng, r in rrecipes.items():
                    if isinstance(r, dict) and _ENGINE_RE.fullmatch(str(eng)):
                        rr = _sanitize_remote_recipe(r)
                        _stamp_ver(rr, catalog_ver)
                        nm["recipes"][eng] = rr
                        applied += 1
                RECIPES["models"].append(nm)
                by_id[nm["id"]] = nm
                continue
            for f in REMOTE_MODEL_FIELDS:
                if f in rm:
                    local[f] = rm[f]
            for eng, r in rrecipes.items():
                if isinstance(r, dict) and _ENGINE_RE.fullmatch(str(eng)):
                    rr = _sanitize_remote_recipe(r)
                    _stamp_ver(rr, catalog_ver)
                    local.setdefault("recipes", {}).setdefault(eng, {}).update(rr)
                    applied += 1
        _apply_recipe_overrides(USER_RECIPE_OVERRIDES)
    return applied, None


def _load_remote_overlay():
    try:
        doc = json.loads(REMOTE_RECIPES_PATH.read_text())
    except Exception:
        return
    apply_recipe_doc(doc.get("payload") or doc, doc.get("catalog_ver") or "")


_load_remote_overlay()


def _ver_key(v):
    """recipe_ver values are YYYY-MM-DD dates (or ints); compare safely."""
    s = str(v or "")
    return [int(p) if p.isdigit() else 0 for p in re.split(r"[^0-9]", s) if p]


def local_recipe_ver(model_id, engine):
    m = find_model(model_id)
    r = (m or {}).get("recipes", {}).get(engine) or {}
    return r.get("recipe_ver") or RECIPES.get("catalog_ver") or ""


def recipe_notices():
    """Remote catalog entries newer than the shipped/applied local recipe."""
    out = {}
    for key, e in RECIPE_REMOTE.get("entries", {}).items():
        parts = key.split(":", 1)
        if len(parts) != 2:
            continue
        mid, eng = parts
        remote_ver = str(e.get("ver") or "")
        if not remote_ver:
            continue
        m = find_model(mid)
        is_new = not m or eng not in (m.get("recipes") or {})
        local_ver = "" if is_new else local_recipe_ver(mid, eng)
        if _ver_key(remote_ver) <= _ver_key(local_ver) and not is_new:
            continue
        out[key] = {
            "model_id": mid, "engine": eng,
            "model_name": (m or {}).get("name") or e.get("model") or mid,
            "local_ver": local_ver, "remote_ver": remote_ver,
            "note": str(e.get("note") or ""),
            "recommended": bool(e.get("recommended")),
            "is_new": is_new,
            "can_apply": bool(RECIPE_REMOTE.get("recipes_url")),
        }
    return out


def recipe_notice_for(model_id, engine):
    return recipe_notices().get(f"{model_id}:{engine}")


def fetch_remote_recipes():
    """Pull the full published recipes document referenced by the manifest."""
    url = RECIPE_REMOTE.get("recipes_url")
    if not url:
        return None, "no remote recipes_url in the update manifest"
    if not _url_ok(url):
        return None, "remote recipes_url must be https"
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read(8 << 20)
        doc = json.loads(raw.decode())
    except Exception as e:
        return None, f"remote recipes fetch failed: {e}"
    if not isinstance(doc, dict) or not isinstance(doc.get("models"), list):
        return None, "remote document is not a recipes catalog"
    return doc, None


def apply_recipe_update(model_id=None, engine=None):
    """Fetch the published catalog and overlay it (or one recipe) locally.

    The merged result is persisted in the state dir, so the update survives
    restarts without touching the install."""
    doc, err = fetch_remote_recipes()
    if err:
        return {"error": err}
    cat_ver = str(doc.get("catalog_ver") or RECIPE_REMOTE.get("catalog_ver") or "")
    if model_id and engine:
        target = None
        for rm in doc["models"]:
            if rm.get("id") == model_id and engine in (rm.get("recipes") or {}):
                target = rm
                break
        if target is None:
            return {"error": f"remote catalog has no recipe for {model_id}:{engine}"}
        doc = {"models": [target]}
    applied, err = apply_recipe_doc(doc, cat_ver)
    if err:
        return {"error": err}
    if not applied:
        return {"error": "nothing newer to apply"}
    try:
        stored = {"catalog_ver": cat_ver,
                  "applied_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "payload": {"models": [
                      m for m in doc["models"]]}}
        if REMOTE_RECIPES_PATH.exists():
            try:
                prev = json.loads(REMOTE_RECIPES_PATH.read_text())
                prev_models = (prev.get("payload") or {}).get("models") or []
                ids = {m.get("id") for m in stored["payload"]["models"]}
                stored["payload"]["models"] = (
                    [m for m in prev_models if m.get("id") not in ids]
                    + stored["payload"]["models"])
                if prev.get("catalog_ver") and _ver_key(prev["catalog_ver"]) > _ver_key(cat_ver):
                    stored["catalog_ver"] = prev["catalog_ver"]
            except Exception:
                pass
        tmp = REMOTE_RECIPES_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(stored, indent=1))
        tmp.rename(REMOTE_RECIPES_PATH)
    except Exception as e:
        return {"error": f"applied in memory but failed to persist: {e}"}
    start_scan_async(force=True)
    return {"ok": True, "applied": applied, "catalog_ver": cat_ver}


def save_override(keys=("scan_dirs",)):
    ov = {}
    try:
        ov = json.loads(OVERRIDE_PATH.read_text())
    except Exception:
        pass
    for k in keys:
        ov[k] = SETTINGS.get(k)
    tmp = OVERRIDE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(ov, indent=2))
    tmp.rename(OVERRIDE_PATH)


# ---------------------------------------------------------------- scan / detect

SKIP_DIRS = {"node_modules", ".git", ".svn", ".hg", "__pycache__", ".venv", "venv",
             "lost+found", "$RECYCLE.BIN", "System Volume Information",
             "site-packages", "dist-packages", ".cache", ".npm", ".cargo", "vendor"}


def scan_roots():
    roots = []
    for r in SETTINGS.get("scan_dirs", []):
        p = Path(r).expanduser().resolve()
        if p in (Path("/"), Path.home().resolve()):
            continue  # refuse filesystem-wide scans
        if p.is_dir() and p not in roots:
            roots.append(p)
    return roots


def read_ctx_from_config(cfgpath):
    try:
        cfg = json.loads(Path(cfgpath).read_text())
    except Exception:
        return None
    # multimodal configs nest the language model under text_config
    for scope in (cfg.get("text_config") or {}, cfg):
        if not isinstance(scope, dict):
            continue
        for key in ("max_position_embeddings", "context_length", "max_seq_len", "n_ctx"):
            v = scope.get(key)
            if isinstance(v, int) and v > 512:
                return v
    return None


def _walk_root(root, items, deadline):
    """Depth-limited iterative walk; returns False when the time budget ran out.

    Classifies every directory: GGUF files are collected individually; model
    directories are tagged openvino (IR xml+bin pairs or the
    openvino_language_model.* prefix) or vllm (HF layout: config.json and/or
    *.safetensors). Model dirs are never descended into."""
    stack = [(root, 0)]
    while stack:
        dirpath, depth = stack.pop()
        if time.time() > deadline:
            return False
        if depth > 8:
            continue  # skip only this subtree; siblings still get scanned
        names = set()
        subdirs = []
        try:
            with os.scandir(dirpath) as it:  # stream entries — a huge dir must not spike memory
                for e in it:
                    try:
                        if e.is_symlink():
                            continue
                        if e.is_dir(follow_symlinks=False):
                            if e.name in SKIP_DIRS or e.name.startswith("."):
                                continue
                            subdirs.append(e.path)
                        elif e.is_file(follow_symlinks=False):
                            n = e.name.lower()
                            names.add(n)
                            if n.endswith(".gguf"):
                                # on a basename collision prefer the shallower
                                # path — deterministic regardless of readdir order
                                prev = items["gguf"].get(n)
                                if prev is None or e.path.count(os.sep) < prev.count(os.sep):
                                    items["gguf"][n] = e.path
                    except OSError:
                        continue
        except (PermissionError, OSError):
            continue
        has_ov = any(n.startswith("openvino_language_model.") for n in names) or \
                 (any(n.endswith(".xml") for n in names) and any(n.endswith(".bin") for n in names))
        has_vllm = "config.json" in names or any(n.endswith(".safetensors") for n in names)
        if has_ov:
            kind = "openvino"
        elif has_vllm:
            kind = "vllm"
            qc = Path(dirpath) / "quantization_config.json"
            if qc.is_file():
                try:
                    if json.loads(qc.read_text()).get("quant_method") == "exl3":
                        kind = "exl3"
                except (OSError, ValueError):
                    pass
        else:
            kind = None
        if kind:
            # a model directory: record it, never descend into its weight files
            cfg = Path(dirpath) / "config.json"
            items["snapshots"].setdefault(Path(dirpath).name.lower(), {
                "path": dirpath,
                "kind": kind,
                "repo_hint": f"{Path(dirpath).parent.name}/{Path(dirpath).name}",
                "ctx": read_ctx_from_config(cfg) if cfg.is_file() else None,
            })
            continue
        for sub in subdirs:
            stack.append((sub, depth + 1))
    return True


def _dir_size_mib(path):
    """Fast size estimate: sum of the files directly inside a model dir."""
    total = 0
    try:
        with os.scandir(path) as it:
            for e in it:
                try:
                    if e.is_file(follow_symlinks=False):
                        total += e.stat().st_size
                except OSError:
                    continue
    except OSError:
        return None
    return round(total / 1048576, 1)


def build_catalog(items):
    """Every artifact found on disk, tagged with its serving format."""
    catalog = {}
    for name, path in items.get("gguf", {}).items():
        try:
            size = round(Path(path).stat().st_size / 1048576, 1)
        except OSError:
            size = None
        catalog[path] = {"kind": "gguf", "name": name, "path": path, "size_mib": size, "ctx": None}
    for name, entry in items.get("snapshots", {}).items():
        if entry["path"] in catalog:
            continue
        catalog[entry["path"]] = {"kind": entry.get("kind", "vllm"), "name": name,
                                  "path": entry["path"],
                                  "size_mib": _dir_size_mib(entry["path"]),
                                  "ctx": entry.get("ctx")}
    return sorted(catalog.values(), key=lambda c: (c["kind"], c["name"]))


def scan_worker():
    items = {"gguf": {}, "snapshots": {}}
    roots = scan_roots()
    deadline = time.time() + 90
    partial = None
    for root in roots:
        try:
            if not _walk_root(root, items, deadline):
                partial = "scan stopped at a time budget; results may be partial"
        except Exception as e:  # never let a scan thread crash the app
            partial = f"scan error: {e}"
    with LOCK:
        SCAN.update({"items": items, "catalog": build_catalog(items),
                     "roots": [str(r) for r in roots],
                     "ts": time.time(), "state": "done", "error": partial})


def start_scan_async(force=False):
    with LOCK:
        stale = time.time() - SCAN.get("ts", 0) > 120
        if SCAN.get("state") == "scanning" or (not force and not stale and SCAN.get("ts", 0)):
            return
        SCAN["state"] = "scanning"
    threading.Thread(target=scan_worker, daemon=True).start()


def scan_blocking():
    """Used by /api/scan: serve cached results, kick a rescan when stale."""
    roots = [str(r) for r in scan_roots()]
    if SCAN.get("roots") != roots or time.time() - SCAN.get("ts", 0) > 120:
        start_scan_async(force=True)
    deadline = time.time() + 5
    with LOCK:
        while SCAN.get("state") == "scanning" and time.time() < deadline:
            LOCK.release()
            time.sleep(0.1)
            LOCK.acquire()
        return dict(SCAN)


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _as_list(v):
    """Recipe match fields accept a string or a list of alternates."""
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if x]
    return [str(v)] if v else []


def _first_nonempty(*vals):
    for v in vals:
        if v:
            return v
    return ""


def detect(model, engine):
    """Match a recipe to an on-disk artifact: exact name first, else same model family."""
    if not model:
        return None
    items = SCAN.get("items") or {"gguf": {}, "snapshots": {}}
    r = model["recipes"].get(engine, {})
    dl = r.get("download", {})
    pool = items["gguf"] if dl.get("kind") == "file" else items["snapshots"]
    # dl.name may carry a repo subdirectory ("dir/file.gguf"); the scan pool
    # is keyed by basename only
    want = Path(dl["name"]).name.lower() if dl.get("name") else ""
    keys = [_norm(k) for k in (_as_list(dl.get("match"))
                               or _as_list(dl.get("search"))
                               or _as_list(dl.get("name"))
                               or _as_list(r.get("search_name"))
                               or _as_list((dl.get("repo") or r.get("source_model") or "").split("/")[-1]))]
    keys = [k for k in keys if k]

    def wrap(name, hit, exact):
        path = hit if isinstance(hit, str) else hit["path"]
        ctx = None if isinstance(hit, str) else hit.get("ctx")
        p = Path(path)
        # mount the artifact's own directory, not the whole scan root
        mount = p.parent if p.is_file() else p
        try:
            size = round(p.stat().st_size / 1048576, 1) if p.is_file() \
                else _dir_size_mib(str(p))
        except OSError:
            size = None
        return {"path": path, "ctx": ctx, "mount_root": str(mount),
                "found_name": name, "variant": not exact, "size_mib": size}

    # direct path check from recipe (for custom/pinned local models)
    for direct_k in ("gguf", "model_path", "source_model"):
        val = r.get(direct_k)
        if val and not str(val).startswith("http"):
            dp = Path(val).expanduser()
            if dp.exists():
                cfg_f = dp / "config.json" if dp.is_dir() else None
                c_val = read_ctx_from_config(cfg_f) if cfg_f and cfg_f.is_file() else r.get("ctx")
                akind = "openvino" if engine == "openvino" else ("exl3" if engine == "exl3" else "vllm")
                return wrap(dp.name, str(dp) if dp.is_file() else {"path": str(dp), "ctx": c_val, "kind": akind}, True)

    if want and want in pool:
        return wrap(want, pool[want], True)
    prefer = [_norm(p) for p in _as_list(dl.get("prefer"))]
    cands = []
    for name, hit in pool.items():
        if dl.get("kind") != "file" and isinstance(hit, dict):
            eng_kind = {"openvino": "openvino", "exl3": "exl3"}.get(engine, "vllm")
            if hit.get("kind") and hit.get("kind") != eng_kind:
                continue
        nn = _norm(name)
        if name.startswith("mmproj"):
            continue  # vision sidecar is never the model itself
        # speculative-decoding draft sidecars (DFlash/MTP) are never the
        # serving artifact — a 1 GB draft must not satisfy a 20 GB model
        want_norm = _norm(want or _first_nonempty(*_as_list(dl.get("search"))) or "")
        if ("dflash" in nn or "draft" in nn) and "dflash" not in want_norm and "draft" not in want_norm:
            continue
        if prefer and not any(p in nn for p in prefer):
            continue  # right model, wrong artifact format for this engine
        if keys and any(k in nn for k in keys):
            score = sum(1 for p in prefer if p and p in nn) - len(nn) / 1000.0
            cands.append((score, name, hit))
    if cands:
        cands.sort(key=lambda t: (-t[0], t[1]))
        _, name, hit = cands[0]
        return wrap(name, hit, name == want)
    return None


def resolve_ctx(model, engine, det):
    r = model["recipes"].get(engine, {})
    cap = r.get("ctx_max")
    if det and det.get("ctx"):
        v = det["ctx"]
        if cap:
            v = min(v, cap)
        return {"value": v, "source": "config.json on disk"}
    v = r.get("ctx", 32768)
    if cap:
        v = min(v, cap)
    return {"value": v, "source": r.get("ctx_note", "cookbook recipe")}


# ---------------------------------------------------------------- HF resolve

def _http_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read(24 << 20))


def hf_files(repo, revision=None):
    if revision:
        data = _http_json(f"{HF}/api/models/{repo}/tree/{urllib.parse.quote(revision, safe='')}?recursive=true&expand=true")
        files = {}
        for node in data:
            if node.get("type") != "file":
                continue
            lfs = node.get("lfs") or {}
            files[node["path"]] = node.get("size") or lfs.get("size")
        return files
    data = _http_json(f"{HF}/api/models/{repo}?blobs=true")
    files = {}
    for sib in data.get("siblings", []):
        lfs = sib.get("lfs") or {}
        size = sib.get("size") or lfs.get("size")
        files[sib["rfilename"]] = size
    return files


# Repositories verified against the public HF API as the correct artifact
# source for the recipes that pin them. Anything else stays manual.
HF_APPROVED_REPOS = {
    "OpenVINO/Qwen3.6-35B-A3B-int4-ov",
    "Intel/Qwen3.6-27B-int4-AutoRound",
    "llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4",
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "agentionai/Signal-3.8-Flash-Next-GGUF",
    "SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16",
    "unsloth/Qwen3.8-27B-GGUF",
    "SergiioB/Qwen3.8-27B-int4-gdn8-ov",
    "OpenVINO/Qwen3.8-27B-int8-ov",
    "turboderp/Qwen3.8-27B-exl3",
    "Qwen/Qwen3.8-27B-FP8",
    "SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym",
    "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF",
    "bottlecapai/ThinkingCap-Qwen3.6-27B-GGUF",
    "meta-models/Muse-Glimmer-30B-GGUF",
    "unsloth/Muse-Glimmer-30B-GGUF",
    "ornith-ai/Ornith-1.0-35B-GGUF",
}


def resolve_repo(dl):
    """Only a recipe-pinned repository is suitable for automatic download."""
    repo = dl.get("repo", "")
    if repo not in HF_APPROVED_REPOS:
        return None, "Automatic download unavailable: recipe has no approved official HF repository. Obtain and inspect the artifact manually."
    rev = dl.get("revision")
    if rev and not re.fullmatch(r"[A-Za-z0-9._-]+", rev):
        return None, "Automatic download unavailable: recipe pins an invalid repository revision."
    return repo, None


def safe_artifact(name):
    """Reject absolute/traversing HF filenames (including Windows separators)."""
    parts = name.split("/")
    return bool(name and not name.startswith("/") and "\\" not in name and ":" not in name and all(p not in ("", ".", "..") for p in parts))


# ---------------------------------------------------------------- hardware

def _read_int(path):
    try:
        return int(Path(path).read_text().strip())
    except Exception:
        return None


def best_default(b70s, devices):
    """Hardware-aware default model + engine for this machine."""
    first = RECIPES["models"][0]
    if b70s:
        vram = sum((d.get("vram_total_mib") or 0) for d in b70s)
        model = find_model("qwen36-35b") or first
        return {"model_id": model["id"], "engine": next(iter(model["recipes"])),
                "reason": f"{len(b70s)}× Arc Pro B70 detected"
                          + (f" · {vram / 1024:.0f} GB total" if vram else "")
                          + (" — the second card stays on dry-run until a two-card topology is validated"
                             if len(b70s) > 1 else "")}
    return {"model_id": first["id"], "engine": next(iter(first["recipes"])),
            "reason": ("Intel GPU detected but no validated B70 — launch is blocked, dry-run only"
                       if devices else "No GPU detected yet — install the Intel driver, dry-run only")}


def hardware_preflight():
    now = time.time()
    if PREFLIGHT_CACHE["data"] and now - PREFLIGHT_CACHE["ts"] < 10:
        return PREFLIGHT_CACHE["data"]
    devices = []
    if IS_WIN:
        result = {"devices": devices, "blockers": ["Windows GPU execution is not validated; use a supported Linux host."],
                  "notes": [], "profile": {"b70_count": 0, "cards": []}}
        PREFLIGHT_CACHE.update({"data": result, "ts": now})
        return result
    # map render nodes to their DRM card so sysfs/hwmon telemetry can be attached
    card_of = {}
    for card in Path("/sys/class/drm").glob("card[0-9]*"):
        try:
            card_of[card.joinpath("device").resolve()] = card.name
        except OSError:
            continue
    for render in sorted(Path("/sys/class/drm").glob("renderD*"), key=lambda p: int(p.name[7:])):
        dev = render / "device"
        try:
            vendor = (dev / "vendor").read_text().strip().lower()
            device = (dev / "device").read_text().strip().lower()
            subsystem_vendor = (dev / "subsystem_vendor").read_text().strip().lower()
            subsystem_device = (dev / "subsystem_device").read_text().strip().lower()
            pci = dev.resolve().name
            label = ""
            if shutil.which("lspci"):
                label = subprocess.run(["lspci", "-s", pci], capture_output=True,
                                       text=True, timeout=3).stdout.strip()
            b70 = (vendor, device, subsystem_vendor, subsystem_device) == ("0x8086", "0xe223", "0x8086", "0x1701")
            vram_total = _read_int(dev / "mem_info_vram_total")
            vram_used = _read_int(dev / "mem_info_vram_used")
            if b70 and not vram_total:
                vram_total = 24 * 1024 * 1024 * 1024
            cap_w = None
            for hm in Path("/sys/class/hwmon").glob("hwmon*"):
                try:
                    if hm.joinpath("device").resolve() != dev.resolve():
                        continue
                    c = _read_int(hm / "power1_cap")
                    if c:
                        cap_w = round(c / 1_000_000)
                    break
                except OSError:
                    continue
            devices.append({"render": f"/dev/dri/{render.name}", "pci": pci,
                            "vendor": vendor, "device": device, "label": label,
                            "card": card_of.get(dev.resolve()),
                            "accessible": os.access(f"/dev/dri/{render.name}", os.R_OK | os.W_OK),
                            "b70": b70,
                            "vram_total_mib": round(vram_total / 1048576, 1) if vram_total else None,
                            "vram_used_mib": round(vram_used / 1048576, 1) if vram_used else None,
                            "power_cap_w": cap_w})
        except (OSError, subprocess.TimeoutExpired):
            continue
    blockers = []
    if not devices:
        blockers.append("No DRM render nodes found; install a working Intel GPU driver.")
    if not any(d["b70"] for d in devices):
        blockers.append("No Arc Pro B70 identified by Intel PCI subsystem ID 8086:1701; other Intel GPUs are unvalidated.")
    if not shutil.which("docker"):
        blockers.append("Docker CLI missing; install Docker and a supported GPU runtime.")
    if os.environ.get("DOCKER_HOST") or os.environ.get("DOCKER_CONTEXT"):
        blockers.append("Docker remote context/environment configured; only a local Docker socket is supported.")
    socket_path = Path(os.environ.get("XDG_RUNTIME_DIR", "/run/user/" + str(os.getuid())) + "/docker.sock")
    if not socket_path.exists():
        socket_path = Path("/var/run/docker.sock")
    try:
        if not stat.S_ISSOCK(socket_path.stat().st_mode) or not os.access(socket_path, os.R_OK | os.W_OK):
            blockers.append("Local Docker socket unavailable or inaccessible.")
    except OSError:
        blockers.append("Local Docker socket unavailable or inaccessible.")
    b70s = [d for d in devices if d["b70"] and d["accessible"]]
    profile = {"b70_count": len(b70s), "cards": [
        {"render": d["render"], "label": d["label"], "vram_total_mib": d["vram_total_mib"],
         "power_cap_w": d["power_cap_w"], "accessible": d["accessible"], "b70": d["b70"]}
        for d in devices]}
    result = {"devices": devices, "blockers": blockers,
              "notes": ["GPU container compatibility and model fit are not proven by this preflight."],
              "profile": profile, "default": best_default(b70s, devices)}
    PREFLIGHT_CACHE.update({"data": result, "ts": now})
    return result


def _vram_mm(pci):
    """Real VRAM usage from xe debugfs (same source as the desktop dashboard).

    Each call sudo-forks `cat`, so results are cached briefly — the telemetry
    poller hits this per GPU every couple of seconds."""
    hit = VRAM_CACHE.get(pci)
    now = time.time()
    if hit and now - hit[0] < 10:
        return hit[1], hit[2]
    used_gb = total_gb = None
    try:
        res = subprocess.run(["sudo", "-n", "cat",
                              f"/sys/kernel/debug/dri/{pci}/tile0/vram_mm"],
                             capture_output=True, text=True, timeout=3)
        if res.returncode == 0:
            avail = total = 0
            for line in res.stdout.splitlines():
                if line.startswith("visible_avail:"):
                    avail = int(line.split()[1].replace("MiB", ""))
                elif line.startswith("visible_size:"):
                    total = int(line.split()[1].replace("MiB", ""))
            if total:
                used_gb, total_gb = round((total - avail) / 1024, 2), round(total / 1024, 1)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    VRAM_CACHE[pci] = (now, used_gb, total_gb)
    return used_gb, total_gb


def _act_freq(pci):
    for rel in ("tile0/gt0/freq0/act_freq", "tile0/gt0/freq0/cur_freq"):
        v = _read_int(Path(f"/sys/bus/pci/devices/{pci}/{rel}"))
        if v:
            return v
    return None


def power_probe():
    """Live per-card watts from xe hwmon energy deltas + temp + real VRAM + clock-based util."""
    out = []
    now = time.time()
    idx = 0
    for hm in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            if (hm / "name").read_text().strip() != "xe":
                continue
            dev = hm.joinpath("device").resolve()
            pci = dev.name
            energy = _read_int(hm / "energy1_input")
            cap = _read_int(hm / "power1_cap")
            temp = _read_int(hm / "temp2_input") or _read_int(hm / "temp1_input")
            used_gb, total_gb = _vram_mm(pci)
            freq = _act_freq(pci)
            maxfreq = None
            for rel in ("tile0/gt0/freq0/max_freq", "tile0/gt0/freq0/rpn_freq"):
                v = _read_int(Path(f"/sys/bus/pci/devices/{pci}/{rel}"))
                if v:
                    maxfreq = v
                    break

            entry = {
                "index": idx,
                "name": f"GPU {idx} Arc Pro B70",
                "pci": pci,
                "active": True,
                "watts": None,
                "cap_w": round(cap / 1_000_000) if cap else 150,
                "temp_c": round(temp / 1000) if temp else None,
                "freq_mhz": freq,
                "util_pct": None,
                "vram_total_gb": total_gb,
                "vram_used_gb": used_gb,
            }
            if freq and maxfreq:
                entry["util_pct"] = max(0, min(100, round(freq / maxfreq * 100)))
            prev = POWER_PREV.get(pci)
            if energy is not None and prev and now > prev[0] and energy >= prev[1]:
                entry["watts"] = round((energy - prev[1]) / (now - prev[0]) / 1_000_000, 1)
            if energy is not None:
                POWER_PREV[pci] = (now, energy)
            out.append(entry)
            idx += 1
        except OSError:
            continue
    return out


# ---------------------------------------------------------------- downloads

def fmt_bytes(n):
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _download_file(url, dest, entry):
    """Stream to dest with HTTP Range resume across network failures (HF/Xet 403s)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.is_symlink():
        raise ValueError("Refusing symlinked partial download")
    counted = 0  # bytes of THIS file already reflected in entry["done"]
    last_err = None
    for attempt in range(4):
        try:
            existing = tmp.stat().st_size if tmp.exists() else 0
            headers = dict(UA)
            if existing:
                headers["Range"] = f"bytes={existing}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                if existing and getattr(resp, "status", 200) != 206:
                    existing = 0  # server ignored Range -> restart this file
                with LOCK:
                    if counted > existing:
                        entry["done"] = entry.get("done", 0) - counted
                        counted = 0
                    if existing and not counted:
                        entry["done"] = entry.get("done", 0) + existing
                        counted = existing
                    total = resp.headers.get("Content-Length")
                    if total and not entry.get("total"):
                        entry["total"] = entry.get("done", 0) + int(total) + (
                            existing if counted != existing else 0)
                last = time.time()
                speed = 0.0
                with open(tmp, "ab" if existing else "wb") as f:
                    while True:
                        if entry.get("cancel"):
                            raise InterruptedError("cancelled")
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        now = time.time()
                        inst = len(chunk) / max(now - last, 1e-6)
                        speed = inst if speed == 0 else 0.8 * speed + 0.2 * inst
                        last = now
                        with LOCK:
                            entry["done"] = entry.get("done", 0) + len(chunk)
                            entry["speed"] = speed
                            total_all = entry.get("total")
                            if total_all:
                                left = max(total_all - entry["done"], 0)
                                entry["eta"] = left / speed if speed > 0 else None
                                entry["pct"] = round(100 * entry["done"] / total_all, 1)
                        counted += len(chunk)
            if dest.exists():
                raise FileExistsError(f"Refusing to overwrite existing artifact: {dest}")
            tmp.rename(dest)
            return
        except InterruptedError:
            raise
        except FileExistsError:
            raise  # not retryable — re-downloading cannot fix a full destination
        except Exception as e:
            last_err = e
            if attempt == 3:
                break
            time.sleep(2 * (attempt + 1))  # resume picks up from .part on retry
    raise RuntimeError(f"download failed after 4 attempts: {last_err}")


def download_worker(did, model, engine):
    entry = DOWNLOADS[did]
    try:
        with LOCK:
            entry["state"] = "resolving"
        dl = model["recipes"][engine]["download"]
        repo, err = resolve_repo(dl)
        if err:
            with LOCK:
                entry["state"] = "error"
                entry["error"] = err
            return
        with LOCK:
            entry["repo"] = repo
        rev = dl.get("revision")
        if rev:
            with LOCK:
                entry["repo"] = f"{repo}@{rev}"
        files = hf_files(repo, rev)
        if dl.get("kind") == "file":
            wanted = [dl["name"]] + list(dl.get("extra_files", []))
            dest_dir = Path(SETTINGS.get("models_dir", "~/models")).expanduser()
        else:
            wanted = list(files.keys())
            root = Path(SETTINGS.get("ovms_repo", "~/models/ovms-repo")).expanduser()
            if engine != "openvino":
                root = Path(SETTINGS.get("models_dir", "~/models")).expanduser()
            dest_dir = root / repo.split("/")[-1]
        if not wanted or any(not safe_artifact(w) for w in wanted):
            raise ValueError("Repository contains unsafe or empty artifact paths")
        if dl.get("kind") == "file" and any(w not in files for w in wanted):
            raise ValueError("Pinned repository does not contain the exact recipe filename")
        dest_dir.mkdir(parents=True, exist_ok=True)
        total_known = sum((files.get(w) or 0) for w in wanted if w in files)
        with LOCK:
            entry.update({"state": "downloading", "files": wanted, "done": 0,
                          "total": total_known or None, "dest": str(dest_dir)})
        for w in wanted:
            dest = dest_dir / w
            if dest.is_symlink() or not dest.parent.resolve().is_relative_to(dest_dir.resolve()):
                raise ValueError("Refusing download through symlink or outside destination")
            if dest.exists():
                if files.get(w) and dest.stat().st_size == files[w]:
                    with LOCK:
                        entry["done"] += files[w]
                    continue
                raise FileExistsError(f"Existing artifact has unknown or mismatched size; move it aside manually: {dest}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = f"{HF}/{repo}/resolve/{urllib.parse.quote(rev or 'main', safe='')}/{urllib.parse.quote(w)}"
            _download_file(url, dest, entry)
            if files.get(w) and dest.stat().st_size != files[w]:
                raise ValueError(f"size mismatch after download ({w}): got "
                                 f"{dest.stat().st_size}, expected {files[w]}")
        with LOCK:
            entry["state"] = "done"
            entry["pct"] = 100
            entry["eta"] = 0
        start_scan_async(force=True)
    except InterruptedError:
        with LOCK:
            entry["state"] = "cancelled"
    except Exception as e:
        with LOCK:
            entry["state"] = "error"
            entry["error"] = str(e)


def start_download(model, engine):
    did = f"{model['id']}-{engine}"
    with LOCK:
        if did in DOWNLOADS and DOWNLOADS[did].get("state") in ("queued", "resolving", "downloading"):
            return did, None
        DOWNLOADS[did] = {"id": did, "model": model["name"], "engine": engine,
                          "state": "queued", "done": 0, "total": None, "pct": 0,
                          "speed": 0, "eta": None,
                          "quant": model["recipes"][engine].get("download", {}).get("quant", "")}
    t = threading.Thread(target=download_worker, args=(did, model, engine), daemon=True)
    t.start()
    return did, None


# ---------------------------------------------------------------- command build

def render_gid(render_node):
    try:
        return os.stat(render_node).st_gid
    except OSError:
        return None


def device_gids(render_node):
    """Group ids needed inside the container: the render node's group and the
    card node's group (often video), since oneCCL enumerates both."""
    gids = []
    for node in [render_node, "/dev/dri/card0", "/dev/dri/card1"]:
        try:
            g = os.stat(node).st_gid
        except OSError:
            continue
        if g not in gids:
            gids.append(g)
    return gids


def pretty(tokens, width=96):
    sep = " ^" if IS_WIN else " \\"
    lines, cur = [], ""
    for tok in tokens:
        quoted = tok if IS_WIN else shlex.quote(tok)
        candidate = (cur + " " + quoted).strip()
        if cur and len(candidate) > width:
            lines.append(cur + sep)
            cur = "    " + quoted
        else:
            cur = candidate
    lines.append(cur)
    return "\n".join(lines)


def find_model(model_id):
    for m in RECIPES["models"]:
        if m["id"] == model_id:
            return m
    return None


def _resolve_draft(val):
    """draft_model may be a host path (with ~) or a bare artifact basename that
    resolves through the last disk scan — so recipes stay machine-agnostic."""
    if not val:
        return None
    p = Path(str(val)).expanduser()
    if p.exists():
        return str(p)
    if p.is_absolute() or "/" in str(val) or "\\" in str(val):
        return None  # explicit path that does not exist — do not guess
    hit = (SCAN.get("items") or {}).get("gguf", {}).get(str(val).lower())
    if isinstance(hit, str):
        return hit
    if isinstance(hit, dict):
        return hit.get("path")
    return None


def container_path(det, fallback):
    """Map an on-disk artifact to its container path and the mount that exposes it.

    GGUF files: the parent directory is mounted at /models, file stays by name.
    Snapshot dirs: the dir itself is mounted AT /models/model so the model always
    has a stable container path regardless of how deep the scan root goes."""
    if not det or not det.get("path"):
        return fallback, None, "/models"
    try:
        p = Path(det["path"])
        if p.is_file():
            src = str(p.parent)
            return "/models/" + p.name, src, "/models"
        return "/models/model", str(p), "/models/model"
    except (ValueError, KeyError, OSError):
        return fallback, None, "/models"


CUSTOM_TEMPLATES = {
    "llamacpp": {"kind": "gguf", "image": "ghcr.io/ggml-org/llama.cpp:server-intel",
                 "ctx": 32768, "power": 150, "download": {"kind": "file", "quant": "GGUF (custom)"}},
    "openvino": {"kind": "ovms", "image": "openvino/model_server:2026.2.1-gpu",
                 "ctx": 20480, "power": 150, "download": {"kind": "snapshot", "quant": "OpenVINO IR (custom)"}},
    "vllm": {"kind": "vllm", "image": "vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f",
             "ctx": 32768, "power": 150, "dtype": "bfloat16", "download": {"kind": "snapshot", "quant": "HF/safetensors (custom)"}},
    "exl3": {"kind": "exl3", "image": "ghcr.io/0xsero/exl3xpu",
             "docker_sock": "/run/b70-exl3-docker.sock", "inner_port": 8100,
             "ctx": 65536, "power": 230, "download": {"kind": "snapshot", "quant": "EXL3 trellis (custom)"}},
}
KIND_ENGINE = {"gguf": "llamacpp", "openvino": "openvino", "vllm": "vllm", "exl3": "exl3"}


def prepare_custom(cfg):
    """Resolve an arbitrary on-disk artifact into a synthetic model+recipe so it
    can flow through the normal build path. Only paths under a configured scan
    root are accepted."""
    warns = []
    raw = (cfg.get("custom_path") or "").strip()
    if not raw:
        return {"error": "custom model path missing", "warnings": warns}
    engine = cfg.get("engine", "vllm")
    if engine not in CUSTOM_TEMPLATES:
        return {"error": f"engine {engine} cannot serve a custom artifact", "warnings": warns}
    p = Path(raw).expanduser().resolve()
    roots = [Path(r).resolve() for r in scan_roots()]
    if not any(p == r or r in p.parents for r in roots):
        return {"error": "custom model path is not under any configured scan root", "warnings": warns}
    if not p.exists():
        return {"error": f"artifact not found on disk: {raw}", "warnings": warns}
    if p.is_file():
        kind = "gguf" if p.suffix.lower() == ".gguf" else None
    else:
        names = {e.name.lower() for e in os.scandir(p) if not e.is_symlink()}
        if any(n.startswith("openvino_language_model.") for n in names) or \
           any(n.endswith(".xml") and (n[:-4] + ".bin") in names for n in names):
            kind = "openvino"
        else:
            kind = "vllm"
            qc = p / "quantization_config.json"
            if qc.is_file():
                try:
                    if json.loads(qc.read_text()).get("quant_method") == "exl3":
                        kind = "exl3"
                except (OSError, ValueError):
                    pass
    if kind is None or KIND_ENGINE[kind] != engine:
        return {"error": f"that artifact is {kind or 'of unknown format'}; pick the matching engine "
                         f"({KIND_ENGINE.get(kind, 'llamacpp')})", "warnings": warns}
    cfg_path = p / "config.json"
    recipe = dict(CUSTOM_TEMPLATES[engine])
    recipe["ctx"] = read_ctx_from_config(cfg_path) or recipe["ctx"]
    model = {"id": "custom", "name": p.name, "arch": f"{kind} · custom artifact",
             "recipes": {engine: recipe}}
    det = {"path": str(p), "mount_root": str(p.parent if p.is_file() else p),
           "found_name": p.name, "variant": False,
           "ctx": read_ctx_from_config(cfg_path)}
    warns.append(f"custom artifact: {p.name} — engine flags follow the generic {engine} template, not a verified recipe")
    return model, engine, det, warns

def _exl3_roots():
    """Isolated dockerd/containerd data roots — user-writable by default so a
    fresh install needs no sudo; override via exl3_data_root in settings."""
    raw = SETTINGS.get("exl3_data_root") or "~/.local/share/b70-exl3"
    root = str(Path(raw).expanduser()).rstrip("/")
    return root + "-containerd", root + "-docker"


EXL3_BRIDGE = "b70-exl3-br0"


def ensure_exl3_stack(dsock):
    """Bring up the isolated dockerd + containerd + bridge for exl3xpu
    (kept off the main dockerd: the image is ~24 GB)."""
    try:
        if (Path(dsock).exists()
                and subprocess.run(["docker", "-H", f"unix://{dsock}", "info"],
                                   capture_output=True, timeout=8).returncode == 0):
            return
    except (OSError, subprocess.TimeoutExpired):
        pass
    # clean stale state from a killed daemon
    ctr_root, docker_root = _exl3_roots()
    subprocess.run(["sudo", "-n", "rm", "-f", dsock], capture_output=True, timeout=10)
    subprocess.run(["sudo", "-n", "rm", "-rf", "/run/b70-exl3-containerd"], capture_output=True, timeout=10)
    subprocess.run(["sudo", "-n", "ip", "link", "add", EXL3_BRIDGE, "type", "bridge"], capture_output=True, timeout=10)
    subprocess.run(["sudo", "-n", "ip", "addr", "add", "172.31.77.1/24", "dev", EXL3_BRIDGE], capture_output=True, timeout=10)
    subprocess.run(["sudo", "-n", "ip", "link", "set", EXL3_BRIDGE, "up"], capture_output=True, timeout=10)
    if subprocess.run(["sudo", "-n", "iptables", "-t", "nat", "-C", "POSTROUTING",
                       "-s", "172.31.77.0/24", "!", "-o", EXL3_BRIDGE, "-j", "MASQUERADE"],
                      capture_output=True, timeout=10).returncode != 0:
        subprocess.run(["sudo", "-n", "iptables", "-t", "nat", "-A", "POSTROUTING",
                        "-s", "172.31.77.0/24", "!", "-o", EXL3_BRIDGE, "-j", "MASQUERADE"], capture_output=True, timeout=10)
    subprocess.run(["sudo", "-n", "bash", "-c",
                    "setsid nohup containerd --root=" + shlex.quote(ctr_root) +
                    " --state=/run/b70-exl3-containerd "
                    "--address=/run/b70-exl3-containerd/containerd.sock "
                    "</dev/null >/var/log/exl3-containerd.log 2>&1 &"],
                   capture_output=True, timeout=10)
    time.sleep(4)
    subprocess.run(["sudo", "-n", "bash", "-c",
                    "setsid nohup dockerd --data-root=" + shlex.quote(docker_root) +
                    " --exec-root=/run/b70-exl3-docker --host=unix://" + shlex.quote(dsock) + " "
                    "--pidfile=/run/b70-exl3-docker.pid --bridge=" + EXL3_BRIDGE + " "
                    "--ip-forward=false --ip-masq=false "
                    "--containerd=/run/b70-exl3-containerd/containerd.sock "
                    "</dev/null >/var/log/exl3-dockerd.log 2>&1 &"],
                   capture_output=True, timeout=10)
    for _ in range(20):
        if Path(dsock).exists():
            try:
                res = subprocess.run(["docker", "-H", f"unix://{dsock}", "info"],
                                     capture_output=True, text=True, timeout=5)
                if res.returncode == 0:
                    break
            except Exception:
                pass
        time.sleep(2)
    subprocess.run(["sudo", "-n", "chown", "root:docker", dsock], capture_output=True, timeout=10)
    subprocess.run(["sudo", "-n", "chmod", "660", dsock], capture_output=True, timeout=10)



def build(cfg):
    if cfg.get("model_id") == "__custom__":
        prepared = prepare_custom(cfg)
        if isinstance(prepared, dict):
            return prepared
        model, engine, custom_det, warns = prepared
        det = custom_det
    else:
        model = find_model(cfg.get("model_id") or cfg.get("model") or "")
        engine = cfg.get("engine", "openvino")
        warns = []
        det = detect(model, engine)
    recipe = (model or {}).get("recipes", {}).get(engine)
    if not model or not recipe:
        return {"error": "no recipe for that model+engine", "warnings": warns}
    ctxres = resolve_ctx(model, engine, det)
    artifact_mib = None
    if det and det.get("path"):
        try:
            p = Path(det["path"])
            artifact_mib = (p.stat().st_size if p.is_file()
                            else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())) / 1048576
        except Exception:
            artifact_mib = None
    if det and det.get("variant"):
        warns.append(f"using your local \"{det.get('found_name')}\" — recipe default was "
                     f"{recipe.get('download', {}).get('name') or recipe.get('download', {}).get('search') or 'the listed quant'}")
    try:
        ctx = int(cfg["ctx"]) if cfg.get("ctx") else ctxres["value"]
        port = int(cfg.get("port") or 8000)
        slots = int(cfg.get("slots") or 1)
        extra = shlex.split(cfg.get("extra", "") or "")
    except (ValueError, TypeError) as exc:
        return {"error": f"Invalid context, port, slots or extra flags: {exc}", "warnings": warns}
    if recipe.get("ctx_max") and ctx > recipe["ctx_max"]:
        warns.append(f"ctx {ctx} > engine ceiling {recipe['ctx_max']} for this recipe — clamping.")
        ctx = recipe["ctx_max"]

    extra_env = {}
    for line in (cfg.get("extra_env") or "").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", k):
                return {"error": "Invalid environment variable name", "warnings": warns}
            if _BLOCKED_ENV.match(k):
                return {"error": f"Environment variable {k} is not allowed (it can redirect code loading or process startup)", "warnings": warns}
            extra_env[k] = v.strip()
    envargs = [x for k, v in extra_env.items() for x in ("-e", f"{k}={v}")]
    gpus = cfg.get("gpus", [0])
    kv = cfg.get("kv") or "recipe default"
    env = {}
    native = False
    kind = recipe.get("kind", "")
    if not isinstance(gpus, list) or not gpus or len(gpus) != len(set(map(str, gpus))) or any(type(g) is not int or g not in (0, 1) for g in gpus):
        return {"error": "Select one or two distinct GPU indices (0 or 1).", "warnings": warns}
    if len(gpus) > 1:
        warns.append("Dual-card B70 execution enabled: multi-GPU tensor-split or tensor-parallel active.")
    if not 1 <= port <= 65535 or not 1 <= slots <= 128 or not 512 <= ctx <= 262144:
        return {"error": "Port, slots, or context out of range.", "warnings": warns}
    preflight = hardware_preflight() if not IS_WIN else {"devices": []}
    render_nodes = [preflight["devices"][g]["render"] for g in gpus if g < len(preflight["devices"])]
    render_node = render_nodes[0] if render_nodes else "/dev/dri"
    sel_dev = preflight["devices"][gpus[0]] if gpus[0] < len(preflight["devices"]) else None
    if sel_dev and sel_dev.get("power_cap_w"):
        watts_cfg = int(cfg.get("power") or recipe.get("power") or 150)
        if watts_cfg > sel_dev["power_cap_w"]:
            warns.append(f"selected {watts_cfg}W target exceeds this card's current {sel_dev['power_cap_w']}W hwmon cap — the launcher never changes power settings; benchmark claims above the cap will not reproduce.")
    if len(gpus) > 1:
        warns.append("Two-card configuration is experimental and requires matching B70 GPUs, verified device order, sufficient PCIe bandwidth, and engine support.")

    if kind == "ovms":
        fallback_mount = cfg.get("models_dir") or SETTINGS.get("ovms_repo", "~/models/ovms-repo")
        if det:
            cpath, mount_src, mount_dst = container_path(det, fallback_mount)
            source_model = cpath[len("/models/"):] or "model"
        else:
            source_model = recipe.get("source_model", "model")
            mount_src = fallback_mount
            mount_dst = "/models"
            warns.append("model not detected on disk — command uses default paths (download it or point 'scan root' at it)")
        tokens = ["docker", "run", "-d", "--rm", "--name", f"b70-{model['id']}-ovms"]
        if not IS_WIN:
            gids = device_gids(render_node)
            tokens += ["--user", f"{os.getuid()}:{gids[0] if gids else os.getuid()}",
                       "--device", "/dev/dri"]
            for g in gids:
                tokens += ["--group-add", str(g)]
        tokens += ["-p", f"127.0.0.1:{port}:{port}", "-v", f"{Path(mount_src).expanduser()}:{mount_dst}:rw"] + envargs + [
                   recipe["image"],
                   "--rest_port", str(port),
                   "--model_repository_path", "/models",
                   "--source_model", source_model,
                   "--task", "text_generation",
                   "--target_device", "GPU",
                   "--enable_prefix_caching", "true"]
        if recipe.get("tool_parser"):
            tokens += ["--tool_parser", recipe["tool_parser"]]
        if recipe.get("reasoning_parser"):
            tokens += ["--reasoning_parser", recipe["reasoning_parser"]]
        if recipe.get("cim_long_ctx") and ctx > 20480:
            tokens += ["--cache_interval_multiplier", str(recipe["cim_long_ctx"])]
        tokens += recipe.get("fixed_flags", []) + extra
        warns.append("OVMS manages KV internally (GPU INT4 KV supported). ctx is advisory here.")
        served_name = source_model
        cname = f"b70-{model['id']}-ovms"

    elif kind.startswith("vllm"):
        fallback = recipe.get("model_path", "/models/model")
        if det:
            model_path, mount_src, mount_dst = container_path(det, fallback)
            if mount_src is None:
                mount_src = cfg.get("models_dir") or SETTINGS.get("models_dir", "~/models")
                mount_dst = "/models"
        else:
            model_path = fallback
            mount_src = cfg.get("models_dir") or SETTINGS.get("models_dir", "~/models")
            mount_dst = "/models"
            warns.append("model not detected on disk — using recipe default path")
        # a single mapped render node enumerates as device 0 inside the container
        selector = "level_zero:" + ",".join(str(i) for i in range(len(gpus)))
        cname = f"b70-{model['id']}-vllm"
        served_name = model["name"]
        use_mtp = bool(recipe.get("spec_tokens")) and cfg.get("mtp", True)

        if kind == "vllm-tp2" and not IS_WIN:
            patches_dir = HERE / "patches"
            tp_size = len(gpus) if len(gpus) > 1 else recipe.get("tp", 2)
            mount_src_path = Path(det["path"]) if det and det.get("path") else Path(mount_src).expanduser()
            tokens = ["docker", "run", "-d", "--rm", "--name", cname,
                      "--device", "/dev/dri", "-v", "/dev/dri:/dev/dri:ro",
                      "--group-add", str(render_gid(render_node) or "render"),
                      "--cap-add", "SYS_PTRACE", "--ipc=host", "--workdir", "/",
                      "-v", f"{mount_src_path}:/model:ro",
                      "-v", f"{patches_dir / 'patch_vllm_worker_affinity.py'}:/patch_affinity.py:ro",
                      "-p", f"127.0.0.1:{port}:{port}",
                      "-e", "VLLM_TARGET_DEVICE=xpu",
                      "-e", "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE",
                      "-e", "B70_MTP_BF16_DRAFT=1",
                      "-e", "VLLM_XPU_ENABLE_XPU_GRAPH=1",
                      "-e", "PYTORCH_ALLOC_CONF=expandable_segments:True",
                      "-e", "CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296",
                      "-e", "CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296",
                      "-e", "CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296",
                      "-e", "CCL_SYCL_ALLTOALL_TMP_BUF=1"] + envargs
            serve = ["vllm", "serve", "/model",
                     "--quantization", "fp8",
                     "--dtype", recipe.get("dtype", "bfloat16"),
                     "--tensor-parallel-size", str(tp_size),
                     "--max-model-len", str(ctx),
                     "--async-scheduling",
                     "--gpu-memory-utilization", "0.90",
                     "--kv-cache-dtype", "fp8",
                     "--port", str(port),
                     "--max-num-seqs", str(slots),
                     "--max-num-batched-tokens", "4096",
                     "--enable-prefix-caching",
                     "--served-model-name", served_name,
                     "--language-model-only"]
            if use_mtp:
                serve += ["--speculative-config",
                          json.dumps({"method": "mtp", "num_speculative_tokens": recipe.get("spec_tokens", 8)})]
            serve += recipe.get("fixed_flags", []) + extra
            script = "set -e; python /patch_affinity.py; exec " + " ".join(shlex.quote(a) for a in serve)
            tokens += ["--entrypoint", "bash", recipe["image"], "-lc", script]
            warns.append("vLLM TP2 dual-card: worker affinity patch + oneCCL threshold pins active; prefix caching enabled.")
        elif kind == "vllm-arext" and not IS_WIN:
            patches_dir = HERE / "patches"
            needed = ("patch_mtp_nightly.py", "patch_mtp_boundary.py", "patch_champion_stack_overlay.py")
            if not all((patches_dir / p).is_file() for p in needed):
                return {"error": "bundled AutoRound/vLLM patch scripts are missing from this install", "warnings": warns}
            model_dir = (det or {}).get("path") or recipe.get("model_path", "")
            if not model_dir or not Path(model_dir).exists():
                return {"error": "AutoRound artifact not found on disk — point the scan root at it or set model_path.",
                        "warnings": warns}
            cname = f"b70-{model['id']}-vllm"
            served_name = model["name"]
            tokens = ["docker", "run", "-d", "--rm", "--name", cname,
                      "--device", "/dev/dri", "-v", "/dev/dri/by-path:/dev/dri/by-path:ro",
                      "--group-add", str(render_gid(render_node) or "render"),
                      "-v", f"{model_dir}:/model:ro",
                      "-v", f"{patches_dir / 'patch_mtp_nightly.py'}:/patch_mtp.py:ro",
                      "-v", f"{patches_dir / 'patch_mtp_boundary.py'}:/patch_boundary.py:ro",
                      "-v", f"{patches_dir / 'patch_champion_stack_overlay.py'}:/patch_gdn_fuse.py:ro",
                      "-p", f"127.0.0.1:{port}:8000",
                      "-e", "VLLM_TARGET_DEVICE=xpu",
                      "-e", "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE",
                      "-e", "PYTORCH_ALLOC_CONF=expandable_segments:True",
                      "-e", "VLLM_WORKER_MULTIPROC_METHOD=spawn",
                      "-e", f"ZE_AFFINITY_MASK={gpus[0]}",
                      "-e", "VLLM_XPU_ENABLE_XPU_GRAPH=1",
                      "-e", "B70_MTP_BF16_DRAFT=1"] + envargs
            serve = ["vllm", "serve", "/model", "--dtype", "bfloat16",
                     "--trust-remote-code", "--kv-cache-dtype", "fp8",
                     "--max-model-len", str(ctx),
                     "--gpu-memory-utilization", "0.88",
                     "--max-num-seqs", str(slots), "--max-num-batched-tokens", "8192",
                     "--served-model-name", served_name,
                     "--host", "0.0.0.0", "--port", "8000"]
            if cfg.get("mtp", True):
                serve += ["--speculative-config",
                          json.dumps({"method": "mtp", "num_speculative_tokens": recipe.get("spec_tokens", 4)})]
            serve += recipe.get("fixed_flags", []) + extra
            script = "set -e; python /patch_mtp.py; python /patch_boundary.py; python /patch_gdn_fuse.py; exec " + " ".join(shlex.quote(a) for a in serve)
            tokens += ["--entrypoint", "bash", recipe["image"], "-lc", script]
            warns.append("AutoRound W4A16: MTP4 + fp8 KV + patched GDN fuse; prefix caching intentionally OFF "
                         "(cache ON corrupts long generations on this stack — campaign finding).")

        elif kind == "vllm-mtp" and not IS_WIN and det:
            patches_dir = HERE / "patches"
            if not (patches_dir / "patch_mtp_nightly.py").is_file() or not (patches_dir / "patch_mtp_boundary.py").is_file():
                return {"error": "bundled MTP patch scripts are missing from this install", "warnings": warns}
            tokens = ["docker", "run", "-d", "--rm", "--name", cname,
                      "--device", "/dev/dri", "-v", "/dev/dri:/dev/dri:ro",
                      "--group-add", str(render_gid(render_node) or "render"),
                      "-v", f"{Path(det['path'])}:/model:ro",
                      "-v", f"{patches_dir / 'patch_mtp_nightly.py'}:/patch_mtp.py:ro",
                      "-v", f"{patches_dir / 'patch_mtp_boundary.py'}:/patch_boundary.py:ro",
                      "-p", f"127.0.0.1:{port}:{port}",
                      "-e", "VLLM_TARGET_DEVICE=xpu",
                      "-e", "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE",
                      "-e", f"ZE_AFFINITY_MASK={gpus[0]}",
                      "-e", "B70_MTP_BF16_DRAFT=1",
                      "-e", "VLLM_XPU_ENABLE_XPU_GRAPH=1",
                      "-e", "PYTORCH_ALLOC_CONF=expandable_segments:True"] + envargs
            serve = ["vllm", "serve", "/model", "--quantization", "gptq", "--dtype", "float16",
                     "--max-model-len", str(ctx),
                     "--gpu-memory-utilization", "0.88" if use_mtp else "0.90",
                     "--kv-cache-dtype", "fp8",
                     "--port", str(port),
                     "--max-num-seqs", str(slots),
                     "--max-num-batched-tokens", "8192",
                     "--enable-prefix-caching",
                     "--served-model-name", served_name,
                     "--language-model-only"]
            if use_mtp:
                serve += ["--speculative-config",
                          json.dumps({"method": "mtp", "num_speculative_tokens": recipe.get("spec_tokens", 4)})]
            serve += extra
            script = "set -e; python /patch_mtp.py; python /patch_boundary.py; exec " + " ".join(shlex.quote(a) for a in serve)
            tokens += ["--entrypoint", "bash", recipe["image"], "-lc", script]
            warns.append("cookbook MTP path: BF16 draft + FP8 KV + patched GDN boundary; prefix caching enabled.")
        else:
            tokens = ["docker", "run", "-d", "--rm", "--name", cname]
            if not IS_WIN:
                tokens += ["--privileged", "--device", "/dev/dri",
                           "-v", f"{Path.home() / '.cache' / 'huggingface'}:/root/.cache/huggingface:rw"]
            else:
                warns.append("Windows GPU passthrough is unvalidated; real launch blocked by preflight.")
            tokens += ["--ipc=host",
                       "-v", f"{Path(mount_src).expanduser()}:{mount_dst}:ro",
                       "-p", f"127.0.0.1:{port}:{port}",
                       "-e", f"ONEAPI_DEVICE_SELECTOR={selector}"] + envargs + [
                       recipe["image"],
                       "--model", model_path,
                       "--dtype", recipe.get("dtype", "bfloat16"),
                       "--gpu-memory-utilization", "0.92",
                       "--max-model-len", str(ctx),
                       "--enable-prefix-caching",
                       "--port", str(port)]
            if use_mtp:
                spec = {"method": "mtp", "num_speculative_tokens": recipe.get("spec_tokens", 4)}
                if recipe.get("speculative"):
                    spec["model"] = recipe["speculative"]
                tokens += ["--speculative-config", json.dumps(spec)]
            if kv == "fp8" or recipe.get("kv") == "fp8":
                tokens += ["--kv-cache-dtype", "fp8"]
            if kind in ("vllm-autoround", "vllm-mtp"):
                tokens += recipe.get("fixed_flags", [])
            if len(gpus) > 1:
                tokens += ["--tensor-parallel-size", str(len(gpus))]
                warns.append("Multi-GPU TP on XPU: verify the level_zero selector string on your driver.")
            if slots > 1:
                tokens += ["--max-num-seqs", str(slots)]
            tokens += extra
        if kind == "vllm-autoround":
            warns.append("FP16 crash guard: dtype float16 + --enforce-eager is intentional (dt_bias crash on BF16 path).")
    elif kind == "exl3":
        if IS_WIN:
            return {"error": "EXL3 XPU is Linux-only (isolated dockerd on /dev/dri).", "warnings": warns}
        model_dir = (det or {}).get("path") or recipe.get("model_path", "")
        if not model_dir or not Path(model_dir).exists():
            return {"error": "EXL3 artifact not found on disk — point the scan root at it or set model_path.",
                    "warnings": warns}
        dsock = recipe.get("docker_sock", "")
        docker_cmd = ["docker"] + (["-H", f"unix://{dsock}"] if dsock else [])
        if dsock:
            if cfg.get("dry_run"):
                # preview must not touch dockerd — especially not a sudo-spawned one
                warns.append("isolated exl3 dockerd auto-starts on launch (needs passwordless sudo)")
            else:
                try:
                    probe = subprocess.run(docker_cmd + ["info"], capture_output=True, timeout=8)
                except (OSError, subprocess.TimeoutExpired):
                    probe = None
                if probe is None or probe.returncode != 0:
                    ensure_exl3_stack(dsock)
                    try:
                        probe = subprocess.run(docker_cmd + ["info"], capture_output=True, timeout=8)
                    except (OSError, subprocess.TimeoutExpired):
                        probe = None
                if probe is None or probe.returncode != 0:
                    return {"error": f"isolated exl3 dockerd unreachable at {dsock} even after auto-start "
                                     "(auto-start needs passwordless sudo; logs at /var/log/exl3-*.log).",
                            "warnings": warns}
        inner_port = int(recipe.get("inner_port", 8100))
        gmu = "0.94" if ctx >= 131072 else "0.90"
        cname = f"b70-{model['id']}-exl3"
        served_name = model["name"]
        tokens = docker_cmd + ["run", "-d", "--rm", "--name", cname,
                 "--device", "/dev/dri",
                 "-v", "/dev/dri/by-path:/dev/dri/by-path:ro",
                 "--group-add", str(render_gid(render_node) or "render"),
                 "-v", f"{model_dir}:/mnt/exl3model:ro",
                 "-p", f"127.0.0.1:{port}:{inner_port}"] + envargs + [
                 recipe["image"],
                 recipe.get("serve_config", "models/qwen3.8-27b-exl3-4.00bpw"),
                 "--gpu", str(gpus[0]),
                 "--model-path", "/mnt/exl3model",
                 "--set", f"vllm.gpu_memory_utilization={gmu}",
                 "--set", f"vllm.max_model_len={ctx}",
                 "--set", "vllm.enable_prefix_caching=true"]
        tokens += recipe.get("fixed_flags", []) + extra
        warns.append(f"EXL3 trellis + native MTP3 on the isolated dockerd; fp8 KV only; gmu {gmu} "
                     "(0.94 required at ≥128K ctx); first load ~5-8 min for kernel compile.")
        if slots > 1:
            warns.append("EXL3 manages concurrency internally; slots is advisory here.")

    elif kind in ("gguf", "gguf-tiered"):
        fallback = recipe.get("gguf", "/models/model.gguf")
        if det:
            gpath, mount_src, mount_dst = container_path(det, fallback)
            if mount_src is None:
                mount_src = cfg.get("models_dir") or SETTINGS.get("models_dir", "~/models")
                mount_dst = "/models"
            host_gpath = det.get("path") or gpath  # native mode needs the host path
        else:
            gpath = fallback
            host_gpath = fallback
            mount_src = cfg.get("models_dir") or SETTINGS.get("models_dir", "~/models")
            mount_dst = "/models"
            warns.append("model not detected on disk — using recipe default path")
        kv_map = {"q5_0/q4_1": ["--cache-type-k", "q5_0", "--cache-type-v", "q4_1"],
                  "q8_0": ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
                  "q8_0/q4_1": ["--cache-type-k", "q8_0", "--cache-type-v", "q4_1"],
                  "f16": ["--cache-type-k", "f16", "--cache-type-v", "f16"]}
        if kv == "f16":
            warns.append("f16 KV wastes VRAM with zero quality gain on these workloads. q8_0 recommended.")
        llama_bin = (recipe.get("llama_bin") or SETTINGS.get("llama_bin") or "").strip()
        if llama_bin:
            llama_bin = str(Path(llama_bin).expanduser())
        if (llama_bin and Path(llama_bin).is_file()) and not cfg.get("use_docker"):
            tokens = [llama_bin]
            native = True
            model_arg = host_gpath
            env.update(extra_env)
            sycl_devs = ",".join(f"SYCL{g}" for g in gpus)
            l0_devs = ",".join(str(g) for g in gpus)
            env.update({
                "ONEAPI_DEVICE_SELECTOR": "level_zero:" + l0_devs,
                "ZES_ENABLE_SYSMAN": "1", "SYCL_CACHE_PERSISTENT": "0",
                "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "0" if kind == "gguf-tiered" else "1",
                "UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS": "1",
                "SYCL_DEVICE_FILTER": "level_zero",
                "ZE_FLAT_DEVICE_HIERARCHY": "COMPOSITE", "ZE_AFFINITY_MASK": l0_devs,
            })
            # oneAPI runtime libs (libsvml, libdnnl, libsycl, …): a fresh user
            # won't have `source setvars.sh` in their shell env, and the SYCL
            # build fails to load without them. Prepend every oneAPI */latest/lib
            # dir we can find so the native binary just works.
            oneapi_dirs = [p for p in
                           sorted(glob.glob("/opt/intel/oneapi/*/latest/lib"))
                           if Path(p).is_dir()]
            if oneapi_dirs:
                ldp = env.get("LD_LIBRARY_PATH") or os.environ.get("LD_LIBRARY_PATH") or ""
                env["LD_LIBRARY_PATH"] = os.pathsep.join(oneapi_dirs + ([ldp] if ldp else []))
            else:
                warns.append("Native llama.cpp: oneAPI libs not found under /opt/intel/oneapi — run inside `source /opt/intel/oneapi/setvars.sh` shell if the binary fails to start.")
            if kind == "gguf-tiered":
                env["LLAMA_ATTN_ROT_DISABLE"] = "1"
            host = "127.0.0.1"  # native: never expose an unauthenticated server to the LAN
        else:
            host = "0.0.0.0"  # inside the container; only 127.0.0.1 is published
            tokens = ["docker", "run", "-d", "--rm", "--name", f"b70-{model['id']}-sycl"]
            if not IS_WIN:
                tokens += ["--device", "/dev/dri"]
                for g in device_gids(render_node):
                    tokens += ["--group-add", str(g)]
            tokens += ["--ipc=host", "-v", f"{Path(mount_src).expanduser()}:{mount_dst}:ro",
                       "-p", f"127.0.0.1:{port}:{port}", "-e", "ZES_ENABLE_SYSMAN=1"] + envargs + [
                       recipe["image"]]
            model_arg = gpath
        tokens += ["-m", model_arg, "-ngl", "99", "--host", host, "--port", str(port),
                   "-c", str(ctx), "--flash-attn", "on", "--metrics"]
        if "--cache-type-k" not in recipe.get("fixed_flags", []):
            tokens += kv_map.get(kv, kv_map.get(recipe.get("kv"), kv_map["q8_0"]))
        if len(gpus) > 1 or recipe.get("tensor_split"):
            ts = recipe.get("tensor_split", "49,51" if len(gpus) == 2 else ",".join(["1"] * len(gpus)))
            tokens += ["--device", ",".join(f"SYCL{g}" for g in gpus),
                       "--tensor-split", ts,
                       "--split-mode", recipe.get("split_mode", "layer")]
            warns.append(f"Dual-GPU tensor split: {ts} (split-mode: {recipe.get('split_mode', 'layer')}).")
        if recipe.get("offload_tensors"):
            tokens += ["-ot", recipe["offload_tensors"]]
            warns.append("3-Tiered Memory: N-gram embedding & MoE boundary blocks offloaded to CPU host RAM (-ot).")
        draft = _resolve_draft(recipe.get("draft_model"))
        if draft:
            draft_arg = draft if native else f"/draft/{Path(draft).name}"
            if not native:
                tokens.insert(tokens.index("-v") if "-v" in tokens else len(tokens), "-v")
                tokens.insert(tokens.index("-v") + 1, f"{Path(draft).parent}:/draft:ro")
            tokens += ["-md", draft_arg, "-ngld", "999",
                       "--spec-type", "draft-mtp",
                       "--spec-draft-n-max", str(recipe.get("spec_tokens", 3)),
                       "--spec-draft-p-min", str(recipe.get("spec_p_min", 0.75)),
                       "--spec-draft-backend-sampling",
                       "--spec-draft-device", recipe.get("draft_device", "SYCL1")]
            warns.append(f"MTP Speculative Decoding: draft model {Path(draft).name} on {recipe.get('draft_device', 'SYCL1')}.")
        if slots > 1:
            tokens += ["-np", str(slots)]
        tokens += recipe.get("fixed_flags", []) + extra
        served_name = Path(gpath).stem
        cname = f"b70-{model['id']}-sycl"
    else:
        return {"error": f"unknown recipe kind {kind}", "warnings": warns}

    watts = int(cfg.get("power") or recipe.get("power") or 150)
    power_cmd = "Not generated: identify the exact GPU hwmon power1_cap and safe board limit manually. The launcher never changes power settings."
    return {
        "cmd": pretty(tokens), "tokens": tokens, "env": env, "warnings": warns,
        "power_cmd": power_cmd, "power": watts, "cname": cname,
        "endpoint": f"http://127.0.0.1:{port}/v1", "served_name": served_name,
        "model_name": model["name"], "engine": engine,
        "ctx": ctx, "ctx_source": "manual override" if cfg.get("ctx") else ctxres["source"],
        "detected": bool(det), "detected_path": (det or {}).get("path"),
        "native": native,
        "artifact_mib": round(artifact_mib, 1) if artifact_mib else None,
    }


# ---------------------------------------------------------------- usage history

SESSION_FIELDS = ("id", "model", "engine", "port", "started")


def _usage_load():
    try:
        data = json.loads(USAGE_PATH.read_text())
        if isinstance(data, dict) and isinstance(data.get("sessions"), list):
            return data
    except Exception:
        pass
    return {"sessions": []}


def _usage_save(data):
    data["sessions"] = data["sessions"][-500:]
    _atomic_write(USAGE_PATH, json.dumps(data, indent=1))


def _atomic_write(path, text):
    """tmp+rename: a crash mid-write must never leave a truncated file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.rename(path)


def record_usage(entry, status):
    """Append a session-scoped record (deltas since this app session started).

    The sess_* seeds advance with every record, so a record is a true delta and
    a duplicate call (stop after exit, double stop) writes zeros, never double
    counts."""
    rec = {k: entry.get(k) for k in SESSION_FIELDS if entry.get(k) is not None}
    tin = int(max((entry.get("tokens_in") or 0) - (entry.get("sess_in") or 0), 0))
    tout = int(max((entry.get("tokens_out") or 0) - (entry.get("sess_out") or 0), 0))
    reqs = int(max((entry.get("requests") or 0) - (entry.get("sess_reqs") or 0), 0))
    rec.update({
        "ended": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "tokens_in": tin,
        "tokens_out": tout,
        "requests": reqs,
        "peak_tok_s": entry.get("peak_tok_s"),
        "artifact_mib": entry.get("artifact_mib"),
    })
    entry["sess_in"] = entry.get("tokens_in") or 0
    entry["sess_out"] = entry.get("tokens_out") or 0
    entry["sess_reqs"] = entry.get("requests") or 0
    if tin or tout or reqs:
        with USAGE_LOCK:
            data = _usage_load()
            data["sessions"].append(rec)
            _usage_save(data)


def usage_summary():
    with USAGE_LOCK:
        data = _usage_load()
    sessions = data["sessions"][-50:]
    totals = {"tokens_in": 0, "tokens_out": 0, "requests": 0, "sessions": len(data["sessions"])}
    for s in data["sessions"]:
        totals["tokens_in"] += s.get("tokens_in") or 0
        totals["tokens_out"] += s.get("tokens_out") or 0
        totals["requests"] += s.get("requests") or 0
    return {"sessions": sessions, "totals": totals}


def persist_state():
    """Persist tracked servers so a restart can re-adopt still-running engines."""
    snap = {}
    with LOCK:
        for rid, e in RUNNING.items():
            if e.get("status") in ("stopped", "dry-run"):
                continue
            snap[rid] = {k: e.get(k) for k in
                         ("id", "model", "model_id", "engine", "cfg", "cname", "endpoint", "port",
                          "native", "log", "started", "cmd", "artifact_mib", "pid",
                          "tokens_in", "tokens_out", "requests",
                          "sess_in", "sess_out", "sess_reqs")}
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(snap, indent=1))
    tmp.rename(STATE_PATH)


# ---------------------------------------------------------------- metrics

TOKS_RE = re.compile(r"([\d.]+)\s*tokens per second|throughput:\s*([\d.]+)\s*tokens/s", re.I)
LLAMA_RUN_RE = re.compile(r"prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens|eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens", re.I)
MEM_RES = [
    (re.compile(r"([\d.]+)\s*(MiB|GiB)\s*VRAM used", re.I), "vram used (log)"),
    (re.compile(r"model weights take\s*([\d.]+)\s*GiB", re.I), "weights GiB (log)"),
    (re.compile(r"buffer size =\s*([\d.]+)\s*(MiB|GiB)", re.I), "buffer (log)"),
]
PROM_RE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|NaN)\s*$")
INTERNAL_KEYS = {"proc", "_prev"}


def _parse_prom(text):
    # same counter name can appear once per label set (finished_reason, model…);
    # token/request accounting needs the SUM across series
    counters = {}
    for line in text.splitlines():
        m = PROM_RE.match(line.strip())
        if m:
            try:
                counters[m.group(1)] = counters.get(m.group(1), 0.0) + float(m.group(2))
            except ValueError:
                continue
    return counters


def _pick_counter(counters, patterns, exclude=()):
    """Best single counter for a token bucket.

    Engines export breakdowns of the same total (by_source, cached, per_pos…);
    summing them would double-count, so prefer the canonical shortest name."""
    cands = {}
    for n, v in counters.items():
        ln = n.lower()
        if not ln.endswith("_total") or "token" not in ln:
            continue
        if "second" in ln or not re.search(patterns, ln):
            continue
        if any(x in ln for x in exclude):
            continue
        cands[n] = v
    if not cands:
        return 0.0
    best = min(cands, key=lambda n: (len(n), n))
    return cands[best]


def _classify_tokens(counters):
    """Prometheus counters -> (tokens_in, tokens_out, requests)."""
    tin = _pick_counter(counters, r"prompt|input|prefill",
                        exclude=("by_source", "cached", "draft", "accepted", "per_pos"))
    tout = _pick_counter(counters, r"generation|generated|output|completion|predict",
                         exclude=("draft", "accepted", "per_pos", "seconds"))
    reqs = 0.0
    for name, val in counters.items():
        n = name.lower()
        if n.endswith("_total") and "request" in n and re.search(r"success|finished|complete|count", n):
            reqs += val
    return tin, tout, reqs


def _engine_prometheus(entry, timeout=3):
    """Scrape the engine's own /metrics endpoint (same port as the API)."""
    port = entry.get("port")
    if not port:
        return None
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics", headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if getattr(resp, "status", 200) != 200:
                return None
            return _parse_prom(resp.read(1 << 22).decode("utf-8", "replace"))
    except Exception:
        return None


def vram_probe():
    """Best real VRAM reading the OS exposes, with its source. None = not exposed."""
    if IS_WIN:
        return None
    for usedp in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_used")):
        try:
            used = int(Path(usedp).read_text().strip())
            totalp = Path(usedp).parent / "mem_info_vram_total"
            total = int(totalp.read_text().strip()) if totalp.exists() else None
            return {"used_mib": round(used / 1048576, 1),
                    "total_mib": round(total / 1048576, 1) if total else None,
                    "source": "sysfs"}
        except Exception:
            continue
    return None


def _account_engine(rid, e, now):
    """Token accounting for one live entry: engine /metrics scrape (source of
    truth) + log-tail fallback; mutates e under LOCK. Network I/O happens
    outside the lock. Returns the per-rid counters dict (or None on skip)."""
    toks = mem = None
    try:
        with open(e["log"], "rb") as lf:
            lf.seek(0, 2)
            size = lf.tell()
            lf.seek(max(0, size - 8000))
            tail = lf.read().decode("utf-8", "replace")
        ms = TOKS_RE.findall(tail)
        if ms:
            toks = ms[-1][0] or ms[-1][1]
        for rx, label in MEM_RES:
            mm = rx.findall(tail)
            if mm:
                mem = f"{label}: {mm[-1][0]} {mm[-1][1] if len(mm[-1]) > 1 else ''}".strip()
                break
        llama_runs = LLAMA_RUN_RE.findall(tail)
    except Exception:
        llama_runs = []

    prom = _engine_prometheus(e, timeout=1.5)
    with LOCK:
        if prom:
            tin, tout, reqs = _classify_tokens(prom)
            if tin or tout or reqs:
                # counters went BACKWARDS to a smaller non-zero value -> the
                # engine restarted under this rid; re-baseline both the total
                # and the session seed. An all-zero scrape (engine mid-warmup)
                # is ignored so a hiccup cannot zero the display.
                if 0 < tin < (e.get("tokens_in") or 0):
                    e["tokens_in"] = e["sess_in"] = tin
                if 0 < tout < (e.get("tokens_out") or 0):
                    e["tokens_out"] = e["sess_out"] = tout
                if 0 < reqs < (e.get("requests") or 0):
                    e["requests"] = e["sess_reqs"] = reqs
                e["tokens_in"] = max(e.get("tokens_in") or 0, tin)
                e["tokens_out"] = max(e.get("tokens_out") or 0, tout)
                e["requests"] = max(e.get("requests") or 0, reqs)
                e["metrics_source"] = "engine /metrics"
        elif llama_runs and e.get("metrics_source") != "engine /metrics":
            pin = sum(int(a or 0) for a, b in llama_runs)
            pout = sum(int(b or 0) for a, b in llama_runs)
            if pin:
                e["tokens_in"] = max(e.get("tokens_in") or 0, pin)
            if pout:
                e["tokens_out"] = max(e.get("tokens_out") or 0, pout)
            e["metrics_source"] = "engine log"

        prev = e.get("_prev")
        if prev and now > prev[0]:
            dt = now - prev[0]
            rate = max((e.get("tokens_out") or 0) - prev[1], 0) / dt
            if rate > 0.01:
                smoothed = (e.get("tok_s") or rate) * 0.6 + rate * 0.4
                e["tok_s"] = round(smoothed, 1)
                e["peak_tok_s"] = round(max(e.get("peak_tok_s") or 0, smoothed), 1)
            elif e.get("tok_s"):
                e["tok_s"] = 0.0  # idle: show zero instead of a stale rate
        e["_prev"] = (now, e.get("tokens_out") or 0)
        e["_metrics_ts"] = now
        return {"toks": toks, "engine_mem": mem,
                "tokens_in": e.get("tokens_in"), "tokens_out": e.get("tokens_out"),
                "requests": e.get("requests"), "tok_s": e.get("tok_s"),
                "metrics_source": e.get("metrics_source")}


def metrics_snapshot():
    """Full debug view: per-server counters + container cpu/mem + power + vram.
    The UI polls /api/power and /api/servers instead — this endpoint is for
    explicit callers, so docker stats only runs while a live server exists."""
    with LOCK:
        items = list(RUNNING.items())
    stats = {}
    if (shutil.which("docker")
            and not (os.environ.get("DOCKER_HOST") or os.environ.get("DOCKER_CONTEXT"))
            and any(e.get("status") not in ("stopped", "dry-run", "stopping") for _, e in items)):
        try:
            r = subprocess.run(
                ["docker", "stats", "--no-stream",
                 "--format", "{{.Name}}|{{.CPUPerc}}|{{.MemUsage}}"],
                capture_output=True, text=True, timeout=8)
            for line in r.stdout.splitlines():
                parts = line.split("|")
                if len(parts) == 3:
                    stats[parts[0]] = {"cpu": parts[1], "mem": parts[2]}
        except Exception:
            pass
    now = time.time()
    out = {"power": power_probe() if not IS_WIN else []}
    for rid, e in items:
        if e.get("status") in ("stopped", "dry-run"):
            continue
        acct = _account_engine(rid, e, now)
        if acct:
            acct["stats"] = stats.get(e.get("cname", ""))
            out[rid] = acct
    vram = vram_probe()
    if not vram:
        ests = [e.get("artifact_mib") for _, e in items if e.get("artifact_mib")]
        if ests:
            vram = {"used_mib": round(max(ests) * 1.1, 1), "total_mib": None,
                    "source": "estimate (file x1.1)"}
    out["vram"] = vram
    return out


def sync_harness_configs(port=8000, model_name="Qwen3.8-27B", model_id="qwen38-27b", engine="vllm"):
    """Sync port and model configuration to client tools (Pi, OMP, Factory Droid)."""
    # 1. Update ~/.pi/agent/models.json
    try:
        pi_cfg = Path.home() / ".pi" / "agent" / "models.json"
        if pi_cfg.exists():
            data = json.loads(pi_cfg.read_text())
            if "providers" in data and isinstance(data["providers"], dict):
                b70 = data["providers"].setdefault("b70-vllm", {})
                b70["baseUrl"] = f"http://127.0.0.1:{port}/v1"
                b70["api"] = "openai-completions"
                b70["apiKey"] = "local-b70"
                models = b70.setdefault("models", [])
                existing_ids = {m["id"] for m in models if isinstance(m, dict) and "id" in m}

                for mid in (model_name, model_id):
                    if mid and mid not in existing_ids:
                        models.append({
                            "id": mid,
                            "name": f"{mid} (Local B70)",
                            "supportsTools": False,
                            "reasoning": True,
                            "contextWindow": 102400,
                            "maxTokens": 16384
                        })
                        existing_ids.add(mid)

                for rm in RECIPES.get("models", []):
                    for mid in (rm.get("name"), rm.get("id")):
                        if mid and mid not in existing_ids:
                            models.append({
                                "id": mid,
                                "name": f"{rm.get('name')} (Local B70)",
                                "supportsTools": False,
                                "reasoning": True,
                                "contextWindow": 102400,
                                "maxTokens": 16384
                            })
                            existing_ids.add(mid)

                _atomic_write(pi_cfg, json.dumps(data, indent=1) + "\n")
    except Exception as exc:
        print(f"Warning: failed updating ~/.pi/agent/models.json: {exc}")

    # 2. Update ~/.omp/agent/models.yml
    try:
        omp_cfg = Path.home() / ".omp" / "agent" / "models.yml"
        if omp_cfg.exists():
            text = omp_cfg.read_text()
            text = re.sub(
                r'(b70-vllm:\s*\n\s*baseUrl:\s*http://127\.0\.0\.1:)\d+(/v1)',
                rf'\g<1>{port}\g<2>',
                text
            )
            for rm in RECIPES.get("models", []):
                for mid in (rm.get("name"), rm.get("id")):
                    # regex-inserted into YAML — refuse anything that could
                    # break the document (newlines, colons, quotes)
                    safe = mid and rm.get("name") and all(
                        re.fullmatch(r"[A-Za-z0-9_.,:+() -]+", str(x))
                        for x in (mid, rm["name"]))
                    if safe and mid not in text and "b70-vllm:" in text:
                        pattern = r'(b70-vllm:\s*\n(?:\s+.*\n)*?\s+models:\s*\n)'
                        m_entry = (
                            f"    - id: {mid}\n"
                            f"      name: {rm.get('name')} (B70 vLLM)\n"
                            f"      input:\n"
                            f"      - text\n"
                            f"      - image\n"
                            f"      supportsTools: false\n"
                            f"      reasoning: true\n"
                            f"      contextWindow: 102400\n"
                            f"      maxTokens: 16384\n"
                        )
                        text = re.sub(pattern, rf'\g<1>{m_entry}', text, count=1)
            _atomic_write(omp_cfg, text)
    except Exception as exc:
        print(f"Warning: failed updating ~/.omp/agent/models.yml: {exc}")

    # 3. Update ~/.factory/settings.json
    try:
        droid_cfg = Path.home() / ".factory" / "settings.json"
        if droid_cfg.exists():
            data = json.loads(droid_cfg.read_text())
            models_dict = data.setdefault("models", {})
            models_dict["desktop-b70"] = {
                "name": "Desktop B70 Loaded Model",
                "provider": "openai",
                "modelId": model_name,
                "baseUrl": f"http://127.0.0.1:{port}/v1"
            }
            custom_models = data.setdefault("customModels", [])
            b70_custom = next((m for m in custom_models if m.get("id") == "custom:Desktop-B70-Loaded-Model-0"), None)
            if b70_custom:
                b70_custom["model"] = model_name
                b70_custom["baseUrl"] = f"http://127.0.0.1:{port}/v1"
                b70_custom["displayName"] = f"{model_name} (B70)"
                b70_custom["provider"] = "openai"
            else:
                custom_models.insert(0, {
                    "model": model_name,
                    "id": "custom:Desktop-B70-Loaded-Model-0",
                    "index": 42,
                    "baseUrl": f"http://127.0.0.1:{port}/v1",
                    "apiKey": "local-b70",
                    "displayName": f"{model_name} (B70)",
                    "maxOutputTokens": 32768,
                    "noImageSupport": True,
                    "provider": "openai"
                })
            favs = data.setdefault("modelFavorites", [])
            cid = "custom:Desktop-B70-Loaded-Model-0"
            if cid in favs:
                favs.remove(cid)
            favs.insert(0, cid)
            _atomic_write(droid_cfg, json.dumps(data, indent=2) + "\n")
    except Exception as exc:
        print(f"Warning: failed updating ~/.factory/settings.json: {exc}")


def harness_line(cfg, built, write_sync=True):
    port = int(cfg.get("port") or built.get("port") or 8000)

    # Resolve target model name and ID
    model_name = built.get("model_name")
    model_id = cfg.get("model_id") or cfg.get("model") or built.get("model_id")
    engine = built.get("engine") or cfg.get("engine") or "vllm"

    # If running server is on this port, prefer its registered model
    with LOCK:
        for rid, e in RUNNING.items():
            if e.get("port") == port and e.get("status") in ("running", "starting", "running (adopted)"):
                if e.get("model"):
                    model_name = e.get("model")
                if e.get("model_id"):
                    model_id = e.get("model_id")
                if e.get("engine"):
                    engine = e.get("engine")
                break

    # If endpoint is live, query /v1/models to see what it actually serves
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models", headers={"Authorization": "Bearer local"})
        with urllib.request.urlopen(req, timeout=0.6) as resp:
            m_data = json.loads(resp.read(1 << 20).decode())
            if m_data.get("data") and len(m_data["data"]) > 0:
                live_id = m_data["data"][0].get("id")
                if live_id:
                    model_name = live_id
    except Exception:
        pass

    # Fallback to recipes lookup
    if not model_name or model_name in ("default", ""):
        m = find_model(model_id or "")
        if m:
            model_name = m["name"]
            model_id = m["id"]
        else:
            model_name = model_id or "Qwen3.8-27B"

    # Synchronize configs for pi, omp, and droid (never on preview/dry-run)
    if write_sync and not cfg.get("dry_run"):
        sync_harness_configs(port, model_name, model_id, engine)

    h_id = cfg.get("harness", "omp")
    if h_id == "webui":
        return "xdg-open http://localhost:3000"

    endpoint = built.get("endpoint") or f"http://127.0.0.1:{port}/v1"
    prefix = "" if IS_WIN else (f"OPENAI_BASE_URL={shlex.quote(str(endpoint))} "
                               f"OPENAI_API_KEY=local "
                               f"OPENAI_MODEL_NAME={shlex.quote(str(model_name or ''))} ")

    # Check for custom override command
    custom_cmd = (cfg.get("harness_cmd") or "").strip()
    if custom_cmd:
        return f"{prefix}{custom_cmd}"

    if h_id == "pi":
        tools_flag = " --no-tools" if engine != "openvino" else ""
        return f"{prefix}pi --model {shlex.quote('b70-vllm/' + str(model_name))}{tools_flag}"
    elif h_id == "omp":
        tools_flag = " --no-tools" if engine != "openvino" else ""
        return f"{prefix}omp --model {shlex.quote('b70-vllm/' + str(model_name))}{tools_flag}"
    elif h_id == "droid":
        return f"{prefix}droid --model custom:Desktop-B70-Loaded-Model-0"
    else:
        hs = {h["id"]: h for h in SETTINGS.get("harnesses", [])}
        h = hs.get(h_id, {"cmd": h_id})
        return f"{prefix}{h.get('cmd', h_id)}"


# ---------------------------------------------------------------- run / stop

def _rid(cfg):
    """Tracked-server key; sanitized — it is also used in log filenames."""
    rid = f"{cfg.get('model_id') or cfg.get('model')}-{cfg.get('engine')}-{cfg.get('port') or 8000}"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", rid)


def launch(cfg, built):
    rid = _rid(cfg)
    port = int(cfg.get("port") or 8000)
    log = LOGDIR / f"{rid}-{int(time.time())}.log"
    model_id_val = cfg.get("model_id") or cfg.get("model") or built.get("model_id")
    entry = {"id": rid, "model": built["model_name"], "model_id": model_id_val,
             "engine": built["engine"],
             "cfg": dict(cfg), "cname": built.get("cname", ""), "endpoint": built["endpoint"],
             "port": port, "native": bool(built.get("native")),
             "cmd": built["cmd"], "artifact_mib": built.get("artifact_mib"),
             "log": str(log), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
             "status": "starting", "proc": None,
             "tokens_in": 0, "tokens_out": 0, "requests": 0,
             "sess_in": 0, "sess_out": 0, "sess_reqs": 0}
    # reserve the rid inside the lock: two concurrent launches of the same
    # model+engine+port must not both spawn an engine and orphan the loser
    with LOCK:
        previous = RUNNING.get(rid)
        if previous and previous.get("status") in ("running", "starting", "running (adopted)", "stopping"):
            raise ValueError("A server with this model, engine, and port is already tracked. Stop it before launching again.")
        RUNNING[rid] = entry
    try:
        if cfg.get("dry_run"):
            entry["status"] = "dry-run"
            with open(log, "ab") as lf:
                lf.write(("[dry-run] " + built["cmd"] + "\n").encode())
        else:
            # the OS is authoritative: try the real bind first
            free = False
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    probe.bind(("127.0.0.1", port))
                    free = True
                except OSError:
                    pass
            if not free:
                owner = None
                with LOCK:
                    for other, oe in RUNNING.items():
                        if other != rid and oe.get("port") == port and oe.get("status") in ("running", "starting", "running (adopted)"):
                            owner = other
                            break
                if owner:
                    raise ValueError(f"Port {port} is held by a running server ({owner}). Stop it first.")
                raise ValueError(f"Port {port} is already in use on this machine (something is bound to it). Pick another port.")
            # port is free — clear any stale tracked entries squatting on it
            with LOCK:
                for other, oe in RUNNING.items():
                    if other != rid and oe.get("port") == port and oe.get("status") not in ("stopped", "dry-run"):
                        oe["status"] = "stopped"
                        oe["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
            env = dict(os.environ)
            env.update(built.get("env") or {})
            with open(log, "ab") as lf:
                lf.write((built["cmd"] + "\n").encode())
                lf.flush()
                entry["proc"] = subprocess.Popen(built["tokens"], env=env, stdout=lf,
                                                 stderr=subprocess.STDOUT, cwd=str(HERE))
                entry["pid"] = entry["proc"].pid
            with LOCK:
                if entry["status"] != "starting":
                    # a stop request landed while the engine was spawning
                    raise ValueError("stopped during launch")
                entry["status"] = "running"
    except Exception:
        proc = entry.get("proc")
        try:
            if proc is not None:
                if proc.poll() is None:
                    proc.terminate()
                try:
                    proc.wait(timeout=15)  # let docker run -d register the container
                except Exception:
                    pass
            cname = entry.get("cname")
            if cname and not entry.get("native") and not cfg.get("dry_run"):
                subprocess.run(["docker", "rm", "-f", cname],
                               capture_output=True, timeout=15)
        except Exception:
            pass
        with LOCK:
            entry["status"] = "stopped"
            entry["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
        raise
    if not cfg.get("dry_run"):
        sync_harness_configs(port, built["model_name"], model_id_val, built.get("engine"))
    persist_state()
    return rid


def adopt_containers():
    """Re-adopt engines from a previous app session or running b70 containers."""
    if IS_WIN:
        return
    adopted = []
    try:
        snap = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    except Exception:
        snap = {}
    if isinstance(snap, dict) and shutil.which("docker"):
        for rid, e in snap.items():
            if not isinstance(e, dict):
                continue  # corrupt state file: never let adoption crash startup
            cname = e.get("cname")
            if not cname or e.get("native"):
                continue  # native engines die with the launcher; only containers re-adopt
            try:
                docker_cmd = ["docker"]
                _r = (find_model(e.get("model_id") or "") or {}).get("recipes", {}).get(
                    (e.get("cfg") or {}).get("engine") or e.get("engine") or "", {})
                if _r.get("docker_sock"):
                    docker_cmd += ["-H", f"unix://{_r['docker_sock']}"]
                r = subprocess.run(docker_cmd + ["inspect", "-f", "{{.State.Running}}", cname],
                                   capture_output=True, text=True, timeout=10)
                if r.returncode != 0 or r.stdout.strip() != "true":
                    continue
            except Exception:
                continue
            entry = dict(e)
            entry["proc"] = None
            entry["status"] = "running"
            entry["sess_in"] = entry.get("tokens_in") or 0
            entry["sess_out"] = entry.get("tokens_out") or 0
            entry["sess_reqs"] = entry.get("requests") or 0
            with LOCK:
                RUNNING[rid] = entry
            adopted.append(f"{e.get('model')}:{e.get('engine')}")
            sync_harness_configs(e.get("port", 8000), e.get("model", "Qwen3.8-27B"), e.get("model_id"), e.get("engine", "vllm"))

    # Also discover any existing active b70-* containers
    if shutil.which("docker"):
        try:
            r = subprocess.run(["docker", "ps", "--format", "{{.Names}}|{{.Image}}|{{.Ports}}"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    parts = line.strip().split("|")
                    if not parts or not parts[0]:
                        continue
                    cname = parts[0]
                    if cname.startswith("b70-"):
                        segs = cname.split("-")
                        eng = segs[-1] if len(segs) >= 3 else "vllm"
                        m_id = "-".join(segs[1:-1]) if len(segs) >= 3 else segs[1]
                        matched_m = next((m for m in RECIPES["models"] if m["id"] == m_id), None)
                        model_name = matched_m["name"] if matched_m else m_id
                        port = 8000
                        if ":8000->" in line or "->8000" in line:
                            port = 8000
                        rid = f"{m_id}-{eng}-{port}"
                        added = False
                        with LOCK:
                            if rid not in RUNNING:
                                RUNNING[rid] = {
                                    "id": rid,
                                    "model": model_name,
                                    "model_id": m_id,
                                    "engine": eng,
                                    "cname": cname,
                                    "endpoint": f"http://127.0.0.1:{port}/v1",
                                    "port": port,
                                    "native": False,
                                    "cmd": "docker",
                                    "log": str(LOGDIR / f"{cname}.log"),
                                    "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "status": "running",
                                    "proc": None,
                                    "tokens_in": 0, "tokens_out": 0, "requests": 0,
                                    "sess_in": 0, "sess_out": 0, "sess_reqs": 0
                                }
                                added = True
                        if added:
                            adopted.append(f"{model_name}:{eng}")
                            sync_harness_configs(port, model_name, m_id, eng)
        except Exception:
            pass

    # Native engines (llama.cpp) also outlive the app — Popen children are not
    # killed on exit. Re-adopt any persisted native entry whose port still serves
    # and whose owning pid can be found, so it stays visible/stoppable.
    if isinstance(snap, dict):
        for rid, e in snap.items():
            if not isinstance(e, dict):
                continue
            if not e.get("native") or e.get("status") in ("stopped", "dry-run"):
                continue
            port = int(e.get("port") or 0)
            if not port:
                continue
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
                urllib.request.urlopen(req, timeout=0.8).close()
            except Exception:
                continue  # nothing serving on that port — engine really is gone
            pid = e.get("pid") if _pid_alive(e.get("pid")) else None
            if pid is None:
                flat = re.sub(r"\\\s*\n\s*", " ", str(e.get("cmd") or ""))
                m_arg = re.search(r"-m\s+(\S+)", flat)
                hint = Path(m_arg.group(1)).stem if m_arg else str(
                    (e.get("cfg") or {}).get("model") or "")
                pid = _pid_for_native(port, hint)
            if pid is None:
                continue
            entry = dict(e)
            entry["proc"] = None
            entry["pid"] = pid
            entry["status"] = "running (adopted)"
            entry["sess_in"] = entry.get("tokens_in") or 0
            entry["sess_out"] = entry.get("tokens_out") or 0
            entry["sess_reqs"] = entry.get("requests") or 0
            with LOCK:
                RUNNING[rid] = entry
            adopted.append(f"{e.get('model')}:{e.get('engine')} (native)")
            sync_harness_configs(port, e.get("model", "model"),
                                 e.get("model_id") or (e.get("cfg") or {}).get("model"),
                                 e.get("engine") or (e.get("cfg") or {}).get("engine") or "")

    if adopted:
        print("re-adopted running engines: " + ", ".join(adopted))
        persist_state()


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (TypeError, ValueError, OSError):
        return False


def _pid_for_native(port, model_hint=""):
    """Find the pid owning a native engine on <port> by scanning /proc cmdlines."""
    want_port = str(port).encode()
    hint = model_hint.encode() if model_hint else b""
    for p in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            raw = Path(p).read_bytes()
        except OSError:
            continue
        if not raw or b"--port" not in raw:
            continue
        parts = raw.split(b"\0")
        port_ok = (b"--port=" + want_port) in parts
        if not port_ok:
            try:
                port_ok = parts[parts.index(b"--port") + 1] == want_port
            except (ValueError, IndexError):
                port_ok = False
        if not port_ok:
            continue
        if hint and hint.lower() not in raw.lower():
            continue
        if not hint and b"llama" not in parts[0].lower():
            continue
        try:
            return int(p.split("/")[2])
        except (IndexError, ValueError):
            continue
    return None


def docker_containers_status(cnames):
    """cname -> {'running': bool, 'exitcode': int|None}; None when docker is unreachable."""
    if not cnames or IS_WIN or not shutil.which("docker"):
        return None
    try:
        fmt = "{{.Name}}|{{.State.Running}}|{{.State.ExitCode}}"
        r = subprocess.run(["docker", "inspect", "--format", fmt, *cnames],
                           capture_output=True, text=True, timeout=15)
    except Exception:
        return None
    out = {}
    for line in r.stdout.splitlines():
        parts = line.split("|")
        if len(parts) == 3:
            out[parts[0].lstrip("/")] = {"running": parts[1].strip() == "true",
                                         "exitcode": parts[2].strip()}
    return out  # names absent from the map are not inspectable (never created / removed)


def servers_snapshot():
    with LOCK:
        tracked = [(rid, e) for rid, e in RUNNING.items()]
        for p in TRANSIENT_PROCS[:]:  # reap closed terminals/browsers (no zombies)
            if p.poll() is not None:
                TRANSIENT_PROCS.remove(p)
    docker_names = {e.get("cname") for _, e in tracked
                    if e.get("cname") and not e.get("native")}
    dstat = None
    if docker_names:
        if time.time() - CONTAINER_CACHE["ts"] > 5:
            CONTAINER_CACHE.update({"data": docker_containers_status(sorted(docker_names)),
                                    "ts": time.time()})
        dstat = CONTAINER_CACHE["data"]
    items = []
    exited_now = []
    probed = []   # (rid, port, items index) — entries that look alive, pending endpoint check
    with LOCK:
        for rid, e in tracked:
            proc = e.get("proc")
            status = e["status"]
            if e.get("native") or not e.get("cname"):
                # native engine process: its lifetime is the server's lifetime
                if proc is not None:
                    rc = proc.poll()
                    status = "running" if rc is None else f"exited ({rc})"
                elif e.get("pid") and not _pid_alive(e["pid"]):
                    # adopted native engine whose pid died while we weren't watching
                    status = "exited (adopted pid gone)"
            elif dstat is not None:
                if e.get("cname") in dstat:
                    st = dstat[e["cname"]]
                    status = "running" if st["running"] else f"exited ({st['exitcode']})"
                elif proc is not None and proc.poll() is not None:
                    # docker answered but the container is not inspectable: it was
                    # never created or already removed — the client rc says why
                    status = f"exited ({proc.poll()})"
                elif proc is None and status.startswith("running"):
                    # adopted container whose engine crashed: --rm removed it
                    status = "exited (removed)"
                elif status == "starting" and proc is not None:
                    status = "starting"
            if status == "stopped":
                continue
            if status == "running":
                # endpoint probe runs AFTER the lock below — a stalled engine
                # must not serialize /api/state, /api/stop, or the log viewer
                probed.append((rid, e.get("port"), len(items)))
            if status.startswith("exited") and not e.get("_report_once"):
                e["_report_once"] = True
                exited_now.append((e, status))
            items.append({k: v for k, v in e.items()
                          if k not in INTERNAL_KEYS and not k.startswith(("sess_", "_"))}
                         | {"status": status})
    # pass 2 — outside the lock: container up ≠ serving; if the OpenAI endpoint
    # doesn't answer yet, the engine is still loading weights / compiling
    # kernels. Ready engines get their token accounting scraped here too (this
    # poll is the only metrics driver — /api/metrics is a debug endpoint).
    now = time.time()
    for rid, port, idx in probed:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                ready = resp.status == 200
            exc = None if ready else f"HTTP {resp.status}"
        except Exception as ex:
            ready, exc = False, ex
        if not ready:
            items[idx]["status"] = "starting"
            with LOCK:
                e = RUNNING.get(rid)
                # log the first readiness failure per server so a wedged
                # engine isn't silently stuck on "starting" forever
                if e is not None and not e.get("_health_logged"):
                    e["_health_logged"] = True
                    print(f"{rid}: waiting for engine endpoint — {exc}")
        else:
            with LOCK:
                e = RUNNING.get(rid)
                due = e is not None and now - (e.get("_metrics_ts") or 0) >= 3
            if due:
                _account_engine(rid, e, now)
    for e, status in exited_now:  # file IO and state writes stay outside the lock
        record_usage(e, status)
    if exited_now:
        persist_state()
    return items


def stop_server(rid):
    with LOCK:
        e = RUNNING.get(rid)
        if not e:
            return False
        e["status"] = "stopping"
        cname = e.get("cname")
        native = bool(e.get("native"))
        proc = e.get("proc")
    # docker/subprocess work happens outside the lock so a hung daemon
    # cannot freeze /api/state for everyone
    if cname and not native:
        docker_cmd = ["docker"]
        eng = (e.get("cfg") or {}).get("engine") or e.get("engine")
        r = (find_model(e.get("model_id") or "") or {}).get("recipes", {}).get(eng or "", {})
        if r.get("docker_sock"):
            docker_cmd += ["-H", f"unix://{r['docker_sock']}"]
        rm = subprocess.run(docker_cmd + ["rm", "-f", cname], capture_output=True, timeout=15)
        if rm.returncode != 0 and r.get("docker_sock"):
            # container may have been launched on the default socket — try that too
            subprocess.run(["docker", "rm", "-f", cname], capture_output=True, timeout=15)
        # wait until the port is actually released (docker-proxy teardown is async)
        port = int(e.get("port") or 0)
        if port:
            for _ in range(30):
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_:
                        s_.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        s_.bind(("127.0.0.1", port))
                    break
                except OSError:
                    time.sleep(0.5)
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            pass
    elif e.get("native") and _pid_alive(e.get("pid")):
        # adopted native engine: no Popen handle, kill by pid
        try:
            os.kill(int(e["pid"]), signal.SIGTERM)
            for _ in range(10):
                if not _pid_alive(e["pid"]):
                    break
                time.sleep(0.5)
            if _pid_alive(e["pid"]):
                os.kill(int(e["pid"]), signal.SIGKILL)
        except OSError:
            pass
    with LOCK:
        e["status"] = "stopped"
        e["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
    record_usage(e, "stopped by user")
    persist_state()
    return True


def open_harness(cfg, built):
    line = harness_line(cfg, built)
    if IS_WIN:
        TRANSIENT_PROCS.append(subprocess.Popen(SETTINGS["terminal_windows"] + [line]))
    else:
        h_id = cfg.get("harness", "")
        if h_id == "webui" or line.startswith("xdg-open"):
            target_url = "http://localhost:3000"
            for p in (3000, 8080):
                try:
                    with socket.socket() as s:
                        s.settimeout(0.2)
                        if s.connect_ex(("127.0.0.1", p)) != 0:
                            continue
                        target_url = f"http://localhost:{p}"
                        break
                except Exception:
                    pass
            TRANSIENT_PROCS.append(subprocess.Popen(["xdg-open", target_url]))
            return f"xdg-open {target_url}"

        inner = line + "; exec sh"
        for cand in ("kitty", "gnome-terminal", "xfce4-terminal", "x-terminal-emulator", "xterm"):
            resolved = shutil.which(cand)
            if not resolved:
                continue
            real_name = Path(resolved).resolve().name
            if "kitty" in real_name or cand == "kitty":
                term = [resolved, "sh", "-c", inner]
            elif "gnome-terminal" in real_name or cand == "gnome-terminal":
                term = [resolved, "--", "sh", "-c", inner]
            elif "xfce4-terminal" in real_name or cand == "xfce4-terminal":
                # -e takes one command string; quote the payload, don't nest '
                term = [resolved, "-e", f"sh -c {shlex.quote(inner)}"]
            else:
                # xterm -e consumes the rest of argv as program + args
                term = [resolved, "-e", "sh", "-c", inner]
            TRANSIENT_PROCS.append(subprocess.Popen(term))
            return line
    return line


# ---------------------------------------------------------------- shutdown

def graceful_shutdown(stop_engines=False):
    with LOCK:
        if SHUTDOWN["started"]:
            return
        SHUTDOWN["started"] = True
    print("shutting down — storing usage…")
    with LOCK:
        entries = list(RUNNING.items())
        for e in DOWNLOADS.values():
            e["cancel"] = True
    for rid, e in entries:
        status = e.get("status", "")
        if status == "stopped":
            continue
        record_usage(e, "stored on exit — engine still running")
        if stop_engines:
            if not e.get("native"):
                cname = e.get("cname")
                if cname:
                    subprocess.run(["docker", "rm", "-f", cname], capture_output=True, timeout=15)
            proc = e.get("proc")
            if proc is not None and proc.poll() is None:
                proc.terminate()
    persist_state()
    if SERVER is not None:
        try:
            SERVER.shutdown()
        except Exception:
            pass


def _sig_handler(signum, frame):
    threading.Thread(target=graceful_shutdown, args=(False,), daemon=True).start()


# ---------------------------------------------------------------- http

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive: the UI polls on a timer

    def log_message(self, format, *args):
        pass

    def _send(self, body, code=200, ctype="application/json"):
        self.send_response(code)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass  # client vanished mid-write; nothing useful left to do

    def _json(self, obj, code=200):
        self._send(json.dumps(obj).encode(), code)

    def _read(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 <= n <= 65536:
            raise ValueError("request body too large")
        obj = json.loads(self.rfile.read(n) or b"{}")
        if not isinstance(obj, dict):
            raise ValueError("expected a JSON object")
        return obj

    def _host_ok(self):
        host = self.headers.get("Host", "")
        return host in ("127.0.0.1", f"127.0.0.1:{self.server.server_port}")

    def _auth(self):
        """GET auth: X-Launcher-Token header, session cookie, or the one-time
        ?token= bootstrap the window/browser is opened with. Loopback is shared
        by every local process, so the token is never served without auth."""
        tok = self.headers.get("X-Launcher-Token")
        if tok and hmac.compare_digest(tok, API_TOKEN):
            return "header"
        m = re.search(r"(?:^|;\s*)b70_token=([^;]+)", self.headers.get("Cookie", ""))
        if m and hmac.compare_digest(m.group(1), API_TOKEN):
            return "cookie"
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        tok = (q.get("token") or [""])[0]
        if tok and hmac.compare_digest(tok, API_TOKEN):
            return "bootstrap"
        return None

    def do_GET(self):
        if not self._host_ok():
            self._json({"error": "invalid Host"}, 403)
            return
        auth = self._auth()
        if not auth:
            self._json({"error": "unauthorized — open the UI via b70-launcher"}, 403)
            return
        parsed_path = urllib.parse.urlparse(self.path).path
        if auth == "bootstrap":
            # one-time hand-off: stash the token in a session cookie, then strip
            # it from the URL so it never lands in history or Referer headers
            self.send_response(302)
            self.send_header("Location", "/")
            self.send_header("Set-Cookie",
                             f"b70_token={API_TOKEN}; Path=/; SameSite=Strict; HttpOnly")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if parsed_path in ("/", "/index.html"):
            body = (HERE / "web" / "index.html").read_bytes().replace(b"__API_TOKEN__", API_TOKEN.encode())
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self' http://localhost:* http://127.0.0.1:* ws://127.0.0.1:* ws://localhost:*; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed_path == "/api/state":
            running = servers_snapshot()     # slow probes stay outside the lock
            preflight = hardware_preflight()
            with LOCK:  # serialize under the lock: recipe overlays mutate RECIPES
                body = json.dumps({"recipes": RECIPES, "settings": SETTINGS,
                                   "running": running, "is_win": IS_WIN,
                                   "version": VERSION, "update": UPDATE_INFO,
                                   "recipe_notices": recipe_notices(),
                                   "recipes_remote": {"catalog_ver": RECIPE_REMOTE.get("catalog_ver"),
                                                      "checked": RECIPE_REMOTE.get("checked")},
                                   "preflight": preflight,
                                   "scan": {"roots": SCAN.get("roots", []),
                                            "state": SCAN.get("state"), "ts": SCAN.get("ts")}})
            self._send(body.encode())
        elif parsed_path == "/api/scan":
            sc = scan_blocking()
            with LOCK:  # detect() reads SCAN+RECIPES; overlays may mutate either
                out = {}
                for m in RECIPES["models"]:
                    out[m["id"]] = {}
                    for eng in m["recipes"]:
                        det = detect(m, eng)
                        out[m["id"]][eng] = {
                            "detected": bool(det),
                            "path": (det or {}).get("path"),
                            "ctx_native": (det or {}).get("ctx"),
                            "size_mib": (det or {}).get("size_mib"),
                        }
            free_gb = None
            try:
                # probe the filesystem the download target actually lives on
                probe = Path(SETTINGS.get("models_dir") or "~").expanduser()
                while not probe.is_dir() and probe != probe.parent:
                    probe = probe.parent
                free_gb = round(shutil.disk_usage(probe).free / 1073741824, 1)
            except OSError:
                pass
            self._json({"state": sc.get("state"), "error": sc.get("error"),
                        "roots": sc.get("roots", []),
                        "catalog": sc.get("catalog", []),
                        "free_gb": free_gb,
                        "matches": out})
        elif parsed_path == "/api/power":
            # light poll path for the telemetry pill — no docker stats, no
            # engine scrapes; vram_mm is cached inside power_probe
            self._json({"power": power_probe() if not IS_WIN else []})
        elif parsed_path == "/api/servers":
            # light poll path for server state + scan-completion signal
            with LOCK:
                scan = {"state": SCAN.get("state"), "ts": SCAN.get("ts")}
            self._json({"running": servers_snapshot(), "scan": scan})
        elif parsed_path == "/api/metrics":
            self._json(metrics_snapshot())
        elif parsed_path == "/api/usage":
            self._json(usage_summary())
        elif self.path.startswith("/api/logs"):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            with LOCK:
                e = RUNNING.get((q.get("id") or [""])[0])
            if not e:
                self._json({"error": "no such server"}, 404)
                return
            try:
                with open(e["log"], "rb") as lf:
                    lf.seek(0, 2)
                    size = lf.tell()
                    lf.seek(max(0, size - 16384))
                    tail = lf.read().decode("utf-8", "replace")
                lines = tail.splitlines()[-60:]
            except Exception:
                lines = []
            self._json({"id": e["id"], "lines": lines})
        elif parsed_path == "/api/downloads":
            with LOCK:
                self._json({did: {k: v for k, v in e.items() if k != "cancel"}
                            for did, e in DOWNLOADS.items()})
        elif self.path.startswith("/assets/"):
            root = (HERE / "web").resolve()
            f = (root / self.path.lstrip("/")).resolve()
            if not f.is_relative_to(root) or not f.is_file():
                self._json({"error": "not found"}, 404)
                return
            ctype = {"svg": "image/svg+xml", "png": "image/png",
                     "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(f.suffix.lstrip("."), "application/octet-stream")
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "max-age=300")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if (not self._host_ok()
                or not hmac.compare_digest(self.headers.get("X-Launcher-Token") or "", API_TOKEN)
                or self.headers.get("Origin") not in (None, f"http://127.0.0.1:{self.server.server_port}")
                or self.headers.get("Content-Type") != "application/json"):
            self._json({"error": "unauthorized request"}, 403)
            return
        try:
            cfg = self._read()
        except Exception:  # bad Content-Length, oversized, malformed JSON
            self._json({"error": "invalid JSON body"}, 400)
            return
        try:
            self._route_post(cfg)
        except Exception as exc:  # a bug in a handler must not drop the connection
            self._json({"error": f"launcher error: {exc}"}, 500)

    def _route_post(self, cfg):
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/build":
            built = build(cfg)
            if "error" not in built:
                # preview only: build the command line without syncing harness configs
                built["harness_line"] = harness_line(cfg, built, write_sync=False)
                m = find_model(cfg.get("model_id", ""))
                if m:
                    built["download"] = m["recipes"].get(cfg.get("engine", ""), {}).get("download", {})
                notice = recipe_notice_for(cfg.get("model_id") or cfg.get("model") or "",
                                           cfg.get("engine") or "")
                if notice:
                    built["recipe_notice"] = notice
            self._json(built)
        elif path == "/api/launch":
            rid0 = _rid(cfg)
            with LOCK:
                prev = RUNNING.get(rid0)
            if prev and prev.get("status") in ("running", "starting", "running (adopted)") and not cfg.get("dry_run"):
                line = open_harness(cfg, {"endpoint": prev.get("endpoint"),
                                          "engine": prev.get("engine"),
                                          "model_name": prev.get("model")})
                self._json({"id": rid0, "already_running": True, "harness_line": line})
                return
            built = build(cfg)
            if "error" in built:
                self._json(built, 400)
                return
            if not cfg.get("dry_run"):
                check = hardware_preflight()
                if check["blockers"] or not built["detected"]:
                    self._json({"error": "; ".join(check["blockers"] or ["Model artifact not detected; download or configure scan roots first."])}, 400)
                    return
                selected = cfg.get("gpus", [0])
                if any(i >= len(check["devices"]) or not check["devices"][i]["b70"] or not check["devices"][i]["accessible"] for i in selected):
                    self._json({"error": "Selected render nodes are not identified accessible B70 GPUs; check permissions and device order."}, 400)
                    return
            try:
                rid = launch(cfg, built)
            except (OSError, ValueError) as exc:
                self._json({"error": str(exc)}, 400)
                return
            out = {"id": rid, "harness_line": harness_line(cfg, built)}
            if cfg.get("dry_run"):
                out.update({k: built.get(k) for k in
                            ("cmd", "warnings", "env", "power_cmd", "detected",
                             "detected_path", "ctx")})
            notice = recipe_notice_for(cfg.get("model_id") or cfg.get("model") or "",
                                       cfg.get("engine") or "")
            if notice:
                out["recipe_notice"] = notice
            self._json(out)
        elif path == "/api/download":
            m = find_model(cfg.get("model_id", ""))
            if not m or cfg.get("engine") not in m["recipes"]:
                self._json({"error": "no such model+engine"}, 400)
                return
            did, err = start_download(m, cfg["engine"])
            self._json({"id": did, "error": err})
        elif path == "/api/stop":
            self._json({"ok": stop_server(cfg.get("id", ""))})
        elif path == "/api/test_prompt":
            prompt = cfg.get("prompt", "Why is dual Intel Arc Pro B70 effective for local MoE inference?")
            try:
                port = int(cfg.get("port", 8000))
                if not (1 <= port <= 65535):
                    raise ValueError
            except (TypeError, ValueError):
                self._json({"error": "invalid port"}, 400)
                return
            req_model = cfg.get("model", "")
            active_model = req_model
            try:
                m_req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
                with urllib.request.urlopen(m_req, timeout=2.0) as m_resp:
                    m_data = json.loads(m_resp.read(1 << 20).decode())
                    if m_data.get("data") and len(m_data["data"]) > 0:
                        active_model = m_data["data"][0]["id"]
            except Exception:
                pass

            t0 = time.time()
            post_data = json.dumps({
                "model": active_model or "default",
                "messages": [{"role": "user", "content": prompt}],
                # reasoning models can spend hundreds of tokens in <think>
                # before emitting content — 120 returns empty text for them
                "max_tokens": 512
            }).encode()

            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=post_data,
                headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=120.0) as resp:
                    data = json.loads(resp.read(8 << 20).decode())
                    dt = round(time.time() - t0, 2)
                    reply = ""
                    tokens = 0
                    if data.get("choices") and len(data["choices"]) > 0:
                        choice = data["choices"][0]
                        msg = choice.get("message", {})
                        reply = msg.get("content", "")
                        if not reply and msg.get("reasoning_content"):
                            # thinking model still reasoning at the token cap
                            reply = "Thinking: " + msg["reasoning_content"]
                            if choice.get("finish_reason") == "length":
                                reply += "\n\n(still thinking — final answer needs more than 512 tokens)"
                    if data.get("usage"):
                        tokens = data["usage"].get("completion_tokens", 0)
                    self._json({
                        "ok": True,
                        "reply": reply or "No text returned by engine",
                        "latency_s": dt,
                        "tokens": tokens,
                        "model": active_model
                    })
            except urllib.error.URLError as e:
                self._json({
                    "ok": False,
                    "error": f"No inference server responding on port {port}. Click 'Launch Model' to start it. ({e})"
                })
            except Exception as e:
                self._json({
                    "ok": False,
                    "error": f"Request failed: {e}"
                })
        elif path == "/api/harness":
            port = cfg.get("port", 8000)
            built = {"endpoint": f"http://127.0.0.1:{port}/v1"}
            try:
                b = build(cfg)
                if "error" not in b:
                    built = b
            except Exception:
                pass
            line = open_harness(cfg, built)
            self._json({"line": line})
        elif path == "/api/recipes/update":
            res = apply_recipe_update(cfg.get("model_id"), cfg.get("engine"))
            self._json(res, 200 if res.get("ok") else 400)
        elif path == "/api/settings":
            roots = cfg.get("scan_dirs")
            if not isinstance(roots, list) or len(roots) > 8:
                self._json({"error": "scan_dirs must be a list of at most 8 directories"}, 400)
                return
            cleaned = []
            for r in roots:
                if not isinstance(r, str) or len(r) > 512:
                    self._json({"error": "invalid scan root"}, 400)
                    return
                p = Path(r).expanduser().resolve()
                if p in (Path("/"), Path.home().resolve()):
                    self._json({"error": "refusing to scan an entire filesystem or home directory"}, 400)
                    return
                if not p.is_dir():
                    self._json({"error": f"not a directory: {r}"}, 400)
                    return
                if p not in cleaned:
                    cleaned.append(p)
            SETTINGS["scan_dirs"] = [str(p) for p in cleaned]
            save_override(("scan_dirs",))
            start_scan_async(force=True)
            self._json({"ok": True, "scan_dirs": SETTINGS["scan_dirs"]})
        elif path == "/api/shutdown":
            stop_engines = bool(cfg.get("stop_engines"))
            self._json({"ok": True})
            threading.Thread(target=graceful_shutdown, args=(stop_engines,), daemon=True).start()
        else:
            self._json({"error": "not found"}, 404)


# ---------------------------------------------------------------- window

def _open_browser(url):
    try:
        for browser in ("chromium", "chromium-browser", "google-chrome", "microsoft-edge"):
            if shutil.which(browser):
                args = [browser, f"--app={url}", "--window-size=1440,900"]
                if os.environ.get("WAYLAND_DISPLAY"):
                    args.append("--ozone-platform=wayland")  # native on GNOME Wayland; "auto" rejected by some builds
                    args.append("--disable-features=Vulkan")  # snap Chromium: Vulkan incompatible with wayland ozone
                TRANSIENT_PROCS.append(subprocess.Popen(args))
                return
        webbrowser.open(url)
    except OSError as exc:
        # no usable browser — keep the server up and say where the UI lives
        print(f"could not open a browser ({exc}) — the UI is at {url}")


def open_window(url, port):
    """Native app window (WebKitGTK child) when possible, else a browser app window.

    Returns the child process handle, or None when a browser was used."""
    if IS_WIN:
        for browser in ("msedge", "chrome"):
            try:
                TRANSIENT_PROCS.append(
                    subprocess.Popen(["cmd", "/c", "start", "", browser, f"--app={url}"]))
                return None
            except OSError:
                continue
        webbrowser.open(url)
        return None
    mode = os.environ.get("B70_LAUNCHER_WINDOW", "auto")
    if mode == "browser":
        _open_browser(url)
        return None
    child = appwindow.open_app_window(HERE / "webwindow.py", url, port, WINDOW_TITLE)
    if child is not None:
        return child
    _open_browser(url)
    return None


def main():
    global WIN_CHILD, SERVER
    ap = argparse.ArgumentParser(description="B70 model launcher")
    ap.add_argument("--port", type=int, default=SETTINGS.get("port", 7570))
    ap.add_argument("--no-open", action="store_true", help="don't open a window")
    ap.add_argument("--browser", action="store_true", help="open in a browser instead of the app window")
    ap.add_argument("--kill", "--force", action="store_true", help="kill any existing launcher instance on this port before starting")
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    # keep the log dir bounded: newest 20 files
    try:
        for old in sorted(LOGDIR.glob("*.log"), key=lambda p: p.stat().st_mtime)[:-20]:
            old.unlink(missing_ok=True)
    except OSError:
        pass

    if args.kill:
        try:
            subprocess.run(["pkill", "-9", "-f", "webwindow.py"], capture_output=True, timeout=2)
            subprocess.run(["fuser", "-k", f"{args.port}/tcp"], capture_output=True, timeout=2)
            time.sleep(0.4)
        except Exception:
            pass

    try:
        srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        print(f"Port {args.port} is busy ({exc}). Freeing port and starting fresh launcher...")
        try:
            subprocess.run(["pkill", "-9", "-f", "webwindow.py"], capture_output=True, timeout=2)
            subprocess.run(["fuser", "-k", f"{args.port}/tcp"], capture_output=True, timeout=3)
            time.sleep(0.5)
            srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
            print(f"Successfully bound port {args.port}.")
        except Exception as retry_exc:
            raise SystemExit(f"Cannot bind {url}: {retry_exc}. Close the existing instance or choose another port.") from retry_exc
    SERVER = srv
    # adoption does docker inspect + engine probes; keep it off the startup path
    # so the window/API come up fast even with a stale state file or dead dockerd
    threading.Thread(target=adopt_containers, daemon=True).start()
    start_scan_async(force=not SCAN.get("ts"))
    print(f"b70-launcher {VERSION} on {url}  (models: {len(RECIPES['models'])})")
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
    if not args.no_open:
        # one-time token bootstrap — the server swaps it for a session cookie
        # and redirects to /, so the token never sits in browser history
        open_url = f"{url}/?token={API_TOKEN}"
        if args.browser:
            _open_browser(open_url)
        else:
            WIN_CHILD = open_window(open_url, args.port)
            if WIN_CHILD is None:
                print("app window unavailable; opened the browser instead")
    if WIN_CHILD is not None:
        def watch_child():
            while not STOP_EVT.wait(1.0):
                if WIN_CHILD.poll() is not None:  # window closed -> quit the app
                    threading.Thread(target=graceful_shutdown, daemon=True).start()
                    return
        threading.Thread(target=watch_child, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        graceful_shutdown(False)
    STOP_EVT.set()
    if WIN_CHILD is not None and WIN_CHILD.poll() is None:
        WIN_CHILD.terminate()
        try:
            WIN_CHILD.wait(timeout=3)
        except Exception:
            WIN_CHILD.kill()
    print("bye")


if __name__ == "__main__":
    main()
