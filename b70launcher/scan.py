"""Disk scan + artifact detection.

SCAN holds the last scan result; detect() maps a recipe onto it. The walk
records each snapshot dir's quantization method (quantization_config.json /
config.json / quantize_config.json) so quant-pinned recipes never match an
artifact the engine cannot load (audit M2).
"""
import os
import re
import time
from pathlib import Path

from . import catalog
from .locks import LOCK

SKIP_DIRS = {"node_modules", ".git", ".svn", ".hg", "__pycache__", ".venv", "venv",
             "lost+found", "$RECYCLE.BIN", "System Volume Information",
             "site-packages", "dist-packages", ".cache", ".npm", ".cargo", "vendor"}

SCAN = {"state": "idle", "items": {"gguf": {}, "snapshots": {}}, "catalog": [], "roots": [], "ts": 0, "error": None}


def scan_roots():
    roots = []
    for r in catalog.SETTINGS.get("scan_dirs", []):
        p = Path(r).expanduser().resolve()
        if p in (Path("/"), Path.home().resolve()):
            continue  # refuse filesystem-wide scans
        if p.is_dir() and p not in roots:
            roots.append(p)
    # download destinations are model locations by definition: always scanned,
    # so a finished download is detectable without extra configuration
    # (recipe-audit #1 — the download→detect loop)
    for key in ("models_dir", "ovms_repo"):
        p = Path(str(catalog.SETTINGS.get(key) or "")).expanduser()
        if p and p != Path("/") and p.is_dir():
            p = p.resolve()
            if p not in roots and p != Path.home().resolve():
                roots.append(p)
    return roots

def read_ctx_from_config(cfgpath):
    try:
        import json
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


def read_quant_method(model_dir):
    """Quantization method recorded in an HF snapshot dir, for quant-pinned
    recipe matching (audit M2). None when the artifact does not say."""
    import json
    d = Path(model_dir)
    try:
        qc = json.loads((d / "quantization_config.json").read_text())
        m = qc.get("quant_method")
        if isinstance(m, str) and m:
            return m.lower()
    except Exception:
        pass
    try:
        cfg = json.loads((d / "config.json").read_text())
        m = (cfg.get("quantization_config") or {}).get("quant_method")
        if isinstance(m, str) and m:
            return m.lower()
    except Exception:
        pass
    # the legacy GPTQ sidecar exists only in GPTQ exports
    if (d / "quantize_config.json").is_file():
        return "gptq"
    return None


def read_exl3_bits(model_dir):
    """Set of trellis bit widths (K) in an EXL3 artifact, from
    quantization_config.json tensor_storage (trellis last dim = 16*K).
    Empty when the file is missing or unreadable."""
    import json
    try:
        qc = json.loads((Path(model_dir) / "quantization_config.json").read_text())
        ks = set()
        for entry in (qc.get("tensor_storage") or {}).values():
            if not isinstance(entry, dict) or entry.get("quant_format") != "exl3":
                continue
            for name, meta in (entry.get("stored_tensors") or {}).items():
                if name.endswith(".trellis"):
                    ks.add(int(meta["shape"][-1]) // 16)
        return ks
    except Exception:
        return set()


def _walk_root(root, items, deadline):
    """Depth-limited iterative walk; returns False when the time budget ran out.

    Classifies every directory: GGUF files are collected individually; model
    directories are tagged openvino (IR xml+bin pairs or the
    openvino_language_model.* prefix) or vllm (HF layout: config.json and/or
    *.safetensors); vllm dirs whose quantization_config says exl3 are tagged
    exl3. Model dirs are never descended into."""
    import json
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
                "quant_method": read_quant_method(dirpath),
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
    catalog_ = {}
    for name, path in items.get("gguf", {}).items():
        try:
            size = round(Path(path).stat().st_size / 1048576, 1)
        except OSError:
            size = None
        catalog_[path] = {"kind": "gguf", "name": name, "path": path, "size_mib": size, "ctx": None}
    for name, entry in items.get("snapshots", {}).items():
        if entry["path"] in catalog_:
            continue
        catalog_[entry["path"]] = {"kind": entry.get("kind", "vllm"), "name": name,
                                   "path": entry["path"],
                                   "size_mib": _dir_size_mib(entry["path"]),
                                   "ctx": entry.get("ctx")}
    return sorted(catalog_.values(), key=lambda c: (c["kind"], c["name"]))


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
    import threading
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


# ---------------------------------------------------------------- matching

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


def _detect_impl(model, engine):
    """Match a recipe to an on-disk artifact: exact identity first, else same
    model family filtered by format/quant. Returns (detection, reason)."""
    if not model:
        return None, "no such model in the catalog"

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
    # identities that count as an EXACT match even without download.name:
    # the recipe's own destination dir / search string / repo basename
    # (audit M1 — the spurious "variant" warning)
    ident = {_norm(x) for x in
             (_as_list(dl.get("dest")) + _as_list(dl.get("search")) + _as_list(dl.get("name"))
              + _as_list(r.get("search_name"))
              + [str(dl.get("repo") or "").split("/")[-1]]
              + _as_list(r.get("source_model") if not str(r.get("source_model") or "").startswith("http") else "")) if x}
    want_quant = dl.get("quant_method")  # quant-pinned recipe (audit M2)

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

    # direct path check from recipe (for custom/pinned local models);
    # relative paths are never resolved against the cwd (audit L5)
    for direct_k in ("gguf", "model_path", "source_model"):
        val = r.get(direct_k)
        if val and not str(val).startswith("http") \
                and (str(val).startswith(("~", "/")) or Path(str(val)).is_absolute()):
            dp = Path(val).expanduser()
            if dp.exists():
                if dp.is_dir():
                    try:
                        files = [p.name.lower() for p in dp.iterdir() if p.is_file()]
                    except OSError:
                        continue
                    if not any(n.endswith((".safetensors", ".bin")) for n in files):
                        continue
                    actual_quant = read_quant_method(dp)
                    if want_quant and actual_quant and actual_quant != str(want_quant).lower():
                        return None, (f"incompatible: recipe path {dp} holds {actual_quant} weights, "
                                      f"this recipe pins {str(want_quant).lower()}")
                cfg_f = dp / "config.json" if dp.is_dir() else None
                c_val = read_ctx_from_config(cfg_f) if cfg_f and cfg_f.is_file() else r.get("ctx")
                akind = "openvino" if engine == "openvino" else ("exl3" if engine == "exl3" else "vllm")
                return wrap(dp.name, str(dp) if dp.is_file() else {"path": str(dp), "ctx": c_val, "kind": akind}, True), ""

    if want and want in pool:
        return wrap(want, pool[want], True), ""
    prefer = [_norm(p) for p in _as_list(dl.get("prefer"))]
    exclude = [_norm(x) for x in _as_list(dl.get("exclude")) if _norm(x)]
    cands = []
    near = []  # family artifacts that exist on disk but cannot serve this recipe
    for name, hit in pool.items():
        nn = _norm(name)
        family = bool(keys and any(k in nn for k in keys))
        if dl.get("kind") != "file" and isinstance(hit, dict):
            eng_kind = {"openvino": "openvino", "exl3": "exl3"}.get(engine, "vllm")
            if hit.get("kind") and hit.get("kind") != eng_kind:
                if family:
                    near.append(f"\"{name}\" is a {hit['kind']} artifact, this engine serves {eng_kind}")
                continue
            # quant-pinned recipes never match an artifact whose recorded
            # quantization method differs (audit M2). An artifact that does
            # not record one falls through to the prefer/exclude filters.
            if want_quant and hit.get("quant_method") \
                    and hit["quant_method"] != str(want_quant).lower():
                if family:
                    near.append(f"\"{name}\" is {hit['quant_method']}, this recipe pins {str(want_quant).lower()}")
                continue
        if name.startswith("mmproj"):
            continue  # vision sidecar is never the model itself
        # speculative-decoding draft sidecars (DFlash/MTP) are never the
        # serving artifact — a 1 GB draft must not satisfy a 20 GB model
        want_norm = _norm(want or _first_nonempty(*_as_list(dl.get("search"))) or "")
        if ("dflash" in nn or "draft" in nn) and "dflash" not in want_norm and "draft" not in want_norm:
            if family:
                near.append(f"\"{name}\" is a speculative-decoding draft sidecar, not the serving artifact")
            continue
        if prefer and not any(p in nn for p in prefer):
            if family:
                near.append(f"\"{name}\" is not one of the artifact formats this recipe serves")
            continue  # right model, wrong artifact format for this engine
        if any(x in nn for x in exclude):
            if family:
                near.append(f"\"{name}\" is a sibling artifact another recipe owns (recipe excludes it)")
            continue
        if family:
            score = sum(1 for p in prefer if p and p in nn) - len(nn) / 1000.0
            cands.append((score, name, hit))
    if cands:
        cands.sort(key=lambda t: (-t[0], t[1]))
        _, name, hit = cands[0]
        exact = (name == want) if want else (_norm(name) in ident)
        return wrap(name, hit, exact), ""
    if near:
        return None, "incompatible: " + "; ".join(dict.fromkeys(near))
    return None, "not downloaded"


def detect(model, engine):
    """Match a recipe to an on-disk artifact (None when nothing matches)."""
    return _detect_impl(model, engine)[0]


def detect_status(model, engine):
    """(detection, reason) for the library's local/download/incompatible state.
    reason is "" when detected, "incompatible: …" when a family artifact
    exists on disk but no variant satisfies the recipe, "not downloaded"
    when nothing matching is on disk."""
    return _detect_impl(model, engine)


def resolve_ctx(model, engine, det):
    """Default context: the recipe's tested value, never above the model's own
    limit (config.json on disk) or the engine ceiling (ctx_max)."""
    r = model["recipes"].get(engine, {})
    cap = r.get("ctx_max")
    disk = (det or {}).get("ctx")
    if r.get("ctx"):
        v, source = r["ctx"], r.get("ctx_note", "cookbook recipe")
        if r.get("kind") != "ovms" and disk and disk < v:
            # OVMS manages KV internally, its IR config is not a limit
            v, source = disk, "config.json on disk"
    elif disk:
        v, source = disk, "config.json on disk"
    else:
        v, source = 32768, "launcher default"
    if cap:
        v = min(v, cap)
    return {"value": v, "source": source}


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


def _find_draft_dir(recipe):
    """Locate a speculative-draft model dir for container mounts.

    recipe["draft"] may hold an absolute "path" or a "search" fragment matched
    (case-insensitive, depth-limited) against configured scan roots. The dir
    must look like a model dir — config.json plus weights — or it is skipped."""
    spec = recipe.get("draft") or {}
    raw = str(spec.get("path") or "").strip()
    if raw:
        p = Path(raw).expanduser()
        return str(p) if p.is_dir() else None
    want = str(spec.get("search") or "").lower()
    if not want:
        return None

    def looks_like_model(d):
        try:
            names = {f.name for f in d.iterdir()}
        except OSError:
            return False
        return "config.json" in names and any(
            n.endswith((".safetensors", ".bin", ".gguf")) for n in names)

    for root in scan_roots():
        stack = [(root, 0)]
        while stack:
            cur, depth = stack.pop()
            if depth > 4:
                continue
            try:
                if cur.is_dir() and want in cur.name.lower() and looks_like_model(cur):
                    return str(cur)
                for child in cur.iterdir():
                    if child.is_dir() and not child.name.startswith("."):
                        stack.append((child, depth + 1))
            except OSError:
                continue
    return None
