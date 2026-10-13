"""Hugging Face downloads: plan → confirm → transfer → cancel.

A download never starts without a confirmed plan, and the plan is computed on
the DESTINATION filesystem: size_bytes from the HF API listing, free_bytes
from the dir the files will land in. An unknown remote size is never treated
as space-safe (a listing that fails to report sizes rejects the confirm).

Recipes may declare a draft sidecar (``draft.download``) — it is fetched into
the same models_dir in the same transfer (recipe-audit #8).
"""
import json
import re
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

from . import catalog, scan
from .locks import LOCK
from .paths import VERSION

HF = "https://huggingface.co"
UA = {"User-Agent": f"b70-launcher/{VERSION}"}

DOWNLOADS = {}  # id -> download entry

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
    "SergiioB/Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16",
    "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF",
    "bottlecapai/ThinkingCap-Qwen3.6-27B-GGUF",
    "meta-models/Muse-Glimmer-30B-GGUF",
    "unsloth/Muse-Glimmer-30B-GGUF",
    "ornith-ai/Ornith-1.0-35B-GGUF",
    "dzannotti/Qwen3.8-Flash-Next-MTP-GGUF",
}

SPACE_HEADROOM = 1.05  # require free >= size * 1.05


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


def fmt_bytes(n):
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _fetch_groups(model, engine):
    """Every (dl, dest_dir, files) group a recipe download consists of: the
    main artifact plus any draft sidecar declared as draft.download."""
    groups = []
    dl = model["recipes"][engine].get("download") or {}
    if dl.get("repo"):
        groups.append((dl, dest_dir_for(dl, engine), "main"))
    draft_dl = ((model["recipes"][engine].get("draft") or {}).get("download") or {})
    if draft_dl.get("repo"):
        groups.append((draft_dl, dest_dir_for(draft_dl, "vllm"), "draft"))
    return groups


def dest_dir_for(dl, engine):
    """Directory the recipe's files download into (before any network call)."""
    if dl.get("kind") == "file":
        return Path(catalog.SETTINGS.get("models_dir", "~/models")).expanduser()
    root = Path(catalog.SETTINGS.get("ovms_repo", "~/models/ovms-repo")).expanduser()
    if engine != "openvino":
        root = Path(catalog.SETTINGS.get("models_dir", "~/models")).expanduser()
    dest_name = dl.get("dest") or (dl.get("repo") or "").split("/")[-1]
    if "/" in dest_name or not safe_artifact(str(dest_name)):
        raise ValueError("Recipe download dest must be a single directory name")
    return root / dest_name


def _probe(path):
    """Nearest existing ancestor of `path` (the dest dir may not exist yet)."""
    probe = Path(path).expanduser()
    while not probe.is_dir() and probe != probe.parent:
        probe = probe.parent
    return probe


def _fs_free(path):
    """Free bytes on the filesystem that will hold `path`."""
    try:
        return shutil.disk_usage(_probe(path)).free
    except OSError:
        return None


def _space_check(needs):
    """Check every destination filesystem against the bytes it will receive.

    needs: [(dest_dir, size_bytes)]. Returns (free bytes on the first
    destination, error or None). Sidecars on another disk count there."""
    per_fs = {}
    for dest, size in needs:
        free = _fs_free(dest)
        try:
            dev = _probe(dest).stat().st_dev
        except OSError:
            free = None
        if free is None:
            return None, f"cannot determine free space on {dest}"
        per_fs.setdefault(dev, [free, 0, str(dest)])[1] += size
    for free, need, dest in per_fs.values():
        if free < need * SPACE_HEADROOM:
            return free, (f"not enough space on {dest}: {fmt_bytes(free)} free for {fmt_bytes(need)} "
                          f"(needs {fmt_bytes(int(need * SPACE_HEADROOM))})")
    return next(iter(per_fs.values()))[0] if per_fs else None, None


def _reject(msg):
    return {"size_bytes": None, "dest_dir": "", "free_bytes": None,
            "enough": False, "error": msg, "groups": []}


def plan_download(model, engine):
    """Confirm-first download plan across every fetch group (main + sidecars).
    Sizes come from the wire listing, free space from the destination
    filesystems. Unknown size → enough=False, never space-safe."""
    recipe = model["recipes"][engine]
    hold = str(recipe.get("operator_hold") or "").strip()
    if hold:
        return _reject(f"recipe is on operator hold: {hold} — downloads are not authorized")
    try:
        groups = _fetch_groups(model, engine)
    except ValueError as e:
        return _reject(str(e))
    if not groups:
        return _reject("recipe has no downloadable artifact")
    out_groups = []
    total_size = 0
    for dl, dest_dir, label in groups:
        repo, err = resolve_repo(dl)
        if err:
            return _reject(err)
        try:
            files = hf_files(repo, dl.get("revision"))
        except Exception as e:
            return _reject(f"could not list repository contents: {e}")
        if dl.get("kind") == "file":
            wanted = [dl["name"]] + list(dl.get("extra_files", []))
        else:
            wanted = list(files.keys())
        if not wanted or any(not safe_artifact(w) for w in wanted):
            return _reject("repository contains unsafe or empty artifact paths")
        if dl.get("kind") == "file" and any(w not in files for w in wanted):
            return _reject("pinned repository does not contain the exact recipe filename")
        sizes = [files.get(w) for w in wanted]
        if any(s is None for s in sizes):
            return _reject("remote file sizes unknown — refusing to start without a size check")
        size = int(sum(sizes))
        total_size += size
        out_groups.append({"repo": repo, "dest_dir": str(dest_dir),
                           "files": wanted, "size_bytes": size, "label": label})
    free, error = _space_check([(g["dest_dir"], g["size_bytes"]) for g in out_groups])
    return {"size_bytes": total_size, "dest_dir": out_groups[0]["dest_dir"],
            "free_bytes": free, "enough": error is None, "error": error,
            "groups": out_groups}


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
    import threading
    entry = DOWNLOADS[did]
    try:
        with LOCK:
            entry["state"] = "resolving"
        groups = _fetch_groups(model, engine)
        resolved = []
        for dl, dest_dir, label in groups:
            repo, err = resolve_repo(dl)
            if err:
                raise ValueError(err)
            files = hf_files(repo, dl.get("revision"))
            if dl.get("kind") == "file":
                wanted = [dl["name"]] + list(dl.get("extra_files", []))
            else:
                wanted = list(files.keys())
            if not wanted or any(not safe_artifact(w) for w in wanted):
                raise ValueError("Repository contains unsafe or empty artifact paths")
            if dl.get("kind") == "file" and any(w not in files for w in wanted):
                raise ValueError("Pinned repository does not contain the exact recipe filename")
            resolved.append((dl, repo, files, wanted, dest_dir, label))
        needs = [(d, sum((f.get(w) or 0) for w in wanted if w in f))
                 for _dl, _r, f, wanted, d, _l in resolved]
        total_known = sum(size for _d, size in needs)
        # disk guard at transfer time, per destination filesystem
        if total_known:
            _free, error = _space_check(needs)
            if error:
                raise ValueError(error)
        dest_dir = resolved[0][4]
        dest_dir.mkdir(parents=True, exist_ok=True)
        with LOCK:
            entry.update({"state": "downloading",
                          "total": total_known or None, "dest": str(dest_dir)})
        for dl, repo, files, wanted, ddir, label in resolved:
            if label != "main":
                entry_note = f"{label} sidecar: {repo}"
                with LOCK:
                    entry["note"] = entry_note
            ddir.mkdir(parents=True, exist_ok=True)
            for w in wanted:
                dest = ddir / w
                if dest.is_symlink() or not dest.parent.resolve().is_relative_to(ddir.resolve()):
                    raise ValueError("Refusing download through symlink or outside destination")
                if dest.exists():
                    if files.get(w) and dest.stat().st_size == files[w]:
                        with LOCK:
                            entry["done"] += files[w]
                        continue
                    raise FileExistsError(f"Existing artifact has unknown or mismatched size; move it aside manually: {dest}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                url = f"{HF}/{repo}/resolve/{urllib.parse.quote(dl.get('revision') or 'main', safe='')}/{urllib.parse.quote(w)}"
                _download_file(url, dest, entry)
                if files.get(w) and dest.stat().st_size != files[w]:
                    raise ValueError(f"size mismatch after download ({w}): got "
                                     f"{dest.stat().st_size}, expected {files[w]}")
        with LOCK:
            entry["state"] = "done"
            entry["pct"] = 100
            entry["eta"] = 0
        scan.start_scan_async(force=True)
    except InterruptedError:
        with LOCK:
            entry["state"] = "cancelled"
    except Exception as e:
        with LOCK:
            entry["state"] = "error"
            entry["error"] = str(e)


def start_download(model, engine, plan=None):
    """Begin the transfer after a confirmed plan. Returns (did, error)."""
    did = f"{model['id']}-{engine}"
    if plan is None:
        plan = plan_download(model, engine)
    if plan.get("error") or not plan.get("enough"):
        return None, plan.get("error") or "download plan rejected (insufficient space)"
    import threading
    with LOCK:
        if did in DOWNLOADS and DOWNLOADS[did].get("state") in ("queued", "resolving", "downloading"):
            return did, None
        DOWNLOADS[did] = {"id": did, "model": model["name"], "engine": engine,
                          "state": "queued", "done": 0, "total": plan.get("size_bytes"),
                          "pct": 0, "speed": 0, "eta": None, "dest": plan.get("dest_dir"),
                          "quant": model["recipes"][engine].get("download", {}).get("quant", "")}
    threading.Thread(target=download_worker, args=(did, model, engine), daemon=True).start()
    return did, None


def cancel_download(did):
    """Cooperative cancel: the worker stops at the next chunk boundary."""
    with LOCK:
        e = DOWNLOADS.get(did)
        if not e:
            return False
        if e.get("state") in ("done", "error", "cancelled"):
            return False
        e["cancel"] = True
    return True


def snapshot():
    with LOCK:
        return {did: {k: v for k, v in e.items() if k != "cancel"}
                for did, e in DOWNLOADS.items()}


def cancel_all():
    with LOCK:
        for e in DOWNLOADS.values():
            e["cancel"] = True
