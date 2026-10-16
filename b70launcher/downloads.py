"""Hugging Face downloads: plan → confirm → transfer → cancel.

A download never starts without a confirmed plan, and the plan is computed on
the DESTINATION filesystem: size_bytes from the HF API listing, free_bytes
from the dir the files will land in. An unknown remote size is never treated
as space-safe (a listing that fails to report sizes rejects the confirm).

Recipes may declare a draft sidecar (``draft.download``) — it is fetched into
the same models_dir in the same transfer (recipe-audit #8).

Bytes already on disk (complete files of the listed size and ``.part``
partials) are resumed, not re-fetched: the plan reports them as
``partial_bytes`` and only ``remaining_bytes`` must fit on the destination.
Live ``speed`` is a sliding-window wall-clock average across all files.

A Hugging Face token (HF_TOKEN, HUGGING_FACE_HUB_TOKEN, $HF_HOME/token or the
huggingface cache token file) is read, never written, and sent only to the HF
origin — redirects to CDN / Xet hosts drop it. Gated repositories without a
token are refused at plan time; a 401/403 mid-transfer stops at once with an
``error_code`` (``gated`` | ``auth``) instead of retrying.
"""
import json
import os
import re
import shutil
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
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
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
}

SPACE_HEADROOM = 1.05  # require free >= remaining bytes * 1.05
SPEED_WINDOW = 5.0  # seconds of wall time the reported speed averages over
SPEED_MIN_SPAN = 1.0  # speed stays 0 until this much wall time is sampled
SAMPLE_EVERY = 0.1  # min seconds between recorded progress samples


GATED_PLAN_ERROR = ("This model is gated on Hugging Face. Accept its terms at {url}, sign in with "
                    "`hf auth login` (or set HF_TOKEN), then try again.")
ACCESS_ERRORS = {
    "gated": "Hugging Face refused the download: accept the model terms at {url} with the account of your token, then resume.",
    "auth": "Hugging Face needs a sign-in for this model: run `hf auth login` (or set HF_TOKEN), then resume.",
}


def hf_token():
    """The user's Hugging Face token, or None. Read-only lookup in the
    huggingface_hub order: env HF_TOKEN, env HUGGING_FACE_HUB_TOKEN,
    $HF_HOME/token, then ${XDG_CACHE_HOME:-~/.cache}/huggingface/token."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(var, "").strip()
        if token:
            return token
    files = []
    if os.environ.get("HF_HOME"):
        files.append(Path(os.environ["HF_HOME"]).expanduser() / "token")
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    files.append(Path(cache).expanduser() / "huggingface" / "token")
    for path in files:
        try:
            token = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if token:
            return token
    return None


def _is_hf(url):
    """True only for the Hugging Face origin itself (scheme + host + port)."""
    got, hf = urllib.parse.urlsplit(url), urllib.parse.urlsplit(HF)
    return (got.scheme, got.netloc.lower()) == (hf.scheme, hf.netloc.lower())


class _HFRedirect(urllib.request.HTTPRedirectHandler):
    """Authorization is an unredirected header, so urllib never copies it;
    re-attach it only when the redirect stays on the HF origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        auth = req.unredirected_hdrs.get("Authorization")
        if new is not None and auth and _is_hf(new.full_url):
            new.add_unredirected_header("Authorization", auth)
        return new


_OPENER = urllib.request.build_opener(_HFRedirect)


def _open(url, headers=None, timeout=30, token=None):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    if token and _is_hf(url):
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    return _OPENER.open(req, timeout=timeout)


class HFAccessError(Exception):
    """HF refused the transfer (401/403); retrying cannot help."""

    def __init__(self, kind):
        super().__init__(kind)
        self.kind = kind


def _http_json(url):
    with _open(url, token=hf_token()) as resp:
        return json.loads(resp.read(24 << 20))


def hf_gated(repo):
    """The repo's gating mode from the model info API: False, "auto" or "manual"."""
    gated = _http_json(f"{HF}/api/models/{repo}").get("gated") or False
    return gated if gated is False else str(gated)


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
    return {"size_bytes": None, "partial_bytes": 0, "remaining_bytes": None,
            "dest_dir": "", "free_bytes": None, "gated": False,
            "hf_token": hf_token() is not None, "repo_url": "",
            "enough": False, "error": msg, "groups": []}


def _on_disk(dest, size):
    """Bytes of one artifact already present at `dest`: `size` for a regular
    file of exactly that size, else the size of a regular `<dest>.part`
    (uncapped — callers decide what an oversized partial means). An existing
    `dest` of the wrong size and any symlink count as 0."""
    try:
        st = dest.lstat()
    except OSError:
        st = None
    if st is not None:
        return size if size and stat.S_ISREG(st.st_mode) and st.st_size == size else 0
    try:
        st = dest.with_suffix(dest.suffix + ".part").lstat()
    except OSError:
        return 0
    return st.st_size if stat.S_ISREG(st.st_mode) else 0


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
    gating = {}  # repo -> gated mode; main and sidecar may share a repo
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
        present = 0
        for w, s in zip(wanted, sizes):
            have = _on_disk(dest_dir / w, s)
            present += have if have <= s else 0  # an oversized .part resumes nothing
        if repo not in gating:
            try:
                gating[repo] = hf_gated(repo)
            except Exception as e:
                return _reject(f"could not read repository info: {e}")
        total_size += size
        out_groups.append({"repo": repo, "dest_dir": str(dest_dir), "files": wanted,
                           "size_bytes": size, "partial_bytes": present, "label": label,
                           "gated": gating[repo], "repo_url": f"{HF}/{repo}"})
    partial = sum(g["partial_bytes"] for g in out_groups)
    free, error = _space_check([(g["dest_dir"], g["size_bytes"] - g["partial_bytes"])
                                for g in out_groups])
    has_token = hf_token() is not None
    gated_group = next((g for g in out_groups if g["gated"]), None)
    shown = gated_group or out_groups[0]
    if gated_group and not has_token:
        error = GATED_PLAN_ERROR.format(url=gated_group["repo_url"])
    return {"size_bytes": total_size, "partial_bytes": partial,
            "remaining_bytes": total_size - partial, "dest_dir": out_groups[0]["dest_dir"],
            "free_bytes": free, "gated": shown["gated"], "hf_token": has_token,
            "repo_url": shown["repo_url"], "enough": error is None, "error": error,
            "groups": out_groups}


def _rate(entry, now):
    """Bytes/s received over the last SPEED_WINDOW seconds of wall time.

    `_samples` holds (monotonic_ts, bytes received this run); the oldest kept
    sample is the last one at or before the window start, so file boundaries
    and slow HTTP connects stay inside the average instead of resetting it."""
    samples = entry.get("_samples")
    if not samples:
        return 0.0
    while len(samples) > 1 and samples[1][0] <= now - SPEED_WINDOW:
        samples.popleft()
    t0, x0 = samples[0]
    span = now - t0
    if span < SPEED_MIN_SPAN:
        return 0.0
    return (entry.get("_xfer", 0) - x0) / span


def _refresh(entry, now):
    """Recompute speed/eta/pct from the samples. Caller holds LOCK."""
    speed = _rate(entry, now)
    entry["speed"] = round(speed, 1)
    total, done = entry.get("total"), entry.get("done", 0)
    if total:
        entry["pct"] = round(min(100 * done / total, 100), 1)
    entry["eta"] = round(max(total - done, 0) / speed, 1) if total and speed > 0 else None


def _advance(entry, n):
    """Count `n` freshly received bytes. Caller holds LOCK."""
    entry["done"] = entry.get("done", 0) + n
    entry["_xfer"] = entry.get("_xfer", 0) + n
    samples = entry.setdefault("_samples", deque())
    now = time.monotonic()
    if not samples or now - samples[-1][0] >= SAMPLE_EVERY:
        samples.append((now, entry["_xfer"]))
        _refresh(entry, now)


def _finish(entry, state, **extra):
    """Terminal transition: stop the clock and drop the speed samples."""
    with LOCK:
        entry.update(state=state, finished=time.time(), speed=0, **extra)
        entry.pop("_samples", None)


def _download_file(url, dest, entry):
    """Stream to dest with HTTP Range resume across network failures (HF/Xet 403s).

    An existing `.part` was already counted into entry["done"] and
    entry["resumed_bytes"] by download_worker before the transfer began.
    A 401/403 raises HFAccessError at once (kind "gated" or "auth"); the
    `.part` stays for a later resume."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.is_symlink():
        raise ValueError("Refusing symlinked partial download")
    counted = tmp.stat().st_size if tmp.exists() else 0  # this file's bytes in entry["done"]
    resumed = counted  # of those, bytes that pre-dated this transfer
    token = hf_token()
    last_err = None
    for attempt in range(4):
        try:
            existing = tmp.stat().st_size if tmp.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            try:
                resp = _open(url, headers, timeout=60, token=token)
            except urllib.error.HTTPError as e:
                if e.code not in (401, 403):
                    raise
                code = e.headers.get("x-error-code", "") if e.headers else ""
                e.close()
                gated = code == "GatedRepo" or (e.code == 403 and token)
                raise HFAccessError("gated" if gated else "auth") from None
            with resp:
                if existing and getattr(resp, "status", 200) != 206:
                    existing = 0  # server ignored Range -> restart this file
                with LOCK:
                    entry["done"] = entry.get("done", 0) + existing - counted
                    counted = existing
                    if resumed and not existing:
                        entry["resumed_bytes"] = entry.get("resumed_bytes", 0) - resumed
                        resumed = 0
                    total = resp.headers.get("Content-Length")
                    if total and not entry.get("total"):
                        entry["total"] = entry["done"] + int(total)
                read = getattr(resp, "read1", resp.read)  # return what arrived: smooth progress, prompt cancel
                with open(tmp, "ab" if existing else "wb") as f:
                    while True:
                        if entry.get("cancel"):
                            raise InterruptedError("cancelled")
                        chunk = read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        counted += len(chunk)
                        with LOCK:
                            _advance(entry, len(chunk))
            if dest.exists():
                raise FileExistsError(f"Refusing to overwrite existing artifact: {dest}")
            tmp.rename(dest)
            return
        except (InterruptedError, HFAccessError):
            raise
        except FileExistsError:
            raise  # not retryable — re-downloading cannot fix a full destination
        except Exception as e:
            last_err = e
            if attempt == 3:
                break
            time.sleep(2 * (attempt + 1))  # resume picks up from .part on retry
    raise RuntimeError(f"download failed after 4 attempts: {last_err}")


def _checked_dest(ddir, name):
    dest = ddir / name
    if dest.is_symlink() or not dest.parent.resolve().is_relative_to(ddir.resolve()):
        raise ValueError("Refusing download through symlink or outside destination")
    return dest


def download_worker(did, model, engine):
    entry = DOWNLOADS[did]
    repo_url = None
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
        # bytes already on disk resume instead of re-downloading; only the
        # remainder needs room on each destination filesystem
        needs, present = [], 0
        for _dl, _r, files, wanted, ddir, _l in resolved:
            size = have = 0
            for w in wanted:
                expected = files.get(w)
                on_disk = _on_disk(_checked_dest(ddir, w), expected)
                if expected and on_disk > expected:
                    raise ValueError(f"Partial download is larger than the remote file; move it aside manually: {ddir / w}.part")
                size += expected or 0
                have += on_disk
            needs.append((ddir, max(size - have, 0)))
            present += have
        total_known = sum(sum((f.get(w) or 0) for w in wanted if w in f)
                          for _dl, _r, f, wanted, _d, _l in resolved)
        # disk guard at transfer time, per destination filesystem
        if total_known:
            _free, error = _space_check(needs)
            if error:
                raise ValueError(error)
        dest_dir = resolved[0][4]
        dest_dir.mkdir(parents=True, exist_ok=True)
        with LOCK:
            entry.update({"state": "downloading", "total": total_known or None,
                          "dest": str(dest_dir), "done": present, "resumed_bytes": present,
                          "files_done": 0, "files_total": sum(len(r[3]) for r in resolved),
                          "_xfer": 0, "_samples": deque([(time.monotonic(), 0)])})
            _refresh(entry, time.monotonic())
        for dl, repo, files, wanted, ddir, label in resolved:
            if label != "main":
                entry_note = f"{label} sidecar: {repo}"
                with LOCK:
                    entry["note"] = entry_note
            ddir.mkdir(parents=True, exist_ok=True)
            for w in wanted:
                repo_url = f"{HF}/{repo}"
                dest = _checked_dest(ddir, w)
                if dest.exists():
                    if files.get(w) and dest.stat().st_size == files[w]:
                        with LOCK:
                            entry["files_done"] += 1  # counted in done/resumed_bytes above
                        continue
                    raise FileExistsError(f"Existing artifact has unknown or mismatched size; move it aside manually: {dest}")
                with LOCK:
                    entry["file"] = w
                dest.parent.mkdir(parents=True, exist_ok=True)
                url = f"{HF}/{repo}/resolve/{urllib.parse.quote(dl.get('revision') or 'main', safe='')}/{urllib.parse.quote(w)}"
                _download_file(url, dest, entry)
                if files.get(w) and dest.stat().st_size != files[w]:
                    raise ValueError(f"size mismatch after download ({w}): got "
                                     f"{dest.stat().st_size}, expected {files[w]}")
                with LOCK:
                    entry["files_done"] += 1
        _finish(entry, "done", pct=100, eta=0, file=None)
        scan.start_scan_async(force=True)
    except InterruptedError:
        _finish(entry, "cancelled", eta=None)
    except HFAccessError as e:
        _finish(entry, "error", eta=None, error_code=e.kind, repo_url=repo_url,
                error=ACCESS_ERRORS[e.kind].format(url=repo_url))
    except Exception as e:
        _finish(entry, "error", eta=None, error=str(e))


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
        DOWNLOADS[did] = {"id": did, "model": model["name"], "model_id": model["id"], "engine": engine,
                          "state": "queued", "done": 0, "total": plan.get("size_bytes"),
                          "pct": 0, "speed": 0, "eta": None, "dest": plan.get("dest_dir"),
                          "quant": model["recipes"][engine].get("download", {}).get("quant", ""),
                          "started": time.time(), "finished": None, "file": None,
                          "files_done": 0, "files_total": sum(len(g["files"]) for g in plan.get("groups", [])),
                          "resumed_bytes": 0}
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
    """Public view of every download; live transfers get speed/eta recomputed
    against the current time so a stalled connection decays toward 0."""
    now = time.monotonic()
    with LOCK:
        out = {}
        for did, e in DOWNLOADS.items():
            if e.get("state") == "downloading" and e.get("_samples"):
                _refresh(e, now)
            out[did] = {k: v for k, v in e.items() if k != "cancel" and not k.startswith("_")}
        return out


def cancel_all():
    with LOCK:
        for e in DOWNLOADS.values():
            e["cancel"] = True
