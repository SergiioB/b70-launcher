#!/usr/bin/env python3
"""b70-launcher — one window from "I want a model" to a running B70 inference server.

Stdlib only. Windows + Linux. Serves the UI, scans disks for known artifacts,
downloads missing ones from Hugging Face with progress, builds cookbook-backed
launch commands, spawns engines, opens a harness terminal.
"""
import argparse
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
RECIPES = json.loads((HERE / "recipes.json").read_text())
SETTINGS = json.loads((HERE / "settings.json").read_text())
LOGDIR = HERE / "logs"
LOGDIR.mkdir(exist_ok=True)
CACHE = HERE / ".resolve-cache.json"

IS_WIN = os.name == "nt"
RUNNING = {}   # id -> server entry
DOWNLOADS = {}  # id -> download entry
SCAN = {"items": {"gguf": {}, "snapshots": {}}, "roots": [], "ts": 0}
LOCK = threading.Lock()

HF = "https://huggingface.co"
UA = {"User-Agent": "b70-launcher/0.2"}


# ---------------------------------------------------------------- scan / detect

def scan_roots():
    roots = []
    for r in SETTINGS.get("scan_dirs", []):
        p = Path(r).expanduser()
        if p.is_dir() and p not in roots:
            roots.append(p)
    return roots


def read_ctx_from_config(cfgpath):
    try:
        cfg = json.loads(Path(cfgpath).read_text())
    except Exception:
        return None
    for key in ("max_position_embeddings", "context_length", "max_seq_len", "n_ctx"):
        v = cfg.get(key)
        if isinstance(v, int) and v > 512:
            return v
    return None


def scan():
    """Find GGUF files and HF snapshot dirs under the scan roots."""
    items = {"gguf": {}, "snapshots": {}}
    roots = scan_roots()
    for root in roots:
        try:
            for p in root.rglob("*.gguf"):
                items["gguf"].setdefault(p.name.lower(), str(p))
            for d in root.rglob("*"):
                if not d.is_dir():
                    continue
                cfg = d / "config.json"
                ov = any(d.glob("openvino_language_model.*"))
                if cfg.exists() or ov:
                    ctx = read_ctx_from_config(cfg) if cfg.exists() else None
                    items["snapshots"][d.name.lower()] = {
                        "path": str(d),
                        "repo_hint": f"{d.parent.name}/{d.name}",
                        "ctx": ctx,
                    }
        except PermissionError:
            continue
    with LOCK:
        SCAN.update({"items": items, "roots": [str(r) for r in roots], "ts": time.time()})
    return items, roots


def detect(model, engine):
    """Match a recipe to an on-disk artifact. Returns dict or None."""
    items = SCAN.get("items") or scan()[0]
    r = model["recipes"].get(engine, {})
    dl = r.get("download", {})
    if dl.get("kind") == "file":
        name = dl.get("name", "").lower()
        p = items["gguf"].get(name)
        return {"path": p, "ctx": None, "mount_root": str(Path(p).parent)} if p else None
    key = (dl.get("search") or r.get("search_name") or "").lower()
    hit = items["snapshots"].get(key)
    if hit:
        p = Path(hit["path"])
        roots = [Path(x) for x in SCAN.get("roots", [])]
        mount = p.parent.parent if p.parent.parent in roots else p.parent
        return {"path": hit["path"], "ctx": hit.get("ctx"), "mount_root": str(mount)}
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
        return json.loads(resp.read())


def _cache_get(key):
    try:
        return json.loads(CACHE.read_text()).get(key)
    except Exception:
        return None


def _cache_put(key, val):
    try:
        data = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    except Exception:
        data = {}
    data[key] = val
    CACHE.write_text(json.dumps(data, indent=1))


def hf_files(repo):
    data = _http_json(f"{HF}/api/models/{repo}?blobs=true")
    files = {}
    for sib in data.get("siblings", []):
        lfs = sib.get("lfs") or {}
        size = sib.get("size") or lfs.get("size")
        files[sib["rfilename"]] = size
    return files


def resolve_repo(dl):
    """Find the HF repo holding the wanted artifact. Grounded repo wins;
    otherwise search HF and verify the expected file exists."""
    if dl.get("repo"):
        return dl["repo"], None
    needle0 = dl.get("search") or dl.get("name") or ""
    cached = _cache_get(needle0)
    if cached:
        return cached, None
    want = (dl.get("name") or "").lower()
    verify = [v.lower() for v in dl.get("verify", [])]
    # search falls back to shorter needles: HF search is fuzzy, quant suffixes kill it
    needles = [needle0]
    stem = needle0
    while "-" in stem:
        stem = stem.rsplit("-", 1)[0]  # Nemotron-3.5-Lightning-Q4_K_M -> Nemotron-3.5-Lightning -> Nemotron-3.5
        needles.append(stem)
        if len(stem.split("-")) <= 1:
            break
    seen = set()
    for needle in needles:
        if needle in seen or not needle:
            continue
        seen.add(needle)
        try:
            results = _http_json(f"{HF}/api/models?search={urllib.parse.quote(needle)}&limit=10")
        except Exception as e:
            return None, f"HF search failed: {e}"
        for cand in results:
            repo = cand.get("modelId") or cand.get("id")
            if not repo:
                continue
            try:
                files = hf_files(repo)
            except Exception:
                continue
            names = [n.lower() for n in files]
            nn = lambda s: s.lower().replace("-", "").replace("_", "")
            if want and (want in names or any(n.endswith(want) for n in names)):
                _cache_put(needle0, repo)
                return repo, None
            if not want and any(v in names for v in verify) and nn(needle) in nn(repo):
                _cache_put(needle0, repo)
                return repo, None
    return None, f"no HF repo found for '{needle0}' — paste the repo id in the artifact row"


# ---------------------------------------------------------------- downloads

def fmt_bytes(n):
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def fmt_eta(sec):
    if sec is None or sec < 0:
        return "--:--"
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _download_file(url, dest, entry):
    """Stream to dest with HTTP Range resume across network failures (HF/Xet 403s)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
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
                        entry["done"] = entry.get("done", 0) + len(chunk)
                        counted += len(chunk)
                        entry["speed"] = speed
                        total_all = entry.get("total")
                        if total_all:
                            left = max(total_all - entry["done"], 0)
                            entry["eta"] = left / speed if speed > 0 else None
                            entry["pct"] = round(100 * entry["done"] / total_all, 1)
            tmp.rename(dest)
            return
        except InterruptedError:
            raise
        except Exception as e:
            last_err = e
            if attempt == 3:
                break
            time.sleep(2 * (attempt + 1))  # resume picks up from .part on retry
    raise RuntimeError(f"download failed after 4 attempts: {last_err}")


def download_worker(did, model, engine):
    entry = DOWNLOADS[did]
    try:
        entry["state"] = "resolving"
        dl = model["recipes"][engine]["download"]
        repo, err = resolve_repo(dl)
        if err:
            entry["state"] = "error"
            entry["error"] = err
            return
        entry["repo"] = repo
        files = hf_files(repo)
        if dl.get("kind") == "file":
            wanted = [dl["name"]] + list(dl.get("extra_files", []))
            dest_dir = Path(SETTINGS.get("models_dir", "~/models")).expanduser()
        else:
            wanted = list(files.keys())
            root = Path(SETTINGS.get("ovms_repo", "~/models/ovms-repo")).expanduser()
            if engine != "openvino":
                root = Path(SETTINGS.get("models_dir", "~/models")).expanduser()
            dest_dir = root / repo.split("/")[-1]
        dest_dir.mkdir(parents=True, exist_ok=True)
        total_known = sum((files.get(w) or 0) for w in wanted if w in files)
        entry.update({"state": "downloading", "files": wanted, "done": 0,
                      "total": total_known or None, "dest": str(dest_dir)})
        for w in wanted:
            dest = dest_dir / w
            if dest.exists() and files.get(w) and dest.stat().st_size == files[w]:
                entry["done"] += files[w]
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = f"{HF}/{repo}/resolve/main/{urllib.parse.quote(w)}"
            _download_file(url, dest, entry)
        entry["state"] = "done"
        entry["pct"] = 100
        entry["eta"] = 0
        scan()
    except InterruptedError:
        entry["state"] = "cancelled"
    except Exception as e:
        entry["state"] = "error"
        entry["error"] = str(e)


def start_download(model, engine):
    did = f"{model['id']}-{engine}"
    with LOCK:
        if did in DOWNLOADS and DOWNLOADS[did].get("state") in ("resolving", "downloading"):
            return did, None
        DOWNLOADS[did] = {"id": did, "model": model["name"], "engine": engine,
                          "state": "queued", "done": 0, "total": None, "pct": 0,
                          "speed": 0, "eta": None,
                          "quant": model["recipes"][engine].get("download", {}).get("quant", "")}
    t = threading.Thread(target=download_worker, args=(did, model, engine), daemon=True)
    t.start()
    return did, None


# ---------------------------------------------------------------- command build

def render_gid():
    if IS_WIN:
        return None
    for dev in sorted(glob.glob("/dev/dri/render*")):
        try:
            return os.stat(dev).st_gid
        except OSError:
            continue
    return None


def pretty(tokens, width=96):
    sep = " ^" if IS_WIN else " \\"
    lines, cur = [], ""
    for tok in tokens:
        candidate = (cur + " " + tok).strip()
        if cur and len(candidate) > width:
            lines.append(cur + sep)
            cur = "    " + tok
        else:
            cur = candidate
    lines.append(cur)
    return "\n".join(lines)


def find_model(model_id):
    for m in RECIPES["models"]:
        if m["id"] == model_id:
            return m
    return None


def container_path(det, fallback):
    """Map an on-disk path to the container's /models mount."""
    if not det or not det.get("path"):
        return fallback, None
    root = Path(det["mount_root"])
    rel = Path(det["path"]).relative_to(root)
    return "/models/" + str(rel).replace(os.sep, "/"), str(root)


def build(cfg):
    model = find_model(cfg.get("model_id", ""))
    engine = cfg.get("engine", "openvino")
    recipe = (model or {}).get("recipes", {}).get(engine)
    warns = []
    if not model or not recipe:
        return {"error": "no recipe for that model+engine", "warnings": warns}

    det = detect(model, engine)
    ctxres = resolve_ctx(model, engine, det)
    artifact_mib = None
    if det and det.get("path"):
        try:
            p = Path(det["path"])
            artifact_mib = (p.stat().st_size if p.is_file()
                            else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())) / 1048576
        except Exception:
            artifact_mib = None
    ctx = int(cfg["ctx"]) if cfg.get("ctx") else ctxres["value"]
    if recipe.get("ctx_max") and ctx > recipe["ctx_max"]:
        warns.append(f"ctx {ctx} > engine ceiling {recipe['ctx_max']} for this recipe — clamping.")
        ctx = recipe["ctx_max"]

    port = int(cfg.get("port") or 8000)
    slots = int(cfg.get("slots") or 1)
    extra = shlex.split(cfg.get("extra", "") or "")
    extra_env = {}
    for line in (cfg.get("extra_env") or "").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            extra_env[k.strip()] = v.strip()
    envargs = [x for k, v in extra_env.items() for x in ("-e", f"{k}={v}")]
    gpus = cfg.get("gpus") or [0]
    split = (cfg.get("split") or "").strip()
    kv = cfg.get("kv") or "recipe default"
    env = {}
    kind = recipe.get("kind", "")

    if kind == "ovms":
        fallback_mount = cfg.get("models_dir") or SETTINGS.get("ovms_repo", "/mnt/models/ovms-repo")
        if det:
            cpath, mount_src = container_path(det, fallback_mount)
            source_model = cpath[len("/models/"):]
        else:
            source_model = recipe["source_model"]
            mount_src = fallback_mount
            warns.append("model not detected on disk — command uses default paths (download it or point 'scan root' at it)")
        tokens = ["docker", "run", "-d", "--rm", "--name", f"b70-{model['id']}-ovms"]
        if not IS_WIN:
            gid = render_gid()
            tokens += ["--user", f"{os.getuid()}:{os.getuid() if gid is None else gid}",
                       "--device", "/dev/dri" if len(gpus) > 1 else f"/dev/dri/renderD{128 + gpus[0]}",
                       "--group-add", str(gid if gid is not None else "render")]
        tokens += ["-p", f"{port}:{port}", "-v", f"{mount_src}:/models:rw",
                   recipe["image"],
                   "--rest_port", str(port),
                   "--model_repository_path", "/models",
                   "--source_model", source_model,
                   "--task", "text_generation",
                   "--target_device", "GPU"]
        if recipe.get("tool_parser"):
            tokens += ["--tool_parser", recipe["tool_parser"]]
        if recipe.get("reasoning_parser"):
            tokens += ["--reasoning_parser", recipe["reasoning_parser"]]
        if recipe.get("cim_long_ctx") and ctx > 20480:
            tokens += ["--cache_interval_multiplier", str(recipe["cim_long_ctx"])]
        tokens += recipe.get("fixed_flags", []) + envargs + extra
        warns.append("OVMS manages KV internally (GPU INT4 KV supported). ctx is advisory here.")
        served_name = source_model
        cname = f"b70-{model['id']}-ovms"

    elif kind.startswith("vllm"):
        fallback = recipe.get("model_path", "/models/model")
        if det:
            model_path, mount_src = container_path(det, fallback)
            if mount_src is None:
                mount_src = cfg.get("models_dir") or SETTINGS.get("models_dir", "/data/models")
        else:
            model_path = fallback
            mount_src = cfg.get("models_dir") or SETTINGS.get("models_dir", "/data/models")
            warns.append("model not detected on disk — using recipe default path")
        # a single mapped render node enumerates as device 0 inside the container
        selector = "level_zero:0,1" if len(gpus) > 1 else "level_zero:0"
        tokens = ["docker", "run", "-d", "--rm", "--name", f"b70-{model['id']}-vllm"]
        if not IS_WIN:
            tokens += ["--device", "/dev/dri" if len(gpus) > 1 else f"/dev/dri/renderD{128 + gpus[0]}",
                       "--group-add", "render",
                       "-v", f"{Path.home() / '.cache' / 'huggingface'}:/root/.cache/huggingface:rw"]
        else:
            warns.append("Windows: docker GPU passthrough runs via WSL2; --device flags omitted.")
        tokens += ["--ipc=host", "--shm-size=16g",
                   "-v", f"{mount_src}:/models:ro",
                   "-p", f"{port}:{port}",
                   "-e", f"ONEAPI_DEVICE_SELECTOR={selector}",
                   "-e", "VLLM_OPENVINO_KVCACHE_SPACE=16",
                   recipe["image"],
                   "--model", model_path,
                   "--dtype", recipe.get("dtype", "bfloat16"),
                   "--gpu-memory-utilization", "0.92",
                   "--max-model-len", str(ctx),
                   "--port", str(port)]
        if recipe.get("speculative") and cfg.get("mtp", True):
            tokens += ["--speculative-model", recipe["speculative"],
                       "--num-speculative-tokens", str(recipe.get("spec_tokens", 4))]
        if kv == "fp8" or recipe.get("kv") == "fp8":
            tokens += ["--kv-cache-dtype", "fp8"]
        if kind == "vllm-autoround":
            tokens += recipe.get("fixed_flags", [])
        if len(gpus) > 1:
            tokens += ["--tensor-parallel-size", str(len(gpus))]
            warns.append("Multi-GPU TP on XPU: verify the level_zero selector string on your driver (per Intel LLM-Scaler TP setup).")
        if slots > 1:
            tokens += ["--max-num-seqs", str(slots)]
        tokens += envargs + extra
        served_name = model["name"]
        cname = f"b70-{model['id']}-vllm"
        if kind == "vllm-autoround":
            warns.append("FP16 crash guard: dtype float16 + --enforce-eager is intentional (dt_bias crash on BF16 path).")

    elif kind == "gguf":
        fallback = recipe.get("gguf", "/models/model.gguf")
        if det:
            gpath, mount_src = container_path(det, fallback)
            if mount_src is None:
                mount_src = cfg.get("models_dir") or SETTINGS.get("models_dir", "/data/models")
        else:
            gpath = fallback
            mount_src = cfg.get("models_dir") or SETTINGS.get("models_dir", "/data/models")
            warns.append("model not detected on disk — using recipe default path")
        kv_map = {"q8_0": ["--cache-type", "q8_0"],
                  "q8_0/q4_1": ["--cache-type-k", "q8_0", "--cache-type-v", "q4_1"],
                  "f16": ["--cache-type", "f16"]}
        if kv == "f16":
            warns.append("f16 KV wastes VRAM with zero quality gain on these workloads. q8_0 recommended.")
        llama_bin = (SETTINGS.get("llama_bin") or "").strip()
        if llama_bin and not cfg.get("use_docker"):
            tokens = [llama_bin]
            env.update(extra_env)
            env.update({
                "ONEAPI_DEVICE_SELECTOR": "level_zero:" + ",".join(str(g) for g in gpus),
                "ZES_ENABLE_SYSMAN": "1", "SYCL_CACHE_PERSISTENT": "0",
                "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "1",
                "UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS": "1",
                "SYCL_DEVICE_FILTER": "level_zero",
                "ZE_FLAT_DEVICE_HIERARCHY": "COMPOSITE", "ZE_AFFINITY_MASK": str(gpus[0]),
            })
            warns.append("Native llama.cpp: run inside `source /opt/intel/oneapi/setvars.sh` shell if libs are missing.")
        else:
            tokens = ["docker", "run", "-d", "--rm", "--name", f"b70-{model['id']}-sycl"]
            if not IS_WIN:
                tokens += ["--device", "/dev/dri" if len(gpus) > 1 else f"/dev/dri/renderD{128 + gpus[0]}",
                           "--group-add", "render"]
            tokens += ["--ipc=host", "-v", f"{mount_src}:/models:ro",
                       "-p", f"{port}:{port}", "-e", "ZES_ENABLE_SYSMAN=1",
                       recipe["image"]]
        tokens += ["-m", gpath, "-ngl", "99", "--host", "0.0.0.0", "--port", str(port),
                   "-c", str(ctx), "--flash-attn"] + kv_map.get(kv, kv_map["q8_0"])
        if len(gpus) > 1:
            tokens += ["-ts", split or ",".join(["1"] * len(gpus))]
            warns.append("Dual-GPU is a capacity play, not speed: PCIe hop costs 10-50% (x4 worst).")
        if slots > 1:
            tokens += ["-np", str(slots)]
            warns.append("-np splits the context across slots (shared -c pool).")
        tokens += recipe.get("fixed_flags", []) + envargs + extra
        served_name = Path(gpath).stem
        cname = f"b70-{model['id']}-sycl"

    else:
        return {"error": f"unknown recipe kind {kind}", "warnings": warns}

    watts = int(cfg.get("power") or recipe.get("power") or 150)
    uw = watts * 1_000_000
    power_cmd = (f"echo {uw} | sudo tee $(grep -lE '^(xe|i915)$' /sys/class/hwmon/hwmon*/name"
                 f" | sed 's|/name$|/power1_cap|')   # {watts}W cap on the GPU only, until reboot")
    return {
        "cmd": pretty(tokens), "tokens": tokens, "env": env, "warnings": warns,
        "power_cmd": power_cmd, "power": watts, "cname": cname,
        "endpoint": f"http://127.0.0.1:{port}/v1", "served_name": served_name,
        "model_name": model["name"], "engine": engine,
        "ctx": ctx, "ctx_source": "manual override" if cfg.get("ctx") else ctxres["source"],
        "detected": bool(det), "detected_path": (det or {}).get("path"),
        "artifact_mib": round(artifact_mib, 1) if artifact_mib else None,
    }


TOKS_RE = re.compile(r"([\d.]+)\s*tokens per second|throughput:\s*([\d.]+)\s*tokens/s", re.I)
MEM_RES = [
    (re.compile(r"([\d.]+)\s*(MiB|GiB)\s*VRAM used", re.I), "vram used (log)"),
    (re.compile(r"model weights take\s*([\d.]+)\s*GiB", re.I), "weights GiB (log)"),
    (re.compile(r"buffer size =\s*([\d.]+)\s*(MiB|GiB)", re.I), "buffer (log)"),
]


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


def metrics_snapshot():
    """Per-server tok/s + memory (from engine logs) + container cpu/mem (docker stats)."""
    stats = {}
    if shutil.which("docker"):
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
    out = {}
    with LOCK:
        items = list(RUNNING.items())
    for rid, e in items:
        toks = mem = None
        try:
            tail = Path(e["log"]).read_text(errors="replace")[-8000:]
            ms = TOKS_RE.findall(tail)
            if ms:
                toks = ms[-1][0] or ms[-1][1]
            for rx, label in MEM_RES:
                mm = rx.findall(tail)
                if mm:
                    mem = f"{label}: {mm[-1][0]} {mm[-1][1] if len(mm[-1]) > 1 else ''}".strip()
                    break
        except Exception:
            pass
        out[rid] = {"toks": toks, "engine_mem": mem,
                    "stats": stats.get(e.get("cname", ""))}
    out["vram"] = vram_probe()
    if not out["vram"]:
        ests = [e.get("artifact_mib") for _, e in items if e.get("artifact_mib")]
        if ests:
            out["vram"] = {"used_mib": round(max(ests) * 1.1, 1), "total_mib": None,
                           "source": "estimate (file x1.1)"}
    return out


def harness_line(cfg, built):
    hs = {h["id"]: h for h in SETTINGS["harnesses"]}
    h = hs.get(cfg.get("harness", "omp"), hs["omp"])
    cmd = (cfg.get("harness_cmd") or "").strip() or h["cmd"]
    prefix = "" if IS_WIN else f"OPENAI_BASE_URL={built['endpoint']} OPENAI_API_KEY=local "
    return f"{prefix}{cmd}"


# ---------------------------------------------------------------- run / stop

def launch(cfg, built):
    rid = f"{cfg.get('model_id')}-{cfg.get('engine')}-{cfg.get('port') or 8000}"
    log = LOGDIR / f"{rid}-{int(time.time())}.log"
    entry = {"id": rid, "model": built["model_name"], "engine": built["engine"],
             "cfg": dict(cfg), "cname": built.get("cname", ""), "endpoint": built["endpoint"], "cmd": built["cmd"],
             "artifact_mib": built.get("artifact_mib"),
             "log": str(log), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
             "status": "starting", "proc": None}
    if cfg.get("dry_run"):
        entry["status"] = "dry-run"
        with open(log, "ab") as lf:
            lf.write(("[dry-run] " + built["cmd"] + "\n").encode())
    else:
        env = dict(os.environ)
        env.update(built.get("env") or {})
        with open(log, "ab") as lf:
            lf.write((built["cmd"] + "\n").encode())
            lf.flush()
            entry["proc"] = subprocess.Popen(built["tokens"], env=env, stdout=lf,
                                             stderr=subprocess.STDOUT, cwd=str(HERE))
        entry["status"] = "running"
    with LOCK:
        RUNNING[rid] = entry
    return rid


def servers_snapshot():
    with LOCK:
        items = []
        for rid, e in RUNNING.items():
            proc = e.get("proc")
            status = e["status"]
            if proc is not None:
                rc = proc.poll()
                status = "running" if rc is None else f"exited ({rc})"
            if status == "stopped":
                continue
            items.append({k: v for k, v in e.items() if k != "proc"} | {"status": status})
        return items


def stop_server(rid):
    with LOCK:
        e = RUNNING.get(rid)
        if not e:
            return False
        proc = e.get("proc")
        cname = e.get("cname")
        if cname:
            subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
        if proc is not None and proc.poll() is None:
            proc.terminate()
        e["status"] = "stopped"
        return True


def open_harness(cfg, built):
    line = harness_line(cfg, built)
    if IS_WIN:
        subprocess.Popen(SETTINGS["terminal_windows"] + [line])
    else:
        term = list(SETTINGS["terminal_linux"])
        for cand in (term[0], "gnome-terminal", "xterm"):
            if shutil.which(cand):
                if cand == "gnome-terminal":
                    term = ["gnome-terminal", "--", "sh", "-c", line + "; exec sh"]
                else:
                    term[0] = cand
                    term = term + [line + "; exec sh"]
                subprocess.Popen(term)
                return line
    return line


# ---------------------------------------------------------------- http

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (HERE / "web" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            self._json({"recipes": RECIPES, "settings": SETTINGS,
                        "running": servers_snapshot(), "is_win": IS_WIN,
                        "scan": {"roots": SCAN.get("roots", [])}})
        elif self.path == "/api/scan":
            scan()
            out = {}
            for m in RECIPES["models"]:
                out[m["id"]] = {}
                for eng in m["recipes"]:
                    det = detect(m, eng)
                    out[m["id"]][eng] = {
                        "detected": bool(det),
                        "path": (det or {}).get("path"),
                        "ctx_native": (det or {}).get("ctx"),
                    }
            self._json({"roots": SCAN.get("roots", []), "matches": out})
        elif self.path == "/api/metrics":
            self._json(metrics_snapshot())
        elif self.path.startswith("/api/logs"):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            with LOCK:
                e = RUNNING.get((q.get("id") or [""])[0])
            if not e:
                self._json({"error": "no such server"}, 404)
                return
            try:
                lines = Path(e["log"]).read_text(errors="replace").splitlines()[-60:]
            except Exception:
                lines = []
            self._json({"id": e["id"], "lines": lines})
        elif self.path == "/api/downloads":
            with LOCK:
                self._json({did: {k: v for k, v in e.items() if k != "cancel"}
                            for did, e in DOWNLOADS.items()})
        elif self.path.startswith("/assets/"):
            root = (HERE / "web").resolve()
            f = (root / self.path.lstrip("/")).resolve()
            if not str(f).startswith(str(root)) or not f.is_file():
                self._json({"error": "not found"}, 404)
                return
            ctype = {"svg": "image/svg+xml", "png": "image/png",
                     "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(f.suffix.lstrip("."), "application/octet-stream")
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        cfg = self._read()
        if self.path == "/api/build":
            built = build(cfg)
            if "error" not in built:
                built["harness_line"] = harness_line(cfg, built)
                m = find_model(cfg.get("model_id", ""))
                if m:
                    built["download"] = m["recipes"].get(cfg.get("engine", ""), {}).get("download", {})
            self._json(built)
        elif self.path == "/api/launch":
            built = build(cfg)
            if "error" in built:
                self._json(built, 400)
                return
            rid = launch(cfg, built)
            self._json({"id": rid, "harness_line": harness_line(cfg, built)})
        elif self.path == "/api/download":
            m = find_model(cfg.get("model_id", ""))
            if not m or cfg.get("engine") not in m["recipes"]:
                self._json({"error": "no such model+engine"}, 400)
                return
            did, err = start_download(m, cfg["engine"])
            self._json({"id": did, "error": err})
        elif self.path == "/api/stop":
            self._json({"ok": stop_server(cfg.get("id", ""))})
        elif self.path == "/api/harness":
            built = build(cfg)
            if "error" in built:
                self._json(built, 400)
                return
            self._json({"line": open_harness(cfg, built)})
        else:
            self._json({"error": "not found"}, 404)


def open_window(url):
    if IS_WIN:
        for browser in ("msedge", "chrome"):
            try:
                subprocess.Popen(["cmd", "/c", "start", "", browser, f"--app={url}"])
                return
            except OSError:
                continue
        webbrowser.open(url)
    else:
        for browser in ("chromium", "google-chrome", "microsoft-edge"):
            if shutil.which(browser):
                subprocess.Popen([browser, f"--app={url}"])
                return
        webbrowser.open(url)


def main():
    ap = argparse.ArgumentParser(description="B70 model launcher")
    ap.add_argument("--port", type=int, default=SETTINGS.get("port", 7570))
    ap.add_argument("--no-open", action="store_true", help="don't open the window")
    args = ap.parse_args()
    scan()
    url = f"http://127.0.0.1:{args.port}"
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        open_window(url)  # already running: bring the window up instead of crashing
        return
    roots = ", ".join(SCAN.get("roots", [])) or "none"
    print(f"b70-launcher on {url}  (models: {len(RECIPES['models'])}, scan roots: {roots})")
    if not args.no_open:
        open_window(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        with LOCK:
            for e in DOWNLOADS.values():
                e["cancel"] = True
            for e in RUNNING.values():
                if e.get("proc") and e["proc"].poll() is None:
                    e["proc"].terminate()
        print("bye")


if __name__ == "__main__":
    main()
