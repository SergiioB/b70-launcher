"""Engine counters and retained usage records."""
import glob
import json
import math
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from .paths import IS_WIN, USAGE_PATH, VERSION, atomic_write as _atomic_write
from .locks import LOCK, USAGE_LOCK
from .hardware import power_probe
from .runtime import RUNNING
UA = {"User-Agent": f"b70-launcher/{VERSION}"}

SESSION_FIELDS = ("id", "model", "engine", "port", "started", "load_s")


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


TOKS_RE = re.compile(r"([\d.]+)\s*tokens per second|throughput:\s*([\d.]+)\s*tokens/s", re.I)


LLAMA_RUN_RE = re.compile(r"prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens|eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens", re.I)


MEM_RES = [
    (re.compile(r"([\d.]+)\s*(MiB|GiB)\s*VRAM used", re.I), "vram used (log)"),
    (re.compile(r"model weights take\s*([\d.]+)\s*GiB", re.I), "weights GiB (log)"),
    (re.compile(r"buffer size =\s*([\d.]+)\s*(MiB|GiB)", re.I), "buffer (log)"),
]


PROM_RE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|NaN)\s*$")


LOAD_PHASES = [
    ("loading weights", r"load_tensors|loading (model|weights)|weights loaded|reading model"),
    ("compiling kernels", r"torch\.compile|compil(ing|ation)|jit|kernel build"),
    ("graph capture", r"captur|cuda ?graph|xpu graph|cudagraph|graph batch"),
    ("warmup", r"warm[- ]?up|warming"),
    ("serving", r"uvicorn running|application startup complete|srv\s+startup|start the main server"),
]


LOAD_PHASE_RES = [(name, re.compile(rx, re.I)) for name, rx in LOAD_PHASES]


def _log_phase(log_path):
    """Current engine load milestone from the log tail, or ""."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 16384))
            tail = f.read().decode("utf-8", "replace")
    except Exception:
        return ""
    phase = ""
    for name, rx in LOAD_PHASE_RES:
        if rx.search(tail):
            phase = name
    return phase


def _parse_prom(text):
    # same counter name can appear once per label set (finished_reason, model…);
    # token/request accounting needs the SUM across series
    counters = {}
    for line in text.splitlines():
        m = PROM_RE.match(line.strip())
        if m:
            try:
                value = float(m.group(2))
                if math.isfinite(value):
                    counters[m.group(1)] = counters.get(m.group(1), 0.0) + value
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
    return int(tin), int(tout), int(reqs)


def _engine_prometheus(entry, timeout=3):
    """Scrape the engine's own /metrics endpoint (same port as the API)."""
    port = entry.get("port")
    if not port:
        return None
    try:
        # Accept: text/plain is the Prometheus scrape ask; Strata's /metrics
        # answers JSON without it (vLLM and llama.cpp send text either way)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics",
                                     headers={**UA, "Accept": "text/plain"})
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
