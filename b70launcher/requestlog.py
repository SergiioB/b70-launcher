"""Per-request speed and energy records for requests that pass through the gateway.

Every finished request becomes one JSON line in ``<state>/request-log.jsonl``:

* ``ttft_s``      seconds to the first generated token (streaming only)
* ``decode_tps``  generated tokens per second after the first token (streaming only);
                  ``overall_tps`` is tokens over wall time and is the only speed a
                  non-streaming request has
* ``prefill_tps`` prompt tokens per second of time-to-first-token (includes queueing and
                  any prefix-cache hit, so a cached prompt reads fast)
* ``card_j``      joules the engine's card(s) drew during the request, from the xe hwmon
                  energy counter. It is the whole card, so a second job on the same card
                  inflates it; ``j_per_ktok`` is derived from it.

The file is a ring: it is trimmed to the newest ``KEEP`` lines when it passes ``TRIM_AT``.
Nothing here ever raises into the request path.
"""
import json
import statistics
import threading
import time

from . import paths

LOG_PATH = paths.DATA / "request-log.jsonl"
KEEP, TRIM_AT = 2000, 2600
_LOCK = threading.Lock()


def _energy_uj(gpus):
    try:
        from . import hardware
        return hardware.energy_uj(gpus)
    except Exception:
        return None


class Span:
    """Timing and energy of one gateway request."""

    def __init__(self, entry):
        self.entry = entry
        self.gpus = list(entry.get("gpus") or [])
        self.t0 = time.monotonic()
        self.first = None
        self.e0 = _energy_uj(self.gpus)
        self.stream = False

    def mark_first(self):
        if self.first is None:
            self.first = time.monotonic()

    def finish(self, client, tin, tout, ok):
        t1 = time.monotonic()
        wall = max(t1 - self.t0, 1e-6)
        e1 = _energy_uj(self.gpus)
        rec = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "client": client,
            "model": self.entry.get("served_name") or self.entry.get("model_name") or self.entry.get("model_id"),
            "engine": self.entry.get("engine"),
            "port": self.entry.get("port"),
            "gpus": self.gpus,
            "ctx": self.entry.get("ctx"),
            "ok": bool(ok),
            "stream": self.first is not None,
            "tokens_in": int(tin or 0),
            "tokens_out": int(tout or 0),
            "wall_s": round(wall, 3),
        }
        if self.first is not None:
            ttft = max(self.first - self.t0, 1e-6)
            rec["ttft_s"] = round(ttft, 3)
            if rec["tokens_in"]:
                rec["prefill_tps"] = round(rec["tokens_in"] / ttft, 1)
            decode_s = t1 - self.first
            if rec["tokens_out"] > 1 and decode_s > 0.05:
                rec["decode_tps"] = round((rec["tokens_out"] - 1) / decode_s, 1)
        if rec["tokens_out"]:
            rec["overall_tps"] = round(rec["tokens_out"] / wall, 1)
        if self.e0 is not None and e1 is not None and e1 >= self.e0:
            joules = (e1 - self.e0) / 1e6
            rec["card_j"] = round(joules, 1)
            rec["avg_w"] = round(joules / wall, 1)
            if rec["tokens_out"]:
                rec["j_per_ktok"] = round(joules / rec["tokens_out"] * 1000, 1)
        return rec


def append(rec):
    """Append one record; trim the ring when it grows past TRIM_AT lines."""
    try:
        with _LOCK:
            paths.DATA.mkdir(parents=True, exist_ok=True)
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            try:
                if LOG_PATH.stat().st_size > TRIM_AT * 260:
                    lines = LOG_PATH.read_text().splitlines()
                    if len(lines) > TRIM_AT:
                        paths.atomic_write(LOG_PATH, "\n".join(lines[-KEEP:]) + "\n")
            except OSError:
                pass
    except OSError:
        pass


def load(limit=None):
    try:
        lines = LOG_PATH.read_text().splitlines()
    except OSError:
        return []
    if limit:
        lines = lines[-limit:]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def _pct(vals, q):
    vals = sorted(vals)
    if not vals:
        return None
    k = (len(vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return round(vals[lo] + (vals[hi] - vals[lo]) * (k - lo), 2)


def summary(recent=25):
    """Per model+engine speed statistics plus the newest requests."""
    recs = load()
    groups = {}
    for r in recs:
        if not r.get("ok"):
            continue
        groups.setdefault((r.get("model"), r.get("engine")), []).append(r)
    rows = []
    for (model, engine), rs in groups.items():
        dec = [r["decode_tps"] for r in rs if r.get("decode_tps")]
        ttft = [r["ttft_s"] for r in rs if r.get("ttft_s")]
        pre = [r["prefill_tps"] for r in rs if r.get("prefill_tps")]
        energy = [r for r in rs if r.get("card_j") is not None and r.get("tokens_out")]
        tout = sum(r["tokens_out"] for r in rs)
        row = {
            "model": model, "engine": engine, "requests": len(rs),
            "tokens_in": sum(r["tokens_in"] for r in rs), "tokens_out": tout,
            "streamed": len(dec),
            "decode_tps_median": _pct(dec, .5), "decode_tps_p10": _pct(dec, .1), "decode_tps_p90": _pct(dec, .9),
            "decode_tps_best": max(dec) if dec else None,
            "ttft_s_median": _pct(ttft, .5), "ttft_s_p90": _pct(ttft, .9),
            "prefill_tps_median": _pct(pre, .5),
            "last": max(r["ts"] for r in rs),
        }
        if energy:
            j = sum(r["card_j"] for r in energy)
            t = sum(r["tokens_out"] for r in energy)
            row["j_per_ktok"] = round(j / t * 1000, 1) if t else None
            row["avg_w"] = round(statistics.fmean(r["avg_w"] for r in energy if r.get("avg_w")), 1)
        rows.append(row)
    rows.sort(key=lambda r: r["last"], reverse=True)
    return {"by_model": rows, "recent": recs[-recent:][::-1], "total_requests": len(recs)}
