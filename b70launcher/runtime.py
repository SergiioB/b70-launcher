"""Owned server lifecycle. Slow process and HTTP work never holds LOCK."""
import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import catalog, hardware, paths
from .builders.common import rid_for
from .locks import LOCK, STATE_LOCK

RUNNING = {}
TRANSIENT_PROCS = []
LIVE_STATES = frozenset(("starting", "running"))
STOP_EVT = threading.Event()
SERVER = None
SHUTDOWN = {"started": False}


def docker_cmd(sock=""):
    return ["docker"] + (["-H", f"unix://{sock}"] if sock else [])


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def parse_port(value):
    """Validated TCP port; ValueError (HTTP 400) for anything else."""
    try:
        port = int(value)
    except (TypeError, ValueError):
        raise ValueError("invalid port") from None
    if not 1 <= port <= 65535:
        raise ValueError("invalid port")
    return port


def port_free(port):
    try:
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", int(port)))
        return True
    except OSError:
        return False


def persist_state():
    with LOCK:
        snap = {rid: {k: v for k, v in e.items() if k not in ("proc", "log_proc") and not k.startswith("_")}
                for rid, e in RUNNING.items() if e.get("status") != "stopped"}
    with STATE_LOCK:
        paths.atomic_write(paths.STATE_PATH, json.dumps(snap, indent=1))


def docker_containers_status(cnames, docker_sock=""):
    if not cnames:
        return {}
    try:
        res = subprocess.run(docker_cmd(docker_sock) + ["inspect", "--format", "{{json .}}", *cnames],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if res.returncode and "No such object" not in res.stderr and "No such container" not in res.stderr:
        return None
    out = {}
    for line in res.stdout.splitlines():
        try:
            obj = json.loads(line)
            state = obj.get("State", {})
            out[obj["Name"].lstrip("/")] = {"running": bool(state.get("Running")),
                                                "exitcode": state.get("ExitCode"), "object": obj}
        except (ValueError, KeyError, TypeError):
            continue
    return out


def engine_models(port, timeout=1):
    with urllib.request.urlopen(f"http://127.0.0.1:{int(port)}/v1/models", timeout=timeout) as response:
        obj = json.loads(response.read(1 << 20))
    return [x["id"] for x in obj.get("data", []) if isinstance(x, dict) and x.get("id")]


def preflight():
    with LOCK:
        live = [dict(e) for e in RUNNING.values() if e.get("status") in LIVE_STATES]
    return hardware.enrich_busy(hardware.hardware_preflight(),
                                [g for e in live for g in e.get("gpus", [])],
                                [e["cname"] for e in live if e.get("cname")],
                                [e.get("pid") for e in live if e.get("native")])


def launch(cfg, built):
    rid = built.get("rid") or rid_for(cfg)
    if cfg.get("dry_run"):
        if not port_free(cfg.get("port") or 8000):
            built.setdefault("warnings", []).append("Port is in use; choose another port before launching.")
        return rid
    if built.get("error"):
        raise ValueError(built["error"])
    port = int(cfg.get("port") or 8000)
    if not port_free(port):
        raise ValueError(f"Port {port} is already in use. Choose another port.")
    log = paths.LOGDIR / f"{rid}-{time.time_ns()}.log"
    entry = {"id": rid, "model_id": cfg.get("model_id") or cfg.get("model"),
             "model": built["model_name"], "model_name": built["model_name"],
             "served_name": built["served_name"], "engine": built["engine"], "kind": built["kind"],
             "cfg": dict(cfg), "gpus": list(built["gpus"]), "ctx": built["ctx"],
             "quant": built.get("quant", ""), "variant": bool(built.get("variant")),
             "found_name": built.get("found_name", ""), "custom": bool(built.get("custom")),
             "recipe_ver": built.get("recipe_ver", ""), "ctx_source": built.get("ctx_source", ""),
             "detected_path": built.get("detected_path"), "artifact_mib": built.get("artifact_mib"),
             "client": cfg.get("harness") or catalog.SETTINGS.get("open_with", "pi"),
             "cname": built.get("cname", ""), "docker_sock": built.get("docker_sock", ""),
             "native": built["native"], "endpoint": built["endpoint"], "port": port,
             "cmd": built["cmd"], "log": str(log), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
             "started_at": time.time(), "launch_ts": time.time(), "ready_at": None,
             "status": "starting", "phase": "starting process", "exitcode": None,
             "proc": None, "adopted": False, "tokens_in": 0, "tokens_out": 0, "requests": 0,
             "test": {"state": "pending"}}
    # Recipe identity is snapshotted above (quant/variant/found_name/recipe_ver/
    # ctx_source/client): later catalog, scan or settings edits must never
    # rewrite what an existing session is. Only served_name is refreshed, by
    # the engine itself, when the endpoint first answers.
    with LOCK:
        old = RUNNING.get(rid)
        if old and old.get("status") in LIVE_STATES | {"stopping", "stop_failed"}:
            raise ValueError("already running")
        if any(e.get("port") == port and e.get("status") in LIVE_STATES | {"stopping"}
               for e in RUNNING.values()):
            raise ValueError(f"Port {port} is reserved by another launch.")
        if any(e.get("status") in LIVE_STATES | {"stopping", "stop_failed"}
               and any(g in e.get("gpus", ()) for g in entry["gpus"])
               for e in RUNNING.values()):
            raise ValueError("Selected GPU is reserved by another launcher server.")
        RUNNING[rid] = entry
    persist_state()
    threading.Thread(target=_start, args=(entry, built), daemon=True).start()
    return rid


def _start(entry, built):
    try:
        if built["kind"] == "exl3":
            from .exl3_stack import ensure_exl3_stack
            entry["phase"] = "starting EXL3 Docker daemon"
            ensure_exl3_stack(entry["docker_sock"])
        image = "" if built.get("native") else built.get("effective_binary")
        if image:
            found = subprocess.run(docker_cmd(entry["docker_sock"]) + ["image", "inspect", image],
                                   capture_output=True, text=True, timeout=15)
            if found.returncode:
                # `docker run` would start a multi-GB pull with no disk-space check
                raise RuntimeError(f"Engine image is not on this machine: {image}. "
                                   f"Check free disk space, then run: docker pull {image}")
        env = dict(os.environ)
        env.update(built.get("env") or {})
        with LOCK:
            if entry["status"] != "starting":
                return
            with open(entry["log"], "ab") as log:
                log.write((built["cmd"] + "\n").encode())
                log.flush()
                entry["proc"] = subprocess.Popen(built["tokens"], env=env, stdout=log,
                                                  stderr=subprocess.STDOUT, cwd=paths.HERE,
                                                  start_new_session=True)
                entry["pid"] = entry["proc"].pid
            entry["phase"] = "loading weights"
        persist_state()
    except Exception as exc:
        with LOCK:
            if entry["status"] == "starting":
                entry.update(status="exited", exitcode=1, error=str(exc))
        with open(entry["log"], "ab") as log:
            log.write((f"Launch failed: {exc}\n").encode())
        persist_state()


def _public(entry):
    out = {k: v for k, v in entry.items()
           if k not in ("proc", "log_proc", "cfg") and not k.startswith(("_", "sess_"))}
    # explicit readiness: the endpoint answered /v1/models with model ids.
    # App-open success is tracked separately (client_open), never by "ready".
    out["ready"] = out.get("status") == "running"
    return out


def servers_snapshot():
    with LOCK:
        entries = list(RUNNING.values())
        for p in TRANSIENT_PROCS[:]:
            if p.poll() is not None:
                TRANSIENT_PROCS.remove(p)
    groups = {}
    for e in entries:
        if e.get("status") in LIVE_STATES and e.get("cname") and not e.get("native"):
            groups.setdefault(e.get("docker_sock", ""), []).append(e["cname"])
    statuses = {sock: docker_containers_status(names, sock) for sock, names in groups.items()}
    changed = False
    for e in entries:
        if e.get("status") not in LIVE_STATES:
            continue
        proc = e.get("proc")
        rc = proc.poll() if proc else None
        if e.get("native") and not proc:
            # Before Popen there is no pid yet: a "starting" entry is mid-launch
            # (env setup, spawn) and must not be reaped as "process exited" —
            # that flip both fakes a failure and re-opens the duplicate-start
            # window (a second launch would see status "exited" and proceed).
            alive = _pid_alive(e["pid"]) if e.get("pid") else e.get("status") == "starting"
        else:
            alive = rc is None
        if not e.get("native"):
            st = statuses.get(e.get("docker_sock", ""))
            container = st.get(e.get("cname")) if st is not None else None
            if container:
                alive, rc = container["running"], container["exitcode"]
            elif e.get("adopted") and st is not None:
                alive, rc = False, None
        if not alive:
            with LOCK:
                if e["status"] in LIVE_STATES:
                    e.update(status="exited", exitcode=rc, phase="process exited")
                    changed = True
            from .telemetry import record_usage
            record_usage(e, "exited")
            continue
        try:
            names = engine_models(e["port"], .5)
            ready = bool(names)
        except Exception:
            ready = False
        test = sync = False
        with LOCK:
            if e["status"] not in LIVE_STATES:
                continue
            if ready:
                e["status"] = "running"
                e["served_name"] = names[0]
                e["phase"] = "ready"
                if e.get("ready_at") is None:
                    e["ready_at"] = time.time()
                    e["load_s"] = round(e["ready_at"] - e.get("launch_ts", e["ready_at"]), 1)
                    changed = True
                    sync = True
                if e["test"].get("state") == "pending" and not e.get("cfg", {}).get("no_test"):
                    e["test"] = {"state": "running"}
                    test = True
            else:
                e["status"] = "starting"
                from .telemetry import _log_phase
                e["phase"] = _log_phase(e.get("log", "")) or e.get("phase", "loading weights")
        if sync:
            from .harness import sync_harness_configs
            sync_harness_configs(e["port"], e["served_name"], e.get("model_id"), e.get("engine"), e.get("ctx") or 32768)
        if test:
            threading.Thread(target=_auto_test, args=(e,), daemon=True).start()
        if ready and time.time() - e.get("_metrics_ts", 0) > 3:
            from .telemetry import _account_engine
            _account_engine(e["id"], e, time.time())
    if changed:
        persist_state()
    with LOCK:
        return [_public(e) for e in RUNNING.values() if e["status"] != "stopped"]


def test_prompt(cfg):
    import re
    with LOCK:
        entry = RUNNING.get(cfg.get("id", ""))
        if not entry and cfg.get("port") is not None:
            # a bare port (CLI `b70 ask`, POST /api/test_prompt {"port": …})
            # still resolves to this launcher's server when one owns the port
            try:
                entry = next((e for e in RUNNING.values()
                              if e.get("port") == parse_port(cfg.get("port"))), None)
            except ValueError:
                entry = None
    if entry and entry.get("status") != "running":
        # honest gating: never dial an endpoint that has not answered
        # /v1/models yet (or whose process already died)
        return {"ok": False, "answer": "", "reasoning": "", "finish_reason": "",
                "http_status": None, "elapsed_s": 0, "tok_s": None, "id": entry.get("id"),
                "error": f"Server {entry.get('id')} is {entry.get('status')} "
                         f"({entry.get('phase') or 'not ready'}); the endpoint is not ready yet."}
    port = parse_port(entry["port"] if entry else cfg.get("port", 8000))
    out = {"ok": False, "answer": "", "reasoning": "", "finish_reason": "", "http_status": None,
           "elapsed_s": 0, "tok_s": None, "error": ""}
    start = time.monotonic()
    try:
        names = engine_models(port, 2)
        model = cfg.get("model") or (names[0] if names else "")
        if not model:
            raise ValueError("Engine returned no model identifiers.")
        body = {"model": model, "messages": [{"role": "user", "content": cfg.get("prompt") or "Reply with exactly B70-OK."}],
                "max_tokens": int(cfg.get("max_tokens", 512)), "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as response:
            obj = json.loads(response.read(8 << 20))
            out["http_status"] = response.status
        choices = obj.get("choices") or []
        choice = choices[0] if choices else {}
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        for match in re.finditer(r"<think>(.*?)</think>", content, re.S):
            reasoning += match.group(1)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
        if "<think>" in content:
            answer, pending = content.split("<think>", 1)
            content, reasoning = answer, reasoning + pending
        out.update(answer=content.strip(), reasoning=reasoning.strip(), finish_reason=choice.get("finish_reason"),
                   model=model, tokens=int((obj.get("usage") or {}).get("completion_tokens") or 0))
        out["ok"] = bool(out["answer"])
        if not out["ok"]:
            out["error"] = "Engine returned reasoning without an answer." if reasoning else "Engine returned no answer text."
    except urllib.error.HTTPError as exc:
        out.update(http_status=exc.code, error=f"Engine on :{port} answered HTTP {exc.code}: {exc.read(2048).decode('utf-8', 'replace')}")
        exc.close()
    except Exception as exc:
        out["error"] = f"Test on :{port} failed: {exc}"
    out["elapsed_s"] = round(time.monotonic() - start, 3)
    if out.get("tokens") and out["elapsed_s"] > 0:
        out["tok_s"] = round(out["tokens"] / out["elapsed_s"], 2)
    return out


def _auto_test(entry):
    result = test_prompt({"id": entry["id"]})
    with LOCK:
        if RUNNING.get(entry["id"]) is entry:
            entry["test"] = dict(result, state="passed" if result["ok"] else "failed")
    persist_state()


def read_logs(rid, offset=None):
    with LOCK:
        entry = RUNNING.get(rid)
    if not entry:
        raise ValueError("no such server")
    try:
        with open(entry["log"], "rb") as stream:
            stream.seek(0, 2)
            end = stream.tell()
            start = max(0, end - 65536) if offset is None else min(max(0, int(offset)), end)
            stream.seek(start)
            data = stream.read(65536)
            pos = stream.tell()
        lines = data.decode("utf-8", "replace").splitlines()
        return {"id": rid, "offset": pos, "lines": lines[-200:] if offset is None else lines, "eof": pos >= end}
    except FileNotFoundError:
        return {"id": rid, "offset": 0, "lines": [], "eof": True}


def stop_server(rid):
    with LOCK:
        e = RUNNING.get(rid)
        if not e:
            return False
        if e["status"] == "stopped":
            return True
        e["status"] = "stopping"
    try:
        if e.get("cname") and not e.get("native"):
            gone = ("No such container", "No such object", "is already in progress")
            for action in (["stop", "--time", "5", e["cname"]], ["rm", e["cname"]]):
                result = subprocess.run(docker_cmd(e.get("docker_sock", "")) + action,
                                        capture_output=True, text=True, timeout=15)
                # --rm containers remove themselves right after `stop`
                if result.returncode and not any(s in result.stderr for s in gone):
                    raise RuntimeError(result.stderr.strip() or "Docker refused to stop container.")
        proc = e.get("proc")
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
        elif e.get("native") and _pid_alive(e.get("pid")):
            os.kill(int(e["pid"]), signal.SIGTERM)
            deadline = time.monotonic() + 5
            while _pid_alive(e["pid"]) and time.monotonic() < deadline:
                time.sleep(.1)
            if _pid_alive(e["pid"]):
                os.kill(int(e["pid"]), signal.SIGKILL)
        with LOCK:
            e.update(status="stopped", ended=time.strftime("%Y-%m-%d %H:%M:%S"))
        from .telemetry import record_usage
        record_usage(e, "stopped by user")
        persist_state()
        return True
    except Exception as exc:
        with LOCK:
            e.update(status="stop_failed", error=str(exc))
        persist_state()
        raise ValueError(f"Stop failed: {exc}") from exc


def adopt_containers():
    try:
        saved = json.loads(paths.STATE_PATH.read_text())
    except (OSError, ValueError):
        saved = {}
    for sock in hardware.recipe_sockets():
        try:
            res = subprocess.run(docker_cmd(sock) + ["ps", "--filter", "label=b70.launcher.rid", "--format", "{{.Names}}"],
                                 capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if res.returncode:
            continue
        states = docker_containers_status(res.stdout.split(), sock) or {}
        for cname, state in states.items():
            obj = state["object"]
            labels = obj.get("Config", {}).get("Labels") or {}
            rid = labels.get("b70.launcher.rid")
            try:
                port = int(labels["b70.launcher.port"])
                if not rid or not 1 <= port <= 65535:
                    continue
            except (ValueError, KeyError):
                continue
            entry = dict(saved.get(rid) or {})
            entry.update(id=rid, cname=cname, docker_sock=sock, port=port,
                         endpoint=f"http://127.0.0.1:{port}/v1", native=False, proc=None,
                         status="starting", adopted=True)
            entry.setdefault("model_id", labels.get("b70.launcher.model", ""))
            entry.setdefault("engine", labels.get("b70.launcher.engine", ""))
            entry.setdefault("model", entry["model_id"])
            entry.setdefault("model_name", entry["model"])
            entry.setdefault("served_name", entry["model"])
            entry.setdefault("gpus", [])
            entry.setdefault("test", {"state": "pending"})
            entry.setdefault("log", str(paths.LOGDIR / f"{rid}-adopted.log"))
            with LOCK:
                if rid not in RUNNING:
                    RUNNING[rid] = entry
            with open(entry["log"], "ab") as log:
                entry["log_proc"] = subprocess.Popen(docker_cmd(sock) + ["logs", "--follow", "--since", str(int(time.time())), cname],
                                                    stdout=log, stderr=subprocess.STDOUT)
    for rid, old in saved.items():
        if old.get("native") and _pid_alive(old.get("pid")):
            try:
                cmd = Path(f"/proc/{int(old['pid'])}/cmdline").read_bytes().replace(b"\0", b" ").decode()
                if "llama-server" not in cmd or str(old.get("port")) not in cmd:
                    continue
            except OSError:
                continue
            old.update(proc=None, adopted=True, status="starting")
            old.setdefault("test", {"state": "pending"})
            with LOCK:
                RUNNING.setdefault(rid, old)
    persist_state()


def graceful_shutdown(stop_engines=False):
    with LOCK:
        if SHUTDOWN["started"]:
            return
        SHUTDOWN["started"] = True
    from . import downloads
    downloads.cancel_all()
    if stop_engines:
        for rid in list(RUNNING):
            try:
                stop_server(rid)
            except ValueError:
                pass
    persist_state()
    STOP_EVT.set()
    if SERVER:
        SERVER.shutdown()


def monitor():
    while not STOP_EVT.wait(2.5):
        servers_snapshot()
