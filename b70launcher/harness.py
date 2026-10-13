"""Client commands and opt-out, context-aware configuration sync."""
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from . import catalog, paths, runtime
from .locks import LOCK, STATE_LOCK


def sync_harness_configs(port=8000, model_name="", model_id="", engine="", ctx=32768):
    if not catalog.SETTINGS.get("sync_clients", True):
        return
    ctx = int(ctx)
    endpoint = f"http://127.0.0.1:{int(port)}/v1"
    model = {"id": model_name, "name": f"{model_name} (Local B70)", "contextWindow": ctx,
             "maxTokens": min(16384, max(1, ctx // 4)), "reasoning": True,
             "supportsTools": engine == "openvino"}
    pi = Path.home() / ".pi/agent/models.json"
    factory = Path.home() / ".factory/settings.json"
    omp = Path.home() / ".omp/agent/models.yml"
    for target in (pi, factory, omp):
        if not target.exists():
            continue
        try:
            with STATE_LOCK:
                if target == omp:
                    text = target.read_text()
                    lines = text.splitlines(keepends=True)
                    # Only replace this launcher's provider block, never scan across providers.
                    indices = [i for i, line in enumerate(lines) if re.fullmatch(r"\s*b70(?:-vllm)?:\s*\n?", line)]
                    if not indices:
                        continue
                    i = indices[0]
                    indent = len(lines[i]) - len(lines[i].lstrip())
                    j = i + 1
                    while j < len(lines):
                        stripped = lines[j].strip()
                        if stripped and not stripped.startswith('#') and len(lines[j]) - len(lines[j].lstrip()) <= indent:
                            break
                        j += 1
                    pad = ' ' * indent
                    quoted = json.dumps(model_name)
                    block = (f'{pad}b70:\n{pad}  baseUrl: {endpoint}\n{pad}  api: openai-completions\n'
                             f'{pad}  apiKey: local-b70\n{pad}  models:\n{pad}    - id: {quoted}\n'
                             f'{pad}      name: {json.dumps(model["name"])}\n{pad}      input: [text]\n'
                             f'{pad}      contextWindow: {ctx}\n{pad}      maxTokens: {model["maxTokens"]}\n'
                             f'{pad}      reasoning: true\n{pad}      supportsTools: {str(model["supportsTools"]).lower()}\n')
                    paths.atomic_write(target, ''.join(lines[:i]) + block + ''.join(lines[j:]))
                else:
                    data = json.loads(target.read_text())
                    if target == pi:
                        providers = data.setdefault("providers", {})
                        providers.pop("b70-vllm", None)
                        providers["b70"] = {"baseUrl": endpoint, "api": "openai-completions", "apiKey": "local-b70", "models": [model]}
                    else:
                        data.setdefault("models", {})["desktop-b70"] = {
                            "name": "Desktop B70 Loaded Model", "provider": "generic-chat-completion-api",
                            "modelId": model_name, "baseUrl": endpoint}
                        custom = data.setdefault("customModels", [])
                        custom[:] = [m for m in custom if m.get("id") != "custom:Desktop-B70-Loaded-Model-0"]
                        custom.insert(0, {"id": "custom:Desktop-B70-Loaded-Model-0", "model": model_name,
                                          "displayName": model["name"], "baseUrl": endpoint, "apiKey": "local-b70",
                                          "provider": "generic-chat-completion-api", "maxOutputTokens": model["maxTokens"],
                                          "noImageSupport": True})
                    paths.atomic_write(target, json.dumps(data, indent=2) + '\n')
        except (OSError, ValueError, TypeError) as exc:
            print(f"Client config unchanged ({target}): {exc}")


def harness_line(cfg, built, write_sync=False):
    port = int(cfg.get("port") or built.get("port") or 8000)
    name = built.get("served_name") or built.get("model_name") or cfg.get("model") or ""
    engine = built.get("engine") or cfg.get("engine") or ""
    endpoint = built.get("endpoint") or f"http://127.0.0.1:{port}/v1"
    client = cfg.get("harness") or catalog.SETTINGS.get("open_with", "pi")
    prefix = (f"OPENAI_BASE_URL={shlex.quote(endpoint)} OPENAI_API_KEY=local-b70 "
              f"OPENAI_MODEL_NAME={shlex.quote(name)} ") if not paths.IS_WIN else ""
    custom = cfg.get("harness_cmd")
    if custom:
        return prefix + str(custom)
    if client == "none":
        return prefix.strip()
    if client == "webui":
        return "xdg-open http://localhost:3000"
    if client in ("pi", "omp"):
        return prefix + client + " --model " + shlex.quote("b70/" + name) + (" --no-tools" if engine != "openvino" else "")
    if client == "droid":
        return prefix + "droid --model custom:Desktop-B70-Loaded-Model-0"
    raise ValueError("Choose Pi, omp, Droid, Web UI or no client.")


TERMINALS = ("kitty", "gnome-terminal", "xfce4-terminal", "x-terminal-emulator", "xterm")
_CLIENTS_CACHE = {"ts": 0.0, "data": {}}


def _display_ok():
    """A GUI is reachable (X11 or Wayland). Headless sessions cannot open
    terminals or browsers even when every binary is installed."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def clients_available():
    """Which target clients this machine can actually open right now.
    Cached briefly: /api/state polls this. webui is an external app the
    launcher does not manage — availability is a live port probe, cached."""
    import socket as _socket
    now = time.monotonic()
    if now - _CLIENTS_CACHE["ts"] < 10:
        return _CLIENTS_CACHE["data"]
    term = None
    for name in TERMINALS:
        binary = shutil.which(name)
        if binary:
            term = name
            break
    webui = False
    try:
        with _socket.create_connection(("127.0.0.1", 3000), timeout=.3):
            webui = True
    except OSError:
        pass
    display = _display_ok()
    data = {
        "pi": {"installed": bool(shutil.which("pi")), "openable": display},
        "omp": {"installed": bool(shutil.which("omp")), "openable": display},
        "droid": {"installed": bool(shutil.which("droid")), "openable": display},
        "webui": {"installed": webui, "openable": display,
                  "url": "http://localhost:3000",
                  "detail": "" if webui else "no server answered on localhost:3000"},
        "none": {"installed": True, "openable": True},
        "terminal": {"installed": bool(term), "name": term or ""},
        "display": {"available": display,
                    "reason": "" if display else "no DISPLAY/WAYLAND_DISPLAY in this session — "
                                                 "copy the client command instead"},
    }
    _CLIENTS_CACHE.update(ts=now, data=data)
    return data


def open_harness(cfg, built):
    """Open the target client for a READY server. Returns {line, opened,
    client}: opened is True only when a client process was actually spawned —
    server readiness (the caller's gate) and app-open success stay separate
    facts, and failures raise ValueError with the exact reason."""
    if built.get("status") != "running":
        raise ValueError("Wait for this server to be ready before opening a client.")
    client = cfg.get("harness") or catalog.SETTINGS.get("open_with", "pi")
    line = harness_line(cfg, built)
    if client == "none":
        return {"line": line, "opened": False, "client": "none"}
    if not _display_ok():
        raise ValueError("No display available in this session (headless) — the server is ready; "
                         "copy the client command and run it where you have a GUI.")
    if paths.IS_WIN:
        argv = catalog.SETTINGS["terminal_windows"] + [line]
    elif client == "webui":
        argv = ["xdg-open", "http://localhost:3000"]
    else:
        argv = None
        for name in TERMINALS:
            binary = shutil.which(name)
            if not binary:
                continue
            command = line + "; exec sh"
            if name == "kitty":
                argv = [binary, "sh", "-c", command]
            elif name == "gnome-terminal":
                argv = [binary, "--", "sh", "-c", command]
            elif name == "xfce4-terminal":
                argv = [binary, "-e", "sh -c " + shlex.quote(command)]
            else:
                argv = [binary, "-e", "sh", "-c", command]
            break
        if argv is None:
            raise ValueError("No terminal found. Copy the client command and run it in a terminal.")
    proc = subprocess.Popen(argv)
    runtime.TRANSIENT_PROCS.append(proc)
    try:
        # terminal emulators legitimately fork+exit 0; a nonzero immediate
        # exit is a real open failure and must not be reported as success
        rc = proc.wait(timeout=1.2)
    except subprocess.TimeoutExpired:
        rc = None
    with LOCK:
        entry = runtime.RUNNING.get(built.get("id", ""))
        if entry is not None:
            entry["client_open"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if rc not in (None, 0):
        raise ValueError(f"Client ({client}) exited immediately with code {rc} — "
                         "copy the client command and run it in a terminal.")
    return {"line": line, "opened": True, "client": client}
