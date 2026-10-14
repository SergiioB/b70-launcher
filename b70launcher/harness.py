"""Client commands and opt-out, context-aware configuration sync.

Agents opened from the launcher (Pi, OMP, Droid, Codex, DeepSeek) are pointed
at the launcher gateway (b70launcher/gateway.py) instead of the raw engine
port, so their requests and tokens are counted per agent. Set
`meter_clients: false` in settings to point them at the engine directly.
"""
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

CLIENT_IDS = ("pi", "omp", "droid", "codex", "deepseek", "webui", "none")
AGENTS = ("pi", "omp", "droid", "codex", "deepseek")
LABELS = {"pi": "Pi", "omp": "OMP", "droid": "Droid", "codex": "Codex", "deepseek": "DeepSeek",
          "webui": "Web UI", "none": "No app"}
BINARIES = {"pi": "pi", "omp": "omp", "droid": "droid", "codex": "codex", "deepseek": "npx"}
TOOL_MARKERS = ("--enable-auto-tool-choice", "tool_call_parser", "tool-call-parser", "--tool_parser")


def daemon_port():
    server = runtime.SERVER
    return server.server_port if server is not None else int(catalog.SETTINGS.get("port", 7570))


def client_endpoint(client, engine_port):
    """Base URL an agent should use: the metering gateway, or the engine."""
    if catalog.SETTINGS.get("meter_clients", True) and runtime.SERVER is not None:
        from .gateway import gateway_url
        return gateway_url(daemon_port(), client if client in AGENTS else "other", engine_port)
    return f"http://127.0.0.1:{int(engine_port)}/v1"


def supports_tools(entry):
    """Tool calls work when the engine was started with a tool parser, or for
    llama.cpp (Jinja chat templates are on by default)."""
    cmd = str(entry.get("cmd") or "")
    if any(m in cmd for m in TOOL_MARKERS):
        return True
    return entry.get("engine") == "llamacpp" and "--no-jinja" not in cmd


def _model_block(model_name, ctx, tools):
    return {"id": model_name, "name": f"{model_name} (Local B70)", "contextWindow": int(ctx),
            "maxTokens": min(16384, max(1, int(ctx) // 4)), "reasoning": True, "supportsTools": bool(tools)}


def _omp_block(pad, endpoint, model):
    return (f'{pad}b70:\n{pad}  baseUrl: {endpoint}\n{pad}  api: openai-completions\n'
            f'{pad}  apiKey: local-b70\n{pad}  models:\n{pad}    - id: {json.dumps(model["id"])}\n'
            f'{pad}      name: {json.dumps(model["name"])}\n{pad}      input: [text]\n'
            f'{pad}      contextWindow: {model["contextWindow"]}\n{pad}      maxTokens: {model["maxTokens"]}\n'
            f'{pad}      reasoning: true\n{pad}      supportsTools: {str(model["supportsTools"]).lower()}\n')


def _sync_omp(target, endpoint, model):
    text = target.read_text() if target.exists() else "providers:\n"
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    # Only replace this launcher's provider block, never scan across providers.
    indices = [i for i, line in enumerate(lines) if re.fullmatch(r"\s*b70(?:-vllm)?:\s*\n?", line)]
    if indices:
        i = indices[0]
        indent = len(lines[i]) - len(lines[i].lstrip())
        j = i + 1
        while j < len(lines):
            stripped = lines[j].strip()
            if stripped and not stripped.startswith('#') and len(lines[j]) - len(lines[j].lstrip()) <= indent:
                break
            j += 1
        new = ''.join(lines[:i]) + _omp_block(' ' * indent, endpoint, model) + ''.join(lines[j:])
    else:
        top = next((k for k, line in enumerate(lines) if re.fullmatch(r"providers:\s*\n?", line)), None)
        if top is None:
            new = ''.join(lines) + "providers:\n" + _omp_block("  ", endpoint, model)
        else:
            new = ''.join(lines[:top + 1]) + _omp_block("  ", endpoint, model) + ''.join(lines[top + 1:])
    paths.atomic_write(target, new)


def sync_harness_configs(port=8000, model_name="", model_id="", engine="", ctx=32768, entry=None):
    """Point each installed agent's B70 entry at this server. A config file is
    created only inside an agent directory that already exists."""
    if not catalog.SETTINGS.get("sync_clients", True):
        return
    tools = supports_tools(entry or {"engine": engine})
    model = _model_block(model_name, ctx, tools)
    home = Path.home()
    targets = {"pi": home / ".pi/agent/models.json", "droid": home / ".factory/settings.json",
               "omp": home / ".omp/agent/models.yml"}
    for client, target in targets.items():
        if not target.parent.is_dir():
            continue
        endpoint = client_endpoint(client, port)
        try:
            with STATE_LOCK:
                if client == "omp":
                    _sync_omp(target, endpoint, model)
                    continue
                data = json.loads(target.read_text()) if target.exists() else {}
                if client == "pi":
                    providers = data.setdefault("providers", {})
                    providers.pop("b70-vllm", None)
                    providers["b70"] = {"baseUrl": endpoint, "api": "openai-completions", "apiKey": "local-b70",
                                        "models": [model]}
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
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            print(f"Client config unchanged ({target}): {exc}")


def _client_argv(client, entry, endpoint, name, ctx):
    """argv + env additions for an agent; None for clients that use a shell line."""
    tools = supports_tools(entry)
    if client in ("pi", "omp"):
        return [client, "--model", "b70/" + name] + ([] if tools else ["--no-tools"]), {}
    if client == "droid":
        return ["droid", "--model", "custom:Desktop-B70-Loaded-Model-0"], {}
    if client == "codex":
        from .codex_client import codex_argv, codex_env
        return codex_argv(endpoint, name, int(ctx), min(16384, max(1, int(ctx) // 4))), codex_env({})
    if client == "deepseek":
        from .dsh_client import dsh_prepare
        return dsh_prepare(endpoint, name, int(ctx), paths.DATA / "dsh")
    return None


def harness_line(cfg, built, write_sync=False):
    port = int(cfg.get("port") or built.get("port") or 8000)
    name = built.get("served_name") or built.get("model_name") or cfg.get("model") or ""
    client = cfg.get("harness") or catalog.SETTINGS.get("open_with", "pi")
    endpoint = client_endpoint(client, port) if client in AGENTS else (built.get("endpoint") or f"http://127.0.0.1:{port}/v1")
    prefix = (f"OPENAI_BASE_URL={shlex.quote(endpoint)} OPENAI_API_KEY=local-b70 "
              f"OPENAI_MODEL_NAME={shlex.quote(name)} ") if not paths.IS_WIN else ""
    custom = cfg.get("harness_cmd")
    if custom:
        return prefix + str(custom)
    if client == "none":
        return prefix.strip()
    if client == "webui":
        return "xdg-open http://localhost:3000"
    if client not in AGENTS:
        raise ValueError("Choose Pi, OMP, Droid, Codex, DeepSeek, Web UI or no client.")
    argv, env = _client_argv(client, built, endpoint, name, built.get("ctx") or cfg.get("ctx") or 32768)
    extra = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
    return prefix + (extra + " " if extra else "") + shlex.join(argv)


TERMINALS = ("kitty", "gnome-terminal", "xfce4-terminal", "x-terminal-emulator", "xterm")
_CLIENTS_CACHE = {"ts": 0.0, "data": {}}


def _display_ok():
    """A GUI is reachable (X11 or Wayland). Headless sessions cannot open
    terminals or browsers even when every binary is installed."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _installed(client):
    if client == "deepseek":
        from .dsh_client import dsh_installed
        return dsh_installed()
    return bool(shutil.which(BINARIES[client]))


def clients_available():
    """Which target clients this machine can actually open right now.
    Cached briefly: /api/state polls this. webui is an external app the
    launcher does not manage — availability is a live port probe, cached."""
    import socket as _socket
    now = time.monotonic()
    if now - _CLIENTS_CACHE["ts"] < 10:
        return _CLIENTS_CACHE["data"]
    term = next((name for name in TERMINALS if shutil.which(name)), None)
    webui = False
    try:
        with _socket.create_connection(("127.0.0.1", 3000), timeout=.3):
            webui = True
    except OSError:
        pass
    display = _display_ok()
    data = {client: {"installed": _installed(client), "openable": display} for client in AGENTS}
    data.update({
        "webui": {"installed": webui, "openable": display,
                  "url": "http://localhost:3000",
                  "detail": "" if webui else "no server answered on localhost:3000"},
        "none": {"installed": True, "openable": True},
        "terminal": {"installed": bool(term), "name": term or ""},
        "display": {"available": display,
                    "reason": "" if display else "no DISPLAY/WAYLAND_DISPLAY in this session — "
                                                 "copy the client command instead"},
    })
    _CLIENTS_CACHE.update(ts=now, data=data)
    return data


def _terminal_argv(title, command):
    for name in TERMINALS:
        binary = shutil.which(name)
        if not binary:
            continue
        if name == "kitty":
            return [binary, "--title", title, "sh", "-c", command]
        if name == "gnome-terminal":
            return [binary, "--title", title, "--", "sh", "-c", command]
        if name == "xfce4-terminal":
            return [binary, "--title", title, "-e", "sh -c " + shlex.quote(command)]
        return [binary, "-e", "sh", "-c", command]
    return None


def open_harness(cfg, built):
    """Open the target client for a READY server. Returns {line, opened,
    client}: opened is True only when a client process was actually spawned —
    server readiness (the caller's gate) and app-open success stay separate
    facts, and failures raise ValueError with the exact reason."""
    if built.get("status") != "running":
        raise ValueError("Wait for this server to be ready before opening a client.")
    client = cfg.get("harness") or catalog.SETTINGS.get("open_with", "pi")
    if client in AGENTS and not _installed(client):
        hint = {"deepseek": "install Node.js (npx) — the DeepSeek harness runs as npx @deepseek-ai/dsh"}.get(
            client, f"`{BINARIES[client]}` is not on PATH")
        raise ValueError(f"{LABELS[client]} is not installed: {hint}.")
    if client in AGENTS:
        # the config files must match this server before the agent reads them
        sync_harness_configs(built["port"], built.get("served_name") or built.get("model_name") or "",
                             built.get("model_id"), built.get("engine"), built.get("ctx") or 32768, entry=built)
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
        # start in the user's work folder; keep the terminal open after the agent exits
        workdir = Path(os.path.expanduser(catalog.SETTINGS.get("agent_dir") or "~"))
        command = (f"cd {shlex.quote(str(workdir))} 2>/dev/null; {line}; "
                   f"printf '\\n%s exited. Press Enter to close.' {shlex.quote(LABELS.get(client, client))}; read _")
        argv = _terminal_argv(f"{LABELS.get(client, client)} · {built.get('served_name') or 'B70'}", command)
        if argv is None:
            raise ValueError("No terminal found. Copy the client command and run it in a terminal.")
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    runtime.TRANSIENT_PROCS.append(proc)
    try:
        # terminal emulators legitimately fork+exit 0; a nonzero immediate
        # exit is a real open failure and must not be reported as success
        rc = proc.wait(timeout=1.2)
    except subprocess.TimeoutExpired:
        rc = None
    if rc not in (None, 0):
        raise ValueError(f"Client ({client}) exited immediately with code {rc} — "
                         "copy the client command and run it in a terminal.")
    with LOCK:
        entry = runtime.RUNNING.get(built.get("id", ""))
        if entry is not None:
            entry["client_open"] = time.strftime("%Y-%m-%d %H:%M:%S")
            entry["client"] = client
    return {"line": line, "opened": True, "client": client}
