"""Client commands and opt-out, context-aware configuration sync."""
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from . import catalog, paths, runtime
from .locks import STATE_LOCK


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


def open_harness(cfg, built):
    if built.get("status") != "running":
        raise ValueError("Wait for this server to be ready before opening a client.")
    client = cfg.get("harness") or catalog.SETTINGS.get("open_with", "pi")
    line = harness_line(cfg, built)
    if client == "none":
        return line
    if paths.IS_WIN:
        argv = catalog.SETTINGS["terminal_windows"] + [line]
    elif client == "webui":
        argv = ["xdg-open", "http://localhost:3000"]
    else:
        argv = None
        for name in ("kitty", "gnome-terminal", "xfce4-terminal", "x-terminal-emulator", "xterm"):
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
    runtime.TRANSIENT_PROCS.append(subprocess.Popen(argv))
    return line
