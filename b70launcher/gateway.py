"""Client gateway: agents opened from the launcher talk to the engine through
http://127.0.0.1:<daemon_port>/gw/<client>/<engine_port>/v1/..., so the
launcher can count requests and tokens per agent.

- Chat/completions requests are forwarded unchanged, except that a streamed
  request without `stream_options.include_usage` gets it added; the extra
  usage-only chunk is then dropped before it reaches the client.
- /v1/responses (Codex) is forwarded to engines with a native Responses API
  after merging developer messages into one system message; engines without
  one (OpenVINO OVMS) get a Responses <-> Chat Completions translation.
- Counts land on the live server entry (`entry["clients"]`) and in
  <state>/client-usage.json (all time, per day, per model).

Loopback only. Requests that carry an Origin header (a web page) are refused:
agents are CLI processes and never send one.
"""
import http.client
import json
import threading
import time

from . import paths
from .locks import LOCK, USAGE_LOCK
from . import runtime

CLIENTS = ("pi", "omp", "droid", "codex", "deepseek", "other")
NATIVE_RESPONSES = {"vllm", "llamacpp", "exl3"}
MAX_BODY = 32 << 20
USAGE_FILE = paths.DATA / "client-usage.json"
_DIRTY = {"pending": False}
_EMPTY = {"requests": 0, "tokens_in": 0, "tokens_out": 0, "errors": 0, "last_used": ""}


def gateway_url(daemon_port, client, engine_port):
    return f"http://127.0.0.1:{int(daemon_port)}/gw/{client}/{int(engine_port)}/v1"


# ---------------------------------------------------------------- accounting
def _load():
    try:
        data = json.loads(USAGE_FILE.read_text())
        if isinstance(data, dict):
            for key in ("clients", "days", "models"):
                data.setdefault(key, {})
            return data
    except (OSError, ValueError):
        pass
    return {"clients": {}, "days": {}, "models": {}}


_STORE = None


def _store():
    global _STORE
    if _STORE is None:
        _STORE = _load()
    return _STORE


def _bump(bucket, tin, tout, ok, now):
    bucket["requests"] = bucket.get("requests", 0) + 1
    bucket["tokens_in"] = bucket.get("tokens_in", 0) + tin
    bucket["tokens_out"] = bucket.get("tokens_out", 0) + tout
    bucket["errors"] = bucket.get("errors", 0) + (0 if ok else 1)
    bucket["last_used"] = now


def record(client, entry, tin, tout, ok):
    """One finished request: update the live entry and the persisted totals."""
    tin, tout = max(int(tin or 0), 0), max(int(tout or 0), 0)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    model = entry.get("served_name") or entry.get("model_name") or entry.get("model_id") or "unknown"
    with LOCK:
        live = runtime.RUNNING.get(entry.get("id", ""))
        if live is not None:
            _bump(live.setdefault("clients", {}).setdefault(client, dict(_EMPTY)), tin, tout, ok, now)
    with USAGE_LOCK:
        data = _store()
        _bump(data["clients"].setdefault(client, dict(_EMPTY)), tin, tout, ok, now)
        day = data["days"].setdefault(now[:10], {})
        _bump(day.setdefault(client, dict(_EMPTY)), tin, tout, ok, now)
        _bump(data["models"].setdefault(model, dict(_EMPTY)), tin, tout, ok, now)
        # keep a year of daily rows
        for old in sorted(data["days"])[:-366]:
            data["days"].pop(old, None)
        if not _DIRTY["pending"]:
            _DIRTY["pending"] = True
            timer = threading.Timer(3.0, flush)
            timer.daemon = True
            timer.start()


def flush():
    with USAGE_LOCK:
        if _STORE is None:
            return
        _DIRTY["pending"] = False
        try:
            paths.atomic_write(USAGE_FILE, json.dumps(_STORE, indent=1))
        except OSError:
            pass


def summary():
    """The `clients`, `today` and `by_model` parts of GET /api/usage."""
    with USAGE_LOCK:
        data = json.loads(json.dumps(_store()))
    return {"clients": data["clients"], "today": data["days"].get(time.strftime("%Y-%m-%d"), {}),
            "by_model": data["models"]}


# ------------------------------------------------------------------- routing
def _find_entry(engine_port):
    with LOCK:
        for e in runtime.RUNNING.values():
            if e.get("port") == engine_port and e.get("status") == "running":
                return dict(e)
    return None


def _error(handler, code, message):
    body = json.dumps({"error": {"message": message, "type": "b70_gateway"}}).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def handle(handler, method):
    """Serve one /gw/... request on the launcher's own HTTP server."""
    parts = handler.path.split("?", 1)[0].split("/")  # ['', 'gw', client, port, 'v1', ...]
    if handler.headers.get("Origin"):
        return _error(handler, 403, "The gateway is for local agents, not web pages.")
    if len(parts) < 5 or parts[4] != "v1" or parts[2] not in CLIENTS or not parts[3].isdigit():
        return _error(handler, 404, "Use /gw/<client>/<engine port>/v1/...")
    client, engine_port = parts[2], int(parts[3])
    rest = "/" + "/".join(parts[4:])
    query = "?" + handler.path.split("?", 1)[1] if "?" in handler.path else ""
    entry = _find_entry(engine_port)
    if entry is None:
        return _error(handler, 503, f"No ready model on port {engine_port}. Start it in B70 Launcher first.")
    body = b""
    if method == "POST":
        try:
            size = int(handler.headers.get("Content-Length", "0"))
        except ValueError:
            size = -1
        if not 0 <= size <= MAX_BODY:
            return _error(handler, 413, "Request body too large or missing Content-Length.")
        body = handler.rfile.read(size)
    try:
        req = json.loads(body) if body else {}
    except ValueError:
        req = None
    if method == "POST" and rest == "/v1/responses" and isinstance(req, dict):
        if entry.get("engine") in NATIVE_RESPONSES:
            return _responses_native(handler, client, entry, req)
        return _responses_translated(handler, client, entry, req)
    if method == "POST" and rest in ("/v1/chat/completions", "/v1/completions") and isinstance(req, dict):
        return _chat(handler, client, entry, rest, req)
    return _passthrough(handler, entry, method, rest + query, body)


# ------------------------------------------------------------------ upstream
def _upstream(entry, method, path, body, timeout=900):
    conn = http.client.HTTPConnection("127.0.0.1", entry["port"], timeout=timeout)
    headers = {"Content-Type": "application/json", "Accept": "*/*"}
    if body:
        headers["Content-Length"] = str(len(body))
    conn.request(method, path, body=body or None, headers=headers)
    return conn, conn.getresponse()


def _start_reply(handler, status, ctype, length=None):
    handler.send_response(status)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Cache-Control", "no-store")
    if length is None:
        handler.send_header("Connection", "close")
        handler.close_connection = True
    else:
        handler.send_header("Content-Length", str(length))
    handler.end_headers()


def _write(handler, data):
    handler.wfile.write(data)
    handler.wfile.flush()


def _passthrough(handler, entry, method, path, body):
    try:
        conn, resp = _upstream(entry, method, path, body, timeout=120)
    except OSError as exc:
        return _error(handler, 502, f"Engine on port {entry['port']} did not answer: {exc}")
    try:
        data = resp.read()
        _start_reply(handler, resp.status, resp.getheader("Content-Type", "application/json"), len(data))
        handler.wfile.write(data)
    finally:
        conn.close()


def _sse_lines(resp):
    """Yield (raw_line_bytes, parsed_json_or_None) for an upstream SSE body."""
    while True:
        line = resp.readline()
        if not line:
            return
        parsed = None
        text = line.strip()
        if text.startswith(b"data:"):
            payload = text[5:].strip()
            if payload and payload != b"[DONE]":
                try:
                    parsed = json.loads(payload)
                except ValueError:
                    parsed = None
        yield line, parsed


def _ok(client, path, resp):
    """True for HTTP 200; otherwise note it in the daemon output."""
    if resp.status != 200:
        print(f"gateway: {client} {path} -> engine HTTP {resp.status}", flush=True)
    return resp.status == 200


def _usage_tokens(usage):
    """(tokens_in, tokens_out) from chat or responses usage, or None."""
    if not isinstance(usage, dict):
        return None
    tin = usage.get("prompt_tokens", usage.get("input_tokens")) or 0
    tout = usage.get("completion_tokens", usage.get("output_tokens")) or 0
    return int(tin), int(tout)


def _chat(handler, client, entry, path, req):
    stream = bool(req.get("stream"))
    injected = False
    if stream:
        opts = req.get("stream_options") if isinstance(req.get("stream_options"), dict) else {}
        if not opts.get("include_usage"):
            req["stream_options"] = dict(opts, include_usage=True)
            injected = True
    body = json.dumps(req).encode()
    try:
        conn, resp = _upstream(entry, "POST", path, body)
    except OSError as exc:
        record(client, entry, 0, 0, False)
        return _error(handler, 502, f"Engine on port {entry['port']} did not answer: {exc}")
    tokens, ok, estimated = None, _ok(client, path, resp), 0
    try:
        ctype = resp.getheader("Content-Type", "")
        if not stream or "event-stream" not in ctype:
            data = resp.read()
            _start_reply(handler, resp.status, ctype or "application/json", len(data))
            handler.wfile.write(data)
            try:
                tokens = _usage_tokens(json.loads(data).get("usage"))
            except (ValueError, AttributeError):
                tokens = None
            return
        _start_reply(handler, resp.status, ctype)
        for line, chunk in _sse_lines(resp):
            if isinstance(chunk, dict):
                if chunk.get("usage"):
                    tokens = _usage_tokens(chunk["usage"]) or tokens
                    if injected and not chunk.get("choices"):
                        continue  # the client did not ask for the usage-only chunk
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content") or delta.get("tool_calls") or delta.get("reasoning_content"):
                        estimated += 1
            _write(handler, line)
    except (BrokenPipeError, ConnectionResetError):
        pass  # the agent cancelled the request (e.g. an aborted title call); not an engine error
    finally:
        conn.close()
        tin, tout = tokens if tokens else (0, estimated)
        record(client, entry, tin, tout, ok)


def _merge_developer(req):
    """Codex sends instructions plus developer messages; several chat templates
    accept only one leading system message (same fix as b70-codex-proxy)."""
    system = []
    if isinstance(req.get("instructions"), str) and req["instructions"].strip():
        system.append(req.pop("instructions"))
    rest = []
    items = req.get("input") if isinstance(req.get("input"), list) else None
    if items is None:
        return req
    for item in items:
        if isinstance(item, dict) and item.get("type", "message") == "message" and item.get("role") in ("developer", "system"):
            content = item.get("content")
            if isinstance(content, str):
                system.append(content)
            else:
                system.extend(c.get("text", "") for c in content or [] if isinstance(c, dict) and c.get("text"))
        else:
            rest.append(item)
    if system:
        rest.insert(0, {"type": "message", "role": "system",
                        "content": [{"type": "input_text", "text": "\n\n".join(system)}]})
    req["input"] = rest
    return req


def _responses_native(handler, client, entry, req):
    req = _merge_developer(req)
    req.pop("store", None)
    req.pop("include", None)
    body = json.dumps(req).encode()
    try:
        conn, resp = _upstream(entry, "POST", "/v1/responses", body)
    except OSError as exc:
        record(client, entry, 0, 0, False)
        return _error(handler, 502, f"Engine on port {entry['port']} did not answer: {exc}")
    tokens, ok = None, _ok(client, "/v1/responses", resp)
    try:
        ctype = resp.getheader("Content-Type", "")
        if "event-stream" not in ctype:
            data = resp.read()
            _start_reply(handler, resp.status, ctype or "application/json", len(data))
            handler.wfile.write(data)
            try:
                tokens = _usage_tokens(json.loads(data).get("usage"))
            except (ValueError, AttributeError):
                pass
            return
        _start_reply(handler, resp.status, ctype)
        for line, event in _sse_lines(resp):
            if isinstance(event, dict) and event.get("type") in ("response.completed", "response.incomplete"):
                tokens = _usage_tokens((event.get("response") or {}).get("usage")) or tokens
            elif isinstance(event, dict) and event.get("type") in ("response.failed", "error"):
                ok = False
            _write(handler, line)
    except (BrokenPipeError, ConnectionResetError):
        pass  # the agent cancelled the request; not an engine error
    finally:
        conn.close()
        record(client, entry, *(tokens or (0, 0)), ok)


def _sse(event):
    return b"event: " + event["type"].encode() + b"\ndata: " + json.dumps(event).encode() + b"\n\n"


def _responses_translated(handler, client, entry, req):
    from .responses_translate import ChatToResponsesStream, chat_completion_to_response, responses_to_chat
    model = entry.get("served_name") or req.get("model") or ""
    chat, tool_types = responses_to_chat(req, model)
    try:
        conn, resp = _upstream(entry, "POST", "/v1/chat/completions", json.dumps(chat).encode())
    except OSError as exc:
        record(client, entry, 0, 0, False)
        return _error(handler, 502, f"Engine on port {entry['port']} did not answer: {exc}")
    tokens, ok = None, _ok(client, "/v1/chat/completions (from /v1/responses)", resp)
    try:
        if resp.status != 200:
            data = resp.read()
            _start_reply(handler, resp.status, resp.getheader("Content-Type", "application/json"), len(data))
            handler.wfile.write(data)
            return
        if not chat.get("stream"):
            out = chat_completion_to_response(json.loads(resp.read()), req.get("model") or model, tool_types)
            tokens = _usage_tokens(out.get("usage"))
            data = json.dumps(out).encode()
            _start_reply(handler, 200, "application/json", len(data))
            handler.wfile.write(data)
            return
        state = ChatToResponsesStream(req.get("model") or model, tool_types)
        _start_reply(handler, 200, "text/event-stream")
        for event in state.start():
            _write(handler, _sse(event))
        for _line, chunk in _sse_lines(resp):
            if isinstance(chunk, dict):
                for event in state.feed(chunk):
                    _write(handler, _sse(event))
        for event in state.finish():
            _write(handler, _sse(event))
        _write(handler, b"data: [DONE]\n\n")
        tokens = _usage_tokens(state.usage)
    except (BrokenPipeError, ConnectionResetError):
        pass  # the agent cancelled the request; not an engine error
    finally:
        conn.close()
        record(client, entry, *(tokens or (0, 0)), ok)
