"""Client gateway: real HTTP through the launcher handler to a mock engine.

Catches: usage not counted per agent, the injected usage-only chunk leaking
to a client that did not ask for it, the persisted totals not written, web
pages using the gateway, and Codex Responses usage not counted.
"""
import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import support_env
except ImportError:
    from tests import support_env

from b70launcher import gateway, http_api, runtime

SEEN = []


class MockEngine(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append((self.path, body))
        self.send_response(200)
        if self.path == "/v1/chat/completions" and body.get("stream"):
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            chunks = [{"choices": [{"index": 0, "delta": {"content": "Hel"}}]},
                      {"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]}]
            if (body.get("stream_options") or {}).get("include_usage"):
                chunks.append({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 2}})
            for c in chunks:
                self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.close_connection = True
            return
        if self.path == "/v1/responses":
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            done = {"type": "response.completed",
                    "response": {"output": [], "usage": {"input_tokens": 40, "output_tokens": 7}}}
            self.wfile.write(b"event: response.completed\ndata: " + json.dumps(done).encode() + b"\n\n")
            self.close_connection = True
            return
        data = json.dumps({"choices": [{"message": {"content": "ok"}}],
                           "usage": {"prompt_tokens": 5, "completion_tokens": 1}}).encode()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class TestGateway(support_env.LauncherStateCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ThreadingHTTPServer(("127.0.0.1", 0), MockEngine)
        cls.launcher = ThreadingHTTPServer(("127.0.0.1", 0), http_api.Handler)
        for srv in (cls.engine, cls.launcher):
            threading.Thread(target=srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        for srv in (cls.engine, cls.launcher):
            srv.shutdown()
            srv.server_close()

    def setUp(self):
        super().setUp()
        SEEN.clear()
        self.eport = self.engine.server_port
        runtime.RUNNING["gw1"] = {"id": "gw1", "status": "running", "port": self.eport, "engine": "llamacpp",
                                  "served_name": "mock-model", "model_id": "m"}
        gateway.USAGE_FILE.unlink(missing_ok=True)
        gateway._STORE = None

    def _post(self, client, path, body, headers=None):
        url = f"http://127.0.0.1:{self.launcher.server_port}/gw/{client}/{self.eport}{path}"
        req = urllib.request.Request(url, json.dumps(body).encode(),
                                     {"Content-Type": "application/json", **(headers or {})})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode()

    def test_streamed_chat_counts_tokens_and_hides_injected_usage(self):
        text = self._post("pi", "/v1/chat/completions", {"model": "mock-model", "stream": True,
                                                          "messages": [{"role": "user", "content": "hi"}]})
        self.assertIn("Hel", text)
        self.assertNotIn("prompt_tokens", text)  # pi did not ask for the usage chunk
        self.assertTrue(SEEN[0][1]["stream_options"]["include_usage"])
        live = runtime.RUNNING["gw1"]["clients"]["pi"]
        self.assertEqual((live["requests"], live["tokens_in"], live["tokens_out"]), (1, 11, 2))
        gateway.flush()
        saved = json.loads(gateway.USAGE_FILE.read_text())
        self.assertEqual(saved["clients"]["pi"]["tokens_out"], 2)
        self.assertEqual(saved["models"]["mock-model"]["requests"], 1)

    def test_client_that_asks_for_usage_still_gets_it(self):
        text = self._post("omp", "/v1/chat/completions", {"stream": True, "stream_options": {"include_usage": True},
                                                           "messages": []})
        self.assertIn("completion_tokens", text)

    def test_codex_responses_usage_counted_and_developer_merged(self):
        self._post("codex", "/v1/responses", {"stream": True, "instructions": "be brief", "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "rules"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}]})
        sent = SEEN[0][1]["input"]
        self.assertEqual(sent[0]["role"], "system")
        self.assertEqual(sent[0]["content"][0]["text"], "be brief\n\nrules")
        self.assertEqual([i["role"] for i in sent], ["system", "user"])
        codex = runtime.RUNNING["gw1"]["clients"]["codex"]
        self.assertEqual((codex["tokens_in"], codex["tokens_out"]), (40, 7))

    def test_codex_on_engine_without_responses_is_translated(self):
        runtime.RUNNING["gw1"]["engine"] = "openvino"
        text = self._post("codex", "/v1/responses", {"stream": True, "model": "mock-model", "input": "hi"})
        self.assertEqual(SEEN[0][0], "/v1/chat/completions")
        events = [json.loads(l[5:]) for l in text.splitlines() if l.startswith("data:") and l[5:].strip() != "[DONE]"]
        deltas = "".join(e.get("delta", "") for e in events if e["type"] == "response.output_text.delta")
        self.assertEqual(deltas, "Hello")
        self.assertEqual(events[-1]["response"]["usage"]["output_tokens"], 2)
        self.assertEqual(runtime.RUNNING["gw1"]["clients"]["codex"]["tokens_in"], 11)

    def test_web_page_origin_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post("pi", "/v1/chat/completions", {"messages": []}, {"Origin": "https://evil.example"})
        self.assertEqual(ctx.exception.code, 403)
        self.assertEqual(SEEN, [])

    def test_no_ready_model_on_port(self):
        runtime.RUNNING["gw1"]["status"] = "starting"
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post("pi", "/v1/chat/completions", {"messages": []})
        self.assertEqual(ctx.exception.code, 503)


if __name__ == "__main__":
    unittest.main()
