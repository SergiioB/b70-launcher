"""Per-request speed records: timing maths, the summary, and the gateway end to end."""
import json
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

try:
    import support_env
except ImportError:
    from tests import support_env

from b70launcher import gateway, http_api, requestlog, runtime


class SlowEngine(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        time.sleep(0.30)                                   # prefill
        role = {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}
        self.wfile.write(b"data: " + json.dumps(role).encode() + b"\n\n")
        self.wfile.flush()
        for i in range(5):                                 # five tokens, 40 ms apart
            time.sleep(0.04)
            c = {"choices": [{"index": 0, "delta": {"content": f"t{i}"}}]}
            self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            self.wfile.flush()
        u = {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 5}}
        self.wfile.write(b"data: " + json.dumps(u).encode() + b"\n\ndata: [DONE]\n\n")
        self.close_connection = True


class TestSpan(unittest.TestCase):
    def test_streaming_record_has_ttft_decode_and_energy(self):
        clock = iter([100.0, 100.5, 102.5])     # t0, first token, finish
        energy = iter([1_000_000, 51_000_000])  # 50 J drawn
        with mock.patch.object(requestlog.time, "monotonic", lambda: next(clock)), \
                mock.patch.object(requestlog, "_energy_uj", lambda g: next(energy)):
            span = requestlog.Span({"gpus": [0], "served_name": "m", "engine": "vllm", "port": 8000})
            span.mark_first()
            rec = span.finish("pi", 1000, 101, True)
        self.assertEqual(rec["ttft_s"], 0.5)
        self.assertEqual(rec["decode_tps"], 50.0)          # 100 tokens over 2.0 s after the first
        self.assertEqual(rec["prefill_tps"], 2000.0)
        self.assertEqual(rec["card_j"], 50.0)
        self.assertEqual(rec["avg_w"], 20.0)
        self.assertEqual(rec["j_per_ktok"], round(50 / 101 * 1000, 1))
        self.assertTrue(rec["stream"])

    def test_non_streaming_has_overall_speed_only(self):
        clock = iter([10.0, 12.0])
        with mock.patch.object(requestlog.time, "monotonic", lambda: next(clock)), \
                mock.patch.object(requestlog, "_energy_uj", lambda g: None):
            rec = requestlog.Span({"gpus": []}).finish("omp", 10, 40, True)
        self.assertNotIn("ttft_s", rec)
        self.assertNotIn("decode_tps", rec)
        self.assertEqual(rec["overall_tps"], 20.0)
        self.assertNotIn("card_j", rec)

    def test_summary_groups_by_model_and_ignores_failures(self):
        requestlog.LOG_PATH.unlink(missing_ok=True)
        for tps in (40.0, 50.0, 60.0):
            requestlog.append({"ts": "2026-11-12 10:00:00", "model": "M", "engine": "e", "ok": True,
                               "tokens_in": 10, "tokens_out": 100, "decode_tps": tps, "ttft_s": 0.2,
                               "card_j": 100.0, "avg_w": 100.0})
        requestlog.append({"ts": "2026-11-12 10:01:00", "model": "M", "engine": "e", "ok": False,
                           "tokens_in": 0, "tokens_out": 0})
        s = requestlog.summary()
        row = s["by_model"][0]
        self.assertEqual((row["requests"], row["decode_tps_median"], row["decode_tps_best"]), (3, 50.0, 60.0))
        self.assertEqual(row["j_per_ktok"], 1000.0)        # 300 J over 300 tokens
        self.assertEqual(s["total_requests"], 4)
        self.assertEqual(s["recent"][0]["ok"], False)      # newest first

    def test_ring_is_trimmed(self):
        requestlog.LOG_PATH.unlink(missing_ok=True)
        with mock.patch.object(requestlog, "TRIM_AT", 30), mock.patch.object(requestlog, "KEEP", 10):
            for i in range(80):
                requestlog.append({"ts": "t", "model": "M", "engine": "e", "ok": True, "tokens_in": 1,
                                   "tokens_out": i, "pad": "x" * 240})
        self.assertLessEqual(len(requestlog.load()), 30)


class TestGatewayRecords(support_env.LauncherStateCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ThreadingHTTPServer(("127.0.0.1", 0), SlowEngine)
        cls.launcher = ThreadingHTTPServer(("127.0.0.1", 0), http_api.Handler)
        for srv in (cls.engine, cls.launcher):
            threading.Thread(target=srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        for srv in (cls.engine, cls.launcher):
            srv.shutdown()
            srv.server_close()

    def test_streamed_request_is_logged_with_real_timings(self):
        requestlog.LOG_PATH.unlink(missing_ok=True)
        port = self.engine.server_port
        runtime.RUNNING["rl1"] = {"id": "rl1", "status": "running", "port": port, "engine": "vllm",
                                  "served_name": "slow", "model_id": "m", "gpus": [0]}
        gateway.USAGE_FILE.unlink(missing_ok=True)
        gateway._STORE = None
        url = f"http://127.0.0.1:{self.launcher.server_port}/gw/pi/{port}/v1/chat/completions"
        req = urllib.request.Request(url, json.dumps({"stream": True, "messages": []}).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        recs = requestlog.load()
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual((r["model"], r["engine"], r["tokens_in"], r["tokens_out"]), ("slow", "vllm", 100, 5))
        self.assertTrue(r["stream"])
        # the role-only chunk at ~0.30 s is not a token; the first text arrives ~0.34 s
        self.assertGreaterEqual(r["ttft_s"], 0.33)
        self.assertLess(r["ttft_s"], 1.0)
        self.assertGreater(r["decode_tps"], 10)
        self.assertLess(r["decode_tps"], 200)


if __name__ == "__main__":
    unittest.main()
