"""Download progress, resume, space accounting and HF auth against local HTTP servers.

A throttled ThreadingHTTPServer with HTTP Range support stands in for
huggingface.co (``downloads.HF`` points at it): it serves the resolve endpoint
and the model-info gating field; the repo listing (hf_files) is patched to the
served files. A second server on 127.0.0.2 plays the CDN for redirect tests.
Nothing touches the network, and no real HF token is ever read.
"""
import json
import os
import re
import shutil
import tempfile
import threading
import time
import unittest
import urllib.parse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

try:
    import support_env
except ImportError:
    from tests import support_env
from b70launcher import catalog, downloads, scan

REPO = "unsloth/Qwen3.8-27B-GGUF"  # an approved repo; dest dir = its basename
MODEL = {"id": "dl-test", "name": "Download Test",
         "recipes": {"llamacpp": {"download": {"repo": REPO, "quant": "Q4_K_M"}}}}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        srv = self.server
        srv.auth.append((self.path, self.headers.get("Authorization")))
        if self.path.startswith("/api/models/"):
            body = json.dumps({"gated": srv.gated}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        name = urllib.parse.unquote(self.path.split("/resolve/", 1)[1].split("/", 1)[1])
        rng = self.headers.get("Range")
        if name in srv.redirect:
            self.send_response(302)
            self.send_header("Location", srv.redirect[name])
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        data = srv.files.get(name)
        if data is None:
            self.send_error(404)
            return
        srv.requests.append((name, rng))
        if name in srv.deny:
            status, header = srv.deny[name]
            self.send_response(status)
            if header:
                self.send_header("x-error-code", header)
                self.send_header("x-error-message", "Access to model is restricted.")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start = 0
        if rng:
            start = int(re.fullmatch(r"bytes=(\d+)-", rng).group(1))
            if start >= len(data):
                self.send_error(416)
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            for i in range(0, len(body), srv.piece):
                self.wfile.write(body[i:i + srv.piece])
                self.wfile.flush()
                if srv.delay:
                    time.sleep(srv.delay)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client cancelled mid-transfer


def _payload(seed, size):
    return bytes((seed * 31 + i) % 251 for i in range(size))


def _server(host):
    server = ThreadingHTTPServer((host, 0), _Handler)
    server.daemon_threads = True
    server.files, server.requests, server.auth = {}, [], []
    server.redirect, server.deny, server.gated = {}, {}, False
    server.piece, server.delay = 4096, 0.0
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HOME", "XDG_CACHE_HOME")


class DownloadCase(support_env.LauncherStateCase):
    def setUp(self):
        super().setUp()
        self.server = _server("127.0.0.1")
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.models = support_env.TMPROOT / f"dl-{self.id().rsplit('.', 1)[-1]}"
        self.models.mkdir()
        self.addCleanup(shutil.rmtree, self.models, True)
        self.dest = self.models / REPO.split("/")[-1]
        self.hf = f"http://127.0.0.1:{self.server.server_port}"
        for patcher in (mock.patch.dict(catalog.SETTINGS, {"models_dir": str(self.models)}),
                        mock.patch.dict(os.environ),
                        mock.patch.object(downloads, "HF", self.hf),
                        mock.patch.object(downloads, "hf_files", side_effect=self._listing),
                        mock.patch.object(scan, "start_scan_async")):
            patcher.start()
            self.addCleanup(patcher.stop)
        for var in TOKEN_ENV:  # no real token reaches the test servers
            os.environ.pop(var, None)

    def _listing(self, repo, revision=None):
        return {name: len(data) for name, data in self.server.files.items()}

    def serve(self, **files):
        self.server.files.update({name.replace("_", "."): data for name, data in files.items()})

    def throttle(self, bytes_per_s, piece=4096):
        self.server.piece, self.server.delay = piece, piece / bytes_per_s

    def start(self):
        plan = downloads.plan_download(MODEL, "llamacpp")
        self.assertTrue(plan["enough"], plan)
        did, error = downloads.start_download(MODEL, "llamacpp", plan)
        self.assertIsNone(error)
        return did

    def wait_until(self, did, pred, timeout=20, every=0.05):
        seen = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            entry = downloads.snapshot()[did]
            seen.append(entry)
            if pred(entry):
                return entry, seen
            time.sleep(every)
        self.fail(f"timed out; last entry {seen[-1] if seen else None}")

    @staticmethod
    def terminal(entry):
        return entry["state"] in ("done", "cancelled", "error")


class TestProgressFields(DownloadCase):
    def test_fields_windowed_speed_across_files_and_decreasing_eta(self):
        nominal = 40 * 1024
        self.serve(a_bin=_payload(1, 32 * 1024), b_bin=_payload(2, 32 * 1024), c_bin=_payload(3, 32 * 1024))
        self.throttle(nominal)
        before = time.time()
        did = self.start()
        final, seen = self.wait_until(did, self.terminal)
        self.assertEqual(final["state"], "done", final.get("error"))
        live = [e for e in seen if e["state"] == "downloading"]
        self.assertTrue(live)
        for e in live:
            for key in ("id", "model", "model_id", "engine", "state", "done", "total", "pct", "speed",
                        "eta", "dest", "quant", "started", "finished", "file", "files_done",
                        "files_total", "resumed_bytes"):
                self.assertIn(key, e)
            self.assertFalse([k for k in e if k.startswith("_") or k == "cancel"])
            self.assertIsNone(e["finished"])
            self.assertEqual(e["files_total"], 3)
        self.assertEqual(final["model_id"], "dl-test")
        self.assertEqual(final["model"], "Download Test")
        self.assertEqual(final["quant"], "Q4_K_M")
        self.assertEqual(final["dest"], str(self.dest))
        # the current file is reported, in order, and files_done trails it
        order = []
        for e in live:
            if e["file"] and (not order or order[-1] != e["file"]):
                order.append(e["file"])
        self.assertEqual(order, ["a.bin", "b.bin", "c.bin"])
        names = ["a.bin", "b.bin", "c.bin"]
        self.assertTrue(all(e["files_done"] in (names.index(e["file"]), names.index(e["file"]) + 1)
                            for e in live if e["file"]))
        # speed: 0 for the first second, then a steady window average that
        # does not drop back to 0 when the next file starts
        self.assertEqual(live[0]["speed"], 0)
        warmed = [e for e in live if e["speed"] > 0]
        self.assertTrue(warmed)
        self.assertTrue(all(e["speed"] > 0 for e in live[live.index(warmed[0]):]))
        self.assertIn("c.bin", [e["file"] for e in warmed])
        first_c = next(e for e in live if e["file"] == "c.bin")
        self.assertGreater(first_c["speed"], 0)
        for e in warmed:
            self.assertLess(e["speed"], nominal * 3)
            self.assertGreater(e["speed"], nominal / 4)
        # eta tracks (total - done) / speed and falls as the transfer progresses
        etas = [e["eta"] for e in warmed]
        self.assertTrue(all(eta is not None for eta in etas))
        for e in warmed:
            self.assertAlmostEqual(e["eta"], (e["total"] - e["done"]) / e["speed"], delta=0.2)
        self.assertLess(etas[-1], etas[0])
        self.assertIsNone(live[0]["eta"])
        # terminal entry stays listed with completion info
        self.assertEqual((final["done"], final["total"], final["pct"]), (96 * 1024, 96 * 1024, 100))
        self.assertEqual((final["files_done"], final["files_total"]), (3, 3))
        self.assertEqual((final["eta"], final["speed"], final["file"]), (0, 0, None))
        self.assertEqual(final["resumed_bytes"], 0)
        self.assertGreaterEqual(final["started"], before)
        self.assertGreaterEqual(final["finished"], final["started"])
        self.assertIn(did, downloads.snapshot())
        for name, data in self.server.files.items():
            self.assertEqual((self.dest / name).read_bytes(), data)


class TestSpeedWindow(unittest.TestCase):
    """The speed is a sliding wall-clock window, independent of chunking."""

    def test_window_average_with_fake_clock(self):
        clock = [100.0]
        entry = {"done": 0, "total": 10_000, "_xfer": 0, "_samples": deque([(100.0, 0)])}
        with mock.patch.object(downloads.time, "monotonic", side_effect=lambda: clock[0]):
            def receive(at, n):
                clock[0] = at
                downloads._advance(entry, n)

            receive(100.5, 500)
            self.assertEqual(entry["speed"], 0)  # < 1 s of samples
            self.assertIsNone(entry["eta"])
            for step in range(1, 11):  # 100 B/s for 10 s, in 0.5 s ticks
                receive(100.5 + step * 0.5, 50)
            self.assertAlmostEqual(entry["speed"], 100, delta=5)
            self.assertAlmostEqual(entry["eta"], (10_000 - entry["done"]) / entry["speed"], delta=0.2)
            for step in range(1, 13):  # then 1000 B/s for 6 s
                receive(105.5 + step * 0.5, 500)
            # the old 100 B/s samples left the 5 s window entirely
            self.assertAlmostEqual(entry["speed"], 1000, delta=50)
            # a stall decays the live value (snapshot recomputes against now):
            # 2.5 s idle leaves 2.5 s of 1000 B/s in the 5 s window
            downloads._refresh(entry, 114.0)
            self.assertAlmostEqual(entry["speed"], 500, delta=10)
            downloads._refresh(entry, 116.5)
            self.assertEqual(entry["speed"], 0)
            self.assertIsNone(entry["eta"])


class TestResume(DownloadCase):
    def test_part_and_complete_files_are_resumed_and_counted(self):
        a, b, c = _payload(4, 20_000), _payload(5, 30_000), _payload(6, 25_000)
        self.serve(a_bin=a, b_bin=b, c_bin=c)
        support_env.write_file(self.dest / "a.bin", a)
        support_env.write_file(self.dest / "b.bin.part", b[:12_345])
        plan = downloads.plan_download(MODEL, "llamacpp")
        self.assertEqual(plan["size_bytes"], 75_000)
        self.assertEqual(plan["partial_bytes"], 20_000 + 12_345)
        self.assertEqual(plan["remaining_bytes"], 75_000 - 32_345)
        did, error = downloads.start_download(MODEL, "llamacpp", plan)
        self.assertIsNone(error)
        final, seen = self.wait_until(did, self.terminal)
        self.assertEqual(final["state"], "done", final.get("error"))
        self.assertEqual(final["resumed_bytes"], 32_345)
        self.assertEqual((final["done"], final["total"]), (75_000, 75_000))
        self.assertEqual((final["files_done"], final["files_total"]), (3, 3))
        self.assertEqual(self.server.requests, [("b.bin", "bytes=12345-"), ("c.bin", None)])
        for name, data in (("a.bin", a), ("b.bin", b), ("c.bin", c)):
            self.assertEqual((self.dest / name).read_bytes(), data)
        self.assertFalse((self.dest / "b.bin.part").exists())
        # already on disk → nothing left to fetch
        plan = downloads.plan_download(MODEL, "llamacpp")
        self.assertEqual((plan["partial_bytes"], plan["remaining_bytes"]), (75_000, 0))

    def test_enough_uses_remaining_bytes(self):
        self.serve(big_bin=_payload(7, 100_000))
        support_env.write_file(self.dest / "big.bin.part", self.server.files["big.bin"][:90_000])
        remaining = 10_000
        free = int(remaining * downloads.SPACE_HEADROOM) + 1
        self.assertLess(free, 100_000 * downloads.SPACE_HEADROOM)  # the full size would not fit
        with mock.patch.object(downloads, "_fs_free", return_value=free):
            plan = downloads.plan_download(MODEL, "llamacpp")
            self.assertEqual((plan["partial_bytes"], plan["remaining_bytes"]), (90_000, remaining))
            self.assertTrue(plan["enough"], plan)
            self.assertIsNone(plan["error"])
            did, error = downloads.start_download(MODEL, "llamacpp", plan)
            self.assertIsNone(error)
            final, _ = self.wait_until(did, self.terminal)
        self.assertEqual(final["state"], "done", final.get("error"))
        self.assertEqual(final["resumed_bytes"], 90_000)
        (self.dest / "big.bin").rename(self.dest / "big.bin.part")
        (self.dest / "big.bin.part").write_bytes(self.server.files["big.bin"][:90_000])
        with mock.patch.object(downloads, "_fs_free", return_value=remaining):
            plan = downloads.plan_download(MODEL, "llamacpp")
        self.assertFalse(plan["enough"])
        self.assertIn("not enough space", plan["error"])

    def test_oversized_part_resumes_nothing(self):
        self.serve(x_bin=_payload(8, 1000))
        support_env.write_file(self.dest / "x.bin.part", _payload(8, 1500))
        plan = downloads.plan_download(MODEL, "llamacpp")
        self.assertEqual((plan["partial_bytes"], plan["remaining_bytes"]), (0, 1000))
        did, _ = downloads.start_download(MODEL, "llamacpp", plan)
        final, _ = self.wait_until(did, self.terminal)
        self.assertEqual(final["state"], "error")
        self.assertIn("larger than the remote file", final["error"])
        self.assertEqual(self.server.requests, [])


class TestCancel(DownloadCase):
    def test_cancel_keeps_part_and_next_start_resumes_it(self):
        data = _payload(9, 96 * 1024)
        self.serve(slow_bin=data)
        self.throttle(32 * 1024)
        did = self.start()
        self.wait_until(did, lambda e: e["done"] >= 16 * 1024)
        self.assertTrue(downloads.cancel_download(did))
        final, _ = self.wait_until(did, self.terminal)
        self.assertEqual(final["state"], "cancelled")
        self.assertIsNotNone(final["finished"])
        self.assertEqual((final["speed"], final["eta"]), (0, None))
        part = self.dest / "slow.bin.part"
        self.assertTrue(part.exists())
        kept = part.stat().st_size
        self.assertGreater(kept, 0)
        self.assertLess(kept, len(data))
        self.assertEqual(final["done"], kept)
        self.assertEqual(part.read_bytes(), data[:kept])
        plan = downloads.plan_download(MODEL, "llamacpp")
        self.assertEqual((plan["partial_bytes"], plan["remaining_bytes"]), (kept, len(data) - kept))
        self.throttle(1 << 30, piece=1 << 16)
        did2, error = downloads.start_download(MODEL, "llamacpp", plan)
        self.assertEqual((did2, error), (did, None))
        final, _ = self.wait_until(did, lambda e: self.terminal(e) and e["finished"] != final["finished"])
        self.assertEqual(final["state"], "done", final.get("error"))
        self.assertEqual(final["resumed_bytes"], kept)
        self.assertEqual(self.server.requests[-1], ("slow.bin", f"bytes={kept}-"))
        self.assertEqual((self.dest / "slow.bin").read_bytes(), data)


class TestHFAccess(DownloadCase):
    def test_gated_plan_without_token_is_refused(self):
        self.serve(m_gguf=_payload(10, 1000))
        self.server.gated = "manual"
        plan = downloads.plan_download(MODEL, "llamacpp")
        repo_url = f"{self.hf}/{REPO}"
        self.assertEqual((plan["gated"], plan["hf_token"], plan["repo_url"]), ("manual", False, repo_url))
        self.assertFalse(plan["enough"])
        self.assertEqual(plan["error"], f"This model is gated on Hugging Face. Accept its terms at {repo_url}, "
                                        "sign in with `hf auth login` (or set HF_TOKEN), then try again.")
        self.assertEqual(plan["size_bytes"], 1000)  # the size is still shown
        self.assertEqual(downloads.start_download(MODEL, "llamacpp", plan), (None, plan["error"]))
        self.assertFalse(downloads.DOWNLOADS)
        os.environ["HF_TOKEN"] = "hf_test_secret"
        plan = downloads.plan_download(MODEL, "llamacpp")
        self.assertEqual((plan["gated"], plan["hf_token"], plan["enough"], plan["error"]),
                         ("manual", True, True, None))
        self.assertNotIn("hf_test_secret", json.dumps(plan))
        self.assertIn(("/api/models/" + REPO, "Bearer hf_test_secret"), self.server.auth)
        self.server.gated = False
        plan = downloads.plan_download(MODEL, "llamacpp")
        self.assertEqual((plan["gated"], plan["repo_url"]), (False, repo_url))

    def test_401_403_stop_after_one_attempt_and_keep_part(self):
        data = _payload(11, 4000)
        part = self.dest / "m.gguf.part"
        cases = (  # (status, x-error-code, token, error_code)
            (401, "GatedRepo", None, "gated"),
            (401, None, None, "auth"),
            (401, None, "hf_bad_token", "auth"),
            (403, None, "hf_test_secret", "gated"),
        )
        for status, header, token, kind in cases:
            with self.subTest(status=status, header=header, token=bool(token)):
                self.server.files, self.server.requests = {"m.gguf": data}, []
                self.server.deny = {"m.gguf": (status, header)}
                os.environ.pop("HF_TOKEN", None)
                if token:
                    os.environ["HF_TOKEN"] = token
                support_env.write_file(part, data[:1500])
                did = self.start()
                final, _ = self.wait_until(did, self.terminal, timeout=5)
                self.assertEqual(final["state"], "error")
                self.assertEqual(final["error_code"], kind)
                self.assertEqual(final["repo_url"], f"{self.hf}/{REPO}")
                self.assertEqual(final["error"], downloads.ACCESS_ERRORS[kind].format(url=f"{self.hf}/{REPO}"))
                self.assertEqual(self.server.requests, [("m.gguf", "bytes=1500-")])  # no retries
                self.assertEqual(part.read_bytes(), data[:1500])
                self.assertIsNotNone(final["finished"])
                if token:
                    self.assertNotIn(token, json.dumps(downloads.snapshot()))
        self.assertEqual(self.server.auth[-1], ("/" + REPO + "/resolve/main/m.gguf", "Bearer hf_test_secret"))

    def test_token_goes_to_hf_only_never_across_redirects(self):
        cdn = _server("127.0.0.2")
        self.addCleanup(cdn.server_close)
        self.addCleanup(cdn.shutdown)
        a, b = _payload(12, 3000), _payload(13, 5000)
        self.serve(a_bin=a, b_bin=b)
        cdn.files = {"b.bin": b}
        self.server.redirect = {
            "a.bin": f"/{REPO}/resolve/main/a.bin.real",  # same HF origin: keeps auth
            "b.bin": f"http://127.0.0.2:{cdn.server_port}/{REPO}/resolve/main/b.bin",  # CDN: drops it
        }
        self.server.files["a.bin.real"] = a
        os.environ["HF_TOKEN"] = "hf_test_secret"
        with mock.patch.object(downloads, "hf_files", return_value={"a.bin": 3000, "b.bin": 5000}):
            did = self.start()
            final, _ = self.wait_until(did, self.terminal)
        self.assertEqual(final["state"], "done", final.get("error"))
        self.assertEqual(((self.dest / "a.bin").read_bytes(), (self.dest / "b.bin").read_bytes()), (a, b))
        hf_auth = dict(self.server.auth)
        self.assertEqual(hf_auth[f"/{REPO}/resolve/main/a.bin"], "Bearer hf_test_secret")
        self.assertEqual(hf_auth[f"/{REPO}/resolve/main/a.bin.real"], "Bearer hf_test_secret")
        self.assertEqual(hf_auth[f"/{REPO}/resolve/main/b.bin"], "Bearer hf_test_secret")
        self.assertEqual(cdn.auth, [(f"/{REPO}/resolve/main/b.bin", None)])
        self.assertEqual(cdn.requests, [("b.bin", None)])

    def test_token_is_never_sent_to_other_hosts(self):
        os.environ["HF_TOKEN"] = "hf_test_secret"
        cdn = _server("127.0.0.2")
        self.addCleanup(cdn.server_close)
        self.addCleanup(cdn.shutdown)
        with downloads._open(f"http://127.0.0.2:{cdn.server_port}/api/models/x", token=downloads.hf_token()):
            pass
        self.assertEqual(cdn.auth, [("/api/models/x", None)])


class TestTokenDiscovery(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=support_env.TMPROOT))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        env = mock.patch.dict(os.environ, {"HOME": str(self.tmp / "home")})
        env.start()
        self.addCleanup(env.stop)
        for var in TOKEN_ENV:
            os.environ.pop(var, None)

    def test_lookup_order(self):
        self.assertIsNone(downloads.hf_token())
        support_env.write_file(self.tmp / "home/.cache/huggingface/token", b"home-cache\n")
        self.assertEqual(downloads.hf_token(), "home-cache")
        support_env.write_file(self.tmp / "xdg/huggingface/token", b"xdg-cache\n")
        os.environ["XDG_CACHE_HOME"] = str(self.tmp / "xdg")
        self.assertEqual(downloads.hf_token(), "xdg-cache")  # XDG_CACHE_HOME honored
        support_env.write_file(self.tmp / "hfhome/token", b"  hf-home  \n")
        os.environ["HF_HOME"] = str(self.tmp / "hfhome")
        self.assertEqual(downloads.hf_token(), "hf-home")
        os.environ["HUGGING_FACE_HUB_TOKEN"] = "hub-env"
        self.assertEqual(downloads.hf_token(), "hub-env")
        os.environ["HF_TOKEN"] = "hf-env"
        self.assertEqual(downloads.hf_token(), "hf-env")  # env over every file
        os.environ["HF_TOKEN"] = "  "
        self.assertEqual(downloads.hf_token(), "hub-env")  # blank values are skipped
        (self.tmp / "hfhome/token").write_text("")
        os.environ.pop("HUGGING_FACE_HUB_TOKEN")
        os.environ.pop("HF_TOKEN")
        self.assertEqual(downloads.hf_token(), "xdg-cache")  # empty file falls through



if __name__ == "__main__":
    unittest.main()
