"""End-to-end smoke tests against a real launcher.py subprocess.

The server runs on an ephemeral high port with HOME/XDG_STATE_HOME pointed
at a temp dir and PATH masked to an empty dir (no docker/sudo/pkill/fuser
resolve inside the child, so no external command path can fire). The update
check URL is a file:// path — no network socket is ever opened.

Run:  cd b70-launcher && python3 -m unittest discover -s tests -v
"""
import http.client
import json
import re
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

try:
    import support_env
except ImportError:
    from tests import support_env

ROOT = support_env.ROOT
TMP = support_env.TMPROOT
free_port = support_env.free_port


class TestServer(unittest.TestCase):
    """One shared server process for the class; test_zz_* shuts it down."""

    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        cls.home = TMP / "srv-home"
        cls.state = TMP / "srv-state"
        cls.home.mkdir(exist_ok=True)
        cls.state.mkdir(exist_ok=True)
        cls.log_path = TMP / "server.log"
        cls.logf = open(cls.log_path, "wb")
        env = support_env.server_env(cls.home, cls.state)
        cls.proc = subprocess.Popen(
            [sys.executable, str(ROOT / "launcher.py"),
             "--no-open", "--port", str(cls.port)],
            stdout=cls.logf, stderr=subprocess.STDOUT,
            cwd=str(ROOT), env=env)
        cls.token = None
        cls._wait_ready()
        cls._fetch_token()
        # artifacts dir used by the settings/scan test
        cls.artdir = TMP / "srv-artifacts"
        support_env.write_file(
            cls.artdir / "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "proc", None) is not None and cls.proc.poll() is None:
            cls.proc.terminate()
            try:
                cls.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.proc.kill()
        if getattr(cls, "logf", None) is not None:
            cls.logf.close()

    @classmethod
    def _wait_ready(cls):
        deadline = time.time() + 15
        while time.time() < deadline:
            if cls.proc.poll() is not None:
                cls.logf.close()
                raise RuntimeError(
                    "server exited during startup:\n"
                    + cls.log_path.read_text(errors="replace"))
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{cls.port}/api/state", timeout=1).close()
                return
            except urllib.error.HTTPError as exc:
                if exc.code == 403:  # auth gate up = server ready
                    return
            except Exception:
                pass
            time.sleep(0.15)
        raise RuntimeError("server did not become ready in 15s")

    @classmethod
    def _fetch_token(cls):
        # GETs require auth — the token file (0600) is the out-of-band channel
        tok_file = cls.state / "b70-launcher" / "token"
        deadline = time.time() + 10
        while time.time() < deadline:
            if tok_file.is_file():
                cls.token = tok_file.read_text().strip()
                if cls.token:
                    return
            time.sleep(0.1)
        raise RuntimeError("server did not write its token file")

    # ---- helpers -------------------------------------------------------

    def get(self, path, auth=True):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}")
        if auth:
            req.add_header("X-Launcher-Token", self.token)
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read() or b"{}")

    _REAL = object()  # sentinel: send the server's real token

    def post(self, path, obj=None, token=_REAL, origin=None,
             ctype="application/json", raw=None):
        data = json.dumps(obj if obj is not None else {}).encode() \
            if raw is None else raw
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data, method="POST")
        tok = self.token if token is self._REAL else token
        if tok is not None:
            req.add_header("X-Launcher-Token", tok)
        if origin is not None:
            req.add_header("Origin", origin)
        req.add_header("Content-Type", ctype)
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read() or b"{}")
            except Exception:
                return e.code, {}
            finally:
                e.close()

    # ---- GETs ----------------------------------------------------------

    def test_state_shape(self):
        code, body = self.get("/api/state")
        self.assertEqual(code, 200)
        expected = re.search(r'^VERSION = "([^"]+)"',
                             (ROOT / "launcher.py").read_text(), re.M).group(1)
        self.assertEqual(body["version"], expected)
        self.assertEqual(len(body["recipes"]["models"]), 9)
        self.assertIn("settings", body)
        self.assertIn("preflight", body)
        self.assertIn("blockers", body["preflight"])
        # PATH was masked — docker cannot be found inside the child
        self.assertTrue(any("Docker CLI missing" in b
                            for b in body["preflight"]["blockers"]))
        self.assertIn("scan", body)
        self.assertIn("recipe_notices", body)
        self.assertIsInstance(body["running"], list)

    def test_index_token_substituted(self):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/")
        req.add_header("X-Launcher-Token", self.token)
        with urllib.request.urlopen(req, timeout=5) as r:
            self.assertEqual(r.status, 200)
            self.assertIn("text/html", r.headers["Content-Type"])
            html = r.read().decode()
        self.assertNotIn("__API_TOKEN__", html)
        self.assertIn(self.token, html)
        self.assertIn("B70", html)

    def test_get_requires_auth(self):
        # the API token must not be served to arbitrary same-box processes:
        # unauthenticated GETs (incl. the index that embeds it) get 403
        for path in ("/", "/api/state", "/api/power", "/api/servers",
                     "/api/scan", "/api/metrics", "/api/usage",
                     "/assets/b70-launcher.svg", "/nope"):
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}{path}", timeout=5)
            cm.exception.close()
            self.assertEqual(cm.exception.code, 403, path)

    def test_get_bootstrap_cookie_flow(self):
        # /?token=... hands off a session cookie and strips the token via 302
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", f"/?token={self.token}")
        resp = conn.getresponse()
        resp.read()
        self.assertEqual(resp.status, 302)
        self.assertEqual(resp.getheader("Location"), "/")
        cookie = resp.getheader("Set-Cookie") or ""
        conn.close()
        self.assertIn("b70_token=", cookie)
        self.assertIn("HttpOnly", cookie)
        # the cookie alone must now authorize GETs
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/state")
        req.add_header("Cookie", cookie.split(";")[0])
        with urllib.request.urlopen(req, timeout=5) as r:
            self.assertEqual(r.status, 200)

    def test_get_wrong_token_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/?token=definitely-wrong", timeout=5)
        cm.exception.close()
        self.assertEqual(cm.exception.code, 403)

    def test_get_404s(self):
        for path in ("/nope", "/api/logs?id=missing",
                     "/assets/../launcher.py", "/assets/../recipes.json"):
            with self.assertRaises(urllib.error.HTTPError) as cm:
                self.get(path)
            cm.exception.close()
            self.assertEqual(cm.exception.code, 404, path)

    def test_bad_host_header_rejected(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("GET", "/api/state", skip_host=True)
        conn.putheader("Host", "evil.example.com")
        conn.endheaders()
        resp = conn.getresponse()
        self.assertEqual(resp.status, 403)
        resp.read()
        conn.close()

    def test_metrics_and_usage_and_downloads(self):
        code, body = self.get("/api/metrics")
        self.assertEqual(code, 200)
        self.assertIn("power", body)
        self.assertIn("vram", body)
        code, body = self.get("/api/usage")
        self.assertEqual(code, 200)
        self.assertIn("sessions", body)
        self.assertIn("totals", body)
        code, body = self.get("/api/downloads")
        self.assertEqual(code, 200)
        self.assertEqual(body, {})

    # ---- POST auth ------------------------------------------------------

    def test_post_requires_token(self):
        for kw in ({"token": None},               # header absent
                   {"token": "wrong-token"},      # wrong value
                   {"token": self.token, "origin": "http://evil.example"},
                   {"token": self.token, "ctype": "text/plain"}):
            code, body = self.post("/api/stop", {"id": "x"}, **kw)
            self.assertEqual(code, 403, kw)
            self.assertIn("unauthorized", body["error"])

    def test_bad_bodies(self):
        self.assertEqual(self.post("/api/build", raw=b"{not json")[0], 400)
        self.assertEqual(self.post("/api/build", raw=b"[1,2]")[0], 400)
        self.assertEqual(self.post("/api/build", raw=b"x" * 70000)[0], 400)

    # ---- functional POSTs ----------------------------------------------

    def test_build_dry(self):
        code, b = self.post("/api/build", {
            "model_id": "qwen36-35b", "engine": "llamacpp",
            "ctx": 8192, "port": free_port()})
        self.assertEqual(code, 200)
        self.assertIn("docker", b["cmd"])
        self.assertIn("--cache-type-k", b["cmd"])
        self.assertIn("-ngl", b["cmd"])
        self.assertTrue(any("not detected" in w for w in b["warnings"]))
        self.assertIn("harness_line", b)

    def test_build_unknown_model(self):
        code, b = self.post("/api/build", {"model_id": "nope", "engine": "x"})
        self.assertIn("error", b)

    def test_launch_dry_run(self):
        port = free_port()
        code, out = self.post("/api/launch", {
            "model_id": "qwen36-35b", "engine": "llamacpp",
            "dry_run": True, "ctx": 8192, "port": port})
        self.assertEqual(code, 200)
        self.assertTrue(out["id"].startswith("qwen36-35b-llamacpp-"))
        self.assertIn("docker", out["cmd"])
        self.assertFalse(out["detected"])
        # dry-runs are pure previews: no tracked entry, no state pollution
        _, state = self.get("/api/state")
        match = [e for e in state["running"] if e["id"] == out["id"]]
        self.assertEqual(len(match), 0)

    def test_launch_real_blocked_by_preflight(self):
        # no dry_run -> preflight runs; with PATH masked the docker blocker
        # guarantees a 400 regardless of hardware
        code, out = self.post("/api/launch", {
            "model_id": "qwen36-35b", "engine": "llamacpp",
            "ctx": 8192, "port": free_port()})
        self.assertEqual(code, 400)
        self.assertIn("error", out)

    def test_recipes_update_error(self):
        code, out = self.post("/api/recipes/update", {})
        self.assertEqual(code, 400)
        self.assertIn("no remote recipes_url", out["error"])

    def test_download_bad_model(self):
        code, out = self.post("/api/download", {"model_id": "nope",
                                                "engine": "x"})
        self.assertEqual(code, 400)
        self.assertIn("no such model+engine", out["error"])

    def test_stop_unknown(self):
        code, out = self.post("/api/stop", {"id": "never-existed"})
        self.assertEqual(code, 200)
        self.assertFalse(out["ok"])

    def test_test_prompt_errors(self):
        code, out = self.post("/api/test_prompt", {"port": "bogus"})
        self.assertEqual(code, 400)
        code, out = self.post("/api/test_prompt", {"port": free_port()})
        self.assertEqual(code, 200)
        self.assertFalse(out["ok"])
        self.assertIn("No inference server responding", out["error"])

    def test_settings_and_scan_flow(self):
        # point a scan root at the fixture artifact, rescan, verify detection
        code, out = self.post("/api/settings",
                              {"scan_dirs": [str(self.artdir)]})
        self.assertEqual(code, 200)
        self.assertTrue(out["ok"])
        code, body = self.get("/api/scan")
        self.assertEqual(code, 200)
        self.assertEqual(body["state"], "done")
        names = [c["name"] for c in body["catalog"]]
        self.assertIn("qwen3.6-35b-a3b-ud-q4_k_xl.gguf", names)
        match = body["matches"]["qwen36-35b"]["llamacpp"]
        self.assertTrue(match["detected"])
        self.assertIn("qwen3.6-35b-a3b-ud-q4_k_xl.gguf", match["path"].lower())
        # a dry-run launch now detects the artifact and mounts its directory
        code, out = self.post("/api/launch", {
            "model_id": "qwen36-35b", "engine": "llamacpp",
            "dry_run": True, "ctx": 8192, "port": free_port()})
        self.assertEqual(code, 200)
        self.assertTrue(out["detected"])
        self.assertIn(str(self.artdir), out["cmd"])

    def test_settings_rejects_dangerous_roots(self):
        for bad in (["/"], "x", [{"bad": 1}], ["x" * 600]):
            code, out = self.post("/api/settings", {"scan_dirs": bad})
            self.assertEqual(code, 400, bad)

    # ---- shutdown (alphabetically last) ---------------------------------

    def test_zz_shutdown(self):
        code, out = self.post("/api/shutdown", {"stop_engines": False})
        self.assertEqual(code, 200)
        self.assertTrue(out["ok"])
        rc = self.proc.wait(timeout=15)
        self.assertEqual(rc, 0)
        with self.assertRaises(Exception):
            urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/api/state", timeout=2)


if __name__ == "__main__":
    unittest.main()
